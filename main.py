import pygame
import sys
import threading
import queue
import time
from src.game.tetris import Tetris
from src.ai.agent import TetrisAgent
import random
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np


class TetrisGame:
    def __init__(self):
        pygame.init()
        self.width = 1200
        self.height = 800
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("AI Tetris")
        self.clock = pygame.time.Clock()
        self.tetris = Tetris()
        self.agent = TetrisAgent()
        self.training_mode = False
        self.game_x = 100
        self.game_y = 50
        self.high_score = 0
        self.save_interval = 50  # Her 50 oyunda bir kaydet
        self.last_save_status = None  # Son kayıt durumunu tut
        # Metrikler için yeni değişkenler
        self.scores = []
        self.lines_cleared_history = []
        self.epsilon_history = []
        self.avg_scores = []  # Her 100 oyundaki ortalama skor
        self.training_start_time = datetime.now()

    def ai_worker(self):

        while True:
            try:
                if not self.ai_queue.empty():
                    state = self.ai_queue.get()
                    action = self.agent.get_action(state)
                    self.action_queue.put(action)
                time.sleep(0.01)  # CPU kullanımını azalt
            except Exception as e:
                print(f"AI Worker Error: {e}")

    def run(self):
        episode = 0
        fall_speed = 50
        last_fall = pygame.time.get_ticks()
        ai_move_delay = 100
        last_ai_move = current_time = pygame.time.get_ticks()

        # State ve action değişkenlerini sınıf değişkeni yapalım
        self.current_state = None
        self.current_action = None

        while True:
            current_time = pygame.time.get_ticks()
            delta_time = current_time - last_fall
            ai_delta = current_time - last_ai_move

            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("\nOyun kapatılıyor... Son durum kaydediliyor...")
                    self.save_training_graphs()  # Grafikleri kaydet
                    self.agent.save_agent()
                    pygame.quit()
                    sys.exit()

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_t:
                        self.training_mode = not self.training_mode
                        print(f"Training Mode: {'ON' if self.training_mode else 'OFF'}")

            # AI kontrolü
            if self.training_mode and not self.tetris.game_over and ai_delta > ai_move_delay:
                # State ve action al
                self.current_state = self.agent.get_state(self.tetris.board)

                # Epsilon-greedy ile action seç
                if random.random() < self.agent.epsilon:
                    self.current_action = random.randint(0, 39)
                else:
                    self.current_action = self.agent.get_action(self.current_state)

                # Hamleyi uygula
                moved = False
                rotation = self.current_action // 10
                target_x = self.current_action % 10

                if rotation > 0:
                    self.tetris.rotate()
                    moved = True
                elif self.tetris.current_piece['x'] < target_x:
                    self.tetris.move(dx=1)
                    moved = True
                elif self.tetris.current_piece['x'] > target_x:
                    self.tetris.move(dx=-1)
                    moved = True

                if moved:
                    last_ai_move = current_time

            # Düşme kontrolü
            if delta_time > fall_speed and not self.tetris.game_over:
                self.tetris.move(dy=1)
                last_fall = current_time

            # Game Over kontrolü
            if self.tetris.game_over:
                if self.training_mode:
                    episode += 1
                    next_state = self.agent.get_state(self.tetris.board)
                    reward = self.tetris.score
                    # Metrikleri güncelle
                    self.scores.append(self.tetris.score)
                    self.lines_cleared_history.append(getattr(self.tetris, 'lines_cleared', 0))
                    self.epsilon_history.append(self.agent.epsilon)

                    # Her 1000 episode'da grafikleri kaydet
                    if episode % 1000 == 0:
                        self.save_training_graphs()
                    print("\n=== GAME OVER ===")
                    print(f"Episode: {episode}")
                    print(f"Score: {reward}")
                    print(f"Current State: {self.current_state}")
                    print(f"Current Action: {self.current_action}")

                    # Deneyimi kaydet
                    if self.current_state is not None and self.current_action is not None:
                        self.agent.remember(self.current_state, self.current_action,
                                            reward, next_state, True)
                        print(f"Memory Size: {len(self.agent.memory)}")

                        # Epsilon güncelle
                        if self.agent.epsilon > self.agent.epsilon_min:
                            old_epsilon = self.agent.epsilon
                            self.agent.epsilon *= self.agent.epsilon_decay
                            print(f"Epsilon: {old_epsilon:.3f} -> {self.agent.epsilon:.3f}")

                            # Eğitim kontrolü
                            if len(self.agent.memory) >= self.agent.min_memory_size:
                                print("\nYeterli deneyim var, eğitim başlıyor...")
                                self.agent.train(batch_size=64)
                            else:
                                print(
                                    f"\nEğitim için {self.agent.min_memory_size - len(self.agent.memory)} deneyim daha gerekli")

                    # Her 50 oyunda bir kaydet
                    if episode % 50 == 0:
                        print("\nOtomatik kayıt yapılıyor...")
                        self.agent.save_agent()

                    # Yeni oyun için hazırlık
                    self.high_score = max(self.high_score, self.tetris.score)
                    self.tetris = Tetris()
                    self.current_state = None
                    self.current_action = None

            # Ekranı güncelle
            self.screen.fill((0, 0, 0))
            self.draw_game()
            pygame.display.flip()
            self.clock.tick(60)

    def draw_game(self):
        # Oyun alanı arkaplanı
        pygame.draw.rect(
            self.screen,
            (50, 50, 50),
            (self.game_x - 5, self.game_y - 5,
             self.tetris.width * self.tetris.block_size + 10,
             self.tetris.height * self.tetris.block_size + 10)
        )

        # Mevcut parçayı çiz
        piece = self.tetris.current_piece
        for y, row in enumerate(piece['shape']):
            for x, cell in enumerate(row):
                if cell:
                    pygame.draw.rect(
                        self.screen,
                        piece['color'],
                        (self.game_x + (piece['x'] + x) * self.tetris.block_size,
                         self.game_y + (piece['y'] + y) * self.tetris.block_size,
                         self.tetris.block_size - 1,
                         self.tetris.block_size - 1)
                    )

        # Yerleşmiş parçaları çiz
        for y in range(self.tetris.height):
            for x in range(self.tetris.width):
                if self.tetris.board[y][x]:
                    pygame.draw.rect(
                        self.screen,
                        (128, 128, 128),
                        (self.game_x + x * self.tetris.block_size,
                         self.game_y + y * self.tetris.block_size,
                         self.tetris.block_size - 1,
                         self.tetris.block_size - 1)
                    )

            # Menü paneli
            menu_x = self.game_x + self.tetris.width * self.tetris.block_size + 100  # Boşluğu artırdık
            menu_y = self.game_y
            menu_width = 300  # Genişliği artırdık
            menu_height = 600  # Yüksekliği artırdık

            # Menü arkaplanı
            pygame.draw.rect(
                self.screen,
                (50, 50, 50),
                (menu_x, menu_y, menu_width, menu_height)
            )

            # Font ayarları
            title_font = pygame.font.Font(None, 48)  # Font boyutunu artırdık
            text_font = pygame.font.Font(None, 36)  # Font boyutunu artırdık

            # Başlık
            title = title_font.render("STATISTICS", True, (255, 255, 255))
            title_rect = title.get_rect(centerx=menu_x + menu_width // 2, y=menu_y + 30)
            self.screen.blit(title, title_rect)

            # Bilgileri yazdır
            texts = [
                ("Score", str(self.tetris.score)),
                ("High Score", str(self.high_score)),
                ("Lines", str(getattr(self.tetris, 'lines_cleared', 0))),
                ("Level", str(getattr(self.tetris, 'level', 1))),
                ("Mode", "AI Training" if self.training_mode else "Manual"),
                ("Epsilon", f"{self.agent.epsilon:.3f}" if self.training_mode else "N/A"),
                ("Experience", f"{len(self.agent.memory)}/2000" if self.training_mode else "N/A")

            ]

            y_offset = 120  # Başlangıç y pozisyonunu artırdık
            for label, value in texts:
                # Etiket
                text_surface = text_font.render(label + ":", True, (200, 200, 200))
                text_rect = text_surface.get_rect(x=menu_x + 30, y=menu_y + y_offset)
                self.screen.blit(text_surface, text_rect)

                # Değer
                value_surface = text_font.render(value, True, (255, 255, 255))
                value_rect = value_surface.get_rect(x=menu_x + 30, y=menu_y + y_offset + 35)
                self.screen.blit(value_surface, value_rect)

                y_offset += 80  # Aralığı artırdık

            # Ayırıcı çizgi
            pygame.draw.line(
                self.screen,
                (200, 200, 200),
                (menu_x + 20, menu_y + y_offset),
                (menu_x + menu_width - 20, menu_y + y_offset),
                2
            )

            # Kontrol bilgileri
            controls_y = menu_y + y_offset + 20
            controls_text = [
                "CONTROLS",
                "",
                "T - Toggle AI",
                "← → - Move",
                "↑ - Rotate",
                "↓ - Soft Drop"
            ]

            for i, text in enumerate(controls_text):
                if i == 0:  # Başlık için farklı font
                    text_surface = text_font.render(text, True, (255, 255, 255))
                else:
                    text_surface = text_font.render(text, True, (200, 200, 200))
                text_rect = text_surface.get_rect(x=menu_x + 30, y=controls_y + i * 35)
                self.screen.blit(text_surface, text_rect)
            status_text = ""
            if self.agent.is_loaded:
                status_text = "Model: Kayıtlı Model (Yüklendi)"
            else:
                status_text = "Model: Yeni Model"

            if self.last_save_status is not None:
                if self.last_save_status:
                    status_text += " | Son Kayıt: Başarılı"
                else:
                    status_text += " | Son Kayıt: HATA!"

            text_surface = text_font.render(status_text, True, (255, 255, 255))
            text_rect = text_surface.get_rect(x=menu_x + 30, y=menu_y + y_offset + 150)
            self.screen.blit(text_surface, text_rect)
            # Reset butonu
            reset_button_x = menu_x + menu_width + 100  # Daha sağa aldık
            reset_button_y = menu_y + menu_height - 600
            reset_button_width = 250  # Biraz daha dar yaptık
            reset_button_height = 40

            # Buton arkaplanı
            button_color = (200, 50, 50) if self.training_mode else (100, 100, 100)
            pygame.draw.rect(
                self.screen,
                button_color,
                (reset_button_x, reset_button_y, reset_button_width, reset_button_height)
            )

            # Buton metni
            reset_text = text_font.render("RESET TRAINING", True, (255, 255, 255))
            text_rect = reset_text.get_rect(
                center=(reset_button_x + reset_button_width // 2,
                        reset_button_y + reset_button_height // 2)
            )
            self.screen.blit(reset_text, text_rect)

            # Onay penceresi değişkenleri (sınıf değişkeni olarak ekleyin)
            if not hasattr(self, 'show_confirm'):
                self.show_confirm = False
                self.confirm_time = 0

            # Mouse tıklama kontrolü
            mouse_pos = pygame.mouse.get_pos()
            mouse_click = pygame.mouse.get_pressed()[0]

            # Mouse butonun üzerinde mi?
            if (reset_button_x <= mouse_pos[0] <= reset_button_x + reset_button_width and
                    reset_button_y <= mouse_pos[1] <= reset_button_y + reset_button_height):
                # Hover efekti
                pygame.draw.rect(
                    self.screen,
                    (min(button_color[0] + 30, 255),
                     min(button_color[1] + 30, 255),
                     min(button_color[2] + 30, 255)),
                    (reset_button_x, reset_button_y, reset_button_width, reset_button_height),
                    3
                )

                # Tıklama kontrolü
                if mouse_click and self.training_mode and not self.show_confirm:
                    self.show_confirm = True
                    self.confirm_time = pygame.time.get_ticks()

            # Onay penceresi gösterimi
            if self.show_confirm:
                # Yarı saydam siyah overlay
                overlay = pygame.Surface((self.width, self.height))
                overlay.fill((0, 0, 0))
                overlay.set_alpha(128)
                self.screen.blit(overlay, (0, 0))

                # Onay penceresi
                confirm_width = 400
                confirm_height = 200
                confirm_x = self.width // 2 - confirm_width // 2
                confirm_y = self.height // 2 - confirm_height // 2

                # Pencere arkaplanı
                pygame.draw.rect(
                    self.screen,
                    (50, 50, 50),
                    (confirm_x, confirm_y, confirm_width, confirm_height)
                )

                # Uyarı metni
                warning_text = text_font.render("Tüm eğitim verilerini silmek", True, (255, 255, 255))
                warning_text2 = text_font.render("istediğinizden emin misiniz?", True, (255, 255, 255))
                text_rect = warning_text.get_rect(centerx=self.width // 2, y=confirm_y + 40)
                text_rect2 = warning_text2.get_rect(centerx=self.width // 2, y=confirm_y + 80)
                self.screen.blit(warning_text, text_rect)
                self.screen.blit(warning_text2, text_rect2)

                # Evet/Hayır butonları
                yes_button = pygame.Rect(confirm_x + 50, confirm_y + 130, 120, 40)
                no_button = pygame.Rect(confirm_x + 230, confirm_y + 130, 120, 40)

                pygame.draw.rect(self.screen, (200, 50, 50), yes_button)
                pygame.draw.rect(self.screen, (100, 100, 100), no_button)

                yes_text = text_font.render("EVET", True, (255, 255, 255))
                no_text = text_font.render("HAYIR", True, (255, 255, 255))

                yes_rect = yes_text.get_rect(center=yes_button.center)
                no_rect = no_text.get_rect(center=no_button.center)

                self.screen.blit(yes_text, yes_rect)
                self.screen.blit(no_text, no_rect)

                # Buton tıklama kontrolü
                if mouse_click:
                    if yes_button.collidepoint(mouse_pos):
                        print("\n=== TÜM KAYITLAR SİLİNİYOR ===")
                        if os.path.exists(f"{self.agent.save_dir}/tetris_model.weights.h5"):
                            os.remove(f"{self.agent.save_dir}/tetris_model.weights.h5")
                        if os.path.exists(f"{self.agent.save_dir}/agent_data.pkl"):
                            os.remove(f"{self.agent.save_dir}/agent_data.pkl")
                        self.agent = TetrisAgent()
                        print("Yeni eğitim başlatıldı!")
                        print("==========================\n")
                        self.show_confirm = False

                    elif no_button.collidepoint(mouse_pos):
                        self.show_confirm = False

    def save_training_graphs(self):
        """Visualize training metrics and save"""
        current_episode = len(self.scores)
        # Episode aralığını belirle
        start_episode = max(0, current_episode - 1000)

        plt.style.use('dark_background')

        # 1. Score Progress
        plt.figure(figsize=(12, 6))
        plt.plot(self.scores[start_episode:current_episode],
                 color='cyan', alpha=0.3, label='Each Game')
        if len(self.scores[start_episode:current_episode]) > 100:
            avg_scores = [np.mean(self.scores[i:i + 100])
                          for i in range(start_episode, current_episode - 100, 100)]
            plt.plot(range(50, current_episode - start_episode - 50, 100), avg_scores,
                     color='cyan', linewidth=2, label='100 Games Average')
        plt.title(f'Score Progress (Episodes {start_episode + 1}-{current_episode})', pad=20)
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True, alpha=0.2)
        plt.savefig(f'training_scores_{start_episode + 1}_{current_episode}.png',
                    bbox_inches='tight', dpi=300)
        plt.close()

        # 2. Epsilon Decay
        plt.figure(figsize=(12, 6))
        plt.plot(self.epsilon_history[start_episode:current_episode],
                 color='magenta', label='Epsilon')
        plt.title(f'Epsilon Decay (Episodes {start_episode + 1}-{current_episode})', pad=20)
        plt.xlabel('Episode')
        plt.ylabel('Epsilon Value')
        plt.legend()
        plt.grid(True, alpha=0.2)
        plt.savefig(f'epsilon_decay_{start_episode + 1}_{current_episode}.png',
                    bbox_inches='tight', dpi=300)
        plt.close()

        # 3. Lines Cleared
        plt.figure(figsize=(12, 6))
        plt.plot(self.lines_cleared_history[start_episode:current_episode],
                 color='green', alpha=0.3, label='Each Game')
        if len(self.lines_cleared_history[start_episode:current_episode]) > 100:
            avg_lines = [np.mean(self.lines_cleared_history[i:i + 100])
                         for i in range(start_episode, current_episode - 100, 100)]
            plt.plot(range(50, current_episode - start_episode - 50, 100), avg_lines,
                     color='green', linewidth=2, label='100 Games Average')
        plt.title(f'Lines Cleared Progress (Episodes {start_episode + 1}-{current_episode})', pad=20)
        plt.xlabel('Episode')
        plt.ylabel('Lines Cleared')
        plt.legend()
        plt.grid(True, alpha=0.2)
        plt.savefig(f'lines_cleared_{start_episode + 1}_{current_episode}.png',
                    bbox_inches='tight', dpi=300)
        plt.close()

        # 4. Training Summary (her zaman tüm veriler)
        total_time = datetime.now() - self.training_start_time
        hours = total_time.total_seconds() / 3600

        plt.figure(figsize=(12, 6))
        plt.text(0.5, 0.8, f'Total Training Time: {hours:.1f} hours',
                 horizontalalignment='center', fontsize=12)
        plt.text(0.5, 0.6, f'Total Episodes: {current_episode}',
                 horizontalalignment='center', fontsize=12)
        plt.text(0.5, 0.4, f'Highest Score: {max(self.scores)}',
                 horizontalalignment='center', fontsize=12)
        plt.text(0.5, 0.2, f'Average Score: {np.mean(self.scores):.1f}',
                 horizontalalignment='center', fontsize=12)
        plt.axis('off')
        plt.savefig(f'training_summary_{current_episode}.png',
                    bbox_inches='tight', dpi=300)
        plt.close()


if __name__ == "__main__":
    game = TetrisGame()
    game.run()