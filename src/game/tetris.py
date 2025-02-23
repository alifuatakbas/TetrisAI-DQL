import pygame
import random

# Tetris parçaları (I, O, T, S, Z, J, L)
SHAPES = [
    [[1, 1, 1, 1]],  # I
    [[1, 1],  # O
     [1, 1]],
    [[0, 1, 0],  # T
     [1, 1, 1]],
    [[0, 1, 1],  # S
     [1, 1, 0]],
    [[1, 1, 0],  # Z
     [0, 1, 1]],
    [[1, 0, 0],  # J
     [1, 1, 1]],
    [[0, 0, 1],  # L
     [1, 1, 1]]
]

COLORS = [
    (0, 255, 255),  # Cyan (I)
    (255, 255, 0),  # Yellow (O)
    (128, 0, 128),  # Purple (T)
    (0, 255, 0),  # Green (S)
    (255, 0, 0),  # Red (Z)
    (0, 0, 255),  # Blue (J)
    (255, 128, 0)  # Orange (L)
]


class Tetris:
    def __init__(self):
        self.width = 10
        self.height = 20
        self.block_size = 30
        self.board = [[0 for _ in range(self.width)] for _ in range(self.height)]
        self.colors = [[None for _ in range(self.width)] for _ in range(self.height)]  # Renk bilgisi
        self.current_piece = self.new_piece()
        self.game_over = False
        self.score = 0
        self.lines_cleared = 0
        self.level = 1

    def new_piece(self):
        # Yeni parça oluştur
        shape_idx = random.randint(0, len(SHAPES) - 1)
        return {
            'shape': SHAPES[shape_idx],
            'color': COLORS[shape_idx],
            'x': self.width // 2 - len(SHAPES[shape_idx][0]) // 2,
            'y': 0
        }

    def move(self, dx=0, dy=0):
        # Parçayı hareket ettir
        self.current_piece['x'] += dx
        self.current_piece['y'] += dy
        if not self.is_valid_position():
            self.current_piece['x'] -= dx
            self.current_piece['y'] -= dy
            if dy > 0:  # Aşağı hareket edemiyorsa, parçayı sabitle
                self.freeze_piece()
                self.clear_lines()
                self.current_piece = self.new_piece()
                if not self.is_valid_position():
                    self.game_over = True

    def rotate(self):
        # Parçayı döndür
        shape = self.current_piece['shape']
        self.current_piece['shape'] = list(zip(*shape[::-1]))  # 90 derece döndür
        if not self.is_valid_position():
            self.current_piece['shape'] = shape

    def is_valid_position(self):
        # Parçanın pozisyonu geçerli mi kontrol et
        for y, row in enumerate(self.current_piece['shape']):
            for x, cell in enumerate(row):
                if cell:
                    new_x = self.current_piece['x'] + x
                    new_y = self.current_piece['y'] + y
                    if (new_x < 0 or new_x >= self.width or
                            new_y >= self.height or
                            (new_y >= 0 and self.board[new_y][new_x])):
                        return False
        return True

    def freeze_piece(self):
        for y, row in enumerate(self.current_piece['shape']):
            for x, cell in enumerate(row):
                if cell:
                    board_y = self.current_piece['y'] + y
                    board_x = self.current_piece['x'] + x
                    if board_y >= 0:
                        self.board[board_y][board_x] = 1
                        self.colors[board_y][board_x] = self.current_piece['color']  # Rengi kaydet

    def clear_lines(self):
        # Dolu satırları temizle
        lines_cleared = 0
        y = self.height - 1
        while y >= 0:
            if all(self.board[y]):  # Satır doluysa
                lines_cleared += 1
                # Üstteki satırları aşağı kaydır
                for ny in range(y, 0, -1):
                    self.board[ny] = self.board[ny - 1][:]
                # En üst satırı temizle
                self.board[0] = [0] * self.width
            else:
                y -= 1

        # Skor hesapla
        if lines_cleared > 0:
            self.score += [0, 40, 100, 300, 1200][lines_cleared]
            return True
        return False

    def copy(self):
        # Mevcut durumun kopyasını oluştur
        new_piece = {
            'shape': [row[:] for row in self.current_piece['shape']],
            'color': self.current_piece['color'],
            'x': self.current_piece['x'],
            'y': self.current_piece['y']
        }
        return new_piece

    def get_board_copy(self):
        # Tahtanın kopyasını oluştur
        return [row[:] for row in self.board]