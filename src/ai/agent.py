import numpy as np
import random
import os
import pickle
from src.ai.model import TetrisModel
import keras


class PrioritizedMemory:
    def __init__(self, maxlen):
        self.memory = []
        self.priorities = []
        self.maxlen = maxlen

    def add(self, priority, experience):
        if len(self.memory) >= self.maxlen:
            # En düşük öncelikli deneyimi sil
            min_priority_idx = np.argmin(self.priorities)
            self.memory.pop(min_priority_idx)
            self.priorities.pop(min_priority_idx)

        self.memory.append(experience)
        self.priorities.append(priority)

    def sample(self, batch_size):
        batch_size = min(batch_size, len(self.memory))
        if batch_size == 0:
            return []

        # Önceliklere göre örnekleme
        probs = np.array(self.priorities) / sum(self.priorities)
        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        return [self.memory[idx] for idx in indices]

    def __len__(self):
        return len(self.memory)


class TetrisAgent:
    def __init__(self):
        # Model ve hafıza
        self.model = TetrisModel()
        self.memory = PrioritizedMemory(maxlen=20000)  # Deque yerine PrioritizedMemory kullan
        self.elite_memory = []
        self.elite_size = 1000
        self.save_dir = "saved_model"  # save_dir'i ekledik

        # Öğrenme parametreleri
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        self.batch_size = 128
        self.min_memory_size = 980

        # Yükleme durumunu kontrol et
        self.is_loaded = self.load_agent()

    def get_exploration_action(self, state):
        """Keşif için rastgele hamle seç"""
        return random.randint(0, self.model.action_size - 1)

    def get_action(self, state):
        """Duruma göre en iyi hareketi seç"""
        if not isinstance(state, np.ndarray):
            state = np.array(state)

        if state.ndim == 1:
            state = state.reshape(1, -1)

        q_values = self.model.predict(state)
        return int(np.argmax(q_values))

    def get_random_action(self):
        """Rastgele bir hareket seç"""
        return random.randint(0, self.model.action_size - 1)

    def remember(self, state, action, reward, next_state, done):
        """Deneyimi öncelikli olarak kaydet"""
        if state is not None and action is not None:
            # State'leri düzgün formata getir
            if isinstance(state, np.ndarray):
                state = state.reshape(-1)  # (4,) boyutuna getir
            if isinstance(next_state, np.ndarray):
                next_state = next_state.reshape(-1)  # (4,) boyutuna getir

            priority = abs(reward) + 1.0
            experience = (state, action, reward, next_state, done)
            self.memory.add(priority, experience)

    def train(self, batch_size):
        """Model eğitimi"""
        if len(self.memory) < self.min_memory_size:
            print(f"Eğitim için bekleniyor... ({len(self.memory)}/{self.min_memory_size} deneyim)")
            return

        print("\n=== EĞİTİM BAŞLADI ===")
        print(f"Batch Size: {batch_size}")
        print(f"Memory Size: {len(self.memory)}")

        # Batch al
        batch = self.memory.sample(batch_size)

        try:
            # State'leri düzgün şekilde reshape et
            states = []
            actions = []
            rewards = []
            next_states = []
            dones = []

            for experience in batch:
                state, action, reward, next_state, done = experience

                # State'leri (4,) boyutuna getir
                if isinstance(state, np.ndarray):
                    state = state.reshape(-1)
                if isinstance(next_state, np.ndarray):
                    next_state = next_state.reshape(-1)

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
                dones.append(done)

            # Numpy array'lerine çevir
            states = np.array(states)
            actions = np.array(actions)
            rewards = np.array(rewards)
            next_states = np.array(next_states)
            dones = np.array(dones)

            # Debug bilgileri
            print(f"States shape: {states.shape}")
            print(f"Actions shape: {actions.shape}")

            # Q-learning güncellemesi
            current_q_values = self.model.predict(states)
            next_q_values = self.model.predict(next_states)

            # Her örnek için hedef Q değerlerini hesapla
            for i in range(batch_size):
                if dones[i]:
                    current_q_values[i][actions[i]] = rewards[i]
                else:
                    current_q_values[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])

            # Model eğitimi
            history = self.model.train(states, current_q_values)
            loss = float(history.history['loss'][0]) if isinstance(history, keras.callbacks.History) else float(history)

            print(f"Loss: {loss:.4f}")
            print("Eğitim tamamlandı!")
            print("===================\n")

            # Epsilon güncelle
            self.epsilon = max(self.epsilon_min,
                               self.epsilon * self.epsilon_decay)
            # Debug bilgileri ekleyelim
            print("\n=== TRAINING DEBUG ===")
            print(f"Sample State: {states[0]}")  # Örnek bir state
            print(f"Sample Reward: {rewards[0]}")  # Örnek bir reward
            print(f"Q-values range: {np.min(current_q_values):.4f} to {np.max(current_q_values):.4f}")
            print(f"Average Q-value: {np.mean(current_q_values):.4f}")
            print("=====================\n")

            # Model eğitimi
            history = self.model.train(states, current_q_values)
            loss = float(history.history['loss'][0]) if isinstance(history, keras.callbacks.History) else float(history)

            print(f"Loss: {loss:.4f}")
            print(f"Epsilon: {self.epsilon:.4f}")

        except Exception as e:
            print(f"Training Error: {e}")
            print("States sample:", states[0] if len(states) > 0 else "No states")
            print("Next states sample:", next_states[0] if len(next_states) > 0 else "No next states")

    def save_agent(self):
        """Model ve deneyim verilerini kaydet"""
        try:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

            # Model ağırlıklarını kaydet
            self.model.save_weights(f"{self.save_dir}/tetris_model.weights.h5")

            # Diğer verileri kaydet
            agent_data = {
                'memory': self.memory.memory,  # PrioritizedMemory'nin memory listesi
                'priorities': self.memory.priorities,  # PrioritizedMemory'nin priorities listesi
                'elite_memory': self.elite_memory,
                'epsilon': self.epsilon
            }

            with open(f"{self.save_dir}/agent_data.pkl", 'wb') as f:
                pickle.dump(agent_data, f)

            print("\n=== KAYIT BAŞARILI ===")
            print(f"Epsilon: {self.epsilon:.3f}")
            print(f"Memory Size: {len(self.memory)}")
            print("=====================\n")
            return True

        except Exception as e:
            print(f"\n!!! KAYIT HATASI: {e} !!!\n")
            return False

    def load_agent(self):
        """Kaydedilmiş model ve deneyim verilerini yükle"""
        try:
            # Dosyaların varlığını kontrol et
            model_exists = os.path.exists(f"{self.save_dir}/tetris_model.weights.h5")
            data_exists = os.path.exists(f"{self.save_dir}/agent_data.pkl")

            if not model_exists or not data_exists:
                print("\n=== KAYITLI MODEL BULUNAMADI ===")
                print("Yeni model ile başlatılıyor...")
                print("================================\n")
                return False

            # Model ağırlıklarını yükle
            self.model.load_weights(f"{self.save_dir}/tetris_model.weights.h5")

            # Diğer verileri yükle
            with open(f"{self.save_dir}/agent_data.pkl", 'rb') as f:
                agent_data = pickle.load(f)  # dump yerine load kullanıyoruz

                # PrioritizedMemory'yi yeniden oluştur
                self.memory = PrioritizedMemory(maxlen=20000)
                self.memory.memory = agent_data['memory']
                self.memory.priorities = agent_data['priorities']

                # Diğer verileri yükle
                self.elite_memory = agent_data['elite_memory']
                self.epsilon = agent_data['epsilon']

            print("\n=== KAYITLI MODEL YÜKLENDİ ===")
            print(f"Epsilon: {self.epsilon:.3f}")
            print(f"Memory Size: {len(self.memory)}")
            print("==============================\n")
            return True

        except Exception as e:
            print(f"\n!!! YÜKLEME HATASI: {e} !!!")
            print("Yeni model ile başlatılıyor...")
            print("==========================\n")
            return False

    def get_reward(self, board, game_over):
        reward = 0

        if not game_over:
            # Temel ödül
            reward += 2.0  # Arttırdık

            # Yükseklik kontrolü
            max_height = self.get_max_height(board)
            if max_height < 10:
                reward += 10.0  # Daha yüksek ödül
            elif max_height > 15:
                reward -= 15.0  # Daha sert ceza

            # Boşluk kontrolü
            holes = self.get_holes(board)
            if holes > 5:  # Çok fazla boşluk varsa
                reward -= holes * 5.0  # Daha sert ceza

            # Yüzey düzgünlüğü
            bumpiness = self.get_bumpiness(board)
            if bumpiness > 3:  # Yüzey çok düzgün değilse
                reward -= bumpiness * 2.0

            # Satır temizleme
            lines = self._get_complete_lines(board)
            if lines > 0:
                reward += lines * 50.0  # Çok daha yüksek ödül

        else:
            reward -= 50.0  # Game over cezasını arttırdık

        return reward

    def get_state(self, board):
        """Başlangıç için daha basit state"""
        board = np.array(board)

        # Özellikleri hesapla
        max_height = self.get_max_height(board)
        holes = self.get_holes(board)
        bumpiness = self.get_bumpiness(board)
        lines = self._get_complete_lines(board)  # Yeni özellik

        # State vektörü (1, 4) şeklinde
        state = np.array([
            max_height / 20.0,  # Yükseklik
            holes / 10.0,  # Boşluklar
            bumpiness / 10.0,  # Yüzey düzgünlüğü
            lines / 4.0  # Tamamlanan satırlar
        ]).reshape(1, -1)

        return state

    def get_max_height(self, board):
        """Yığının maksimum yüksekliğini hesapla"""
        # Listeyi numpy array'e çevir
        board = np.array(board)

        # Her sütunun en üst bloğunu bul
        heights = []
        for col in range(len(board[0])):  # Sütun sayısı
            for row in range(len(board)):  # Satır sayısı
                if board[row][col] > 0:  # Blok bulundu
                    heights.append(len(board) - row)
                    break
            else:  # Sütun boş
                heights.append(0)
        return max(heights) if heights else 0

    def _get_heights(self, board):
        """Her sütunun yüksekliğini hesapla"""
        heights = []
        for col in range(len(board[0])):  # Her sütun için
            for row in range(len(board)):  # Yukarıdan aşağıya kontrol
                if board[row][col]:  # Dolu hücre bulundu
                    heights.append(len(board) - row)
                    break
            else:  # Sütun boş
                heights.append(0)
        return heights

    def _get_holes(self, board):
        """Boşlukları say (üstü dolu altı boş olan yerler)"""
        holes = 0
        for col in range(len(board[0])):
            block_found = False
            for row in range(len(board)):
                if board[row][col]:
                    block_found = True
                elif block_found:
                    holes += 1
        return holes

    def get_bumpiness(self, board):
        """Yüzey düzgünlüğünü hesapla"""
        board = np.array(board)
        heights = []
        for col in range(len(board[0])):
            for row in range(len(board)):
                if board[row][col] > 0:
                    heights.append(len(board) - row)
                    break
            else:
                heights.append(0)

        bumpiness = 0
        for i in range(len(heights) - 1):
            bumpiness += abs(heights[i] - heights[i + 1])
        return bumpiness

    def _get_complete_lines(self, board):
        """Tamamlanmış satırları say"""
        lines = 0
        for row in board:
            if all(cell for cell in row):  # Satır tamamen dolu
                lines += 1
        return lines

    def get_holes(self, board):
        """Boşlukları say"""
        board = np.array(board)
        return self._get_holes(board)  # Zaten var olan _get_holes'u kullanalım