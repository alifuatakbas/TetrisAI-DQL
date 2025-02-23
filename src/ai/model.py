import tensorflow as tf
import numpy as np
import keras
import os
import threading
from functools import lru_cache

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class TetrisModel:
    def __init__(self):
        self.state_size = 4
        self.action_size = 40
        self.model = self._build_model()
        self.prediction_cache = {}
        self.cache_lock = threading.Lock()

    def save_weights(self, filepath):
        """Model ağırlıklarını kaydet"""
        self.model.save_weights(filepath)

    def load_weights(self, filepath):
        """Model ağırlıklarını yükle"""
        self.model.load_weights(filepath)

    def _build_model(self):
        inputs = keras.Input(shape=(self.state_size,))
        x = keras.layers.Dense(64, activation='relu')(inputs)
        x = keras.layers.Dense(64, activation='relu')(x)
        outputs = keras.layers.Dense(self.action_size, activation='linear')(x)

        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse'
        )

        # Warmup
        dummy_state = np.zeros((1, self.state_size))
        model.predict(dummy_state, verbose=0)

        return model

    @lru_cache(maxsize=1000)
    def _cached_predict(self, state_tuple):
        """Önbellekli tahmin"""
        state_array = np.array(state_tuple).reshape(1, -1)
        return tuple(self.model.predict(state_array, verbose=0)[0])

    def predict(self, state):
        """Hızlı ve önbellekli tahmin"""
        try:
            if not isinstance(state, np.ndarray):
                state = np.array(state)

            if state.ndim == 1:
                # Tek durum için
                state = state.reshape(1, -1)  # (1, state_size) şekline getir
                return self.model.predict(state, verbose=0)[0]  # İlk (ve tek) tahmini döndür
            else:
                # Batch tahmin
                return self.model.predict(state, verbose=0)  # Tüm batch tahminlerini döndür

        except Exception as e:
            print(f"Predict Error: {e}")
            if state.ndim == 1:
                return np.zeros(self.action_size)
            else:
                return np.zeros((state.shape[0], self.action_size))

    def train(self, states, targets):
        """Modeli eğit ve önbelleği temizle"""
        history = self.model.fit(states, targets, verbose=0, batch_size=32)
        self._cached_predict.cache_clear()
        return history.history['loss'][0]  # Direkt loss değerini döndür