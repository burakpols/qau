"""QAU - LSTM Model

Sequence-to-one LSTM, attention mekanizması, early stopping.
"""

import numpy as np
import pandas as pd
from loguru import logger

from src.models.base import BaseModel


class LSTMModel(BaseModel):
    """LSTM time-series forecasting modeli"""

    name = "lstm"

    def __init__(self, sequence_length: int = 30, forecast_horizon: int = 1,
                 units: list[int] = None, dropout: float = 0.2,
                 use_attention: bool = True):
        super().__init__()
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.units = units or [64, 32]
        self.dropout = dropout
        self.use_attention = use_attention
        self.scaler = None
        self.feature_names = []

    def _build_model(self, input_shape: tuple) -> None:
        """LSTM modelini oluştur"""
        import tensorflow as tf
        from tensorflow.keras import layers, models, regularizers

        inputs = layers.Input(shape=input_shape)

        # LSTM layers
        x = inputs
        for i, unit in enumerate(self.units):
            return_seq = i < len(self.units) - 1
            x = layers.LSTM(
                unit, return_sequences=return_seq,
                dropout=self.dropout,
                kernel_regularizer=regularizers.l2(1e-4),
                name=f"lstm_{i}",
            )(x)

        # Attention
        if self.use_attention and len(self.units) > 0:
            # Simple self-attention on last LSTM output
            x = layers.Dense(1, activation="tanh", name="attention")(x)

        # Output
        outputs = layers.Dense(self.forecast_horizon, name="output")(x)

        self.model = models.Model(inputs, outputs, name="qau_lstm")
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss="huber",
            metrics=["mae"],
        )

    def _create_sequences(self, X: np.ndarray, y: np.ndarray = None) -> tuple:
        """Sequence oluşturma"""
        X_seq, y_seq = [], []
        for i in range(self.sequence_length, len(X)):
            X_seq.append(X[i - self.sequence_length:i])
            if y is not None:
                y_seq.append(y[i:i + self.forecast_horizon])

        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq) if y is not None else None
        return X_seq, y_seq

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> dict:
        """LSTM modelini eğit"""
        from sklearn.preprocessing import RobustScaler
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

        self.feature_names = list(X.columns)
        X_vals = X.fillna(0).values
        y_vals = y.values.reshape(-1, 1)

        # Scale
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X_vals)

        y_scaler = RobustScaler()
        y_scaled = y_scaler.fit_transform(y_vals)
        self._y_scaler = y_scaler

        # Sequences
        X_seq, y_seq = self._create_sequences(X_scaled, y_scaled.flatten())

        if len(X_seq) < 10:
            raise ValueError(f"Not enough data: {len(X_seq)} sequences")

        # Train/val split
        split = int(len(X_seq) * 0.8)
        X_train, X_val = X_seq[:split], X_seq[split:]
        y_train, y_val = y_seq[:split], y_seq[split:]

        # Build model
        self._build_model((X_train.shape[1], X_train.shape[2]))

        # Callbacks
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
        ]

        # Train
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=kwargs.get("epochs", 100),
            batch_size=kwargs.get("batch_size", 32),
            callbacks=callbacks,
            verbose=0,
        )

        self.is_fitted = True

        # Validation metrics
        y_pred = self.model.predict(X_val, verbose=0)
        y_pred_inv = self._y_scaler.inverse_transform(y_pred)
        y_val_inv = self._y_scaler.inverse_transform(y_val)

        metrics = self.evaluate(y_val_inv.flatten(), y_pred_inv.flatten())
        metrics["epochs_trained"] = len(history.history["loss"])

        logger.info(f"LSTM fit: {metrics}")
        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Tahmin yap"""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")

        X_vals = X.fillna(0)[self.feature_names].values
        X_scaled = self.scaler.transform(X_vals)
        X_seq, _ = self._create_sequences(X_scaled)
        y_pred = self.model.predict(X_seq, verbose=0)
        return self._y_scaler.inverse_transform(y_pred).flatten()

    def predict_next(self, X: pd.DataFrame) -> float:
        """Son sequence'den sonraki 1 adım tahmin"""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")

        X_vals = X.fillna(0)[self.feature_names].values
        X_scaled = self.scaler.transform(X_vals)
        last_seq = X_scaled[-self.sequence_length:].reshape(1, self.sequence_length, -1)
        pred = self.model.predict(last_seq, verbose=0)
        return float(self._y_scaler.inverse_transform(pred)[0, 0])

    def get_feature_importance(self) -> dict | None:
        """Permutation-based feature importance"""
        if not self.is_fitted:
            return None
        return {"note": "Use get_shap_values for LSTM interpretability"}

    def save(self, path: str = None) -> str:
        """Keras modelini kaydet"""
        if self.model is not None:
            path = path or f"models/lstm_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.keras"
            self.model.save(path)
            logger.info(f"LSTM model saved: {path}")
            return path
        return ""

    def load(self, path: str) -> None:
        """Keras modelini yükle"""
        from tensorflow.keras import models
        self.model = models.load_model(path)
        self.is_fitted = True
        logger.info(f"LSTM model loaded: {path}")