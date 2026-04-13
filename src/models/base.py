"""QAU - Model Temel Sınıfı"""

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


class BaseModel(ABC):
    """Tüm modeller için temel sınıf"""

    name: str = "base"

    def __init__(self):
        self.model = None
        self.is_fitted = False
        self.last_train_date = None
        self.metrics = {}

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> dict:
        """Modeli eğit"""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Tahmin yap"""
        pass

    @abstractmethod
    def get_feature_importance(self) -> dict | None:
        """Feature importance döndür"""
        return None

    def save(self, path: str = None) -> str:
        """Modeli kaydet"""
        import joblib

        path = path or f"models/{self.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        logger.info(f"Model saved: {path}")
        return path

    def load(self, path: str) -> None:
        """Modeli yükle"""
        import joblib

        self.model = joblib.load(path)
        self.is_fitted = True
        logger.info(f"Model loaded: {path}")

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Tahmin performansını değerlendir"""
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        # Direction accuracy
        if len(y_true) > 1:
            actual_dir = np.diff(y_true) > 0
            pred_dir = np.diff(y_pred) > 0
            direction_acc = np.mean(actual_dir == pred_dir) * 100
        else:
            direction_acc = 0.0

        self.metrics = {
            "mae": mae, "rmse": rmse, "r2": r2,
            "mape": mape, "direction_accuracy": direction_acc,
        }
        return self.metrics