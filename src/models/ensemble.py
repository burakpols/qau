"""QAU - Ensemble Model

Ağırlıklı ortalama, stacking, direction voting ile çoklu model kombinasyonu.
"""

import numpy as np
import pandas as pd
from loguru import logger

from src.models.base import BaseModel


class EnsembleModel(BaseModel):
    """Çoklu model ensemble"""

    name = "ensemble"

    def __init__(self, models: dict[str, BaseModel] = None, method: str = "weighted"):
        super().__init__()
        self.models = models or {}
        self.method = method  # "weighted", "stacking", "voting"
        self.weights = {}
        self.meta_model = None

    def add_model(self, name: str, model: BaseModel, weight: float = 1.0):
        """Model ekle"""
        self.models[name] = model
        self.weights[name] = weight

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> dict:
        """Tüm modelleri eğit ve ağırlıkları optimize et"""
        all_metrics = {}

        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            try:
                if model.name == "arima":
                    # ARIMA farklı API'ye sahip
                    df = X.copy()
                    df["close"] = y.values
                    metrics = model.fit(df)
                else:
                    metrics = model.fit(X, y)

                all_metrics[name] = metrics
                logger.info(f"{name} trained: {metrics}")
            except Exception as e:
                logger.error(f"{name} training failed: {e}")
                all_metrics[name] = {"error": str(e)}

        # Ağırlıkları optimize et
        if self.method == "weighted":
            self._optimize_weights(X, y)

        self.is_fitted = True
        return all_metrics

    def _optimize_weights(self, X: pd.DataFrame, y: pd.Series):
        """Validation performansına göre ağırlıkları optimize et"""
        total_score = 0
        scores = {}

        for name, model in self.models.items():
            if not model.is_fitted:
                continue
            try:
                pred = self._predict_single(model, X)
                if len(pred) == len(y):
                    from sklearn.metrics import r2_score
                    score = max(r2_score(y.values[:len(pred)], pred), 0.01)
                    scores[name] = score
                    total_score += score
            except Exception as e:
                logger.warning(f"Weight optimization failed for {name}: {e}")
                scores[name] = 0.01
                total_score += 0.01

        # Normalize weights
        for name in scores:
            self.weights[name] = scores[name] / total_score

        logger.info(f"Ensemble weights: {self.weights}")

    def _predict_single(self, model: BaseModel, X: pd.DataFrame) -> np.ndarray:
        """Tek model tahmini"""
        if model.name == "arima":
            return model.predict(steps=len(X))
        return model.predict(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Ensemble tahmin"""
        if not self.is_fitted:
            raise RuntimeError("Ensemble not fitted")

        predictions = {}
        for name, model in self.models.items():
            if not model.is_fitted:
                continue
            try:
                pred = self._predict_single(model, X)
                predictions[name] = pred
            except Exception as e:
                logger.warning(f"Prediction failed for {name}: {e}")

        if not predictions:
            raise RuntimeError("No models could predict")

        if self.method == "weighted":
            return self._weighted_average(predictions)
        elif self.method == "voting":
            return self._direction_voting(predictions, X)
        else:
            return self._weighted_average(predictions)

    def _weighted_average(self, predictions: dict[str, np.ndarray]) -> np.ndarray:
        """Ağırlıklı ortalama"""
        min_len = min(len(p) for p in predictions.values())
        result = np.zeros(min_len)
        total_weight = 0

        for name, pred in predictions.items():
            w = self.weights.get(name, 1.0 / len(predictions))
            result += w * pred[:min_len]
            total_weight += w

        return result / total_weight if total_weight > 0 else result

    def _direction_voting(self, predictions: dict[str, np.ndarray],
                          X: pd.DataFrame) -> np.ndarray:
        """Yön oylaması - çoğunluğun yönüne göre tahmin"""
        avg = self._weighted_average(predictions)

        # Her modelin yön tahmini
        directions = []
        for name, pred in predictions.items():
            if len(pred) > 0:
                directions.append(1 if pred[-1] > pred[0] else -1)

        # Majority vote
        direction = np.sign(sum(directions)) if directions else 1
        avg[-1] = avg[-2] + direction * abs(avg[-1] - avg[-2])

        return avg

    def get_feature_importance(self) -> dict | None:
        """Model ağırlıklarını döndür"""
        return self.weights.copy()

    def get_prediction_report(self, X: pd.DataFrame) -> dict:
        """Detaylı tahmin raporu"""
        predictions = {}
        for name, model in self.models.items():
            if not model.is_fitted:
                continue
            try:
                pred = self._predict_single(model, X)
                predictions[name] = {
                    "value": float(pred[-1]) if len(pred) > 0 else None,
                    "weight": self.weights.get(name, 0),
                    "direction": "↑" if len(pred) > 1 and pred[-1] > pred[-2] else "↓",
                }
            except Exception as e:
                predictions[name] = {"error": str(e)}

        ensemble_pred = self.predict(X)

        return {
            "individual_predictions": predictions,
            "ensemble_prediction": float(ensemble_pred[-1]),
            "method": self.method,
        }