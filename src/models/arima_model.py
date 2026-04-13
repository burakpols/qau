"""QAU - ARIMA/SARIMAX Model

Auto ARIMA ile otomatik parametre seçimi, SARIMAX ile exogenous variables,
walk-forward validation.
"""

import warnings

import numpy as np
import pandas as pd
from loguru import logger

from src.models.base import BaseModel

warnings.filterwarnings("ignore")


class ARIMAModel(BaseModel):
    """ARIMA/SARIMAX modeli"""

    name = "arima"

    def __init__(self, auto: bool = True, seasonal: bool = True, forecast_horizon: int = 1):
        super().__init__()
        self.auto = auto
        self.seasonal = seasonal
        self.forecast_horizon = forecast_horizon
        self.order = (1, 1, 1)
        self.seasonal_order = (0, 0, 0, 0)
        self.exog_cols = []

    def fit(self, df: pd.DataFrame, target_col: str = "close",
            exog_cols: list[str] = None, **kwargs) -> dict:
        """ARIMA modelini eğit

        Args:
            df: date ve target_col içeren DataFrame
            target_col: Hedef değişken
            exog_cols: Exogenous değişkenler (SARIMAX için)
        """
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found")

        y = df[target_col].dropna()
        self.exog_cols = exog_cols or []

        # Exogenous veri
        X = None
        if self.exog_cols:
            available = [c for c in self.exog_cols if c in df.columns]
            if available:
                X = df[available].loc[y.index]
                X = X.fillna(method="ffill").fillna(0)

        try:
            if self.auto:
                self._auto_arima(y, X)

            # SARIMAX fit
            self.model = SARIMAX(
                y, exog=X,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)

            self.is_fitted = True
            self.last_train_date = df["date"].max() if "date" in df.columns else None

            metrics = {
                "aic": self.model.aic,
                "bic": self.model.bic,
                "order": self.order,
                "seasonal_order": self.seasonal_order,
            }
            logger.info(f"ARIMA fit: order={self.order}, AIC={self.model.aic:.2f}")
            return metrics

        except Exception as e:
            logger.error(f"ARIMA fit error: {e}")
            # Fallback: basit ARIMA(1,1,1)
            self.order = (1, 1, 1)
            self.seasonal_order = (0, 0, 0, 0)
            self.model = SARIMAX(y, order=self.order, enforce_stationarity=False).fit(disp=False)
            self.is_fitted = True
            return {"aic": self.model.aic, "order": self.order, "fallback": True}

    def _auto_arima(self, y: pd.Series, X: pd.DataFrame = None):
        """Auto ARIMA ile en iyi parametreleri bul"""
        try:
            import pmdarima as pm

            stepwise_model = pm.auto_arima(
                y, exogenous=X,
                seasonal=self.seasonal,
                m=5,  # İş günü frekansı
                d=None,  # Otomatik belirle
                max_p=5, max_q=5,
                max_P=2, max_Q=2,
                trace=False,
                error_action="ignore",
                suppress_warnings=True,
                stepwise=True,
            )

            self.order = stepwise_model.order
            if self.seasonal:
                self.seasonal_order = stepwise_model.seasonal_order or (0, 0, 0, 0)

            logger.info(f"Auto ARIMA: order={self.order}, seasonal={self.seasonal_order}")

        except ImportError:
            logger.warning("pmdarima not installed, using default order (1,1,1)")
            self.order = (1, 1, 1)
        except Exception as e:
            logger.warning(f"Auto ARIMA failed: {e}, using default order")

    def predict(self, steps: int = 1, exog: pd.DataFrame = None) -> np.ndarray:
        """Gelecek tahmini yap"""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")

        forecast = self.model.get_forecast(
            steps=steps,
            exog=exog if self.exog_cols else None,
        )

        self._last_forecast = forecast
        return forecast.predicted_mean.values

    def predict_with_ci(self, steps: int = 1, alpha: float = 0.05, exog=None) -> dict:
        """Güven aralığı ile tahmin"""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")

        forecast = self.model.get_forecast(
            steps=steps, exog=exog if self.exog_cols else None,
        )

        ci = forecast.conf_int(alpha=alpha)
        return {
            "mean": forecast.predicted_mean.values,
            "lower": ci.iloc[:, 0].values,
            "upper": ci.iloc[:, 1].values,
        }

    def walk_forward_validation(self, df: pd.DataFrame, target_col: str = "close",
                                 train_size: int = 252, steps: int = 1) -> dict:
        """Walk-forward validation"""
        y = df[target_col].dropna().values
        n = len(y)

        if n < train_size + steps:
            logger.error("Not enough data for walk-forward validation")
            return {}

        predictions = []
        actuals = []

        for i in range(train_size, n - steps + 1):
            train = y[:i]
            try:
                from statsmodels.tsa.statespace.sarimax import SARIMAX
                model = SARIMAX(train, order=self.order, seasonal_order=self.seasonal_order,
                               enforce_stationarity=False).fit(disp=False)
                pred = model.forecast(steps=steps)
                predictions.append(pred[-1])
                actuals.append(y[i + steps - 1])
            except Exception:
                continue

        predictions = np.array(predictions)
        actuals = np.array(actuals)

        return self.evaluate(actuals, predictions)

    def get_feature_importance(self) -> dict | None:
        return None

    def get_diagnostics(self) -> dict:
        """Model diyagnostikleri"""
        if not self.is_fitted:
            return {}

        return {
            "aic": self.model.aic,
            "bic": self.model.bic,
            "hqic": self.model.hqic,
            "order": self.order,
            "seasonal_order": self.seasonal_order,
            "residuals_mean": float(self.model.resid.mean()),
            "residuals_std": float(self.model.resid.std()),
        }