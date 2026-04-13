"""QAU - XGBoost Model

Optuna hiperparametre optimizasyonu, yön sınıflandırma, SHAP açıklanabilirlik.
"""

import numpy as np
import pandas as pd
from loguru import logger

from src.models.base import BaseModel


class XGBoostModel(BaseModel):
    """XGBoost regresyon ve sınıflandırma modeli"""

    name = "xgboost"

    def __init__(self, task: str = "regression", optimize: bool = True):
        super().__init__()
        self.task = task  # "regression" veya "classification"
        self.optimize = optimize
        self.best_params = {}
        self.feature_names = []

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> dict:
        """XGBoost modelini eğit"""
        import xgboost as xgb
        from sklearn.model_selection import TimeSeriesSplit

        self.feature_names = list(X.columns)
        X = X.fillna(0)

        # Zaman serisi cross-validation
        tscv = TimeSeriesSplit(n_splits=5)

        if self.optimize:
            self.best_params = self._optuna_optimize(X, y, tscv)

        # Daha muhafazakar parametreler - overfitting önlemek için
        params = {
            "n_estimators": 200,
            "max_depth": 4,  # Daha sığ
            "learning_rate": 0.03,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "colsample_bylevel": 0.7,
            "reg_alpha": 1.0,  # Daha güçlü regularization
            "reg_lambda": 5.0,
            "min_child_weight": 5,
            "gamma": 0.1,
            "random_state": 42,
        }
        params.update(self.best_params)

        if self.task == "classification":
            params["objective"] = "binary:logistic"
            params["eval_metric"] = "logloss"
            params["scale_pos_weight"] = 1  # Balance classes
            self.model = xgb.XGBClassifier(**params)
        else:
            params["objective"] = "reg:squarederror"
            params["eval_metric"] = "mae"  # MAE daha robust
            self.model = xgb.XGBRegressor(**params)

        # Train/validation split (son %20 validation)
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        # Early stopping
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        self.is_fitted = True

        # Validation metrikleri
        y_pred = self.model.predict(X_val)
        metrics = self.evaluate(y_val.values, y_pred)
        metrics["best_params"] = str(self.best_params) if self.best_params else "default"

        logger.info(f"XGBoost fit: task={self.task}, metrics={metrics}")
        return metrics

    def _optuna_optimize(self, X: pd.DataFrame, y: pd.Series, tscv) -> dict:
        """Optuna ile hiperparametre optimizasyonu"""
        try:
            import optuna
            import xgboost as xgb
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            def objective(trial):
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                    "random_state": 42,
                }

                if self.task == "classification":
                    params["objective"] = "binary:logistic"
                    model = xgb.XGBClassifier(**params)
                else:
                    params["objective"] = "reg:squarederror"
                    model = xgb.XGBRegressor(**params)

                scores = []
                for train_idx, val_idx in tscv.split(X):
                    X_t, X_v = X.iloc[train_idx], X.iloc[val_idx]
                    y_t, y_v = y.iloc[train_idx], y.iloc[val_idx]
                    model.fit(X_t, y_t, verbose=False)
                    pred = model.predict(X_v)

                    if self.task == "classification":
                        from sklearn.metrics import accuracy_score
                        scores.append(accuracy_score(y_v, pred))
                    else:
                        from sklearn.metrics import mean_squared_error
                        scores.append(-np.sqrt(mean_squared_error(y_v, pred)))

                return np.mean(scores)

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=50, show_progress_bar=False)

            logger.info(f"Optuna best value: {study.best_value:.4f}")
            return study.best_params

        except ImportError:
            logger.warning("Optuna not installed, using default params")
            return {}
        except Exception as e:
            logger.warning(f"Optuna optimization failed: {e}")
            return {}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Tahmin yap"""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        X = X.fillna(0)
        return self.model.predict(X[self.feature_names])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray | None:
        """Sınıflandırma olasılıkları - sadece classification için"""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        if self.task != "classification":
            return None
        X = X.fillna(0)
        return self.model.predict_proba(X[self.feature_names])

    def get_feature_importance(self) -> dict | None:
        """Feature importance"""
        if not self.is_fitted:
            return None
        importance = self.model.feature_importances_
        return dict(zip(self.feature_names, importance.tolist()))

    def get_shap_values(self, X: pd.DataFrame) -> dict | None:
        """SHAP değerlerini hesapla"""
        try:
            import shap
            X = X.fillna(0)
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X[self.feature_names])
            return {
                "shap_values": shap_values,
                "feature_names": self.feature_names,
                "base_value": explainer.expected_value,
            }
        except ImportError:
            logger.warning("SHAP not installed")
            return None
        except Exception as e:
            logger.error(f"SHAP error: {e}")
            return None

    def predict_next(self, df: pd.DataFrame) -> float:
        """Sonraki gün için tek tahmin"""
        X = df.drop(columns=["close"]).select_dtypes(include="number")
        X = X.fillna(0)
        X = X.iloc[[-1]]  # Son satır
        
        if self.task == "classification":
            # Sınıf 1 (yükseliş) olasılığı
            proba = self.predict_proba(X)
            if proba is not None:
                return float(proba[0, 1])
        return float(self.model.predict(X[self.feature_names])[0])
