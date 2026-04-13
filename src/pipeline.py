"""QAU - Ana Pipeline

Günlük veri çekme → feature engineering → tahmin → sinyal → rapor akışı.
"""

import pandas as pd
from loguru import logger

from src.data.db import db
from src.data.fetcher import fetcher
from src.data.processor import processor
from src.data.news_fetcher import NewsFetcher, run_news_pipeline
from src.models.arima_model import ARIMAModel
from src.models.xgboost_model import XGBoostModel
from src.models.ensemble import EnsembleModel
from src.backtest.engine import BacktestEngine
from src.llm.assistant import GoldAssistant
from src.portfolio.manager import PortfolioManager


class QAUPipeline:
    """QAU ana pipeline orchestration"""

    def __init__(self):
        self.ensemble = EnsembleModel(method="weighted")
        self.backtest = BacktestEngine()
        self.assistant = GoldAssistant()
        self.portfolio = PortfolioManager()
        self._models = {}
        self._features_df = None
        self._last_predictions = {}

    def initialize(self) -> pd.DataFrame:
        """Veritabanı şemasını oluştur, veri çek, feature engineering yap"""
        logger.info("=== QAU Initialization ===")

        # 1. DB şemasını oluştur
        db.init_schema()

        # 2. Verileri çek
        results = fetcher.update_all()
        logger.info(f"Data fetch results: {results}")

        # 3. Haberleri çek ve kaydet
        try:
            news_fetcher = NewsFetcher()
            if news_fetcher.api_key:
                articles = news_fetcher.fetch_news(days_back=3)
                news_fetcher.save_to_db(articles)
                logger.info(f"Haber fetch: {len(articles)} haber çekildi")
            else:
                logger.warning("NEWS_API_KEY yok - haberler atlanıyor")
        except Exception as e:
            logger.warning(f"Haber fetch hatası: {e}")

        # 4. Feature engineering
        self._features_df = processor.build_features()
        logger.info(f"Features: {len(self._features_df)} rows, {len(self._features_df.columns)} cols")

        return self._features_df

    def train_models(self, days: int = 365) -> dict:
        """Tüm modelleri eğit"""
        if self._features_df is None:
            self._features_df = processor.build_features()

        df = self._features_df.tail(days)
        
        # Hem regression hem classification modelleri
        X = df.drop(columns=["close", "target_close", "target_return", "target_direction"]).select_dtypes(include="number")
        
        # Regression: return tahmini
        y_return = df["target_return"].dropna()
        X_return = X.loc[y_return.index]
        
        # Classification: direction tahmini
        y_direction = df["target_direction"].dropna()
        X_direction = X.loc[y_direction.index]

        # ARIMA - return için
        logger.info("Training ARIMA...")
        arima = ARIMAModel()
        arima_df = df[["close"]].copy()
        arima_metrics = arima.fit(arima_df)
        self._models["arima"] = arima
        logger.info(f"ARIMA: {arima_metrics}")

        # XGBoost - direction classification (daha kolay ve faydalı)
        logger.info("Training XGBoost (Classification)...")
        xgb = XGBoostModel(task="classification", optimize=True)
        xgb_metrics = xgb.fit(X_direction, y_direction.astype(int))
        self._models["xgboost"] = xgb
        logger.info(f"XGBoost: {xgb_metrics}")

        return {
            "arima": arima_metrics,
            "xgboost": xgb_metrics,
        }

    def run_daily(self) -> dict:
        """Günlük çalışma akışı: veri güncelle → tahmin → sinyal → rapor"""
        logger.info("=== QAU Daily Run ===")

        # 1. Veri güncelle
        fetcher.update_all()

        # 2. Haberleri güncelle (LLM filtreleme ile)
        news_summary = None
        try:
            total_news, relevant_news = run_news_pipeline(
                llm_client=self.assistant.llm if hasattr(self.assistant, 'llm') else None,
                days_back=3
            )
            news_summary = {
                "total": total_news,
                "relevant": relevant_news,
            }
            logger.info(f"Haber güncelleme: {relevant_news}/{total_news} ilgili haber")
        except Exception as e:
            logger.warning(f"Haber güncelleme hatası: {e}")

        # 3. Feature güncelle
        self._features_df = processor.build_features()
        df = self._features_df

        if df.empty:
            logger.error("No data available")
            return {"error": "No data"}

        current_price = float(df["close"].iloc[-1])

        # 4. Tahminler
        predictions = {"current_price": current_price}
        X = df.drop(columns=["close"]).select_dtypes(include="number")

        for name, model in self._models.items():
            try:
                if model.is_fitted:
                    if name == "arima":
                        pred = model.predict(steps=1)
                        predictions[f"{name}_prediction"] = float(pred[0])
                    elif name == "xgboost":
                        pred_next = model.predict_next(df)
                        predictions[f"{name}_prediction"] = float(pred_next)
            except Exception as e:
                logger.warning(f"Prediction failed for {name}: {e}")

        # XGBoost probability'yi direction'a çevir
        xgb_prob = predictions.get("xgboost_prediction", 0.5)
        if 0 <= xgb_prob <= 1:
            # 0.5 üzeri = UP (class 1), 0.5 altı = DOWN (class 0)
            predictions["xgboost_direction"] = "UP" if xgb_prob > 0.5 else "DOWN"
            predictions["xgboost_confidence"] = abs(xgb_prob - 0.5) * 2  # 0-1 arası confidence
        else:
            predictions["xgboost_direction"] = "NEUTRAL"
            predictions["xgboost_confidence"] = 0
        
        # Ensemble prediction = XGBoost probability
        predictions["ensemble_prediction"] = xgb_prob

        self._last_predictions = predictions

        # 5. Sentiment (haberlerden + LLM filtreleme sonrası)
        sentiment = None
        try:
            # Veritabanından son 3 günün ilgili haberlerini al
            from src.data.db import get_db_connection
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT title, summary, relevance_score 
                FROM news_sentiment 
                WHERE date >= CURRENT_DATE - INTERVAL '3 days'
                AND relevance_score > 0.5
                ORDER BY relevance_score DESC
                LIMIT 20
            """)
            news_rows = cursor.fetchall()
            cursor.close()
            conn.close()

            if news_rows:
                db_news = [
                    {"title": row[0], "summary": row[1], "relevance_score": row[2]}
                    for row in news_rows
                ]
                from src.sentiment.analyzer import SentimentAnalyzer
                analyzer = SentimentAnalyzer()
                sentiment = analyzer.get_daily_sentiment(db_news)
            else:
                # Fallback: demo haberler
                from src.sentiment.analyzer import SentimentAnalyzer
                analyzer = SentimentAnalyzer()
                demo_news = [
                    {"title": "Altın fiyatları yükselişe geçti", "source": "demo"},
                    {"title": "Fed faiz kararı altını etkiledi", "source": "demo"},
                ]
                sentiment = analyzer.get_daily_sentiment(demo_news)
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")

        # 6. Sinyal üret
        signal = self.assistant.generate_signal(predictions, sentiment)

        # 7. Portföy güncelle
        if signal["signal"] in ("BUY", "SELL"):
            self.portfolio.execute_signal(
                signal["signal"], current_price, signal["confidence"]
            )

        # 8. LLM Analiz raporu (haber özeti ile)
        market_data = {
            "current_price": current_price,
            "date": str(df.index[-1]) if isinstance(df.index[-1], pd.Timestamp) else str(df.index[-1]),
            "news_summary": news_summary,
        }
        analysis = self.assistant.analyze(market_data, predictions, sentiment)

        result = {
            "current_price": current_price,
            "predictions": predictions,
            "signal": signal,
            "sentiment": sentiment,
            "news": news_summary,
            "portfolio": self.portfolio.get_status(),
            "analysis": analysis,
        }

        logger.info(f"Signal: {signal['signal']} ({signal['confidence']:.0f}%) | "
                     f"Price: ₺{current_price:,.2f} | Pred: ₺{predictions.get('ensemble_prediction', 0):,.2f}")

        return result

    def run_backtest(self, days: int = 180) -> dict:
        """Backtest çalıştır"""
        logger.info("=== QAU Backtest ===")

        if self._features_df is None:
            self._features_df = processor.build_features()

        df = self._features_df.tail(days)
        prices = df["close"]

        # Model tahminlerinden sinyaller üret
        X = df.drop(columns=["close"]).select_dtypes(include="number")

        strategies = {}
        for name, model in self._models.items():
            try:
                if model.is_fitted:
                    if name == "arima":
                        pred = model.predict(steps=len(df))
                        if len(pred) >= len(prices):
                            pred = pred[:len(prices)]
                    else:
                        pred = model.predict(X)

                    if len(pred) >= len(prices):
                        pred = pred[:len(prices)]
                        pred_series = pd.Series(pred, index=prices.index[:len(pred)])
                        signals = self.backtest.generate_signals_from_predictions(
                            prices, pred_series
                        )
                        strategies[name] = signals
            except Exception as e:
                logger.error(f"Backtest strategy {name} failed: {e}")

        result = self.backtest.compare_strategies(prices, strategies)
        logger.info(f"\n{result.to_string()}")

        return result.to_dict()

    def get_status(self) -> dict:
        """Sistem durumu"""
        return {
            "models_trained": {n: m.is_fitted for n, m in self._models.items()},
            "ensemble_fitted": self.ensemble.is_fitted,
            "portfolio": self.portfolio.get_status(),
            "last_predictions": self._last_predictions,
        }


# Singleton
pipeline = QAUPipeline()