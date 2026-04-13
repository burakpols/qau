"""QAU - Gelişmiş Feature Engineering

Teknik göstergeler, cross-market features, makro features,
sentiment entegrasyonu ve regime detection.
"""

import numpy as np
import pandas as pd
from loguru import logger

try:
    from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
    from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
    from ta.volatility import BollingerBands, AverageTrueRange
    from ta.volume import OnBalanceVolumeIndicator, MFIIndicator
    HAS_TA = True
except ImportError:
    HAS_TA = False

from src.data.db import db


class FeatureProcessor:
    """Gelişmiş feature engineering pipeline"""

    def build_features(self) -> pd.DataFrame:
        """Veritabanından tüm verileri çek ve feature engineering yap"""
        logger.info("Building features...")

        # 1. Gram altın TL fiyatlarını çek
        gold_df = db.query_df("SELECT * FROM gold_prices ORDER BY date")
        if gold_df.empty:
            logger.error("No gold price data found in database")
            return pd.DataFrame()

        gold_df["date"] = pd.to_datetime(gold_df["date"])

        # 2. Cross-market verileri ekle
        for table, col_name in [("xau_usd", "xau_usd_close"), ("usd_try", "usd_try_close"), ("brent_oil", "brent_close")]:
            try:
                mdf = db.query_df(f"SELECT date, close AS {col_name} FROM {table} ORDER BY date")
                if not mdf.empty:
                    mdf["date"] = pd.to_datetime(mdf["date"])
                    gold_df = gold_df.merge(mdf, on="date", how="left")
            except Exception as e:
                logger.warning(f"Could not merge {table}: {e}")

        # 3. Makro verileri ekle
        try:
            macro_df = db.query_df("SELECT * FROM macro_indicators ORDER BY date")
            if not macro_df.empty:
                macro_df["date"] = pd.to_datetime(macro_df["date"])
                gold_df = gold_df.merge(macro_df, on="date", how="left")
        except Exception as e:
            logger.warning(f"Could not merge macro data: {e}")

        # 4. Sentiment verisini ekle
        try:
            sent_df = db.query_df("""
                SELECT date, AVG(sentiment_score) AS sentiment_score, COUNT(*) AS news_count
                FROM news_sentiment GROUP BY date ORDER BY date
            """)
            if not sent_df.empty:
                sent_df["date"] = pd.to_datetime(sent_df["date"])
                gold_df = gold_df.merge(sent_df, on="date", how="left")
        except Exception:
            pass

        # 5. Teknik göstergeler
        gold_df = self._add_technical_indicators(gold_df)

        # 6. Cross-market features
        gold_df = self._add_cross_market_features(gold_df)

        # 7. Makro features
        gold_df = self._add_macro_features(gold_df)

        # 8. Sentiment features
        gold_df = self._add_sentiment_features(gold_df)

        # 9. Regime detection
        gold_df = self._add_regime_detection(gold_df)

        # 10. Target değişkenler
        gold_df = self._add_targets(gold_df)

        # Temizlik
        gold_df = gold_df.sort_values("date").reset_index(drop=True)

        # Feature veritabanına kaydet (sadece features tablosundaki kolonları)
        if not gold_df.empty:
            save_df = gold_df.copy()
            save_df["date"] = save_df["date"].dt.date
            
            # Sadece features tablosundaki kolonları tut
            feature_columns = [
                'date', 'close', 'return_1d', 'return_5d', 'return_20d',
                'volatility_5d', 'volatility_20d', 'sma_5', 'sma_10', 'sma_20',
                'sma_50', 'sma_200', 'ema_12', 'ema_26', 'macd', 'macd_signal',
                'macd_hist', 'rsi_14', 'stoch_k', 'stoch_d', 'bb_upper', 'bb_lower',
                'bb_width', 'bb_pct', 'atr_14', 'adx_14', 'cci_14', 'willr_14',
                'obv', 'mfi_14', 'xau_usd_close', 'usd_try_close', 'brent_close',
                'xau_usd_return_1d', 'usd_try_return_1d', 'brent_return_1d',
                'gold_usd_ratio', 'repo_rate', 'cpi_annual', 'm2_money_supply',
                'net_reserves', 'real_interest_rate', 'sentiment_score',
                'sentiment_ma7', 'news_count', 'news_count_7d', 'regime',
                'target_close', 'target_return', 'target_direction'
            ]
            save_df = save_df[[c for c in feature_columns if c in save_df.columns]]
            db.insert_df("features", save_df, on_conflict="DO UPDATE")

        logger.info(f"Built {len(gold_df)} rows with {len(gold_df.columns)} features")
        return gold_df

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Teknik analiz göstergeleri ekle"""
        if not HAS_TA:
            logger.warning("ta library not installed, using basic indicators")
            return self._add_basic_indicators(df)

        close = df["close"]
        high = df["high"] if "high" in df.columns else close
        low = df["low"] if "low" in df.columns else close
        volume = df["volume"] if "volume" in df.columns else pd.Series(0, index=df.index)

        try:
            # Moving Averages
            for w in [5, 10, 20, 50, 200]:
                if len(df) >= w:
                    df[f"sma_{w}"] = SMAIndicator(close=close, window=w).sma_indicator()

            df["ema_12"] = EMAIndicator(close=close, window=12).ema_indicator()
            df["ema_26"] = EMAIndicator(close=close, window=26).ema_indicator()

            # MACD
            macd = MACD(close=close)
            df["macd"] = macd.macd()
            df["macd_signal"] = macd.macd_signal()
            df["macd_hist"] = macd.macd_diff()

            # RSI
            df["rsi_14"] = RSIIndicator(close=close, window=14).rsi()

            # Stochastic
            stoch = StochasticOscillator(high=high, low=low, close=close)
            df["stoch_k"] = stoch.stoch()
            df["stoch_d"] = stoch.stoch_signal()

            # Bollinger Bands
            bb = BollingerBands(close=close)
            df["bb_upper"] = bb.bollinger_hband()
            df["bb_lower"] = bb.bollinger_lband()
            df["bb_width"] = bb.bollinger_wband()
            df["bb_pct"] = bb.bollinger_pband()

            # ATR
            df["atr_14"] = AverageTrueRange(high=high, low=low, close=close).average_true_range()

            # ADX
            if len(df) >= 28:
                df["adx_14"] = ADXIndicator(high=high, low=low, close=close).adx()

            # CCI (Commodity Channel Index) - Manuel hesaplama
            typical_price = (high + low + close) / 3
            sma_tp = typical_price.rolling(14).mean()
            mad = typical_price.rolling(14).apply(lambda x: np.abs(x - x.mean()).mean())
            df["cci_14"] = (typical_price - sma_tp) / (0.015 * mad)

            # Williams %R
            if len(df) >= 14:
                df["willr_14"] = WilliamsRIndicator(high=high, low=low, close=close).williams_r()

            # OBV
            df["obv"] = OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()

            # MFI
            if volume.sum() > 0:
                df["mfi_14"] = MFIIndicator(high=high, low=low, close=close, volume=volume).money_flow_index()

        except Exception as e:
            logger.error(f"Error adding technical indicators: {e}")

        return df

    def _add_basic_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """ta kütüphanesi yoksa basit göstergeler"""
        close = df["close"]

        for w in [5, 10, 20, 50, 200]:
            if len(df) >= w:
                df[f"sma_{w}"] = close.rolling(w).mean()

        df["return_1d"] = close.pct_change(1)
        df["return_5d"] = close.pct_change(5)
        df["return_20d"] = close.pct_change(20)
        df["volatility_5d"] = close.pct_change().rolling(5).std()
        df["volatility_20d"] = close.pct_change().rolling(20).std()

        return df

    def _add_cross_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cross-market features (XAU/USD, USD/TRY, Brent)"""
        for col in ["xau_usd_close", "usd_try_close", "brent_close"]:
            if col in df.columns:
                base = col.replace("_close", "_return_1d")
                df[base] = df[col].pct_change(1)

        # Gram altın / Ons altın oranı (premium/discount)
        if "xau_usd_close" in df.columns and "usd_try_close" in df.columns:
            df["gold_usd_ratio"] = df["close"] / (df["xau_usd_close"] * df["usd_try_close"] / 31.1035)

        return df

    def _add_macro_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Makroekonomik features"""
        if "repo_rate" in df.columns and "cpi_annual" in df.columns:
            df["real_interest_rate"] = df["repo_rate"] - df["cpi_annual"]

        return df

    def _add_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sentiment features"""
        if "sentiment_score" in df.columns:
            df["sentiment_score"] = df["sentiment_score"].fillna(0)
            df["sentiment_ma7"] = df["sentiment_score"].rolling(7, min_periods=1).mean()

        if "news_count" in df.columns:
            df["news_count_7d"] = df["news_count"].rolling(7, min_periods=1).sum()

        return df

    def _add_regime_detection(self, df: pd.DataFrame) -> pd.DataFrame:
        """Piyasa rejimi tespiti (0=normal, 1=yükseliş, 2=düşüş, 3=volatile)"""
        if "close" not in df.columns or len(df) < 50:
            df["regime"] = 0
            return df

        returns = df["close"].pct_change()
        vol = returns.rolling(20).std()
        trend = df["close"].rolling(50).mean() - df["close"].rolling(200).mean()

        vol_threshold = vol.quantile(0.75) if len(vol.dropna()) > 10 else vol.median()

        regime = pd.Series(0, index=df.index)
        regime[trend > 0] = 1  # Yükseliş
        regime[trend < 0] = 2  # Düşüş
        regime[vol > vol_threshold] = 3  # Volatil

        df["regime"] = regime
        return df

    def _add_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tahmin hedefi değişkenleri"""
        df["target_close"] = df["close"].shift(-1)
        df["target_return"] = df["close"].pct_change().shift(-1)
        df["target_direction"] = (df["target_return"] > 0).astype(int)
        return df

    def prepare_model_input(self, df: pd.DataFrame, target_col: str = "target_close") -> tuple:
        """Model eğitimi için X, y hazırla"""
        feature_cols = [c for c in df.columns if c not in [
            "date", "target_close", "target_return", "target_direction", "source"
        ]]

        df = df.dropna(subset=[target_col])

        X = df[feature_cols].select_dtypes(include=[np.number])
        y = df[target_col]

        return X, y


# Singleton
processor = FeatureProcessor()