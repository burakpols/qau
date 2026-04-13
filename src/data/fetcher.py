"""QAU - Veri Çekme Modülü

Yahoo Finance, Bigpara, TCMB EVDS ve Brent Petrol'den veri çekme.
"""

from datetime import datetime, timedelta

import pandas as pd
import requests
from loguru import logger

try:
    import yfinance as yf
except ImportError:
    yf = None

from src.config import data_config
from src.data.db import db


class GoldDataFetcher:
    """Altın fiyatları ve ilgili verileri çeken sınıf"""

    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

    # =========================================================================
    # Yahoo Finance - XAU/USD (Ons Altın)
    # =========================================================================
    def fetch_xau_usd(self, period: str = None) -> pd.DataFrame:
        """Yahoo Finance'den ons altın (XAU/USD) günlük verilerini çek"""
        if yf is None:
            logger.error("yfinance not installed. Run: pip install yfinance")
            return pd.DataFrame()

        period = period or f"{data_config.HISTORY_YEARS}y"

        try:
            ticker = yf.Ticker(data_config.XAU_USD_SYMBOL)
            df = ticker.history(period=period, interval=data_config.FETCH_INTERVAL)

            if df.empty:
                logger.warning("No XAU/USD data fetched from Yahoo Finance")
                return df

            df = df.rename(columns={
                "Open": "open", "High": "high", "Low": "low",
                "Close": "close", "Volume": "volume",
            })
            df.index = df.index.tz_localize(None).date
            df = df[["open", "high", "low", "close", "volume"]]
            df.index.name = "date"
            df = df.reset_index()
            df["source"] = "yahoo_finance"

            logger.info(f"Fetched {len(df)} rows of XAU/USD data")
            return df

        except Exception as e:
            logger.error(f"Error fetching XAU/USD: {e}")
            return pd.DataFrame()

    # =========================================================================
    # Yahoo Finance - USD/TRY
    # =========================================================================
    def fetch_usd_try(self, period: str = None) -> pd.DataFrame:
        """Yahoo Finance'den USD/TRY günlük kur verilerini çek"""
        if yf is None:
            logger.error("yfinance not installed")
            return pd.DataFrame()

        period = period or f"{data_config.HISTORY_YEARS}y"

        try:
            ticker = yf.Ticker(data_config.USD_TRY_SYMBOL)
            df = ticker.history(period=period, interval=data_config.FETCH_INTERVAL)

            if df.empty:
                logger.warning("No USD/TRY data fetched from Yahoo Finance")
                return df

            df = df.rename(columns={
                "Open": "open", "High": "high", "Low": "low", "Close": "close",
            })
            df.index = df.index.tz_localize(None).date
            df = df[["open", "high", "low", "close"]]
            df.index.name = "date"
            df = df.reset_index()
            df["source"] = "yahoo_finance"

            logger.info(f"Fetched {len(df)} rows of USD/TRY data")
            return df

        except Exception as e:
            logger.error(f"Error fetching USD/TRY: {e}")
            return pd.DataFrame()

    # =========================================================================
    # Yahoo Finance - Brent Petrol
    # =========================================================================
    def fetch_brent_oil(self, period: str = None) -> pd.DataFrame:
        """Yahoo Finance'den Brent petrol günlük verilerini çek"""
        if yf is None:
            logger.error("yfinance not installed")
            return pd.DataFrame()

        period = period or f"{data_config.HISTORY_YEARS}y"

        try:
            ticker = yf.Ticker(data_config.BRENT_SYMBOL)
            df = ticker.history(period=period, interval=data_config.FETCH_INTERVAL)

            if df.empty:
                logger.warning("No Brent oil data fetched from Yahoo Finance")
                return df

            df = df.rename(columns={
                "Open": "open", "High": "high", "Low": "low",
                "Close": "close", "Volume": "volume",
            })
            df.index = df.index.tz_localize(None).date
            df = df[["open", "high", "low", "close", "volume"]]
            df.index.name = "date"
            df = df.reset_index()
            df["source"] = "yahoo_finance"

            logger.info(f"Fetched {len(df)} rows of Brent oil data")
            return df

        except Exception as e:
            logger.error(f"Error fetching Brent oil: {e}")
            return pd.DataFrame()

    # =========================================================================
    # AltınKaynak - Gram Altın TL (Canlı Fiyat)
    # =========================================================================
    def fetch_altinkaynak(self) -> pd.DataFrame:
        """AltınKaynak API'den gram altın TL canlı fiyatını çek"""
        try:
            url = "https://static.altinkaynak.com/Gold"
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()

            data = response.json()
            if not data:
                logger.warning("Empty response from AltınKaynak")
                return pd.DataFrame()

            # Gram altın (24 Ayar) - Kod: "GA"
            gram_gold = None
            for item in data:
                if item.get("Kod") == "GA":
                    gram_gold = item
                    break

            if not gram_gold:
                logger.warning("Gram gold not found in AltınKaynak response")
                return pd.DataFrame()

            # Türk formatındaki sayıları çevir (6.757,60 -> 6757.60)
            def parse_turkish_number(val):
                if not val or val == "0":
                    return None
                return float(val.replace(".", "").replace(",", "."))

            today = datetime.now().date()
            row = {
                "date": today,
                "open": parse_turkish_number(gram_gold.get("Alis")),
                "high": parse_turkish_number(gram_gold.get("Satis")),
                "low": parse_turkish_number(gram_gold.get("Alis")),
                "close": parse_turkish_number(gram_gold.get("Satis")),
                "change_pct": gram_gold.get("Change", 0),
                "source": "altinkaynak",
                "fetched_at": datetime.now().isoformat(),
            }

            df = pd.DataFrame([row])
            logger.info(f"Fetched Gram Altın from AltınKaynak: Alış={row['open']}, Satış={row['close']}")
            return df

        except Exception as e:
            logger.error(f"Error fetching from AltınKaynak: {e}")
            return self._calculate_gold_from_yahoo()

    # =========================================================================
    # Bigpara - Gram Altın TL (Fallback)
    # =========================================================================
    def fetch_gold_prices_bigpara(self) -> pd.DataFrame:
        """Bigpara'dan gram altın TL geçmiş verilerini çek (scraping) - FALLBACK"""
        try:
            url = "https://www.bigpara.com/altin/gunluk-altin-tablosu/"
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()

            tables = pd.read_html(response.text)
            if not tables:
                logger.warning("No tables found on Bigpara page")
                return pd.DataFrame()

            df = tables[0]
            df = df.rename(columns={
                df.columns[0]: "date",
                df.columns[1]: "open",
                df.columns[2]: "high",
                df.columns[3]: "low",
                df.columns[4]: "close",
            })

            df["date"] = pd.to_datetime(df["date"], format="mixed", dayfirst=True).dt.date

            for col in ["open", "high", "low", "close"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(
                        df[col].astype(str).str.replace(",", ".").str.replace(r"[^\d.]", "", regex=True),
                        errors="coerce",
                    )

            df = df[["date", "open", "high", "low", "close"]].dropna()
            df["volume"] = 0
            df["source"] = "bigpara"

            logger.info(f"Fetched {len(df)} rows of gold prices from Bigpara")
            return df

        except Exception as e:
            logger.error(f"Error fetching from Bigpara: {e}")
            return self._calculate_gold_from_yahoo()

    def _calculate_gold_from_yahoo(self) -> pd.DataFrame:
        """Yahoo Finance verilerinden gram altın TL fiyatını hesapla (fallback)

        Gram Altın TL = (XAU/USD * USD/TRY) / 31.1035
        """
        logger.info("Calculating gram gold TRY from XAU/USD and USD/TRY...")

        xau_df = self.fetch_xau_usd()
        usd_df = self.fetch_usd_try()

        if xau_df.empty or usd_df.empty:
            logger.error("Cannot calculate gold prices - missing source data")
            return pd.DataFrame()

        merged = pd.merge(xau_df, usd_df, on="date", suffixes=("_xau", "_usd"))

        grams_per_ounce = 31.1035
        df = pd.DataFrame()
        df["date"] = merged["date"]
        df["open"] = (merged["open_xau"] * merged["open_usd"]) / grams_per_ounce
        df["high"] = (merged["high_xau"] * merged["high_usd"]) / grams_per_ounce
        df["low"] = (merged["low_xau"] * merged["low_usd"]) / grams_per_ounce
        df["close"] = (merged["close_xau"] * merged["close_usd"]) / grams_per_ounce
        df["volume"] = merged.get("volume", 0)
        df["source"] = "calculated_yahoo"

        df = df.dropna()
        logger.info(f"Calculated {len(df)} rows of gram gold TRY prices")
        return df

    # =========================================================================
    # TCMB EVDS - Makroekonomik Göstergeler
    # =========================================================================
    def fetch_evds_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """TCMB EVDS API'den makroekonomik verileri çek

        EVDS Seri Kodları:
        - TP.KTFTKR: Faiz oranı (politika faizi)
        - TP.FG.J0:  TÜFE yıllık
        - TP.FG.J01: TÜFE aylık
        - TP.UG.J0:  ÜFE yıllık
        - TP.UG.J01: ÜFE aylık
        - TP.YP.PDS: M2 para arzı
        - TP.AB.C1:  Cari işlemler dengesi
        """
        if not data_config.EVDS_API_KEY:
            logger.warning("EVDS API key not configured. Skipping macro data fetch.")
            return pd.DataFrame()

        end_date = end_date or datetime.now().strftime("%d-%m-%Y")
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365 * data_config.HISTORY_YEARS)).strftime("%d-%m-%Y")

        # EVDS seri kodları (TCMB EVDS API)
        series_codes = [
            "TP.KTFTKR.OR4",    # Faiz oranı
            "TP.KTGLK.YR4",     # Geç likidite penceresi
            "TP.FG.J0",         # TÜFE yıllık
            "TP.FG.J01",        # TÜFE aylık
            "TP.UG.J0",         # ÜFE yıllık
            "TP.UG.J01",        # ÜFE aylık
            "TP.YP.PDS.C008",   # M2 para arzı
            "TP.AB.C1.A001",    # Cari işlemler
        ]

        series_str = "-".join(series_codes)

        try:
            url = f"{data_config.EVDS_BASE_URL}series={series_str}"
            url += f"&startDate={start_date}&endDate={end_date}"
            url += f"&type=json&key={data_config.EVDS_API_KEY}"

            response = requests.get(url, timeout=30, verify=False)

            if response.status_code != 200:
                logger.error(f"EVDS API error: {response.status_code} - {response.text[:200]}")
                return pd.DataFrame()

            data = response.json()

            if "items" not in data:
                logger.warning("No items in EVDS response")
                return pd.DataFrame()

            rows = []
            for item in data["items"]:
                row = {
                    "date": pd.to_datetime(item.get("Tarih", ""), format="%m-%Y", errors="coerce"),
                    "repo_rate": self._safe_float(item.get("TP_KTFTKR_OR4")),
                    "late_liquidity": self._safe_float(item.get("TP_KTGLK_YR4")),
                    "cpi_annual": self._safe_float(item.get("TP_FG_J0")),
                    "cpi_monthly": self._safe_float(item.get("TP_FG_J01")),
                    "ppi_annual": self._safe_float(item.get("TP_UG_J0")),
                    "ppi_monthly": self._safe_float(item.get("TP_UG_J01")),
                    "m2_money_supply": self._safe_float(item.get("TP_YP_PDS_C008")),
                    "current_account": self._safe_float(item.get("TP_AB_C1_A001")),
                }
                rows.append(row)

            df = pd.DataFrame(rows)
            df = df.dropna(subset=["date"])

            if df.empty:
                logger.warning("EVDS returned empty data")
                return pd.DataFrame()

            # Aylık veriyi günlüğe forward-fill ile genişlet
            df = df.set_index("date").resample("D").ffill().reset_index()
            df["date"] = df["date"].dt.date

            # Eksik sütunları ekle
            for col in ["net_reserves", "bisti_rate", "credit_volume"]:
                if col not in df.columns:
                    df[col] = None

            logger.info(f"Fetched {len(df)} rows of EVDS macro data")
            return df

        except Exception as e:
            logger.error(f"Error fetching EVDS data: {e}")
            return pd.DataFrame()

    @staticmethod
    def _safe_float(value) -> float | None:
        """Değeri güvenle float'a çevir"""
        if value is None or value == "" or value == "-":
            return None
        try:
            return float(str(value).replace(",", "."))
        except (ValueError, TypeError):
            return None

    # =========================================================================
    # Veritabanına Kaydet
    # =========================================================================
    def update_all(self) -> dict:
        """Tüm veri kaynaklarını güncelle ve veritabanına kaydet"""
        results = {}

        # 1. XAU/USD
        logger.info("=== Fetching XAU/USD ===")
        xau_df = self.fetch_xau_usd()
        if not xau_df.empty:
            count = db.insert_df("xau_usd", xau_df, on_conflict="DO UPDATE")
            results["xau_usd"] = count
        else:
            results["xau_usd"] = 0

        # 2. USD/TRY
        logger.info("=== Fetching USD/TRY ===")
        usd_df = self.fetch_usd_try()
        if not usd_df.empty:
            count = db.insert_df("usd_try", usd_df, on_conflict="DO UPDATE")
            results["usd_try"] = count
        else:
            results["usd_try"] = 0

        # 3. Brent Petrol
        logger.info("=== Fetching Brent Oil ===")
        brent_df = self.fetch_brent_oil()
        if not brent_df.empty:
            count = db.insert_df("brent_oil", brent_df, on_conflict="DO UPDATE")
            results["brent_oil"] = count
        else:
            results["brent_oil"] = 0

        # 4. Gram Altın TL (AltınKaynak API - Canlı)
        logger.info("=== Fetching Gold Prices from AltınKaynak ===")
        altinkaynak_df = self.fetch_altinkaynak()
        if not altinkaynak_df.empty:
            count = db.insert_df("gold_prices", altinkaynak_df, on_conflict="DO UPDATE")
            results["gold_prices"] = count
        else:
            results["gold_prices"] = 0

        # 5. TCMB EVDS Makro Veriler
        logger.info("=== Fetching EVDS Macro Data ===")
        macro_df = self.fetch_evds_data()
        if not macro_df.empty:
            count = db.insert_df("macro_indicators", macro_df, on_conflict="DO UPDATE")
            results["macro_indicators"] = count
        else:
            results["macro_indicators"] = 0

        logger.info(f"=== Data update complete: {results} ===")
        return results


# Singleton
fetcher = GoldDataFetcher()