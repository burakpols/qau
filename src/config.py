"""QAU - Gram Altın Tahmin Sistemi Konfigürasyonu"""

import os
from pathlib import Path
from dotenv import load_dotenv

# .env dosyasını yükle
load_dotenv()

# Proje kök dizini
BASE_DIR = Path(__file__).resolve().parent.parent


# =============================================================================
# Database Ayarları
# =============================================================================
class DatabaseConfig:
    HOST = os.getenv("DB_HOST", "localhost")
    PORT = int(os.getenv("DB_PORT", 5432))
    NAME = os.getenv("DB_NAME", "qau")
    USER = os.getenv("DB_USER", "qau_user")
    PASSWORD = os.getenv("DB_PASSWORD", "qau_password")

    @property
    def url(self) -> str:
        return f"postgresql://{self.USER}:{self.PASSWORD}@{self.HOST}:{self.PORT}/{self.NAME}"

    @property
    def async_url(self) -> str:
        return f"postgresql+asyncpg://{self.USER}:{self.PASSWORD}@{self.HOST}:{self.PORT}/{self.NAME}"


# =============================================================================
# API Keys
# =============================================================================
class APIConfig:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")


# =============================================================================
# Veri Kaynakları
# =============================================================================
class DataSourceConfig:
    # Yahoo Finance sembolleri
    XAU_USD_SYMBOL = "GC=F"  # Altın futures
    USD_TRY_SYMBOL = "TRY=X"  # USD/TRY kuru

    # Bigpara
    BIGPARA_BASE_URL = os.getenv("BIGPARA_BASE_URL", "https://www.bigpara.com")
    BIGPARA_GOLD_URL = f"{BIGPARA_BASE_URL}/altin/gunluk-altin-tablosu/"

    # TCMB EVDS
    EVDS_API_KEY = os.getenv("EVDS_API_KEY", "")
    EVDS_BASE_URL = "https://evds2.tcmb.gov.tr/service/evds/"

    # NewsAPI
    NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
    NEWS_API_BASE_URL = "https://newsapi.org/v2"
    
    # CollectAPI News
    COLLECTAPI_KEY = os.getenv("COLLECTAPI_KEY", "7owGgfJbjhTE6hIErBVrvB:2avGGXIj0da11f5Ay7SBYm")
    COLLECTAPI_BASE_URL = "https://api.collectapi.com"
    COLLECTAPI_NEWS_TAGS = ["economy", "politics", "world", "business"]
    

    # News fetch queries - geniş kapsamlı haberler
    NEWS_QUERIES = [
        "economics OR economy",
        "politics OR political",
        "war OR conflict OR attack",
        "central bank OR Fed OR ECB",
        "inflation OR interest rate",
        "oil OR crude OR commodity",
        "Turkey economy",
        "global markets",
    ]
    NEWS_KEYWORDS = "USA OR China OR Russia OR Iran OR EU OR Germany OR Turkey"
    NEWS_LANGUAGE = "en"
    NEWS_PAGE_SIZE = 50  # Her query başına max haber sayısı

    # Brent Petrol
    BRENT_SYMBOL = "BZ=F"  # Brent Crude Futures

    # Veri çekme ayarları
    HISTORY_YEARS = 7  # Kaç yıllık geçmiş veri çekilecek
    FETCH_INTERVAL = "1d"  # Günlük veri


# =============================================================================
# Model Ayarları
# =============================================================================
class ModelConfig:
    # ARIMA
    ARIMA_ORDER = (1, 1, 1)
    ARIMA_SEASONAL_ORDER = (1, 1, 1, 7)  # Haftalık mevsimsellik
    ARIMA_AUTO = True  # Auto ARIMA kullan
    ARIMA_WALK_FORWARD_WINDOW = 90  # Walk-forward pencere büyüklüğü

    # XGBoost
    XGBOOST_PARAMS = {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
    }
    XGBOOST_OPTUNA_TRIALS = 50  # Optuna hyperparameter search deneme sayısı
    XGBOOST_USE_CLASSIFIER = True  # Yön tahmini classifier ekle

    # LSTM
    LSTM_SEQUENCE_LENGTH = 30  # 30 günlük pencere
    LSTM_EPOCHS = 100
    LSTM_BATCH_SIZE = 32
    LSTM_HIDDEN_SIZE = 128
    LSTM_NUM_LAYERS = 2
    LSTM_USE_ATTENTION = True  # Attention mekanizması
    LSTM_USE_BIDIRECTIONAL = True  # Bidirectional LSTM

    # Ensemble
    ENSEMBLE_WEIGHTS = {
        "arima": 0.15,
        "xgboost": 0.45,
        "lstm": 0.40,
    }
    ENSEMBLE_USE_STACKING = True  # Meta-learner stacking
    ENSEMBLE_DYNAMIC_WEIGHTS = True  # Performansa göre dinamik ağırlık

    # Eğitim
    TRAIN_TEST_SPLIT = 0.8
    CROSS_VALIDATION_FOLDS = 5

    # Backtesting
    BACKTEST_WALK_FORWARD = True
    BACKTEST_INITIAL_WINDOW = 252  # 1 yıl iş günü
    BACKTEST_STEP = 21  # 1 ay adım


# =============================================================================
# Portföy Ayarları
# =============================================================================
class PortfolioConfig:
    INITIAL_CAPITAL = 100000.0  # TL
    RISK_FREE_RATE = 0.40  # Yıllık risksiz getiri oranı (mevduat)
    MAX_POSITION_RATIO = 0.30  # Tek pozisyonda max %30


# =============================================================================
# Logging
# =============================================================================
class LogConfig:
    LOG_DIR = BASE_DIR / "logs"
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<line>{line}</line> - <level>{message}</level>"


# =============================================================================
# Zamanlama
# =============================================================================
class ScheduleConfig:
    DAILY_FETCH_HOUR = "18:30"  # Piyasa kapanışı sonrası veri çekme
    DAILY_PREDICTION_HOUR = "19:00"  # Tahmin çalışma saati
    DAILY_REPORT_HOUR = "19:30"  # Rapor gönderim saati


# Singleton instances
db_config = DatabaseConfig()
api_config = APIConfig()
data_config = DataSourceConfig()
model_config = ModelConfig()
portfolio_config = PortfolioConfig()
log_config = LogConfig()
schedule_config = ScheduleConfig()

# Convenience alias
settings = api_config
