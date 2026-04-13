-- QAU Database Schema
-- Gram Altın Tahmin Sistemi

-- TimescaleDB extension'ını etkinleştir
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- =============================================================================
-- Gram Altın TL Fiyatları
-- =============================================================================
CREATE TABLE IF NOT EXISTS gold_prices (
    date        DATE NOT NULL,
    open        DECIMAL(12,4),
    high        DECIMAL(12,4),
    low         DECIMAL(12,4),
    close       DECIMAL(12,4),
    volume      BIGINT DEFAULT 0,
    source      VARCHAR(50) DEFAULT 'bigpara',
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(date, source)
);

SELECT create_hypertable('gold_prices', 'date', if_not_exists => TRUE);

-- =============================================================================
-- USD/TRY Kurları
-- =============================================================================
CREATE TABLE IF NOT EXISTS usd_try (
    date        DATE NOT NULL,
    open        DECIMAL(12,6),
    high        DECIMAL(12,6),
    low         DECIMAL(12,6),
    close       DECIMAL(12,6),
    source      VARCHAR(50) DEFAULT 'yahoo_finance',
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(date, source)
);

SELECT create_hypertable('usd_try', 'date', if_not_exists => TRUE);

-- =============================================================================
-- Ons Altın (XAU/USD)
-- =============================================================================
CREATE TABLE IF NOT EXISTS xau_usd (
    date        DATE NOT NULL,
    open        DECIMAL(12,4),
    high        DECIMAL(12,4),
    low         DECIMAL(12,4),
    close       DECIMAL(12,4),
    volume      BIGINT DEFAULT 0,
    source      VARCHAR(50) DEFAULT 'yahoo_finance',
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(date, source)
);

SELECT create_hypertable('xau_usd', 'date', if_not_exists => TRUE);

-- =============================================================================
-- Makro Ekonomik Göstergeler
-- =============================================================================
CREATE TABLE IF NOT EXISTS macro_indicators (
    date            DATE NOT NULL,
    indicator_code  VARCHAR(20) NOT NULL,
    value           DECIMAL(14,4),
    source          VARCHAR(50) DEFAULT 'tcmb',
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(date, indicator_code, source)
);

SELECT create_hypertable('macro_indicators', 'date', if_not_exists => TRUE);

-- =============================================================================
-- Haber Sentiment Analizi
-- =============================================================================
CREATE TABLE IF NOT EXISTS news_sentiment (
    id              SERIAL,
    date            DATE NOT NULL,
    title           TEXT,
    content         TEXT,
    source          VARCHAR(100),
    url             TEXT,
    sentiment_score DECIMAL(5,4),  -- -1.0 ile 1.0 arası
    sentiment_label VARCHAR(20),    -- positive, negative, neutral
    keywords        TEXT[],
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(date, title, source)
);

-- =============================================================================
-- Model Tahminleri
-- =============================================================================
CREATE TABLE IF NOT EXISTS predictions (
    id              SERIAL,
    date            DATE NOT NULL,          -- Tahmin edilen tarih
    model_name      VARCHAR(50) NOT NULL,   -- arima, xgboost, lstm, ensemble
    predicted_close DECIMAL(12,4),
    confidence_low  DECIMAL(12,4),          -- Güven aralığı alt
    confidence_high DECIMAL(12,4),          -- Güven aralığı üst
    actual_close    DECIMAL(12,4),          -- Gerçekleşen (sonradan güncellenir)
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(date, model_name)
);

SELECT create_hypertable('predictions', 'date', if_not_exists => TRUE);

-- =============================================================================
-- AL/SAT Sinyalleri
-- =============================================================================
CREATE TABLE IF NOT EXISTS signals (
    id              SERIAL,
    date            DATE NOT NULL,
    signal_type     VARCHAR(10) NOT NULL,   -- BUY, SELL, HOLD
    confidence      DECIMAL(5,2),            -- 0-100 arası güven skoru
    predicted_price DECIMAL(12,4),
    reasoning       TEXT,                    -- LLM açıklaması
    model_name      VARCHAR(50) DEFAULT 'ensemble',
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =============================================================================
-- Portföy İşlemleri
-- =============================================================================
CREATE TABLE IF NOT EXISTS portfolio (
    id              SERIAL,
    date            DATE NOT NULL,
    action          VARCHAR(10) NOT NULL,    -- BUY, SELL
    quantity_gram   DECIMAL(10,4) NOT NULL,  -- Gram miktarı
    price_per_gram  DECIMAL(12,4) NOT NULL,  -- Gram başına TL fiyatı
    total_amount    DECIMAL(14,4) NOT NULL,  -- Toplam TL tutar
    fee             DECIMAL(10,4) DEFAULT 0,
    notes           TEXT,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =============================================================================
-- Portföy Durumu (Snapshot)
-- =============================================================================
CREATE TABLE IF NOT EXISTS portfolio_status (
    id                  SERIAL,
    date                DATE NOT NULL,
    total_grams         DECIMAL(10,4),      -- Toplam gram altın
    avg_buy_price       DECIMAL(12,4),      -- Ortalama alış fiyatı
    current_price       DECIMAL(12,4),      -- Güncel gram altın fiyatı
    portfolio_value     DECIMAL(14,4),       -- Portföy TL değeri
    cash_balance        DECIMAL(14,4),      -- Nakit TL bakiye
    total_assets        DECIMAL(14,4),      -- Toplam varlık (altın + nakit)
    pnl                 DECIMAL(14,4),       -- Kar/Zarar
    pnl_percentage      DECIMAL(7,4),       -- Kar/Zarar %
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(date)
);

-- =============================================================================
-- İndeksler
-- =============================================================================
CREATE INDEX idx_gold_prices_date ON gold_prices (date DESC);
CREATE INDEX idx_usd_try_date ON usd_try (date DESC);
CREATE INDEX idx_xau_usd_date ON xau_usd (date DESC);
CREATE INDEX idx_predictions_date_model ON predictions (date DESC, model_name);
CREATE INDEX idx_signals_date ON signals (date DESC);
CREATE INDEX idx_portfolio_date ON portfolio (date DESC);
CREATE INDEX idx_news_date ON news_sentiment (date DESC);
CREATE INDEX idx_macro_date_code ON macro_indicators (date DESC, indicator_code);