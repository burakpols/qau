-- QAU - Gram Altın Tahmin Sistemi Veritabanı Şeması
-- PostgreSQL (plain tables, no TimescaleDB)

-- =============================================================================
-- Fiyat Verileri
-- =============================================================================

CREATE TABLE IF NOT EXISTS gold_prices (
    date        DATE NOT NULL,
    open        DOUBLE PRECISION,
    high        DOUBLE PRECISION,
    low         DOUBLE PRECISION,
    close       DOUBLE PRECISION,
    volume      DOUBLE PRECISION DEFAULT 0,
    source      VARCHAR(50) DEFAULT 'altinkaynak',
    change_pct  DOUBLE PRECISION,
    fetched_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (date)
);

CREATE TABLE IF NOT EXISTS xau_usd (
    date        DATE NOT NULL,
    open        DOUBLE PRECISION,
    high        DOUBLE PRECISION,
    low         DOUBLE PRECISION,
    close       DOUBLE PRECISION,
    volume      DOUBLE PRECISION DEFAULT 0,
    source      VARCHAR(50) DEFAULT 'yahoo_finance',
    PRIMARY KEY (date)
);

CREATE TABLE IF NOT EXISTS usd_try (
    date        DATE NOT NULL,
    open        DOUBLE PRECISION,
    high        DOUBLE PRECISION,
    low         DOUBLE PRECISION,
    close       DOUBLE PRECISION,
    source      VARCHAR(50) DEFAULT 'yahoo_finance',
    PRIMARY KEY (date)
);

CREATE TABLE IF NOT EXISTS brent_oil (
    date        DATE NOT NULL,
    open        DOUBLE PRECISION,
    high        DOUBLE PRECISION,
    low         DOUBLE PRECISION,
    close       DOUBLE PRECISION,
    volume      DOUBLE PRECISION DEFAULT 0,
    source      VARCHAR(50) DEFAULT 'yahoo_finance',
    PRIMARY KEY (date)
);

-- =============================================================================
-- TCMB EVDS Makroekonomik Veriler
-- =============================================================================

CREATE TABLE IF NOT EXISTS macro_indicators (
    date            DATE NOT NULL,
    repo_rate        DOUBLE PRECISION,
    late_liquidity   DOUBLE PRECISION,
    bisti_rate       DOUBLE PRECISION,
    cpi_annual       DOUBLE PRECISION,
    cpi_monthly      DOUBLE PRECISION,
    ppi_annual       DOUBLE PRECISION,
    ppi_monthly      DOUBLE PRECISION,
    m2_money_supply  DOUBLE PRECISION,
    net_reserves     DOUBLE PRECISION,
    current_account  DOUBLE PRECISION,
    credit_volume    DOUBLE PRECISION,
    PRIMARY KEY (date)
);

-- =============================================================================
-- Feature Engineering Çıktısı
-- =============================================================================

CREATE TABLE IF NOT EXISTS features (
    date                DATE NOT NULL,
    close               DOUBLE PRECISION,
    return_1d           DOUBLE PRECISION,
    return_5d           DOUBLE PRECISION,
    return_20d          DOUBLE PRECISION,
    volatility_5d       DOUBLE PRECISION,
    volatility_20d      DOUBLE PRECISION,
    sma_5               DOUBLE PRECISION,
    sma_10              DOUBLE PRECISION,
    sma_20              DOUBLE PRECISION,
    sma_50              DOUBLE PRECISION,
    sma_200             DOUBLE PRECISION,
    ema_12              DOUBLE PRECISION,
    ema_26              DOUBLE PRECISION,
    macd                DOUBLE PRECISION,
    macd_signal         DOUBLE PRECISION,
    macd_hist           DOUBLE PRECISION,
    rsi_14              DOUBLE PRECISION,
    stoch_k             DOUBLE PRECISION,
    stoch_d             DOUBLE PRECISION,
    bb_upper            DOUBLE PRECISION,
    bb_lower            DOUBLE PRECISION,
    bb_width            DOUBLE PRECISION,
    bb_pct              DOUBLE PRECISION,
    atr_14              DOUBLE PRECISION,
    adx_14              DOUBLE PRECISION,
    cci_14              DOUBLE PRECISION,
    willr_14            DOUBLE PRECISION,
    obv                 DOUBLE PRECISION,
    mfi_14              DOUBLE PRECISION,
    xau_usd_close       DOUBLE PRECISION,
    usd_try_close       DOUBLE PRECISION,
    brent_close         DOUBLE PRECISION,
    xau_usd_return_1d   DOUBLE PRECISION,
    usd_try_return_1d   DOUBLE PRECISION,
    brent_return_1d     DOUBLE PRECISION,
    gold_usd_ratio      DOUBLE PRECISION,
    repo_rate            DOUBLE PRECISION,
    cpi_annual           DOUBLE PRECISION,
    m2_money_supply      DOUBLE PRECISION,
    net_reserves         DOUBLE PRECISION,
    real_interest_rate   DOUBLE PRECISION,
    sentiment_score      DOUBLE PRECISION,
    sentiment_ma7        DOUBLE PRECISION,
    news_count           DOUBLE PRECISION,
    news_count_7d        DOUBLE PRECISION,
    regime               INTEGER DEFAULT 0,
    target_close         DOUBLE PRECISION,
    target_return        DOUBLE PRECISION,
    target_direction     INTEGER,
    PRIMARY KEY (date)
);

-- =============================================================================
-- Model Tahminleri
-- =============================================================================

CREATE TABLE IF NOT EXISTS predictions (
    id              SERIAL,
    date            DATE NOT NULL,
    model_name      VARCHAR(50) NOT NULL,
    predicted_close DOUBLE PRECISION,
    confidence_low  DOUBLE PRECISION,
    confidence_high DOUBLE PRECISION,
    actual_close    DOUBLE PRECISION,
    direction_pred  INTEGER,
    direction_actual INTEGER,
    created_at      TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (date, model_name)
);

-- =============================================================================
-- Model Performans Kayıtları
-- =============================================================================

CREATE TABLE IF NOT EXISTS model_registry (
    id              SERIAL PRIMARY KEY,
    model_name      VARCHAR(50) NOT NULL,
    version         VARCHAR(20) NOT NULL,
    params          JSONB,
    metrics         JSONB,
    feature_importance JSONB,
    trained_at      TIMESTAMP DEFAULT NOW(),
    is_active       BOOLEAN DEFAULT TRUE,
    UNIQUE(model_name, version)
);

-- =============================================================================
-- Portföy Durumu
-- =============================================================================

CREATE TABLE IF NOT EXISTS portfolio_status (
    date            DATE NOT NULL,
    total_grams     DOUBLE PRECISION DEFAULT 0,
    avg_buy_price   DOUBLE PRECISION DEFAULT 0,
    current_price   DOUBLE PRECISION,
    portfolio_value DOUBLE PRECISION,
    cash_balance    DOUBLE PRECISION,
    total_assets    DOUBLE PRECISION,
    pnl             DOUBLE PRECISION,
    pnl_percentage  DOUBLE PRECISION,
    PRIMARY KEY (date)
);

-- =============================================================================
-- Portföy İşlemleri
-- =============================================================================

CREATE TABLE IF NOT EXISTS portfolio_trades (
    id              SERIAL PRIMARY KEY,
    date            DATE NOT NULL,
    action          VARCHAR(10) NOT NULL,
    grams           DOUBLE PRECISION NOT NULL,
    price           DOUBLE PRECISION NOT NULL,
    amount_tl       DOUBLE PRECISION NOT NULL,
    reason          TEXT,
    model_confidence DOUBLE PRECISION,
    created_at      TIMESTAMP DEFAULT NOW()
);

-- =============================================================================
-- Haber / Sentiment Verileri
-- =============================================================================

CREATE TABLE IF NOT EXISTS news_sentiment (
    id                SERIAL PRIMARY KEY,
    date              DATE NOT NULL,
    source            VARCHAR(100),
    title             TEXT,
    summary           TEXT,
    url               TEXT,
    sentiment_score   DOUBLE PRECISION,
    relevance_score   DOUBLE PRECISION DEFAULT 0.5,
    gold_relevant     BOOLEAN DEFAULT FALSE,
    impact_direction  VARCHAR(20),  -- positive, negative, neutral
    impact_reason     TEXT,
    keywords          TEXT[],
    created_at        TIMESTAMP DEFAULT NOW(),
    CONSTRAINT valid_impact_direction CHECK (impact_direction IN ('positive', 'negative', 'neutral', NULL))
);

-- =============================================================================
-- İndeksler
-- =============================================================================

CREATE INDEX IF NOT EXISTS idx_predictions_model ON predictions(model_name, date DESC);
CREATE INDEX IF NOT EXISTS idx_features_date ON features(date DESC);
CREATE INDEX IF NOT EXISTS idx_news_date ON news_sentiment(date DESC);
CREATE INDEX IF NOT EXISTS idx_macro_date ON macro_indicators(date DESC);
CREATE INDEX IF NOT EXISTS idx_trades_date ON portfolio_trades(date DESC);