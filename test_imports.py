"""QAU - Import Test Script"""

def test_imports():
    """Tüm modüllerin import edilebildiğini kontrol et"""
    errors = []

    # Config
    try:
        from src.config import db_config, data_config, log_config
        print("✅ src.config")
    except Exception as e:
        errors.append(f"❌ src.config: {e}")

    # Database
    try:
        from src.data.db import db
        print("✅ src.data.db")
    except Exception as e:
        errors.append(f"❌ src.data.db: {e}")

    # Fetcher
    try:
        from src.data.fetcher import fetcher
        print("✅ src.data.fetcher")
    except Exception as e:
        errors.append(f"❌ src.data.fetcher: {e}")

    # Processor
    try:
        from src.data.processor import processor
        print("✅ src.data.processor")
    except Exception as e:
        errors.append(f"❌ src.data.processor: {e}")

    # Pipeline
    try:
        from src.pipeline import pipeline
        print("✅ src.pipeline")
    except Exception as e:
        errors.append(f"❌ src.pipeline: {e}")

    # Models
    for name in ["arima_model", "xgboost_model", "lstm_model", "ensemble", "base"]:
        try:
            mod = __import__(f"src.models.{name}", fromlist=[""])
            print(f"✅ src.models.{name}")
        except Exception as e:
            errors.append(f"❌ src.models.{name}: {e}")

    # Sentiment
    try:
        from src.sentiment.analyzer import SentimentAnalyzer
        print("✅ src.sentiment.analyzer")
    except Exception as e:
        errors.append(f"❌ src.sentiment.analyzer: {e}")

    # Backtest
    try:
        from src.backtest.engine import BacktestEngine
        print("✅ src.backtest.engine")
    except Exception as e:
        errors.append(f"❌ src.backtest.engine: {e}")

    # LLM
    try:
        from src.llm.assistant import GoldAssistant
        print("✅ src.llm.assistant")
    except Exception as e:
        errors.append(f"❌ src.llm.assistant: {e}")

    # Portfolio
    try:
        from src.portfolio.manager import PortfolioManager
        print("✅ src.portfolio.manager")
    except Exception as e:
        errors.append(f"❌ src.portfolio.manager: {e}")

    print(f"\n{'='*40}")
    if errors:
        print(f"❌ {len(errors)} errors found:")
        for e in errors:
            print(f"  {e}")
    else:
        print("✅ All imports successful!")


if __name__ == "__main__":
    test_imports()