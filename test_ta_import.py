import traceback
import sys
sys.path.insert(0, '.')

try:
    from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
    from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
    from ta.volatility import BollingerBands, AverageTrueRange
    from ta.volume import OnBalanceVolumeIndicator, MFIIndicator
    from ta.others import CCIIndicator
    print("TA IMPORT SUCCESS")
except ImportError as e:
    print("TA IMPORT FAILED:", e)
    traceback.print_exc()

# Test in processor context
print("\n--- Testing processor import ---")
try:
    from src.data import db  # noqa
    from src.data.processor import HAS_TA
    print("HAS_TA after db import:", HAS_TA)
except Exception as e:
    print("Error:", e)
    traceback.print_exc()