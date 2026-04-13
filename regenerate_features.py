"""Regenerate features from gold_prices"""
from src.data.processor import FeatureProcessor

engineer = FeatureProcessor()
df = engineer.build_features()
print(f"Generated {len(df)} feature rows")
print(f"Sample columns: {list(df.columns[:10])}")
print(df.tail(3))