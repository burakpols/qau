"""Calculate gram gold from XAU/USD and USD/TRY"""
from src.data.db import db
import pandas as pd

# XAU/USD ve USD/TRY verilerini çek
xau_usd = db.query_df("SELECT date, close FROM xau_usd ORDER BY date")
usd_try = db.query_df("SELECT date, close FROM usd_try ORDER BY date")

print(f"XAU/USD: {len(xau_usd)} rows")
print(f"USD/TRY: {len(usd_try)} rows")

# Merge
xau_usd.columns = ['date', 'xau_usd_close']
usd_try.columns = ['date', 'usd_try_close']

gold_df = xau_usd.merge(usd_try, on='date', how='inner')
print(f"Merged: {len(gold_df)} rows")

# Gram altın hesapla (1 troy ounce = 31.1035 gram)
gold_df['close'] = gold_df['xau_usd_close'] * gold_df['usd_try_close'] / 31.1035
gold_df['source'] = 'calculated'
gold_df['open'] = gold_df['close']
gold_df['high'] = gold_df['close']
gold_df['low'] = gold_df['close']
gold_df['volume'] = 0
gold_df['change_pct'] = 0

# Sadece gerekli kolonları seç
gold_df = gold_df[['date', 'open', 'high', 'low', 'close', 'volume', 'source', 'change_pct']]

print(f"Calculated gold prices: {len(gold_df)} rows")
print(f"Sample:\n{gold_df.tail()}")

# gold_prices tablosuna kaydet
db.insert_df("gold_prices", gold_df, on_conflict="DO UPDATE")
print("Saved to gold_prices table")

# Verify
result = db.execute("SELECT COUNT(*) as cnt FROM gold_prices")
for row in result:
    print(f"gold_prices now has {row['cnt']} rows")