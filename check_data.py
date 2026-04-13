"""Check data counts"""
from src.data.db import db

# Check gold prices
result = db.execute("SELECT COUNT(*) as cnt, MIN(date) as min_date, MAX(date) as max_date FROM gold_prices")
print("Gold prices table:")
for row in result:
    print(f"  Count: {row['cnt']}, Min: {row['min_date']}, Max: {row['max_date']}")

# Check XAU/USD
result2 = db.execute("SELECT COUNT(*) as cnt, MIN(date) as min_date, MAX(date) as max_date FROM xau_usd")
print("XAU/USD table:")
for row in result2:
    print(f"  Count: {row['cnt']}, Min: {row['min_date']}, Max: {row['max_date']}")

# Check all tables
tables = ['gold_prices', 'xau_usd', 'usd_try', 'brent_oil', 'macro_indicators', 'news_sentiment']
for t in tables:
    try:
        result = db.execute(f"SELECT COUNT(*) as cnt FROM {t}")
        for row in result:
            print(f"{t}: {row['cnt']} rows")
    except Exception as e:
        print(f"{t}: Error - {e}")