"""Test full data pipeline: fetch + insert + verify"""
from src.data.db import db
from src.data.fetcher import fetcher
from loguru import logger

# 1. Fetch and insert XAU/USD
logger.info("=== Testing XAU/USD ===")
df = fetcher.fetch_xau_usd(period='5d')
logger.info(f"Fetched {len(df)} rows")
if not df.empty:
    count = db.insert_df('xau_usd', df, on_conflict='DO UPDATE')
    logger.info(f"Inserted {count} rows into xau_usd")

# 2. Fetch and insert USD/TRY
logger.info("=== Testing USD/TRY ===")
df = fetcher.fetch_usd_try(period='5d')
logger.info(f"Fetched {len(df)} rows")
if not df.empty:
    count = db.insert_df('usd_try', df, on_conflict='DO UPDATE')
    logger.info(f"Inserted {count} rows into usd_try")

# 3. Fetch and insert Brent Oil
logger.info("=== Testing Brent Oil ===")
df = fetcher.fetch_brent_oil(period='5d')
logger.info(f"Fetched {len(df)} rows")
if not df.empty:
    count = db.insert_df('brent_oil', df, on_conflict='DO UPDATE')
    logger.info(f"Inserted {count} rows into brent_oil")

# 4. Fetch and insert Gram Altın
logger.info("=== Testing Gram Altın ===")
df = fetcher.fetch_gold_prices_bigpara()
logger.info(f"Fetched {len(df)} rows")
if not df.empty:
    count = db.insert_df('gold_prices', df, on_conflict='DO UPDATE')
    logger.info(f"Inserted {count} rows into gold_prices")

# 5. Verify data
logger.info("=== Verifying ===")
for t in ['xau_usd', 'usd_try', 'brent_oil', 'gold_prices']:
    r = db.execute(f"SELECT count(*) as cnt, MAX(date) as latest, MAX(close) as max_close FROM {t}")
    row = r[0]
    logger.info(f"{t}: {row['cnt']} rows, latest={row['latest']}, max_close={row['max_close']}")

# 6. Test processor
logger.info("=== Testing Processor ===")
from src.data.processor import processor
features_df = processor.build_features()
logger.info(f"Features shape: {features_df.shape}")
if not features_df.empty:
    logger.info(f"Feature columns: {list(features_df.columns[:10])}...")
    count = db.insert_df('features', features_df, on_conflict='DO UPDATE')
    logger.info(f"Inserted {count} rows into features")

logger.info("✅ Pipeline test complete!")