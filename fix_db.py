"""Fix database constraints - TimescaleDB hypertable converted PRIMARY KEY(date)
into UNIQUE(date, source) which breaks ON CONFLICT (date). 
Drop and recreate tables with proper constraints."""

from src.data.db import db
from loguru import logger

# Drop all tables in correct order (respecting dependencies)
tables_to_fix = [
    'gold_weekly', 'gold_monthly',  # materialized views first
    'features', 'predictions', 'portfolio_status', 'portfolio_trades', 
    'news_sentiment', 'model_registry',
    'macro_indicators', 'gold_prices', 'brent_oil', 'usd_try', 'xau_usd',
]

with db.engine.begin() as conn:
    from sqlalchemy import text
    
    # Drop materialized views
    for mv in ['gold_weekly', 'gold_monthly']:
        try:
            conn.execute(text(f'DROP MATERIALIZED VIEW IF EXISTS {mv} CASCADE'))
            logger.info(f"Dropped materialized view {mv}")
        except Exception as e:
            logger.warning(f"Could not drop {mv}: {e}")
    
    # Drop tables
    for t in tables_to_fix:
        if t in ['gold_weekly', 'gold_monthly']:
            continue
        try:
            conn.execute(text(f'DROP TABLE IF EXISTS {t} CASCADE'))
            logger.info(f"Dropped table {t}")
        except Exception as e:
            logger.warning(f"Could not drop {t}: {e}")

# Now re-init schema
db.init_schema()

# Verify constraints
for t in ['xau_usd', 'usd_try', 'brent_oil', 'gold_prices', 'macro_indicators']:
    r = db.execute(f"SELECT constraint_name, constraint_type FROM information_schema.table_constraints WHERE table_name = '{t}'")
    r2 = db.execute(f"SELECT indexname, indexdef FROM pg_indexes WHERE tablename = '{t}'")
    print(f"\n{t}:")
    print(f"  Constraints: {r}")
    print(f"  Indexes: {r2}")

print("\n✅ Database fixed!")