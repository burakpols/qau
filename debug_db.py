from src.data.db import db

result = db.execute("SELECT constraint_name, constraint_type FROM information_schema.table_constraints WHERE table_name = 'xau_usd'")
print('xau_usd constraints:', result)

result2 = db.execute("SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'xau_usd'")
print('xau_usd indexes:', result2)

for t in ['xau_usd', 'usd_try', 'brent_oil', 'gold_prices', 'macro_indicators']:
    r = db.execute(f"SELECT count(*) as cnt FROM {t}")
    print(f'{t}: {r[0]["cnt"]} rows')