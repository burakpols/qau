"""Check current database schema"""
from src.data.db import db

# Check features columns
print("=== Features table columns ===")
result = db.execute("""
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'features'
ORDER BY ordinal_position
""")
for row in result:
    print(f"  {row['column_name']}: {row['data_type']}")

# Check gold_prices columns
print("\n=== Gold prices table columns ===")
result = db.execute("""
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'gold_prices'
ORDER BY ordinal_position
""")
for row in result:
    print(f"  {row['column_name']}: {row['data_type']}")