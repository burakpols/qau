"""QAU - PostgreSQL + TimescaleDB Veritabanı Yönetimi"""

from pathlib import Path

import pandas as pd
from loguru import logger
from sqlalchemy import create_engine, text

from src.config import db_config


class Database:
    """PostgreSQL veritabanı yönetim sınıfı"""

    def __init__(self):
        self._engine = None

    @property
    def engine(self):
        if self._engine is None:
            self._engine = create_engine(
                db_config.url,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True,
            )
        return self._engine

    def init_schema(self) -> None:
        """Veritabanı şemasını oluştur (schema.sql dosyasından)"""
        schema_path = Path(__file__).parent / "schema.sql"
        if not schema_path.exists():
            logger.error(f"Schema file not found: {schema_path}")
            return

        schema_sql = schema_path.read_text(encoding="utf-8")

        # Parse statements
        statements = []
        current = []
        for line in schema_sql.split("\n"):
            stripped = line.strip()
            if stripped.startswith("--") or not stripped:
                continue
            current.append(line)
            if stripped.endswith(";"):
                statements.append("\n".join(current))
                current = []

        # Run each statement in its own transaction to avoid cascading failures
        errors = 0
        for stmt in statements:
            stmt = stmt.strip()
            if not stmt:
                continue
            try:
                with self.engine.begin() as conn:
                    conn.execute(text(stmt))
            except Exception as e:
                err_str = str(e).lower()
                if "already exists" in err_str:
                    pass  # Table/index already exists, fine
                else:
                    logger.warning(f"Schema stmt skipped: {str(e)[:120]}")
                    errors += 1

        if errors:
            logger.warning(f"Schema initialized with {errors} warnings")
        else:
            logger.info("Database schema initialized successfully")

    def execute(self, query: str, params: dict = None) -> list[dict]:
        """SQL sorgusu çalıştır ve sonuç döndür"""
        with self.engine.connect() as conn:
            result = conn.execute(text(query), params or {})
            return [dict(row._mapping) for row in result]

    def query_df(self, query: str, params: dict = None) -> pd.DataFrame:
        """SQL sorgusu çalıştır ve DataFrame döndür"""
        with self.engine.connect() as conn:
            return pd.read_sql(text(query), conn, params=params or {})

    def insert_df(self, table: str, df: pd.DataFrame, on_conflict: str = "DO NOTHING") -> int:
        """DataFrame'i veritabanına kaydet (upsert)"""
        if df.empty:
            return 0

        try:
            with self.engine.begin() as conn:
                # Kullanılabilir sütunları al
                cols = list(df.columns)
                placeholders = ", ".join([f":{c}" for c in cols])
                col_names = ", ".join(cols)

                if on_conflict == "DO UPDATE":
                    update_cols = [c for c in cols if c != "date"]
                    update_stmt = ", ".join([f"{c} = EXCLUDED.{c}" for c in update_cols])
                    sql = f"INSERT INTO {table} ({col_names}) VALUES ({placeholders}) ON CONFLICT (date) DO UPDATE SET {update_stmt}"
                else:
                    sql = f"INSERT INTO {table} ({col_names}) VALUES ({placeholders}) ON CONFLICT (date) {on_conflict}"

                rows = df.to_dict("records")
                conn.execute(text(sql), rows)
                return len(rows)

        except Exception as e:
            logger.error(f"Error inserting into {table}: {e}")
            return 0

    def get_latest_date(self, table: str) -> str | None:
        """Tablodaki en son tarih"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(f"SELECT MAX(date) FROM {table}"))
                row = result.fetchone()
                return str(row[0]) if row and row[0] else None
        except Exception:
            return None

    def table_exists(self, table: str) -> bool:
        """Tablo var mı kontrol et"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(
                    text("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = :name)"),
                    {"name": table},
                )
                return result.scalar()
        except Exception:
            return False


# Singleton
db = Database()


def get_db_connection():
    """Direct psycopg2 connection for news_fetcher compatibility"""
    import psycopg2
    return psycopg2.connect(
        host=db_config.HOST,
        port=db_config.PORT,
        database=db_config.NAME,
        user=db_config.USER,
        password=db_config.PASSWORD
    )
