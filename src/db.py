import os
from contextlib import contextmanager

import psycopg2
import pandas as pd


def dsn() -> str:
    """Возврат адреса базы данных."""
    return (
        f"postgresql://{os.environ.get('DB_USER', 'admin')}:"
        f"{os.environ.get('DB_PASSWORD', '1')}@"
        f"{os.environ.get('DB_HOST', 'localhost')}:"
        f"{os.environ.get('DB_PORT', '5433')}/"
        f"{os.environ.get('DB_NAME', 'database_ml_project')}"
    )

@contextmanager
def connect():
    """Контекстны менеджер для создания соединения к базе данных."""
    conn = psycopg2.connect(dsn())
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def read_frame(conn, query: str, params=None):
    """Возврат датафрейма по SQL запросу."""
    with conn.cursor() as cur:
        cur.execute(query, params)
        columns = [item[0] for item in cur.description]
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=columns)
