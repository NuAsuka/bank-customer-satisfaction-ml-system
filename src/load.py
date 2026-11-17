import pandas as pd
from psycopg2.extras import execute_values

TIMEZONE = "Europe/Moscow"
REVIEW_COLUMNS = ["id_client", "review_dttm", "review_source", "review_text"]
ENRICHMENT_COLUMNS = ["id_client", "review_dttm", "review_source", "product", "review_emotion"]


def _value(value):
    """Приведение nan, пустых строк или строк состоящих только из пробелов к None, 
    остальные значения не меняются
    """
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        value = value.strip()
        return value or None
    return value


def read_reviews(path) -> pd.DataFrame:
    """Загрузка отзывов"""
    df = pd.read_csv(path)
    for column in REVIEW_COLUMNS:
        if column not in df.columns:
            raise ValueError(f"в файле нет колонки {column}")
    df["review_dttm"] = pd.to_datetime(df["review_dttm"]).dt.tz_localize(TIMEZONE)
    return df


def read_enrichment(path) -> pd.DataFrame:
    """Загрузка разметки"""
    df = pd.read_csv(path)
    for column in ENRICHMENT_COLUMNS:
        if column not in df.columns:
            raise ValueError(f"в файле нет колонки {column}")
    df["review_dttm"] = pd.to_datetime(df["review_dttm"]).dt.tz_localize(TIMEZONE)
    return df


def period(df: pd.DataFrame) -> tuple:
    """Получение картёжа с максимальным и минимальным значением даты в переданном датафрейме."""
    dates = df["review_dttm"].dt.date
    return dates.min(), dates.max()


def load_reviews(conn, batch_review_id: int, path) -> dict:
    """Загрузка отзывов в raw слой базы данных.
    Проставляются id батча загрузки.
    """
    df = read_reviews(path)
    start, end = period(df)
    rows = [
        (
            _value(row.id_client),
            row.review_dttm,
            batch_review_id,
            _value(row.review_source),
            _value(row.review_text),
        )
        for row in df.itertuples(index=False)
    ]
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM raw.review WHERE batch_review_id = %s", (batch_review_id,))
        before = cur.fetchone()[0]
        execute_values(
            cur,
            """
            INSERT INTO raw.review
                (id_client, review_dttm, batch_review_id, review_source, review_text)
            VALUES %s
            ON CONFLICT (id_client, review_dttm) DO NOTHING
            """,
            rows,
            page_size=1000,
        )
        cur.execute("SELECT count(*) FROM raw.review WHERE batch_review_id = %s", (batch_review_id,))
        inserted = cur.fetchone()[0] - before
    conn.commit()
    return {"rows": len(rows), "inserted": int(inserted), "period_start": start, "period_end": end}


def load_enrichment(conn, batch_enrichment_id: int, path) -> dict:
    """Загрузка разметки в raw слой базы данных."""
    df = read_enrichment(path)
    start, end = period(df)
    rows = [
        (
            _value(row.id_client),
            row.review_dttm,
            batch_enrichment_id,
            _value(row.review_source),
            _value(row.product),
            _value(row.review_emotion),
        )
        for row in df.itertuples(index=False)
    ]
    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM raw.review_enrichment WHERE batch_enrichment_id = %s",
            (batch_enrichment_id,),
        )
        before = cur.fetchone()[0]
        execute_values(
            cur,
            """
            INSERT INTO raw.review_enrichment
                (id_client, review_dttm, batch_enrichment_id, review_source, product, review_emotion)
            VALUES %s
            ON CONFLICT (id_client, review_dttm) DO UPDATE
               SET batch_enrichment_id = EXCLUDED.batch_enrichment_id,
                   review_source = COALESCE(EXCLUDED.review_source, raw.review_enrichment.review_source),
                   product = COALESCE(EXCLUDED.product, raw.review_enrichment.product),
                   review_emotion = COALESCE(EXCLUDED.review_emotion, raw.review_enrichment.review_emotion),
                   received_at = now()
            """,
            rows,
            page_size=1000,
        )
        cur.execute(
            "SELECT count(*) FROM raw.review_enrichment WHERE batch_enrichment_id = %s",
            (batch_enrichment_id,),
        )
        inserted = cur.fetchone()[0] - before
    conn.commit()
    return {"rows": len(rows), "inserted": int(inserted), "period_start": start, "period_end": end}
