from datetime import timedelta

import pandas as pd

from .db import read_frame
from psycopg2.extras import execute_values

MIN_TEXT_LEN = 20
MAX_AGE_DAYS = 3 * 365
EMOTIONS = frozenset({"Позитивный", "Негативный"})

PRODUCTS = frozenset({
    "дебетовая карта", "premium", "взр", "осаго", "не определено", "расчетный счет",
    "кредитная карта", "брокерский счет", "каско", "автокредит в салонах", "подписки",
    "онлайн-бухгалтерия", "кредит наличными", "мобайл", "автокредит наличными", "ифл",
    "вклад", "lifestyle", "кредит на покупку", "торговый эквайринг", "кредиты",
    "интернет эквайринг", "иис", "кредит наличными под залог недвижимости", "банкоматы",
    "бизнес-залы", "путешествия", "collection", "долями", "кредитование покупателей",
    "накопительный счет", "вэд", "рефинансирование ипотеки", "junior", "зарплатный проект",
    "ипотека на вторичном рынке", "кубышка", "кредит наличными под залог авто",
    "онлайн-касса", "инвесткопилка", "бухгалтерское обслуживание", "бизнес-карта",
    "ипотечное страхование", "private", "рефинансирование автокредита", "рефинансирование",
    "рассрочка по кнопке", "выгода", "образовательный кредит", "совместный счет",
    "ипотека на первичном рынке", "рефинансирование кн", "нс", "овернайт", "отели",
    "кредит наличными на образование", "самозанятые", "депозит",
    "рефинансирование ипотеки для сотрудника", "инвестиции для бизнеса",
})

SOURCES = frozenset({
    "appstore", "asn", "banki.ru", "instagramm", "irecommend", "otzovik", "sravni.ru",
    "twitter", "vc", "youtube", "вконтакте", "гугл карты", "иная площадка", "нет",
    "одноклассники", "пикабу", "пульс", "суперапп", "форум banki.ru", "яндекс карты",
})


def normalize_product(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip().lower()
    return text or None


def normalize_source(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip().lower()
    return text or None


def _reason_review(row, seen_texts, batch_date):
    text = row.review_text
    if text is None or (isinstance(text, float) and pd.isna(text)) or not str(text).strip():
        return "пустой текст"
    text = str(text).strip()
    if len(text) < MIN_TEXT_LEN:
        return "текст короче 20 символов"
    if text in seen_texts:
        return "повтор текста"
    moment = row.review_dttm
    if moment is None or pd.isna(moment):
        return "дата не разобрана"
    moment_date = moment.date()
    if moment_date > batch_date:
        return "дата в будущем"
    if moment_date < batch_date - timedelta(days=MAX_AGE_DAYS):
        return "дата старше 3 лет"
    if normalize_source(row.review_source) not in SOURCES:
        return "площадка вне справочника"
    return None


def check_reviews(conn, batch_review_id: int) -> tuple[int, int]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT loaded_at::date FROM etl.review_batches WHERE batch_review_id = %s",
            (batch_review_id,),
        )
        row = cur.fetchone()
    if row is None:
        raise ValueError(f"порция {batch_review_id} не зарегистрирована")
    batch_date = row[0]

    df = read_frame(
        conn,
        "SELECT id_client, review_dttm, review_source, review_text "
        "FROM raw.review WHERE batch_review_id = %s ORDER BY review_dttm",
        (batch_review_id,),
    )
    other = read_frame(
        conn,
        "SELECT review_text FROM raw.review WHERE batch_review_id <> %s AND review_text IS NOT NULL",
        (batch_review_id,),
    )
    seen = {str(value).strip() for value in other["review_text"].tolist()}

    accepted = []
    rejected = []
    for row in df.itertuples(index=False):
        reason = _reason_review(row, seen, batch_date)
        if reason:
            rejected.append((row.id_client, row.review_dttm, batch_review_id, reason))
            continue
        text = str(row.review_text).strip()
        seen.add(text)
        accepted.append(
            (
                row.id_client,
                row.review_dttm,
                batch_review_id,
                normalize_source(row.review_source),
                text,
            )
        )

    with conn.cursor() as cur:
        if accepted:
            execute_values(
                cur,
                """
                INSERT INTO staging.review_clean
                    (id_client, review_dttm, batch_review_id, review_source, review_text)
                VALUES %s
                ON CONFLICT (id_client, review_dttm) DO UPDATE
                   SET batch_review_id = EXCLUDED.batch_review_id,
                       review_source = EXCLUDED.review_source,
                       review_text = EXCLUDED.review_text
                """,
                accepted,
                page_size=1000,
            )
        if rejected:
            execute_values(
                cur,
                """
                INSERT INTO quarantine.review
                    (id_client, review_dttm, batch_review_id, reason_text)
                VALUES %s
                ON CONFLICT (id_client, review_dttm) DO UPDATE
                   SET reason_text = EXCLUDED.reason_text,
                       quarantine_at = now()
                """,
                rejected,
                page_size=1000,
            )
    conn.commit()
    return len(accepted), len(rejected)


def _reason_enrichment(row, known_reviews):
    if (row.id_client, row.review_dttm) not in known_reviews:
        return "отзыв не найден в raw.review"
    if normalize_product(row.product) not in PRODUCTS:
        return "продукт вне справочника"
    emotion = None if row.review_emotion is None else str(row.review_emotion).strip()
    if emotion not in EMOTIONS:
        return "эмоция вне справочника"
    source = normalize_source(row.review_source)
    if source is not None and source not in SOURCES:
        return "площадка вне справочника"
    return None


def check_enrichment(conn, batch_enrichment_id: int) -> tuple[int, int]:
    df = read_frame(
        conn,
        "SELECT id_client, review_dttm, review_source, product, review_emotion "
        "FROM raw.review_enrichment WHERE batch_enrichment_id = %s ORDER BY review_dttm",
        (batch_enrichment_id,),
    )
    known = read_frame(conn, "SELECT id_client, review_dttm FROM raw.review")
    known_reviews = set(zip(known["id_client"], known["review_dttm"]))

    accepted = []
    rejected = []
    for row in df.itertuples(index=False):
        reason = _reason_enrichment(row, known_reviews)
        if reason:
            rejected.append((row.id_client, row.review_dttm, batch_enrichment_id, reason))
            continue
        accepted.append(
            (
                row.id_client,
                row.review_dttm,
                batch_enrichment_id,
                normalize_source(row.review_source),
                normalize_product(row.product),
                str(row.review_emotion).strip(),
            )
        )

    with conn.cursor() as cur:
        if accepted:
            execute_values(
                cur,
                """
                INSERT INTO staging.review_enrichment_clean
                    (id_client, review_dttm, batch_enrichment_id, review_source, product, review_emotion)
                VALUES %s
                ON CONFLICT (id_client, review_dttm) DO UPDATE
                   SET batch_enrichment_id = EXCLUDED.batch_enrichment_id,
                       review_source = EXCLUDED.review_source,
                       product = EXCLUDED.product,
                       review_emotion = EXCLUDED.review_emotion
                """,
                accepted,
                page_size=1000,
            )
        if rejected:
            execute_values(
                cur,
                """
                INSERT INTO quarantine.review_enrichment
                    (id_client, review_dttm, batch_enrichment_id, reason_text)
                VALUES %s
                ON CONFLICT (id_client, review_dttm) DO UPDATE
                   SET reason_text = EXCLUDED.reason_text,
                       quarantine_at = now()
                """,
                rejected,
                page_size=1000,
            )
    conn.commit()
    return len(accepted), len(rejected)
