#!/usr/bin/env python3
"""Загрузка history.csv в нормализованную ER-модель staging (PostgreSQL).

Порядок загрузки:
    справочники -> company/business_line/product -> theme/subtheme
    -> client -> review.

Скрипт рассчитан на первичную загрузку истории: перед вставкой таблицы
очищаются, поэтому повторный запуск приводит БД к тому же состоянию.

Запуск:
    python src/pipeline/load_history.py data/incoming/history.csv
    python src/pipeline/load_history.py data/incoming/history.csv \
        --dsn postgresql://admin:1@localhost:5433/database
"""

from __future__ import annotations

import argparse
import os

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

SCHEMA = "staging"
DEFAULT_DSN = "postgresql://admin:1@localhost:5433/database"
TIMEZONE = "Europe/Moscow"  # время источника без таймзоны трактуем как МСК

COLUMNS = [
    "review_dttm", "finish_dttm", "id_client", "company", "review_mark",
    "review_emotion", "review_source", "business_line", "product", "reason",
    "review_theme", "subtheme", "review_text", "solution_flg", "gender_cd",
    "education_level_cd", "marital_status_cd", "children_cnt",
    "citizenship_country", "segment_name", "subscription_important_flg",
    "new_flg", "influencer_flg", "age_segment", "csat_score",
]

CLIENT_ATTRS = [
    "gender_cd", "education_level_cd", "marital_status_cd", "children_cnt",
    "citizenship_country", "segment_name", "subscription_important_flg",
    "new_flg", "influencer_flg", "age_segment",
]

TABLES = [
    "review", "client", "product", "business_line", "company",
    "review_subtheme", "review_theme", "review_source", "review_emotion",
    "review_mark", "review_reason", "review_solution", "gender",
    "education_level", "marital_status", "country", "client_segment",
    "age_segment",
]


# ---------------------------------------------------------------------------
# Приведение значений
# ---------------------------------------------------------------------------

def is_null(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def clean_text(value) -> str | None:
    """Убирает пробелы по краям и повторные пробелы, сохраняя регистр."""
    if is_null(value):
        return None
    text = " ".join(str(value).split())
    return text or None


def norm_text(value) -> str | None:
    """Нормализованный код справочника: регистр и пробелы не различаются."""
    text = clean_text(value)
    return text.casefold() if text else None


def clean_long_text(value) -> str | None:
    """Текст отзыва: убираем только внешние пробелы, переносы строк сохраняем."""
    if is_null(value):
        return None
    text = str(value).strip()
    return text or None


def parse_int(value) -> int | None:
    if is_null(value):
        return None
    try:
        number = float(str(value).replace(",", "."))
    except ValueError:
        return None
    return int(number) if number.is_integer() else None


def parse_bool(value) -> bool | None:
    if is_null(value):
        return None
    text = norm_text(value)
    if text in {"1", "1.0", "true", "t", "yes", "y", "да"}:
        return True
    if text in {"0", "0.0", "false", "f", "no", "n", "нет"}:
        return False
    return None


def parse_mark_value(value) -> int | None:
    """Числовая часть review_mark; у значений «без оценки» её нет."""
    if is_null(value):
        return None
    try:
        number = float(str(value).replace(",", "."))
    except ValueError:
        return None
    if number.is_integer() and 1 <= number <= 5:
        return int(number)
    return None


def parse_datetime(series: pd.Series) -> pd.Series:
    """Разбирает дату и явно приводит её к часовому поясу источника."""
    parsed = pd.to_datetime(series, errors="coerce", format="mixed")
    if parsed.dt.tz is None:
        return parsed.dt.tz_localize(TIMEZONE)
    return parsed.dt.tz_convert(TIMEZONE)


# ---------------------------------------------------------------------------
# Загрузка справочников
# ---------------------------------------------------------------------------

def load_lookup(
    cur,
    table: str,
    id_col: str,
    code_col: str,
    name_col: str,
    values,
) -> dict[str, int]:
    """Заполняет плоский справочник и возвращает соответствие code -> id."""
    unique: dict[str, str] = {}
    for value in values:
        name = clean_text(value)
        code = norm_text(name)
        if code is not None:
            unique.setdefault(code, name)

    if not unique:
        return {}

    rows = sorted(unique.items())
    execute_values(
        cur,
        f"INSERT INTO {SCHEMA}.{table} ({code_col}, {name_col}) VALUES %s "
        f"ON CONFLICT ({code_col}) DO NOTHING",
        rows,
        page_size=500,
    )
    cur.execute(f"SELECT {id_col}, {code_col} FROM {SCHEMA}.{table}")
    return {code: pk for pk, code in cur.fetchall()}


def load_marks(cur, values) -> dict[str, int]:
    """Справочник оценок: код, отображаемое значение и число 1–5."""
    unique: dict[str, tuple[str, int | None]] = {}
    for value in values:
        name = clean_text(value)
        code = norm_text(name)
        if code is not None:
            unique.setdefault(code, (name, parse_mark_value(name)))

    if not unique:
        return {}

    rows = sorted(
        (code, name, mark_value)
        for code, (name, mark_value) in unique.items()
    )
    execute_values(
        cur,
        f"INSERT INTO {SCHEMA}.review_mark (mark_code, mark_name, mark_value) "
        "VALUES %s ON CONFLICT (mark_code) DO NOTHING",
        rows,
        page_size=500,
    )
    cur.execute(f"SELECT mark_id, mark_code FROM {SCHEMA}.review_mark")
    return {code: pk for pk, code in cur.fetchall()}


def load_products(cur, frame: pd.DataFrame) -> dict[tuple[str, str, str], int]:
    """Иерархия company -> business_line -> product."""
    companies: dict[str, str] = {}
    lines: dict[tuple[str, str], str] = {}
    products: dict[tuple[str, str, str], str] = {}

    for company, line, product in zip(
        frame["company"], frame["business_line"], frame["product"]
    ):
        company_name = clean_text(company)
        line_name = clean_text(line)
        product_name = clean_text(product)
        if not (company_name and line_name and product_name):
            continue
        company_code = norm_text(company_name)
        line_code = norm_text(line_name)
        product_code = norm_text(product_name)
        companies.setdefault(company_code, company_name)
        lines.setdefault((company_code, line_code), line_name)
        products.setdefault((company_code, line_code, product_code), product_name)

    execute_values(
        cur,
        f"INSERT INTO {SCHEMA}.company (company_code, company_name) VALUES %s "
        "ON CONFLICT (company_code) DO NOTHING",
        sorted(companies.items()),
        page_size=500,
    )
    cur.execute(f"SELECT company_id, company_code FROM {SCHEMA}.company")
    company_ids = {code: pk for pk, code in cur.fetchall()}

    execute_values(
        cur,
        f"INSERT INTO {SCHEMA}.business_line "
        "(business_line_code, business_line_name, company_id) VALUES %s "
        "ON CONFLICT (company_id, business_line_code) DO NOTHING",
        [
            (line_code, line_name, company_ids[company_code])
            for (company_code, line_code), line_name in sorted(lines.items())
        ],
        page_size=500,
    )
    cur.execute(
        f"SELECT bl.business_line_id, bl.business_line_code, c.company_code "
        f"FROM {SCHEMA}.business_line bl "
        f"JOIN {SCHEMA}.company c USING (company_id)"
    )
    line_ids = {
        (company_code, line_code): pk
        for pk, line_code, company_code in cur.fetchall()
    }

    execute_values(
        cur,
        f"INSERT INTO {SCHEMA}.product "
        "(product_code, product_name, business_line_id) VALUES %s "
        "ON CONFLICT (business_line_id, product_code) DO NOTHING",
        [
            (product_code, product_name, line_ids[(company_code, line_code)])
            for (company_code, line_code, product_code), product_name
            in sorted(products.items())
        ],
        page_size=500,
    )
    cur.execute(
        f"SELECT p.product_id, p.product_code, bl.business_line_code, "
        f"c.company_code "
        f"FROM {SCHEMA}.product p "
        f"JOIN {SCHEMA}.business_line bl USING (business_line_id) "
        f"JOIN {SCHEMA}.company c USING (company_id)"
    )
    return {
        (company_code, line_code, product_code): pk
        for pk, product_code, line_code, company_code in cur.fetchall()
    }


def load_subthemes(cur, frame: pd.DataFrame) -> dict[tuple[str, str], int]:
    """Иерархия review_theme -> review_subtheme."""
    themes: dict[str, str] = {}
    subthemes: dict[tuple[str, str], str] = {}

    for theme, subtheme in zip(frame["review_theme"], frame["subtheme"]):
        theme_name = clean_text(theme)
        subtheme_name = clean_text(subtheme)
        if not (theme_name and subtheme_name):
            continue
        theme_code = norm_text(theme_name)
        subtheme_code = norm_text(subtheme_name)
        themes.setdefault(theme_code, theme_name)
        subthemes.setdefault((theme_code, subtheme_code), subtheme_name)

    execute_values(
        cur,
        f"INSERT INTO {SCHEMA}.review_theme (theme_code, theme_name) VALUES %s "
        "ON CONFLICT (theme_code) DO NOTHING",
        sorted(themes.items()),
        page_size=500,
    )
    cur.execute(f"SELECT theme_id, theme_code FROM {SCHEMA}.review_theme")
    theme_ids = {code: pk for pk, code in cur.fetchall()}

    execute_values(
        cur,
        f"INSERT INTO {SCHEMA}.review_subtheme "
        "(subtheme_code, subtheme_name, theme_id) VALUES %s "
        "ON CONFLICT (theme_id, subtheme_code) DO NOTHING",
        [
            (subtheme_code, subtheme_name, theme_ids[theme_code])
            for (theme_code, subtheme_code), subtheme_name
            in sorted(subthemes.items())
        ],
        page_size=500,
    )
    cur.execute(
        f"SELECT s.subtheme_id, s.subtheme_code, t.theme_code "
        f"FROM {SCHEMA}.review_subtheme s "
        f"JOIN {SCHEMA}.review_theme t USING (theme_id)"
    )
    return {
        (theme_code, subtheme_code): pk
        for pk, subtheme_code, theme_code in cur.fetchall()
    }


def load_clients(
    cur,
    frame: pd.DataFrame,
    gender_ids: dict[str, int],
    education_ids: dict[str, int],
    marital_ids: dict[str, int],
    country_ids: dict[str, int],
    segment_ids: dict[str, int],
    age_ids: dict[str, int],
) -> dict[str, int]:
    """Одна строка на клиента; атрибуты — из первого по времени отзыва."""
    clients = (
        frame[["id_client", "review_dttm", *CLIENT_ATTRS]]
        .sort_values(["id_client", "review_dttm"], kind="stable")
        .drop_duplicates(subset=["id_client"], keep="first")
    )

    def lookup(mapping: dict[str, int], value) -> int | None:
        code = norm_text(value)
        return mapping.get(code) if code else None

    rows = []
    for row in clients.to_dict("records"):
        rows.append((
            clean_text(row["id_client"]),
            lookup(gender_ids, row["gender_cd"]),
            lookup(education_ids, row["education_level_cd"]),
            lookup(marital_ids, row["marital_status_cd"]),
            parse_int(row["children_cnt"]),
            lookup(country_ids, row["citizenship_country"]),
            lookup(segment_ids, row["segment_name"]),
            parse_bool(row["subscription_important_flg"]),
            parse_bool(row["new_flg"]),
            parse_bool(row["influencer_flg"]),
            lookup(age_ids, row["age_segment"]),
            row["review_dttm"],
        ))

    execute_values(
        cur,
        f"INSERT INTO {SCHEMA}.client "
        "(client_external_id, gender_id, education_level_id, marital_status_id, "
        " children_cnt, country_id, client_segment_id, "
        " subscription_important_flg, new_flg, influencer_flg, "
        " age_segment_id, first_review_dttm) VALUES %s "
        "ON CONFLICT (client_external_id) DO NOTHING",
        rows,
        page_size=1000,
    )
    cur.execute(
        f"SELECT client_id, client_external_id FROM {SCHEMA}.client"
    )
    return {external_id: pk for pk, external_id in cur.fetchall()}


# ---------------------------------------------------------------------------
# Основной сценарий
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Загрузка history.csv в ER-модель staging"
    )
    parser.add_argument(
        "csv_path", nargs="?", default="data/incoming/history.csv"
    )
    parser.add_argument(
        "--dsn", default=os.environ.get("DATABASE_URL", DEFAULT_DSN)
    )
    args = parser.parse_args()

    frame = pd.read_csv(
        args.csv_path, dtype=str, keep_default_na=False, na_values=[""]
    )
    missing = [column for column in COLUMNS if column not in frame.columns]
    if missing:
        raise SystemExit(f"В файле нет колонок: {missing}")

    frame = frame[COLUMNS].copy()
    frame["review_dttm"] = parse_datetime(frame["review_dttm"])
    frame["finish_dttm"] = parse_datetime(frame["finish_dttm"])
    frame = frame[
        frame["review_dttm"].notna()
        & frame["id_client"].notna()
        & frame["company"].notna()
        & frame["business_line"].notna()
        & frame["product"].notna()
        & frame["review_source"].notna()
        & frame["reason"].notna()
        & frame["review_theme"].notna()
        & frame["subtheme"].notna()
    ].reset_index(drop=True)

    conn = psycopg2.connect(args.dsn)
    try:
        with conn.cursor() as cur:
            cur.execute("SET TIME ZONE %s", (TIMEZONE,))
            cur.execute(
                "TRUNCATE "
                + ", ".join(f"{SCHEMA}.{table}" for table in TABLES)
                + " RESTART IDENTITY CASCADE"
            )

            gender_ids = load_lookup(
                cur, "gender", "gender_id", "gender_code", "gender_name",
                frame["gender_cd"],
            )
            education_ids = load_lookup(
                cur, "education_level", "education_level_id",
                "education_level_code", "education_level_name",
                frame["education_level_cd"],
            )
            marital_ids = load_lookup(
                cur, "marital_status", "marital_status_id",
                "marital_status_code", "marital_status_name",
                frame["marital_status_cd"],
            )
            country_ids = load_lookup(
                cur, "country", "country_id", "country_code", "country_name",
                frame["citizenship_country"],
            )
            segment_ids = load_lookup(
                cur, "client_segment", "client_segment_id",
                "client_segment_code", "client_segment_name",
                frame["segment_name"],
            )
            age_ids = load_lookup(
                cur, "age_segment", "age_segment_id", "age_segment_code",
                "age_segment_name", frame["age_segment"],
            )

            products = load_products(cur, frame)
            subthemes = load_subthemes(cur, frame)

            source_ids = load_lookup(
                cur, "review_source", "source_id", "source_code",
                "source_name", frame["review_source"],
            )
            emotion_ids = load_lookup(
                cur, "review_emotion", "emotion_id", "emotion_code",
                "emotion_name", frame["review_emotion"],
            )
            reason_ids = load_lookup(
                cur, "review_reason", "reason_id", "reason_code",
                "reason_name", frame["reason"],
            )
            solution_ids = load_lookup(
                cur, "review_solution", "solution_id", "solution_code",
                "solution_name", frame["solution_flg"],
            )
            mark_ids = load_marks(cur, frame["review_mark"])

            clients = load_clients(
                cur, frame, gender_ids, education_ids, marital_ids,
                country_ids, segment_ids, age_ids,
            )

            def optional_id(mapping: dict[str, int], value) -> int | None:
                code = norm_text(value)
                return mapping.get(code) if code else None

            rows = []
            skipped = 0
            for row in frame.to_dict("records"):
                company_code = norm_text(row["company"])
                line_code = norm_text(row["business_line"])
                product_code = norm_text(row["product"])
                theme_code = norm_text(row["review_theme"])
                subtheme_code = norm_text(row["subtheme"])
                source_code = norm_text(row["review_source"])
                reason_code = norm_text(row["reason"])

                client_id = clients.get(clean_text(row["id_client"]))
                product_id = products.get(
                    (company_code, line_code, product_code)
                )
                subtheme_id = subthemes.get((theme_code, subtheme_code))
                source_id = source_ids.get(source_code)
                reason_id = reason_ids.get(reason_code)

                if None in (client_id, product_id, subtheme_id,
                            source_id, reason_id):
                    skipped += 1
                    continue

                finish_dttm = row["finish_dttm"]
                if is_null(finish_dttm) or pd.isna(finish_dttm):
                    finish_dttm = None

                rows.append((
                    client_id,
                    row["review_dttm"],
                    finish_dttm,
                    product_id,
                    source_id,
                    reason_id,
                    subtheme_id,
                    optional_id(emotion_ids, row["review_emotion"]),
                    optional_id(mark_ids, row["review_mark"]),
                    optional_id(solution_ids, row["solution_flg"]),
                    clean_long_text(row["review_text"]),
                    parse_int(row["csat_score"]),
                ))

            execute_values(
                cur,
                f"INSERT INTO {SCHEMA}.review "
                "(client_id, review_dttm, finish_dttm, product_id, source_id, "
                " reason_id, subtheme_id, emotion_id, mark_id, solution_id, "
                " review_text, csat_score) VALUES %s "
                "ON CONFLICT (client_id, review_dttm, source_id) DO NOTHING",
                rows,
                page_size=1000,
            )

            cur.execute(
                f"SELECT {', '.join(f'(SELECT count(*) FROM {SCHEMA}.{table})' for table in TABLES)}"
            )
            counts = cur.fetchone()
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    print(f"Строк в файле          {len(frame):>7,}")
    if skipped:
        print(f"Пропущено строк        {skipped:>7,}  (не найдены обязательные справочники)")
    for table, count in zip(TABLES, counts):
        print(f"{SCHEMA + '.' + table:<22} {count:>7,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
