import json
from pathlib import Path

import joblib
import pandas as pd
from psycopg2.extras import execute_values

from .db import read_frame

from .quality import EMOTIONS

PROJECT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_DIR / "models"
NEGATIVE = "Негативный"
POSITIVE = "Позитивный"


def load_current(models_dir=MODELS_DIR) -> dict:
    path = Path(models_dir) / "current.json"
    if not path.exists():
        raise FileNotFoundError(f"нет указателя действующей модели: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_model(models_dir=MODELS_DIR):
    current = load_current(models_dir)
    return joblib.load(Path(models_dir) / current["artifact"]), current


def predict_batch(
    conn,
    batch_review_id: int,
    models_dir=MODELS_DIR,
    threshold: float = 0.5,
) -> int:
    pipe, current = load_model(models_dir)
    version = current["model_version"]
    df = read_frame(
        conn,
        """
        SELECT c.id_client, c.review_dttm, c.review_source, c.review_text
          FROM staging.review_clean c
          LEFT JOIN staging.ml_predictions p
                 ON p.id_client = c.id_client
                AND p.review_dttm = c.review_dttm
                AND p.model_version = %s
         WHERE c.batch_review_id = %s
           AND p.id_client IS NULL
         ORDER BY c.review_dttm
        """,
        (version, batch_review_id),
    )
    if df.empty:
        return 0

    df["review_source"] = df["review_source"].fillna("нет").astype(str)
    df["review_text"] = df["review_text"].astype(str)
    probabilities = pipe.predict_proba(df[["review_text", "review_source"]])[:, 1]

    rows = []
    for row, probability in zip(df.itertuples(index=False), probabilities):
        label = NEGATIVE if probability >= threshold else POSITIVE
        if label not in EMOTIONS:
            raise ValueError(label)
        rows.append((row.id_client, row.review_dttm, version, batch_review_id, float(probability), label))

    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO staging.ml_predictions
                (id_client, review_dttm, model_version, batch_review_id,
                 review_emotion_prob, review_emotion_predict)
            VALUES %s
            ON CONFLICT (id_client, review_dttm, model_version) DO NOTHING
            """,
            rows,
            page_size=1000,
        )
    conn.commit()
    return len(rows)
