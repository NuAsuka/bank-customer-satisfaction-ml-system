from contextlib import contextmanager
from dataclasses import dataclass


STAGES = ("load", "quality", "features", "scoring", "train", "kpi")


@dataclass
class Step:
    run_id: int
    rows_in: int | None = None
    rows_out: int | None = None
    rows_rejected: int | None = None


@contextmanager
def run_step(
    conn,
    stage: str,
    batch_review_id: int | None = None,
    batch_enrichment_id: int | None = None,
    attempt: int = 1,
    model_version: str | None = None,
):
    if stage not in STAGES:
        raise ValueError(f"неизвестная стадия: {stage}")
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO etl.run_log
                (batch_review_id, batch_enrichment_id, stage, status, attempt, model_version)
            VALUES (%s, %s, %s, 'running', %s, %s)
            RETURNING run_id
            """,
            (batch_review_id, batch_enrichment_id, stage, attempt, model_version),
        )
        step = Step(run_id=cur.fetchone()[0])
    conn.commit()
    try:
        yield step
    except Exception as exc:
        conn.rollback()
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE etl.run_log
                   SET status = 'failed', finished_at = now(), message = %s
                 WHERE run_id = %s
                """,
                (str(exc)[:1000], step.run_id),
            )
        conn.commit()
        raise
    else:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE etl.run_log
                   SET status = 'success', finished_at = now(),
                       rows_in = %s, rows_out = %s, rows_rejected = %s
                 WHERE run_id = %s
                """,
                (step.rows_in, step.rows_out, step.rows_rejected, step.run_id),
            )
        conn.commit()
