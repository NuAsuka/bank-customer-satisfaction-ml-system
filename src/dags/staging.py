import os
import sys
from datetime import timedelta

import pendulum
from airflow.sdk import dag, task

PROJECT_PATH = os.environ.get("PROJECT_PATH", "/opt/airflow/project")
if PROJECT_PATH not in sys.path:
    sys.path.insert(0, PROJECT_PATH)

from src import batches, features, quality
from src.db import connect
from src.runlog import run_step


@dag(
    dag_id="staging",
    schedule="*/15 * * * *",
    start_date=pendulum.datetime(2025, 1, 1, tz="Europe/Moscow"),
    catchup=False,
    max_active_runs=1,
    tags=["ml"],
)
def staging():

    @task
    def pending_review() -> list[int]:
        with connect() as conn:
            return batches.by_status(conn, "review", "processing")

    @task
    def pending_enrichment() -> list[int]:
        with connect() as conn:
            return batches.by_status(conn, "enrichment", "processing")

    @task(retries=2, retry_delay=timedelta(minutes=1))
    def stage_review(batch_review_id: int) -> int:
        with connect() as conn:
            with run_step(conn, "quality", batch_review_id=batch_review_id) as step:
                accepted, rejected = quality.check_reviews(conn, batch_review_id)
                step.rows_out = accepted
                step.rows_rejected = rejected
            with run_step(conn, "features", batch_review_id=batch_review_id) as step:
                step.rows_out = features.build(conn, batch_review_id=batch_review_id)
            batches.set_status(conn, "review", batch_review_id, "done")
        return batch_review_id

    @task(retries=2, retry_delay=timedelta(minutes=1))
    def stage_enrichment(batch_enrichment_id: int) -> int:
        with connect() as conn:
            with run_step(conn, "quality", batch_enrichment_id=batch_enrichment_id) as step:
                accepted, rejected = quality.check_enrichment(conn, batch_enrichment_id)
                step.rows_out = accepted
                step.rows_rejected = rejected
            with run_step(conn, "features", batch_enrichment_id=batch_enrichment_id) as step:
                step.rows_out = features.build(conn, batch_enrichment_id=batch_enrichment_id)
            batches.set_status(conn, "enrichment", batch_enrichment_id, "done")
        return batch_enrichment_id

    stage_review.expand(batch_review_id=pending_review())
    stage_enrichment.expand(batch_enrichment_id=pending_enrichment())


staging()
