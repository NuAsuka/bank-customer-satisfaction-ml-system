import os
import sys
from datetime import timedelta

import pendulum
from airflow.sdk import dag, task

PROJECT_PATH = os.environ.get("PROJECT_PATH", "/opt/airflow/project")
if PROJECT_PATH not in sys.path:
    sys.path.insert(0, PROJECT_PATH)

from src import batches, load
from src.db import connect
from src.runlog import run_step

LABELS_DIR = os.environ.get("LABELS_DIR", "/opt/airflow/project/data/incoming/labels")
SKIP_STATUSES = ("processing", "done")


@dag(
    dag_id="enrichment_ingestion",
    schedule="*/15 * * * *",
    start_date=pendulum.datetime(2025, 1, 1, tz="Europe/Moscow"),
    catchup=False,
    max_active_runs=1,
    tags=["ml"],
)
def enrichment_ingestion():

    @task(retries=1, retry_delay=timedelta(minutes=1))
    def ingest() -> int | None:
        with connect() as conn:
            for path in batches.incoming(LABELS_DIR):
                review = batches.by_name(conn, "review", path.name)
                if review is None or review[1] not in SKIP_STATUSES:
                    print(f"{path.name}: отзывы порции ещё не загружены, пропуск")
                    continue
                found = batches.lookup(conn, "enrichment", path)
                if found is not None and found[1] in SKIP_STATUSES:
                    print(f"{path.name}: порция {found[0]} в статусе {found[1]}, пропуск")
                    continue
                batch_id = found[0] if found else batches.register(conn, "enrichment", path)
                with run_step(conn, "load", batch_enrichment_id=batch_id) as step:
                    result = load.load_enrichment(conn, batch_id, path)
                    step.rows_in = result["rows"]
                    step.rows_out = result["inserted"]
                    batches.set_status(
                        conn,
                        "enrichment",
                        batch_id,
                        "processing",
                        result["inserted"],
                        result["period_start"],
                        result["period_end"],
                    )
                print(f"{path.name}: порция {batch_id}, вставлено {result['inserted']}")
                return batch_id
        print("новых файлов нет")
        return None

    ingest()


enrichment_ingestion()
