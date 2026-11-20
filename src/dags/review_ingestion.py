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

FEATURES_DIR = os.environ.get("FEATURES_DIR", "/opt/airflow/project/data/incoming/features")
SKIP_STATUSES = ("processing", "done")


@dag(
    dag_id="review_ingestion",
    schedule="*/15 * * * *",
    start_date=pendulum.datetime(2025, 1, 1, tz="Europe/Moscow"),
    catchup=False,
    max_active_runs=1,
    doc_md="Загрузка данных из датасетов в raw слой хранилища данных."
)
def review_ingestion():

    @task(retries=1, retry_delay=timedelta(minutes=1))
    def ingest() -> int | None:
        with connect() as conn:
            for path in batches.incoming(FEATURES_DIR):
                found = batches.lookup(conn, "review", path)
                if found is not None and found[1] in SKIP_STATUSES:
                    print(f"{path.name}: порция {found[0]} в статусе {found[1]}, пропуск")
                    continue
                batch_id = found[0] if found else batches.register(conn, "review", path)
                with run_step(conn, "load", batch_review_id=batch_id) as step:
                    result = load.load_reviews(conn, batch_id, path)
                    step.rows_in = result["rows"]
                    step.rows_out = result["inserted"]
                    batches.set_status(
                        conn,
                        "review",
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


review_ingestion()
