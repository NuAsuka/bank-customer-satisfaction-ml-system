import os
import sys

import pendulum
from airflow.sdk import dag, task

PROJECT_PATH = os.environ.get("PROJECT_PATH", "/opt/airflow/project")
if PROJECT_PATH not in sys.path:
    sys.path.insert(0, PROJECT_PATH)

from src import experiments, train_final
from src.db import connect
from src.runlog import run_step


@dag(
    dag_id="retrain",
    schedule=None,
    start_date=pendulum.datetime(2025, 1, 1, tz="Europe/Moscow"),
    catchup=False,
    max_active_runs=1,
    tags=["ml"],
)
def retrain():

    @task(retries=1, retry_delay=300)
    def train_model() -> dict:
        version = train_final.next_version()
        with connect() as conn:
            data = experiments.load_from_db(conn)
            with run_step(conn, "train", model_version=version) as step:
                meta = train_final.train(data=data, version=version)
                step.rows_in = meta["n_train"]
                step.rows_out = meta["n_train"]
        return meta

    train_model()


retrain()
