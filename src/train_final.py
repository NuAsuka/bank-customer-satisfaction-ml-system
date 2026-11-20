import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import sklearn

from . import experiments
from .models import SEED, make_pipeline

PROJECT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_DIR / "models"


def next_version(models_dir=MODELS_DIR) -> str:
    directory = Path(models_dir)
    directory.mkdir(parents=True, exist_ok=True)
    numbers = []
    for path in directory.glob("model_v*.joblib"):
        suffix = path.stem.split("_v")[-1]
        if suffix.isdigit():
            numbers.append(int(suffix))
    return f"v{max(numbers, default=0) + 1}"


def train(
    data=None,
    conn=None,
    entry: dict | None = None,
    version: str | None = None,
    models_dir=MODELS_DIR,
    log_path=experiments.LOG_PATH,
) -> dict:
    if data is None:
        if conn is None:
            raise ValueError("нужны либо данные, либо соединение с базой")
        data = experiments.load_from_db(conn)
    if entry is None:
        entry = experiments.select_best(log_path=log_path)

    df, y = experiments.prepare(data)
    n_train = experiments.train_size(len(df), entry["cv"]["test_size"])
    X = df[experiments.FEATURES].iloc[:n_train]
    y_train = y[:n_train]

    use_numeric = entry.get("use_numeric", True)
    pipe = make_pipeline(
        entry["family"], text_repr=entry["text_repr"], use_numeric=use_numeric, **(entry["params"] or {})
    )
    pipe.fit(X, y_train)

    version = version or next_version(models_dir)
    directory = Path(models_dir)
    directory.mkdir(parents=True, exist_ok=True)
    artifact = directory / f"model_{version}.joblib"
    joblib.dump(pipe, artifact)

    meta = {
        "model_version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "run_id": entry["run_id"],
        "family": entry["family"],
        "params": entry["params"],
        "text_repr": entry["text_repr"],
        "use_numeric": use_numeric,
        "features": experiments.FEATURES,
        "n_train": int(n_train),
        "n_test": int(len(df) - n_train),
        "seed": SEED,
        "sklearn_version": sklearn.__version__,
        "cv": entry["cv"],
        "roc_auc_mean": entry["roc_auc_mean"],
        "roc_auc_std": entry["roc_auc_std"],
        "pr_auc_mean": entry["pr_auc_mean"],
        "f2_mean": entry["f2_mean"],
        "brier_mean": entry["brier_mean"],
        "artifact": artifact.name,
    }
    (directory / f"model_{version}.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    current = {
        "model_version": version,
        "artifact": artifact.name,
        "meta": f"model_{version}.json",
        "updated_at": meta["trained_at"],
    }
    (directory / "current.json").write_text(
        json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return meta
