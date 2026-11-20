import hashlib
import io
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline

from . import metrics
from .models import FAMILIES, SEED, coefficients, complexity_rank, grid_for, make_pipeline

NEGATIVE = "Негативный"
EMOTIONS = ("Позитивный", "Негативный")
FEATURES = ["review_text", "review_source"]
MIN_TEXT_LEN = 20

PROJECT_DIR = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = PROJECT_DIR / "experiments"
FIGURES_DIR = PROJECT_DIR / "reports" / "figures"
LOG_PATH = EXPERIMENTS_DIR / "log.jsonl"


def load_from_db(conn) -> pd.DataFrame:
    from .db import read_frame

    return read_frame(
        conn,
        """
        SELECT id_client, review_dttm, review_source, review_text, product, review_emotion
          FROM staging.training_set
         WHERE review_emotion IS NOT NULL
         ORDER BY review_dttm
        """,
    )


def load_from_csv(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["review_dttm", "review_source", "review_text", "review_emotion"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"в файле нет колонок: {missing}")
    df = df[required].copy()
    df["review_dttm"] = pd.to_datetime(df["review_dttm"])
    return df


def prepare(data: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    df = data.copy()
    df = df[df["review_emotion"].astype(str).str.strip().isin(EMOTIONS)]
    df = df[df["review_text"].notna()]
    df = df.assign(review_text=df["review_text"].astype(str).str.strip())
    df = df[df["review_text"].str.len() >= MIN_TEXT_LEN]
    df = df[df["review_dttm"].notna()]
    df = df.assign(
        review_source=df["review_source"].fillna("нет").astype(str).str.strip().str.lower()
    )
    df = df.sort_values("review_dttm", kind="mergesort")
    df = df.drop_duplicates(subset=["review_text"], keep="first")
    df = df.reset_index(drop=True)
    y = (df["review_emotion"].astype(str).str.strip() == NEGATIVE).astype(int).to_numpy()
    return df, y


def train_size(n_rows: int, test_size: float = 0.2) -> int:
    return int(n_rows * (1 - test_size))


def _run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{stamp}-{uuid.uuid4().hex[:6]}"


def _append_log(entry: dict, log_path) -> None:
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _write_json(path, payload) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(target)


def run_experiment(
    data: pd.DataFrame,
    family: str,
    params: dict | None = None,
    n_splits: int = 5,
    test_size: float = 0.2,
    text_repr: str | None = None,
    use_numeric: bool = True,
    seed: int = SEED,
    data_source: str = "staging.training_set",
    notes: str | None = None,
    log_path=LOG_PATH,
) -> dict:
    if family not in FAMILIES:
        raise ValueError(f"неизвестное семейство: {family}")
    df, y = prepare(data)
    n_train = train_size(len(df), test_size)
    X = df[FEATURES]
    y_train = y[:n_train]

    splitter = TimeSeriesSplit(n_splits=n_splits)
    folds = []
    for number, (train_idx, valid_idx) in enumerate(splitter.split(X.iloc[:n_train]), start=1):
        pipe = make_pipeline(family, text_repr=text_repr, use_numeric=use_numeric, **(params or {}))
        started = time.perf_counter()
        pipe.fit(X.iloc[train_idx], y_train[train_idx])
        fit_seconds = time.perf_counter() - started
        prob = pipe.predict_proba(X.iloc[valid_idx])[:, 1]
        scores = metrics.evaluate(y_train[valid_idx], prob)
        folds.append(
            {
                "fold": number,
                "n_train": int(len(train_idx)),
                "n_valid": int(len(valid_idx)),
                "fit_seconds": round(fit_seconds, 3),
                **scores,
            }
        )

    summary = metrics.summarize(folds)
    entry = {
        "run_id": _run_id(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "family": family,
        "params": params or {},
        "text_repr": text_repr or FAMILIES[family]["text_repr"],
        "use_numeric": bool(use_numeric),
        "features": FEATURES,
        "cv": {"scheme": "TimeSeriesSplit", "n_splits": n_splits, "test_size": test_size},
        "n_rows": int(len(df)),
        "n_train": int(n_train),
        "n_test": int(len(df) - n_train),
        "data_source": data_source,
        "seed": seed,
        "folds": folds,
        "notes": notes,
        **summary,
    }
    _append_log(entry, log_path)
    return entry


def read_log(log_path=LOG_PATH) -> list[dict]:
    path = Path(log_path)
    if not path.exists():
        return []
    entries = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def build_tables(log_path=LOG_PATH, out_dir=EXPERIMENTS_DIR) -> dict:
    entries = read_log(log_path)
    if not entries:
        return {"runs": 0, "folds": 0}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    runs = []
    folds = []
    for entry in entries:
        runs.append(
            {
                "run_id": entry["run_id"],
                "created_at": entry["created_at"],
                "family": entry["family"],
                "text_repr": entry["text_repr"],
                "use_numeric": entry.get("use_numeric", True),
                "params": json.dumps(entry["params"], ensure_ascii=False),
                "notes": entry.get("notes"),
                "n_splits": entry["cv"]["n_splits"],
                "n_rows": entry["n_rows"],
                "n_train": entry["n_train"],
                "roc_auc_mean": entry["roc_auc_mean"],
                "roc_auc_std": entry["roc_auc_std"],
                "pr_auc_mean": entry["pr_auc_mean"],
                "f2_mean": entry["f2_mean"],
                "precision_mean": entry["precision_mean"],
                "recall_mean": entry["recall_mean"],
                "brier_mean": entry["brier_mean"],
                "negative_share_mean": entry["negative_share_mean"],
                "fit_seconds_mean": float(np.mean([fold["fit_seconds"] for fold in entry["folds"]])),
            }
        )
        for fold in entry["folds"]:
            folds.append({"run_id": entry["run_id"], "family": entry["family"], **fold})

    pd.DataFrame(runs).to_csv(out / "runs.csv", index=False)
    pd.DataFrame(folds).to_csv(out / "folds.csv", index=False)
    return {"runs": len(runs), "folds": len(folds)}


def select_best(entries: list[dict] | None = None, log_path=LOG_PATH, tolerance: float = 0.005) -> dict:
    data = entries if entries is not None else read_log(log_path)
    if not data:
        raise ValueError("журнал экспериментов пуст")
    best = max(entry["roc_auc_mean"] for entry in data)
    candidates = [entry for entry in data if best - entry["roc_auc_mean"] <= tolerance]
    candidates.sort(key=lambda entry: (complexity_rank(entry["family"]), entry["roc_auc_std"]))
    return candidates[0]


def tune_family(
    data: pd.DataFrame,
    family: str,
    n_splits: int = 3,
    test_size: float = 0.2,
    seed: int = SEED,
) -> dict:
    df, y = prepare(data)
    n_train = train_size(len(df), test_size)
    X = df[FEATURES].iloc[:n_train]
    y_train = y[:n_train]
    search = GridSearchCV(
        make_pipeline(family),
        grid_for(family),
        scoring="roc_auc",
        cv=TimeSeriesSplit(n_splits=n_splits),
        n_jobs=-1,
    )
    search.fit(X, y_train)
    best = {key.replace("clf__", ""): value for key, value in search.best_params_.items()}
    return {"family": family, "best_params": best, "best_score": float(search.best_score_)}


def check_reproducibility(
    data: pd.DataFrame,
    family: str,
    params: dict | None = None,
    text_repr: str | None = None,
    use_numeric: bool = True,
    test_size: float = 0.2,
    seed: int = SEED,
    sample: int = 500,
) -> dict:
    df, y = prepare(data)
    n_train = train_size(len(df), test_size)
    X = df[FEATURES].iloc[:n_train]
    y_train = y[:n_train]

    digests = []
    predictions = []
    for _ in range(2):
        pipe = make_pipeline(family, text_repr=text_repr, use_numeric=use_numeric, **(params or {}))
        pipe.fit(X, y_train)
        buffer = io.BytesIO()
        joblib.dump(pipe, buffer)
        digests.append(hashlib.sha256(buffer.getvalue()).hexdigest())
        predictions.append(pipe.predict_proba(X.iloc[:sample])[:, 1])

    difference = float(np.max(np.abs(predictions[0] - predictions[1])))
    return {
        "family": family,
        "params": params or {},
        "use_numeric": bool(use_numeric),
        "identical": bool(difference == 0.0),
        "max_probability_difference": difference,
        "artifact_sha256": digests[0],
        "artifact_identical": digests[0] == digests[1],
        "seed": seed,
    }


def extract_coefficients(model_path, top_n: int = 20) -> dict:
    pipe = joblib.load(model_path)
    pairs = coefficients(pipe)
    if not pairs:
        return {"positive": [], "negative": []}
    positive = sorted(pairs, key=lambda item: item[1], reverse=True)[:top_n]
    negative = sorted(pairs, key=lambda item: item[1])[:top_n]
    return {
        "positive": [{"feature": name, "weight": weight} for name, weight in positive],
        "negative": [{"feature": name, "weight": weight} for name, weight in negative],
    }


def run_baselines(data: pd.DataFrame, test_size: float = 0.2) -> dict:
    df, y = prepare(data)
    n_train = train_size(len(df), test_size)
    y_train = y[:n_train]
    lengths = df["review_text"].str.len().to_numpy(dtype=float)[:n_train]

    majority = metrics.baseline_majority(y_train)
    share = float(np.mean(y_train))
    majority["accuracy"] = float(max(share, 1.0 - share))
    majority["negative_share"] = share

    length = metrics.baseline_length(lengths, y_train)
    predicted = (lengths > length["threshold"]).astype(int)
    length["accuracy"] = float(np.mean(predicted == y_train))
    return {
        "n_rows": int(len(df)),
        "n_train": int(n_train),
        "n_test": int(len(df) - n_train),
        "negative_share_train": share,
        "negative_share_test": float(np.mean(y[n_train:])),
        "majority": majority,
        "length": length,
    }


def _block(family: str, X_train, X_valid, text_repr=None, use_numeric=True):
    steps = make_pipeline(family, text_repr=text_repr, use_numeric=use_numeric).steps[:-1]
    prep = Pipeline(steps)
    prep.fit(X_train)
    return prep.transform(X_train), prep.transform(X_valid)


def _last_block(data: pd.DataFrame, n_splits: int = 5, test_size: float = 0.2):
    df, y = prepare(data)
    n_train = train_size(len(df), test_size)
    X = df[FEATURES].iloc[:n_train]
    y_train = y[:n_train]
    train_idx, valid_idx = list(TimeSeriesSplit(n_splits=n_splits).split(X))[-1]
    return X, y_train, train_idx, valid_idx


def regularization_curve(
    data: pd.DataFrame,
    family: str = "logistic_regression",
    penalties=("l1", "l2"),
    cs=(0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0),
    n_splits: int = 5,
    test_size: float = 0.2,
) -> list[dict]:
    from sklearn.linear_model import LogisticRegression

    X, y_train, train_idx, valid_idx = _last_block(data, n_splits, test_size)
    z_train, z_valid = _block(family, X.iloc[train_idx], X.iloc[valid_idx])
    rows = []
    for penalty in penalties:
        for strength in cs:
            model = LogisticRegression(
                l1_ratio=1.0 if penalty == "l1" else 0.0,
                C=strength,
                solver="liblinear",
                max_iter=2000,
                random_state=SEED,
            )
            model.fit(z_train, y_train[train_idx])
            scores = metrics.evaluate(
                y_train[valid_idx], model.predict_proba(z_valid)[:, 1]
            )
            rows.append(
                {
                    "penalty": penalty,
                    "C": float(strength),
                    "roc_auc": scores["roc_auc"],
                    "f2": scores["f2"],
                    "recall": scores["recall"],
                    "precision": scores["precision"],
                    "nonzero": int(np.count_nonzero(model.coef_)),
                    "total": int(model.coef_.size),
                }
            )
    return rows


def depth_curve(
    data: pd.DataFrame,
    depths=range(1, 16),
    family: str = "decision_tree",
    n_splits: int = 5,
    test_size: float = 0.2,
) -> list[dict]:
    from sklearn.metrics import roc_auc_score
    from sklearn.tree import DecisionTreeClassifier

    X, y_train, train_idx, valid_idx = _last_block(data, n_splits, test_size)
    z_train, z_valid = _block(family, X.iloc[train_idx], X.iloc[valid_idx])
    rows = []
    for depth in depths:
        model = DecisionTreeClassifier(max_depth=depth, random_state=SEED)
        model.fit(z_train, y_train[train_idx])
        train_score = roc_auc_score(y_train[train_idx], model.predict_proba(z_train)[:, 1])
        valid_score = roc_auc_score(y_train[valid_idx], model.predict_proba(z_valid)[:, 1])
        rows.append({"depth": int(depth), "train_roc_auc": float(train_score), "valid_roc_auc": float(valid_score)})
    return rows


def feature_benefit(
    data: pd.DataFrame,
    families=("logistic_regression", "gradient_boosting"),
    n_splits: int = 5,
    seed: int = SEED,
    data_source: str = "staging.training_set",
    log_path=LOG_PATH,
) -> list[dict]:
    entries = []
    for family in families:
        for use_numeric in (True, False):
            entries.append(
                run_experiment(
                    data,
                    family,
                    n_splits=n_splits,
                    seed=seed,
                    use_numeric=use_numeric,
                    notes="польза признаков",
                    data_source=data_source,
                    log_path=log_path,
                )
            )
    return entries


def write_decision(best: dict, candidates: list[dict], path=EXPERIMENTS_DIR / "decision.md") -> str:
    lines = [
        "# Решение о выборе итоговой модели",
        "",
        f"Выбран запуск {best['run_id']}: семейство {best['family']}, параметры {json.dumps(best['params'], ensure_ascii=False)}.",
        "",
        f"Средний ROC-AUC на кросс-валидации {best['roc_auc_mean']:.4f} плюс-минус {best['roc_auc_std']:.4f}, "
        f"PR-AUC {best['pr_auc_mean']:.4f}, F2 при пороге 0,5 {best['f2_mean']:.4f}.",
        "",
        "Правило выбора: наибольший средний ROC-AUC, при различии менее 0,005 предпочтение более простой модели.",
        "",
        "| запуск | семейство | параметры | ROC-AUC | PR-AUC | F2 |",
        "|---|---|---|---|---|---|",
    ]
    for entry in sorted(candidates, key=lambda item: item["roc_auc_mean"], reverse=True):
        params = json.dumps(entry["params"], ensure_ascii=False) or "по умолчанию"
        lines.append(
            f"| {entry['run_id']} | {entry['family']} | {params} | "
            f"{entry['roc_auc_mean']:.4f} | {entry['pr_auc_mean']:.4f} | {entry['f2_mean']:.4f} |"
        )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(target)


def run_suite(
    data: pd.DataFrame,
    families: list[str] | None = None,
    n_splits: int = 5,
    seed: int = SEED,
    data_source: str = "staging.training_set",
    log_path=LOG_PATH,
) -> list[dict]:
    selected = families or list(FAMILIES)
    entries = []
    for family in selected:
        entries.append(
            run_experiment(
                data,
                family,
                n_splits=n_splits,
                seed=seed,
                data_source=data_source,
                log_path=log_path,
            )
        )
    return entries


def run_plan(
    data: pd.DataFrame,
    n_splits: int = 5,
    seed: int = SEED,
    data_source: str = "staging.training_set",
    log_path=LOG_PATH,
    out_dir=EXPERIMENTS_DIR,
) -> dict:
    from . import plots
    from .train_final import MODELS_DIR, train as train_final

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if Path(log_path).exists():
        Path(log_path).unlink()

    print("базовые решения", flush=True)
    baselines = run_baselines(data)
    _write_json(out / "baselines.json", baselines)

    print("семейства моделей", flush=True)
    entries = run_suite(data, n_splits=n_splits, seed=seed, data_source=data_source, log_path=log_path)

    print("линейная модель с L1", flush=True)
    l1_entry = run_experiment(
        data,
        "logistic_regression",
        {"l1_ratio": 1.0, "C": 1.0},
        n_splits=n_splits,
        seed=seed,
        notes="L1",
        data_source=data_source,
        log_path=log_path,
    )
    candidates = entries + [l1_entry]

    ranked = sorted(candidates, key=lambda entry: entry["roc_auc_mean"], reverse=True)
    best_per_family = []
    seen = set()
    for entry in ranked:
        if entry["family"] in seen:
            continue
        seen.add(entry["family"])
        best_per_family.append(entry)
    top = best_per_family[:2]

    print("подбор гиперпараметров", flush=True)
    tuned = []
    for entry in top:
        result = tune_family(data, entry["family"], n_splits=3, seed=seed)
        tuned_entry = run_experiment(
            data,
            entry["family"],
            result["best_params"],
            n_splits=n_splits,
            seed=seed,
            notes="после подбора",
            data_source=data_source,
            log_path=log_path,
        )
        tuned.append({"tuning": result, "entry": tuned_entry})
        candidates.append(tuned_entry)

    print("польза признаков", flush=True)
    benefit = feature_benefit(data, n_splits=n_splits, seed=seed, data_source=data_source, log_path=log_path)

    print("сводные таблицы", flush=True)
    tables = build_tables(log_path=log_path, out_dir=out)

    best = select_best(entries=candidates)
    decision = write_decision(best, candidates, out / "decision.md")

    print("кривые регуляризации и глубины", flush=True)
    curves = {
        "regularization": regularization_curve(data, n_splits=n_splits),
        "depth": depth_curve(data, n_splits=n_splits),
    }
    _write_json(out / "curves.json", curves)
    plots.save_regularization_curve(curves["regularization"])
    plots.save_depth_curve(curves["depth"])

    print("проверка повторяемости", flush=True)
    reproducibility = check_reproducibility(
        data,
        best["family"],
        best["params"],
        text_repr=best["text_repr"],
        use_numeric=best["use_numeric"],
        seed=seed,
    )
    _write_json(out / "reproducibility.json", reproducibility)

    print("обучение итоговой модели", flush=True)
    meta = train_final(data=data, entry=best, log_path=log_path)

    print("графики", flush=True)
    runs = pd.read_csv(out / "runs.csv")
    plots.save_family_comparison(runs[runs["notes"].fillna("") != "польза признаков"])
    pipe = joblib.load(Path(MODELS_DIR) / meta["artifact"])
    plots.save_coefficients(coefficients(pipe))
    plots.save_model_curves(data, best, n_splits=n_splits)

    return {
        "baselines": baselines,
        "runs": tables["runs"],
        "folds": tables["folds"],
        "best": {
            "run_id": best["run_id"],
            "family": best["family"],
            "params": best["params"],
            "roc_auc_mean": best["roc_auc_mean"],
        },
        "tuned": tuned,
        "reproducibility": reproducibility,
        "model": {"version": meta["model_version"], "artifact": meta["artifact"]},
        "decision": decision,
    }
