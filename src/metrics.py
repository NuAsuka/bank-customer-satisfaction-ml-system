import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

NEGATIVE = "Негативный"


def to_binary(labels) -> np.ndarray:
    return np.asarray([1 if str(value).strip() == NEGATIVE else 0 for value in labels])


def evaluate(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "f2": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "negative_share": float(np.mean(y_true)),
    }


def baseline_majority(y_true) -> dict:
    y_true = np.asarray(y_true).astype(int)
    share = float(np.mean(y_true))
    prob = np.full(len(y_true), share)
    return evaluate(y_true, prob)


def baseline_length(lengths, y_true) -> dict:
    lengths = np.asarray(lengths, dtype=float)
    y_true = np.asarray(y_true).astype(int)
    threshold = float(np.median(lengths))
    prob = (lengths > threshold).astype(float)
    scores = evaluate(y_true, prob)
    scores["threshold"] = threshold
    return scores


def summarize(folds: list[dict], keys=("roc_auc", "pr_auc", "f2", "precision", "recall", "brier", "negative_share")) -> dict:
    summary = {}
    for key in keys:
        values = np.asarray([fold[key] for fold in folds], dtype=float)
        summary[f"{key}_mean"] = float(values.mean())
        summary[f"{key}_std"] = float(values.std(ddof=0))
    return summary
