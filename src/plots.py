from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

PROJECT_DIR = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_DIR / "reports" / "figures"

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "figure.autolayout": True,
    }
)


def save_roc_pr(y_true, y_prob, out_dir=FIGURES_DIR, prefix: str = "model") -> str:
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    share = float(np.mean(np.asarray(y_true)))

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(fpr, tpr, color="#1f77b4", label=f"ROC-AUC = {auc:.3f}")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="#999999", label="случайное ранжирование")
    axes[0].set_xlabel("Доля ложных срабатываний")
    axes[0].set_ylabel("Доля верно найденных негативных")
    axes[0].set_title("ROC-кривая")
    axes[0].legend(loc="lower right")

    axes[1].plot(recall, precision, color="#2ca02c", label=f"PR-AUC = {ap:.3f}")
    axes[1].axhline(share, linestyle="--", color="#999999", label=f"доля негатива {share:.3f}")
    axes[1].set_xlabel("Полнота")
    axes[1].set_ylabel("Точность")
    axes[1].set_title("Кривая точность-полнота")
    axes[1].legend(loc="upper right")

    figure.tight_layout()
    path = directory / f"{prefix}_roc_pr.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return str(path)


def save_family_comparison(runs: pd.DataFrame, out_path=FIGURES_DIR / "families_roc_auc.png") -> str:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = runs.sort_values("roc_auc_mean").reset_index(drop=True)
    labels = data["name"] if "name" in data.columns else data["family"]
    figure, axis = plt.subplots(figsize=(9, 0.55 * len(data) + 1.6))
    positions = np.arange(len(data))
    axis.barh(
        positions,
        data["roc_auc_mean"],
        xerr=data["roc_auc_std"],
        capsize=3,
        color="#1f77b4",
        error_kw={"elinewidth": 1, "ecolor": "#333333"},
    )
    axis.set_yticks(positions)
    axis.set_yticklabels(labels, fontsize=10)
    axis.set_xlabel("ROC-AUC на кросс-валидации")
    axis.set_title("Сравнение моделей по ROC-AUC")
    low = max(0.4, float(data["roc_auc_mean"].min()) - 0.04)
    axis.set_xlim(low, float(data["roc_auc_mean"].max()) + 0.025)
    for position, value, spread in zip(positions, data["roc_auc_mean"], data["roc_auc_std"]):
        axis.text(value + spread + 0.003, position, f"{value:.3f}".replace(".", ","), va="center", fontsize=9)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return str(path)


def save_coefficients(pairs, out_path=FIGURES_DIR / "coefficients.png", top_n: int = 15) -> str:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    selected = pairs[:top_n]
    names = [name.split("__", 1)[-1] for name, _ in selected]
    weights = [weight for _, weight in selected]

    figure, axis = plt.subplots(figsize=(9, max(4, 0.4 * len(selected))))
    positions = np.arange(len(selected))
    colors = ["#d62728" if weight > 0 else "#1f77b4" for weight in weights]
    axis.barh(positions, weights, color=colors)
    axis.set_yticks(positions)
    axis.set_yticklabels(names, fontsize=8)
    axis.invert_yaxis()
    axis.axvline(0, color="#333333", linewidth=0.8)
    axis.set_xlabel("Вес признака")
    axis.set_title("Наиболее влиятельные признаки логистической регрессии")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return str(path)


def save_regularization_curve(rows, out_path=FIGURES_DIR / "regularization.png") -> str:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = pd.DataFrame(rows)
    styles = {"l1": ("#d62728", "L1"), "l2": ("#1f77b4", "L2")}

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for penalty, (color, label) in styles.items():
        part = data[data["penalty"] == penalty]
        if part.empty:
            continue
        axes[0].plot(part["C"], part["roc_auc"], marker="o", color=color, label=label)
        axes[1].plot(part["C"], part["nonzero"], marker="o", color=color, label=label)
    axes[0].set_xscale("log")
    axes[1].set_xscale("log")
    axes[0].set_xlabel("Сила регуляризации C")
    axes[0].set_ylabel("ROC-AUC на проверочном блоке")
    axes[0].set_title("Влияние регуляризации на качество")
    axes[0].legend()
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Сила регуляризации C")
    axes[1].set_ylabel("Число ненулевых коэффициентов")
    axes[1].set_title("Влияние регуляризации на число признаков")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return str(path)


def save_depth_curve(rows, out_path=FIGURES_DIR / "depth.png") -> str:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = pd.DataFrame(rows)
    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.plot(data["depth"], data["train_roc_auc"], marker="o", color="#1f77b4", label="обучающая часть")
    axis.plot(data["depth"], data["valid_roc_auc"], marker="s", color="#d62728", label="проверочный блок")
    axis.set_xlabel("Максимальная глубина дерева")
    axis.set_ylabel("ROC-AUC")
    axis.set_title("Качество дерева решений в зависимости от глубины")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return str(path)


def save_model_curves(data, entry: dict, n_splits: int = 5, out_dir=FIGURES_DIR) -> str:
    from . import experiments

    X, y_train, train_idx, valid_idx = experiments._last_block(data, n_splits=n_splits)
    from sklearn.pipeline import Pipeline
    from .models import make_pipeline

    steps = make_pipeline(
        entry["family"], text_repr=entry["text_repr"], use_numeric=entry["use_numeric"], **(entry["params"] or {})
    ).steps
    pipeline = Pipeline(steps)
    pipeline.fit(X.iloc[train_idx], y_train[train_idx])
    probability = pipeline.predict_proba(X.iloc[valid_idx])[:, 1]
    return save_roc_pr(y_train[valid_idx], probability, out_dir=out_dir, prefix="final_model")
