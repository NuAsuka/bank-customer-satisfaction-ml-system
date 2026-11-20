import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

SEED = 42
TEXT_COLUMN = "review_text"
SOURCE_COLUMN = "review_source"
NUM_COLS = ["words_cnt", "avg_word_len", "exclaim_cnt"]
SVD_COMPONENTS = 100


def _logistic(**params):
    params.setdefault("l1_ratio", 0.0)
    return LogisticRegression(max_iter=2000, solver="liblinear", random_state=SEED, **params)


FAMILIES = {
    "logistic_regression": {
        "text_repr": "tfidf",
        "rank": 0,
        "build": _logistic,
        "grid": {"C": [0.1, 1.0, 10.0], "l1_ratio": [1.0, 0.0]},
    },
    "naive_bayes": {
        "text_repr": "tfidf",
        "rank": 1,
        "build": lambda **params: ComplementNB(**params),
        "grid": {"alpha": [0.1, 0.5, 1.0]},
    },
    "knn": {
        "text_repr": "svd",
        "rank": 2,
        "build": lambda **params: KNeighborsClassifier(n_neighbors=25, n_jobs=-1, **params),
        "grid": {"n_neighbors": [15, 25, 41]},
    },
    "decision_tree": {
        "text_repr": "svd",
        "rank": 3,
        "build": lambda **params: DecisionTreeClassifier(random_state=SEED, **params),
        "grid": {"max_depth": [10, 30, None], "min_samples_leaf": [1, 5]},
    },
    "random_forest": {
        "text_repr": "svd",
        "rank": 4,
        "build": lambda **params: RandomForestClassifier(
            n_estimators=300, n_jobs=-1, random_state=SEED, **params
        ),
        "grid": {"max_depth": [20, None], "min_samples_leaf": [1, 5]},
    },
    "gradient_boosting": {
        "text_repr": "svd",
        "rank": 5,
        "build": lambda **params: HistGradientBoostingClassifier(random_state=SEED, **params),
        "grid": {"max_iter": [200, 400], "learning_rate": [0.05, 0.1]},
    },
}


def add_text_features(frame):
    texts = frame[TEXT_COLUMN].fillna("").astype(str)
    counts = texts.str.split().str.len().fillna(0).astype(float)
    letters = texts.str.count(r"\S").astype(float)
    result = frame.copy()
    result["words_cnt"] = counts
    result["exclaim_cnt"] = texts.str.count(r"!").astype(float)
    result["avg_word_len"] = np.where(counts > 0, letters / counts.replace(0, 1), 0.0)
    return result


def make_pipeline(
    family: str,
    text_repr: str | None = None,
    use_numeric: bool = True,
    **params,
) -> Pipeline:
    if family not in FAMILIES:
        raise ValueError(f"неизвестное семейство: {family}")
    spec = FAMILIES[family]
    representation = text_repr or spec["text_repr"]
    sparse = representation != "svd"

    text_steps = [
        (
            "tfidf",
            TfidfVectorizer(
                max_features=30000,
                ngram_range=(1, 2),
                min_df=2,
                sublinear_tf=True,
            ),
        )
    ]
    if not sparse:
        text_steps.append(("svd", TruncatedSVD(n_components=SVD_COMPONENTS, random_state=SEED)))

    transformers = []
    if use_numeric:
        scaler = MinMaxScaler() if family == "naive_bayes" else StandardScaler()
        transformers.append(("num", Pipeline([("scale", scaler)]), NUM_COLS))
    transformers.append(("text", Pipeline(text_steps), TEXT_COLUMN))
    transformers.append(
        ("source", OneHotEncoder(handle_unknown="ignore", sparse_output=sparse), [SOURCE_COLUMN])
    )

    steps = [
        ("features", FunctionTransformer(add_text_features)),
        ("prep", ColumnTransformer(transformers)),
    ]
    if family == "knn":
        steps.append(("scale", StandardScaler()))
    steps.append(("clf", spec["build"](**params)))
    return Pipeline(steps)


def grid_for(family: str) -> dict:
    return {f"clf__{key}": values for key, values in FAMILIES[family]["grid"].items()}


def complexity_rank(family: str) -> int:
    return FAMILIES[family]["rank"]


def feature_names(pipe: Pipeline) -> list[str]:
    return list(pipe.named_steps["prep"].get_feature_names_out())


def coefficients(pipe: Pipeline) -> list[tuple[str, float]]:
    clf = pipe.named_steps["clf"]
    names = feature_names(pipe)
    if hasattr(clf, "coef_"):
        weights = np.ravel(clf.coef_)
    elif hasattr(clf, "feature_log_prob_"):
        weights = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
    else:
        return []
    pairs = [(str(name), float(weight)) for name, weight in zip(names, weights)]
    pairs.sort(key=lambda item: abs(item[1]), reverse=True)
    return pairs
