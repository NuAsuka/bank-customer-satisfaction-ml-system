import argparse
import json

import pandas as pd

from . import batches, experiments, features, load, predict, quality, train_final
from .db import connect
from .runlog import run_step


def _ingest_review(args) -> None:
    with connect() as conn:
        batch_id = args.batch_id or batches.register(conn, "review", args.file)
        with run_step(conn, "load", batch_review_id=batch_id) as step:
            result = load.load_reviews(conn, batch_id, args.file)
            step.rows_in = result["rows"]
            step.rows_out = result["inserted"]
            batches.set_status(
                conn, "review", batch_id, "processing",
                result["inserted"], result["period_start"], result["period_end"],
            )
        with run_step(conn, "quality", batch_review_id=batch_id) as step:
            accepted, rejected = quality.check_reviews(conn, batch_id)
            step.rows_out = accepted
            step.rows_rejected = rejected
        with run_step(conn, "features", batch_review_id=batch_id) as step:
            step.rows_out = features.build(conn, batch_review_id=batch_id)
        batches.set_status(conn, "review", batch_id, "done")
    print(json.dumps({"batch_review_id": batch_id}, ensure_ascii=False))


def _ingest_enrichment(args) -> None:
    with connect() as conn:
        batch_id = args.enrichment_id or batches.register(conn, "enrichment", args.file)
        with run_step(conn, "load", batch_enrichment_id=batch_id) as step:
            result = load.load_enrichment(conn, batch_id, args.file)
            step.rows_in = result["rows"]
            step.rows_out = result["inserted"]
            batches.set_status(
                conn, "enrichment", batch_id, "processing",
                result["inserted"], result["period_start"], result["period_end"],
            )
        with run_step(conn, "quality", batch_enrichment_id=batch_id) as step:
            accepted, rejected = quality.check_enrichment(conn, batch_id)
            step.rows_out = accepted
            step.rows_rejected = rejected
        with run_step(conn, "features", batch_enrichment_id=batch_id) as step:
            step.rows_out = features.build(conn, batch_enrichment_id=batch_id)
        batches.set_status(conn, "enrichment", batch_id, "done")
    print(json.dumps({"batch_enrichment_id": batch_id}, ensure_ascii=False))


def _features(args) -> None:
    with connect() as conn:
        with run_step(conn, "features", batch_review_id=args.batch_id, batch_enrichment_id=args.enrichment_id) as step:
            step.rows_out = features.build(
                conn,
                batch_review_id=args.batch_id,
                batch_enrichment_id=args.enrichment_id,
            )
    print(json.dumps({"rows": step.rows_out}, ensure_ascii=False))


def _dataset(args):
    if args.csv:
        return experiments.load_from_csv(args.csv)
    with connect() as conn:
        return experiments.load_from_db(conn)


def _source(args) -> str:
    return f"csv:{args.csv}" if args.csv else "staging.training_set"


def _experiments(args) -> None:
    data = _dataset(args)
    if args.limit:
        data = data.head(args.limit)
    params = json.loads(args.params) if args.params else None
    if args.family:
        entry = experiments.run_experiment(
            data,
            args.family,
            params=params,
            n_splits=args.n_splits,
            use_numeric=not args.no_numeric,
            notes=args.note,
            data_source=_source(args),
        )
        print(
            json.dumps(
                {
                    "run_id": entry["run_id"],
                    "family": entry["family"],
                    "roc_auc_mean": entry["roc_auc_mean"],
                },
                ensure_ascii=False,
            )
        )
        return
    families = args.families.split(",") if args.families else None
    entries = experiments.run_suite(data, families=families, n_splits=args.n_splits, data_source=_source(args))
    tables = experiments.build_tables()
    best = experiments.select_best(entries)
    print(json.dumps({"runs": len(entries), "tables": tables, "best_family": best["family"]}, ensure_ascii=False))


def _baselines(args) -> None:
    result = experiments.run_baselines(_dataset(args))
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _curves(args) -> None:
    from . import plots

    data = _dataset(args)
    curves = {
        "regularization": experiments.regularization_curve(data),
        "depth": experiments.depth_curve(data),
    }
    path = experiments._write_json(experiments.EXPERIMENTS_DIR / "curves.json", curves)
    print(
        json.dumps(
            {
                "curves": path,
                "regularization": plots.save_regularization_curve(curves["regularization"]),
                "depth": plots.save_depth_curve(curves["depth"]),
            },
            ensure_ascii=False,
        )
    )


def _repro(args) -> None:
    data = _dataset(args)
    entries = experiments.read_log()
    if args.family:
        entry = {"family": args.family, "params": {}, "text_repr": None, "use_numeric": True}
    else:
        entry = experiments.select_best(entries)
    result = experiments.check_reproducibility(
        data,
        entry["family"],
        entry.get("params") or None,
        text_repr=entry.get("text_repr"),
        use_numeric=entry.get("use_numeric", True),
    )
    experiments._write_json(experiments.EXPERIMENTS_DIR / "reproducibility.json", result)
    print(json.dumps(result, ensure_ascii=False))


def _suite(args) -> None:
    data = _dataset(args)
    result = experiments.run_plan(data, n_splits=args.n_splits, data_source=_source(args))
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _tune(args) -> None:
    data = _dataset(args)
    result = experiments.tune_family(data, args.family, n_splits=args.n_splits)
    print(json.dumps(result, ensure_ascii=False))


def _train(args) -> None:
    data = _dataset(args)
    meta = train_final.train(data=data, version=args.version)
    print(json.dumps({"model_version": meta["model_version"], "family": meta["family"]}, ensure_ascii=False))


def _predict(args) -> None:
    with connect() as conn:
        with run_step(conn, "scoring", batch_review_id=args.batch_id) as step:
            step.rows_out = predict.predict_batch(conn, args.batch_id)
    print(json.dumps({"rows": step.rows_out}, ensure_ascii=False))


def _plots(args) -> None:
    from . import plots

    data = _dataset(args)
    entry = experiments.select_best()
    frames = pd.read_csv(experiments.EXPERIMENTS_DIR / "runs.csv")
    print(
        json.dumps(
            {
                "families": plots.save_family_comparison(
                    frames[frames["notes"].fillna("") != "польза признаков"]
                ),
                "curves": plots.save_model_curves(data, entry),
            },
            ensure_ascii=False,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="src.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_review = subparsers.add_parser("ingest-review")
    ingest_review.add_argument("--file", required=True)
    ingest_review.add_argument("--batch-id", type=int)
    ingest_review.set_defaults(func=_ingest_review)

    ingest_enrichment = subparsers.add_parser("ingest-enrichment")
    ingest_enrichment.add_argument("--file", required=True)
    ingest_enrichment.add_argument("--enrichment-id", type=int)
    ingest_enrichment.set_defaults(func=_ingest_enrichment)

    features_parser = subparsers.add_parser("features")
    features_parser.add_argument("--batch-id", type=int)
    features_parser.add_argument("--enrichment-id", type=int)
    features_parser.set_defaults(func=_features)

    experiments_parser = subparsers.add_parser("experiments")
    experiments_parser.add_argument("--families")
    experiments_parser.add_argument("--family")
    experiments_parser.add_argument("--params")
    experiments_parser.add_argument("--note")
    experiments_parser.add_argument("--no-numeric", action="store_true")
    experiments_parser.add_argument("--n-splits", type=int, default=5)
    experiments_parser.add_argument("--csv")
    experiments_parser.add_argument("--limit", type=int)
    experiments_parser.set_defaults(func=_experiments)

    baselines_parser = subparsers.add_parser("baselines")
    baselines_parser.add_argument("--csv")
    baselines_parser.set_defaults(func=_baselines)

    curves_parser = subparsers.add_parser("curves")
    curves_parser.add_argument("--csv")
    curves_parser.set_defaults(func=_curves)

    repro_parser = subparsers.add_parser("repro")
    repro_parser.add_argument("--family")
    repro_parser.add_argument("--csv")
    repro_parser.set_defaults(func=_repro)

    suite_parser = subparsers.add_parser("suite")
    suite_parser.add_argument("--csv")
    suite_parser.add_argument("--n-splits", type=int, default=5)
    suite_parser.set_defaults(func=_suite)

    tune_parser = subparsers.add_parser("tune")
    tune_parser.add_argument("--family", required=True)
    tune_parser.add_argument("--n-splits", type=int, default=3)
    tune_parser.add_argument("--csv")
    tune_parser.set_defaults(func=_tune)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--version")
    train_parser.add_argument("--csv")
    train_parser.set_defaults(func=_train)

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--batch-id", type=int, required=True)
    predict_parser.set_defaults(func=_predict)

    plots_parser = subparsers.add_parser("plots")
    plots_parser.add_argument("--csv")
    plots_parser.set_defaults(func=_plots)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
