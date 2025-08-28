"""Train the proxy score and write the artifact the API is allowed to serve.

Nothing in this module runs on import. A test can call `run` on a small
frame. The real entry point is `python -m credit_risk.train`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline

from credit_risk.config import TrainConfig, load_config
from credit_risk.data import load_transactions, validate_transactions
from credit_risk.evaluate import (
    CAVEAT,
    choose_threshold,
    classification_metrics,
    policy_cost,
    write_reports,
)
from credit_risk.features import aggregate_features, assert_features_allowed
from credit_risk.split import assign_cohorts, slice_window
from credit_risk.target import RFM_COLUMNS, assign_proxy, customer_rfm, fit_proxy
from credit_risk.woe import WOEEncoder

logger = logging.getLogger(__name__)


@dataclass
class Candidate:
    name: str
    params: dict
    val_roc_auc: float
    val_pr_auc: float
    val_scores: np.ndarray


def run(config: TrainConfig, transactions: pd.DataFrame | None = None) -> dict:
    """Fit the proxy and the champion, then write reports and the artifact.

    The test cohort is not used to choose the model or the cutoff. It is
    scored after both are fixed. The RFM oracle is fit only so the report
    can show the AUC that the label inputs would have produced.
    """
    np.random.seed(config.seed)
    frame = (
        validate_transactions(transactions)
        if transactions is not None
        else load_transactions(config.raw_path)
    )
    cohorts = assign_cohorts(frame, config.val_fraction, config.test_fraction)
    # Windows end when the next cohort starts. Test customers have no
    # earlier transactions, so their window is the whole file.
    fit_tx = slice_window(frame, cohorts.fit_ids, cohorts.val_start)
    val_tx = slice_window(frame, cohorts.val_ids, cohorts.test_start)
    test_tx = slice_window(frame, cohorts.test_ids, None)
    test_snapshot = cohorts.data_end + pd.Timedelta(days=1)

    fit_rfm = customer_rfm(fit_tx, cohorts.val_start, config.min_transactions)
    val_rfm = customer_rfm(val_tx, cohorts.test_start, config.min_transactions)
    test_rfm = customer_rfm(test_tx, test_snapshot, config.min_transactions)
    proxy = fit_proxy(fit_rfm, config.n_clusters, config.seed)
    fit_labels = assign_proxy(fit_rfm, proxy)
    val_labels = assign_proxy(val_rfm, proxy)
    test_labels = assign_proxy(test_rfm, proxy)

    fit_x, levels = aggregate_features(
        fit_tx,
        levels=None,
        min_level_share=config.min_level_share,
        timezone_offset_hours=config.timezone_offset_hours,
    )
    val_x, _levels = aggregate_features(
        val_tx,
        levels=levels,
        min_level_share=config.min_level_share,
        timezone_offset_hours=config.timezone_offset_hours,
    )
    test_x, _levels = aggregate_features(
        test_tx,
        levels=levels,
        min_level_share=config.min_level_share,
        timezone_offset_hours=config.timezone_offset_hours,
    )

    x_fit, y_fit, id_fit = _align(fit_x, fit_labels)
    x_val, y_val, id_val = _align(val_x, val_labels)
    x_test, y_test, id_test = _align(test_x, test_labels)
    _require_both_classes(y_fit, "fit")
    _require_both_classes(y_val, "validation")
    _require_both_classes(y_test, "test")
    feature_columns = list(x_fit.columns)
    assert_features_allowed(feature_columns)
    if list(x_val.columns) != feature_columns or list(x_test.columns) != feature_columns:
        raise RuntimeError("Feature columns differ across cohorts")

    candidates = _grid(config, x_fit, y_fit, x_val, y_val)
    champion = _select(candidates, config.prefer_logistic_margin)
    threshold, validation_cost = choose_threshold(
        y_val, champion.val_scores, config.fp_cost, config.fn_cost
    )
    best_by_family = {
        "logistic_regression": _best_of(candidates, "logistic_regression"),
        "random_forest": _best_of(candidates, "random_forest"),
    }
    x_train = pd.concat([x_fit, x_val], axis=0, ignore_index=True)
    y_train = np.concatenate([y_fit, y_val])
    refit = {
        name: _fit_pipeline(config, name, candidate.params, x_train, y_train)
        for name, candidate in best_by_family.items()
    }
    test_scores = {
        name: _positive_proba(pipeline, x_test) for name, pipeline in refit.items()
    }
    thresholds = {
        name: choose_threshold(y_val, candidate.val_scores, config.fp_cost, config.fn_cost)[0]
        for name, candidate in best_by_family.items()
    }
    # The served cutoff is the champion's. The other model is reported at
    # its own validation cutoff so a different score scale is not punished.
    prevalence = float(y_train.mean())
    baseline_scores = np.full(y_test.shape[0], prevalence)
    oracle_scores = _oracle_scores(fit_rfm, id_fit, y_fit, test_rfm, id_test, config.seed)

    family_metrics = {}
    for name, scores in test_scores.items():
        family_metrics[name] = classification_metrics(
            y_test, scores, thresholds[name], config.fp_cost, config.fn_cost
        )
    family_metrics["baseline"] = classification_metrics(
        y_test, baseline_scores, 0.5, config.fp_cost, config.fn_cost
    )
    family_metrics["rfm_oracle_diagnostic"] = classification_metrics(
        y_test, oracle_scores, 0.5, config.fp_cost, config.fn_cost
    )

    served = refit[champion.name]
    importance = _importance(served, x_test, y_test, config)
    information_value = _information_value(config, x_train, y_train, feature_columns)
    version = _version(champion.name, champion.params, feature_columns, config.seed, threshold)
    trained_at = datetime.now(timezone.utc).isoformat()
    metrics = _metrics_document(
        config=config,
        frame=frame,
        cohorts=cohorts,
        proxy=proxy,
        counts=_counts(cohorts, y_fit, y_val, y_test),
        candidates=candidates,
        champion=champion,
        threshold=threshold,
        thresholds=thresholds,
        validation_cost=validation_cost,
        family_metrics=family_metrics,
        y_test=y_test,
        information_value=information_value,
        importance=importance,
        version=version,
        trained_at=trained_at,
    )
    reports = _path(config.reports_dir)
    write_reports(
        reports,
        metrics,
        {
            "y_test": y_test,
            "scores": {
                "baseline": baseline_scores,
                "logistic_regression": test_scores["logistic_regression"],
                "random_forest": test_scores["random_forest"],
                "rfm_oracle_diagnostic": oracle_scores,
            },
            "champion": champion.name,
            "y_val": y_val,
            "val_scores": champion.val_scores,
            "threshold": threshold,
            "importances": importance,
            "fp_cost": config.fp_cost,
            "fn_cost": config.fn_cost,
        },
    )
    artifact = _path(config.artifact_path)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": served,
            "feature_columns": feature_columns,
            "threshold": float(threshold),
            "model_name": champion.name,
            "model_version": version,
            "trained_at": trained_at,
            "seed": config.seed,
            "params": champion.params,
            "levels": levels,
            "caveat": CAVEAT,
        },
        artifact,
        compress=3,
    )
    logger.info(
        "Served %s %s. Test ROC-AUC %.3f. Oracle ROC-AUC %.3f. Artifact %s",
        champion.name,
        version,
        family_metrics[champion.name]["roc_auc"],
        family_metrics["rfm_oracle_diagnostic"]["roc_auc"],
        artifact,
    )
    return metrics


def _align(features: pd.DataFrame, labels: pd.DataFrame):
    merged = labels[["CustomerId", "is_high_risk"]].merge(features, on="CustomerId", how="inner")
    if merged.empty:
        raise ValueError("Features and labels share no customers")
    # Sort so a rerun feeds the estimators the same row order.
    merged = merged.sort_values("CustomerId").reset_index(drop=True)
    columns = [column for column in features.columns if column != "CustomerId"]
    return (
        merged.loc[:, columns],
        merged["is_high_risk"].to_numpy(dtype=int),
        merged["CustomerId"].astype(str).tolist(),
    )


def _require_both_classes(y: np.ndarray, name: str) -> None:
    classes = np.unique(y)
    if classes.size < 2:
        raise RuntimeError(
            f"The {name} cohort has a single class {classes.tolist()}. "
            "Refusing to invent a metric."
        )


def _grid(config, x_fit, y_fit, x_val, y_val) -> list[Candidate]:
    candidates = []
    for value in config.lr_C:
        params = {"C": float(value)}
        pipeline = _fit_pipeline(config, "logistic_regression", params, x_fit, y_fit)
        scores = _positive_proba(pipeline, x_val)
        candidates.append(
            Candidate(
                "logistic_regression",
                params,
                float(roc_auc_score(y_val, scores)),
                float(average_precision_score(y_val, scores)),
                scores,
            )
        )
    for depth in config.rf_max_depth:
        for leaf in config.rf_min_samples_leaf:
            params = {"max_depth": int(depth), "min_samples_leaf": int(leaf)}
            pipeline = _fit_pipeline(config, "random_forest", params, x_fit, y_fit)
            scores = _positive_proba(pipeline, x_val)
            candidates.append(
                Candidate(
                    "random_forest",
                    params,
                    float(roc_auc_score(y_val, scores)),
                    float(average_precision_score(y_val, scores)),
                    scores,
                )
            )
    if not candidates:
        raise ValueError("The model grid is empty")
    return candidates


def _select(candidates: list[Candidate], margin: float) -> Candidate:
    """Highest validation ROC-AUC. Logistic regression wins inside `margin`.

    A linear weight-of-evidence model is the one a reviewer can read. It
    is preferred only when the forest is not actually better on the
    validation cohort.
    """
    best = max(candidate.val_roc_auc for candidate in candidates)
    eligible = [candidate for candidate in candidates if best - candidate.val_roc_auc <= margin]
    logistic = [candidate for candidate in eligible if candidate.name == "logistic_regression"]
    pool = logistic or eligible
    return max(pool, key=lambda candidate: (candidate.val_roc_auc, candidate.val_pr_auc))


def _best_of(candidates: list[Candidate], name: str) -> Candidate:
    family = [candidate for candidate in candidates if candidate.name == name]
    if not family:
        raise ValueError(f"No candidate for {name}")
    return max(family, key=lambda candidate: (candidate.val_roc_auc, candidate.val_pr_auc))


def _fit_pipeline(config, name: str, params: dict, x, y) -> Pipeline:
    if name == "logistic_regression":
        pipeline = Pipeline(
            [
                ("woe", WOEEncoder(n_bins=config.woe_bins)),
                (
                    "clf",
                    LogisticRegression(
                        C=params["C"],
                        class_weight="balanced",
                        max_iter=1000,
                        solver="lbfgs",
                        random_state=config.seed,
                    ),
                ),
            ]
        )
    elif name == "random_forest":
        pipeline = Pipeline(
            [
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=config.rf_n_estimators,
                        max_depth=params["max_depth"],
                        min_samples_leaf=params["min_samples_leaf"],
                        class_weight="balanced_subsample",
                        random_state=config.seed,
                        n_jobs=config.rf_n_jobs,
                    ),
                )
            ]
        )
    else:
        raise ValueError(f"Unknown model {name}")
    pipeline.fit(x, y)
    return pipeline


def _positive_proba(pipeline: Pipeline, features: pd.DataFrame) -> np.ndarray:
    classes = list(pipeline.named_steps["clf"].classes_)
    if 1 not in classes:
        raise RuntimeError(f"Classifier classes {classes} do not include the positive label")
    probabilities = pipeline.predict_proba(features)
    return probabilities[:, classes.index(1)]


def _oracle_scores(fit_rfm, id_fit, y_fit, test_rfm, id_test, seed: int) -> np.ndarray:
    """Random forest on the three label inputs. Not a model we would serve.

    If this AUC is high and the champion AUC is high too, the champion is
    still not using these columns. The gap between them is the point of
    the diagnostic.
    """
    train = fit_rfm.set_index("CustomerId").loc[id_fit, RFM_COLUMNS]
    test = test_rfm.set_index("CustomerId").loc[id_test, RFM_COLUMNS]
    oracle = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        min_samples_leaf=5,
        random_state=seed,
        n_jobs=1,
    )
    oracle.fit(train, y_fit)
    probabilities = oracle.predict_proba(test)
    classes = list(oracle.classes_)
    return probabilities[:, classes.index(1)]


def _importance(pipeline, features, y, config: TrainConfig) -> list[dict]:
    result = permutation_importance(
        pipeline,
        features,
        y,
        scoring="roc_auc",
        n_repeats=config.permutation_repeats,
        random_state=config.seed,
        n_jobs=1,
    )
    rows = [
        {
            "feature": column,
            "mean": float(result.importances_mean[index]),
            "std": float(result.importances_std[index]),
        }
        for index, column in enumerate(features.columns)
    ]
    rows.sort(key=lambda row: row["mean"], reverse=True)
    return rows


def _information_value(config, features, y, columns) -> list[dict]:
    encoder = WOEEncoder(n_bins=config.woe_bins)
    encoder.fit(features, y)
    rows = [{"feature": column, "iv": float(encoder.iv_[column])} for column in columns]
    rows.sort(key=lambda row: row["iv"], reverse=True)
    return rows


def _version(name: str, params: dict, columns: list[str], seed: int, threshold: float) -> str:
    payload = json.dumps(
        {
            "model": name,
            "params": params,
            "features": columns,
            "seed": seed,
            "threshold": round(float(threshold), 4),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def _counts(cohorts, y_fit, y_val, y_test) -> dict:
    def block(assigned, y):
        positives = int(np.sum(y))
        return {
            "assigned": int(assigned),
            "scored": int(y.size),
            "positives": positives,
            "prevalence": float(positives / y.size),
        }

    return {
        "fit": block(len(cohorts.fit_ids), y_fit),
        "validation": block(len(cohorts.val_ids), y_val),
        "test": block(len(cohorts.test_ids), y_test),
    }


def _metrics_document(
    config,
    frame,
    cohorts,
    proxy,
    counts,
    candidates,
    champion,
    threshold,
    thresholds,
    validation_cost,
    family_metrics,
    y_test,
    information_value,
    importance,
    version,
    trained_at,
) -> dict:
    return {
        "caveat": CAVEAT,
        "seed": config.seed,
        "model_version": version,
        "trained_at": trained_at,
        "champion": champion.name,
        "data": {
            "n_transactions": int(len(frame)),
            "n_customers": int(frame["CustomerId"].nunique()),
            "time_min": frame["TransactionStartTime"].min().isoformat(),
            "time_max": frame["TransactionStartTime"].max().isoformat(),
            "val_start": pd.Timestamp(cohorts.val_start).isoformat(),
            "test_start": pd.Timestamp(cohorts.test_start).isoformat(),
            "min_transactions": config.min_transactions,
            **counts,
        },
        "proxy": {
            "n_clusters": proxy.n_clusters,
            "high_risk_cluster": proxy.high_risk_cluster,
            "risk_scores": proxy.risk_scores,
            "scaled_centers": proxy.kmeans.cluster_centers_.tolist(),
            "center_order": list(RFM_COLUMNS),
        },
        "selection": {
            "rule": (
                "Highest validation ROC-AUC. Logistic regression is preferred when "
                f"it is within {config.prefer_logistic_margin} of the best score."
            ),
            "prefer_logistic_margin": config.prefer_logistic_margin,
            "chosen_model": champion.name,
            "chosen_params": champion.params,
            "validation_roc_auc": {
                "logistic_regression": _best_of(candidates, "logistic_regression").val_roc_auc,
                "random_forest": _best_of(candidates, "random_forest").val_roc_auc,
            },
            "grid": [
                {
                    "model": candidate.name,
                    "params": candidate.params,
                    "val_roc_auc": candidate.val_roc_auc,
                    "val_pr_auc": candidate.val_pr_auc,
                }
                for candidate in candidates
            ],
        },
        "threshold": {
            "champion": threshold,
            "fp_cost": config.fp_cost,
            "fn_cost": config.fn_cost,
            "chosen_on": "validation",
            "validation_cost": validation_cost,
            "by_model": thresholds,
        },
        "test": {
            **family_metrics,
            "cost": {
                "champion": family_metrics[champion.name]["cost"],
                "all_low_risk": policy_cost(y_test, False, config.fp_cost, config.fn_cost),
                "all_high_risk": policy_cost(y_test, True, config.fp_cost, config.fn_cost),
            },
        },
        "information_value": information_value,
        "permutation_importance": importance,
    }


def _path(value: str) -> Path:
    return Path(value)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Train the proxy credit-risk score")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    metrics = run(load_config(args.config))
    champion = metrics["champion"]
    test = metrics["test"][champion]
    oracle = metrics["test"]["rfm_oracle_diagnostic"]
    print(
        f"champion={champion} test_roc_auc={test['roc_auc']:.4f} "
        f"test_pr_auc={test['pr_auc']:.4f} test_ks={test['ks']:.4f} "
        f"oracle_roc_auc={oracle['roc_auc']:.4f}"
    )


if __name__ == "__main__":
    main()
