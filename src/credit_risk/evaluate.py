"""Holdout metrics, plots, and the written report.

The numbers describe agreement with the proxy segment. The caveat is
written into the same files as the metrics so a table cannot be copied
out without it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

CAVEAT = (
    "The label is a KMeans segment of recency fraction and log activity rates. "
    "It is not a loan default. Metrics measure whether other behavior tracks that "
    "segment on later customers. They are not a probability of default."
)

DISPLAY_NAMES = {
    "baseline": "Constant baseline",
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "rfm_oracle_diagnostic": "RFM oracle (diagnostic)",
}


def ks_statistic(y_true, y_score) -> float:
    """Maximum gap between the cumulative score distributions of the two classes."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    if np.unique(y_true).size < 2 or np.unique(y_score).size < 2:
        return 0.0
    false_positive, true_positive, _thresholds = roc_curve(y_true, y_score)
    return float(np.max(true_positive - false_positive))


def classification_metrics(
    y_true, y_score, threshold: float, fp_cost: float, fn_cost: float
) -> dict:
    """Ranking metrics, plus the confusion matrix at one cutoff."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    predicted = (y_score >= threshold).astype(int)
    constant = np.unique(y_score).size < 2
    roc_auc = 0.5 if constant else float(roc_auc_score(y_true, y_score))
    true_positive = int(np.sum((predicted == 1) & (y_true == 1)))
    true_negative = int(np.sum((predicted == 0) & (y_true == 0)))
    false_positive = int(np.sum((predicted == 1) & (y_true == 0)))
    false_negative = int(np.sum((predicted == 0) & (y_true == 1)))
    return {
        "n": int(y_true.size),
        "positives": int(y_true.sum()),
        "roc_auc": roc_auc,
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "ks": 0.0 if constant else ks_statistic(y_true, y_score),
        "brier": float(brier_score_loss(y_true, y_score)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "threshold": float(threshold),
        "tp": true_positive,
        "tn": true_negative,
        "fp": false_positive,
        "fn": false_negative,
        "cost": float(false_positive * fp_cost + false_negative * fn_cost),
    }


def choose_threshold(y_true, y_score, fp_cost: float, fn_cost: float) -> tuple[float, float]:
    """Cutoff on this cohort that minimizes fp_cost * FP + fn_cost * FN.

    The costs are assumptions. On a tie the lower cutoff wins, which flags
    more customers when the two costs do not separate the choices.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    best_threshold = 0.5
    best_cost = math.inf
    positives = int(y_true.sum())
    for threshold in np.round(np.linspace(0.01, 0.99, 99), 2):
        predicted = y_score >= threshold
        false_positive = int(np.sum(predicted & (y_true == 0)))
        false_negative = int(positives - np.sum(predicted & (y_true == 1)))
        cost = false_positive * fp_cost + false_negative * fn_cost
        if cost < best_cost or (math.isclose(cost, best_cost) and threshold < best_threshold):
            best_cost = float(cost)
            best_threshold = float(threshold)
    return best_threshold, best_cost


def policy_cost(y_true, predict_positive: bool, fp_cost: float, fn_cost: float) -> float:
    """Cost of calling every customer high risk, or every customer low risk."""
    y_true = np.asarray(y_true).astype(int)
    if predict_positive:
        return float(np.sum(y_true == 0) * fp_cost)
    return float(np.sum(y_true == 1) * fn_cost)


def write_reports(directory: Path, metrics: dict, plots: dict) -> None:
    """Write metrics.json, the two markdown reports, and the figures."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "metrics.json").write_text(json.dumps(_jsonable(metrics), indent=2) + "\n")
    (directory / "evaluation.md").write_text(render_evaluation(metrics))
    (directory / "model_card.md").write_text(render_model_card(metrics))
    write_figures(directory / "figures", plots)


def write_figures(directory: Path, plots: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    y_test = np.asarray(plots["y_test"]).astype(int)
    scores = plots["scores"]
    champion = plots["champion"]
    _plot_roc(directory / "roc_curve.png", y_test, scores)
    _plot_pr(directory / "pr_curve.png", y_test, scores)
    _plot_calibration(directory / "calibration.png", y_test, scores[champion])
    _plot_ks(directory / "ks_curve.png", y_test, scores[champion])
    _plot_cost(
        directory / "threshold_cost.png",
        plots["y_val"],
        plots["val_scores"],
        plots["threshold"],
        plots["fp_cost"],
        plots["fn_cost"],
    )
    _plot_importance(directory / "feature_importance.png", plots["importances"])
    plt.close("all")


def render_evaluation(metrics: dict) -> str:
    """Human-readable holdout report. Numbers come from `metrics`, not from memory."""
    data = metrics["data"]
    test = metrics["test"]
    lines = [
        "# Evaluation",
        "",
        CAVEAT,
        "",
        "## Data",
        "",
        f"- Transactions: {data['n_transactions']}",
        f"- Customers in the file: {data['n_customers']}",
        f"- Window: {data['time_min']} to {data['time_max']}",
        f"- Validation cohort starts: {data['val_start']}",
        f"- Test cohort starts: {data['test_start']}",
        f"- Minimum transactions inside the cohort window: {data['min_transactions']}",
        "",
        "| Cohort | Assigned | Scored | Positives | Prevalence |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name in ("fit", "validation", "test"):
        block = data[name]
        lines.append(
            f"| {name} | {block['assigned']} | {block['scored']} | "
            f"{block['positives']} | {block['prevalence']:.3f} |"
        )
    proxy = metrics["proxy"]
    lines.extend(
        [
            "",
            "## Proxy label",
            "",
            f"KMeans with {proxy['n_clusters']} clusters on standardized "
            + ", ".join(proxy["center_order"])
            + f". High-risk cluster id: {proxy['high_risk_cluster']}.",
            "",
            "Scaled centers follow that column order. "
            "The risk score of a center is z_recency - z_frequency - z_monetary.",
            "",
            "| Cluster | Risk score | Scaled center |",
            "| ---: | ---: | --- |",
        ]
    )
    for index, (score, center) in enumerate(zip(proxy["risk_scores"], proxy["scaled_centers"])):
        center_text = ", ".join(f"{value:.3f}" for value in center)
        lines.append(f"| {index} | {score:.3f} | {center_text} |")
    lines.extend(
        [
            "",
            "## Model selection",
            "",
            metrics["selection"]["rule"],
            "",
            (
                f"Champion: `{metrics['champion']}` with parameters "
                f"`{metrics['selection']['chosen_params']}`."
            ),
            "",
            "| Model | Best validation ROC-AUC |",
            "| --- | ---: |",
        ]
    )
    for name, score in metrics["selection"]["validation_roc_auc"].items():
        lines.append(f"| {DISPLAY_NAMES.get(name, name)} | {score:.3f} |")
    lines.extend(
        [
            "",
            "The grid, scored on the validation cohort:",
            "",
            "| Model | Parameters | Validation ROC-AUC | Validation PR-AUC |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for row in metrics["selection"]["grid"]:
        lines.append(
            f"| {row['model']} | `{row['params']}` | {row['val_roc_auc']:.3f} | "
            f"{row['val_pr_auc']:.3f} |"
        )
    threshold = metrics["threshold"]
    lines.extend(
        [
            "",
            "## Threshold",
            "",
            (
                f"False positive cost {threshold['fp_cost']}, "
                f"false negative cost {threshold['fn_cost']}. "
                "Both are assumptions, not a measured loss."
            ),
            f"Champion cutoff {threshold['champion']:.2f}, chosen on the validation scores "
            f"of the pre-refit champion (validation cost {threshold['validation_cost']:.1f}).",
            "",
            "## Test cohort",
            "",
            "The champion was refit on the fit and validation customers before this table. "
            "The RFM oracle is not a candidate. It is the same label predicted from the "
            "three columns that define it, so its AUC shows what a leaked feature set can print.",
            "",
            "| Model | ROC-AUC | PR-AUC | KS | Brier | Precision | Recall | Cost |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name in ("baseline", "logistic_regression", "random_forest", "rfm_oracle_diagnostic"):
        row = test[name]
        lines.append(
            f"| {DISPLAY_NAMES.get(name, name)} | {row['roc_auc']:.3f} | {row['pr_auc']:.3f} | "
            f"{row['ks']:.3f} | {row['brier']:.3f} | {row['precision']:.3f} | "
            f"{row['recall']:.3f} | {row['cost']:.1f} |"
        )
    cost = test["cost"]
    lines.extend(
        [
            "",
            f"Test cost of the champion at its cutoff: {cost['champion']:.1f}. "
            f"Call everyone low risk: {cost['all_low_risk']:.1f}. "
            f"Call everyone high risk: {cost['all_high_risk']:.1f}.",
            "",
            "## Information value",
            "",
            "Diagnostic only. Columns were not dropped on information value, "
            "because that selection would be fit on the same rows as the model.",
            "",
            "| Feature | IV |",
            "| --- | ---: |",
        ]
    )
    for row in metrics["information_value"]:
        lines.append(f"| {row['feature']} | {row['iv']:.4f} |")
    lines.extend(
        [
            "",
            "## Permutation importance",
            "",
            "Mean drop in test ROC-AUC when the column is shuffled. "
            "Descriptive. Not used to select features or the champion.",
            "",
            "| Feature | Mean drop | Std |",
            "| --- | ---: | ---: |",
        ]
    )
    for row in metrics["permutation_importance"]:
        lines.append(f"| {row['feature']} | {row['mean']:.4f} | {row['std']:.4f} |")
    lines.extend(["", "Figures are in `reports/figures/`.", ""])
    return "\n".join(lines)


def render_model_card(metrics: dict) -> str:
    """Model card for the artifact that the API serves."""
    data = metrics["data"]
    champion = metrics["champion"]
    row = metrics["test"][champion]
    oracle = metrics["test"]["rfm_oracle_diagnostic"]
    baseline = metrics["test"]["baseline"]
    threshold = metrics["threshold"]
    return "\n".join(
        [
            "# Model card",
            "",
            "## Summary",
            "",
            f"Served model: `{champion}` `{metrics['model_version']}`.",
            f"Seed {metrics['seed']}. Trained at {metrics['trained_at']}.",
            "",
            CAVEAT,
            "",
            "## Intended use",
            "",
            "Rank Xente customers by how much their non-RFM behavior resembles "
            "the high-risk engagement segment, for a research or portfolio demo "
            "of a proxy credit score. The audience is a reviewer who can read "
            "the label definition.",
            "",
            "## Out of scope",
            "",
            "- A probability of default, a Basel PD, or a decision to decline credit.",
            (
                "- Customers with fewer than "
                f"{data['min_transactions']} transactions in the scoring window."
            ),
            "- Any population other than this Ugandan eCommerce sample "
            "(currency and country are constants).",
            "",
            "## Data and label",
            "",
            f"Public Xente training transactions, {data['time_min']} to {data['time_max']}, "
            f"{data['n_transactions']} rows, {data['n_customers']} customers. "
            "The raw file is not in git.",
            "",
            "The label is the high-risk KMeans cluster of standardized recency "
            "fraction, log frequency rate, and log monetary rate. "
            f"Cluster id {metrics['proxy']['high_risk_cluster']} "
            f"of {metrics['proxy']['n_clusters']}. "
            "The segmenter was fit on the fit cohort only and then applied, unchanged, "
            "to validation and test customers.",
            "",
            f"Scored customers: fit {data['fit']['scored']}, "
            f"validation {data['validation']['scored']}, test {data['test']['scored']}.",
            "",
            "## Features",
            "",
            "Mix and shape only: debit share, local night and weekend share, hour dispersion, "
            "gap regularity, entropy of the product and channel mix, ticket-size dispersion, and "
            "shares of category, channel, provider, and pricing level. "
            "RFM totals and rates, transaction counts, and FraudResult are not inputs.",
            "",
            "## Training",
            "",
            "Customers are ordered by first transaction. The latest slice is the test cohort, "
            "the slice before that is validation, and the rest is the fit cohort. "
            "Logistic regression (weight of evidence, class_weight balanced) and a random "
            "forest (class_weight balanced_subsample) are compared on validation ROC-AUC. "
            "Logistic regression wins a tie within "
            f"{metrics['selection']['prefer_logistic_margin']}. "
            "The champion is refit on fit plus validation. "
            f"Parameters: `{metrics['selection']['chosen_params']}`.",
            "",
            "class_weight='balanced' moves scores away from observed frequencies. "
            "Use them to rank and to apply the cutoff. Do not quote them as a PD.",
            "",
            "## Holdout metrics",
            "",
            "Rounded to 3 decimals. Unrounded values are in `reports/metrics.json`.",
            "",
            "| | ROC-AUC | PR-AUC | KS | Brier |",
            "| --- | ---: | ---: | ---: | ---: |",
            f"| Constant baseline | {baseline['roc_auc']:.3f} | {baseline['pr_auc']:.3f} | "
            f"{baseline['ks']:.3f} | {baseline['brier']:.3f} |",
            f"| Champion ({champion}) | {row['roc_auc']:.3f} | {row['pr_auc']:.3f} | "
            f"{row['ks']:.3f} | {row['brier']:.3f} |",
            f"| RFM oracle, not served | {oracle['roc_auc']:.3f} | {oracle['pr_auc']:.3f} | "
            f"{oracle['ks']:.3f} | {oracle['brier']:.3f} |",
            "",
            f"Test customers: {row['n']}. Positives: {row['positives']}.",
            f"At cutoff {threshold['champion']:.2f}: precision {row['precision']:.3f}, "
            f"recall {row['recall']:.3f}, "
            f"TP {row['tp']}, FP {row['fp']}, TN {row['tn']}, FN {row['fn']}.",
            "",
            f"Assumed cost is {threshold['fp_cost']} per false positive and "
            f"{threshold['fn_cost']} per false negative. "
            f"Champion test cost {metrics['test']['cost']['champion']:.1f}, "
            f"against {metrics['test']['cost']['all_low_risk']:.1f} for calling everyone low risk "
            f"and {metrics['test']['cost']['all_high_risk']:.1f} for calling everyone high risk.",
            "",
            "## Limitations",
            "",
            "- Low engagement is not default. The segment flags quiet files, including "
            "customers who have not had long to look active.",
            "- The history is about three months. This is not a stability test across seasons.",
            "- The 1:5 cost ratio is a placeholder.",
            "- Rates reduce the effect of unequal window lengths. They do not remove it.",
            "- K = 3 is a choice, not a discovered number of credit grades.",
            "",
            "## Reproduce",
            "",
            "`PYTHONPATH=src python -m credit_risk.train --config configs/config.yaml`",
            "",
        ]
    )


def _plot_roc(path: Path, y_true, scores: dict) -> None:
    figure, axis = plt.subplots(figsize=(7, 5))
    for name, score in scores.items():
        if np.unique(score).size < 2:
            axis.plot([0, 1], [0, 1], linestyle="--", label=DISPLAY_NAMES.get(name, name))
            continue
        false_positive, true_positive, _thresholds = roc_curve(y_true, score)
        style = "--" if name == "rfm_oracle_diagnostic" else "-"
        axis.plot(
            false_positive,
            true_positive,
            linestyle=style,
            label=f"{DISPLAY_NAMES.get(name, name)} ({roc_auc_score(y_true, score):.3f})",
        )
    axis.plot([0, 1], [0, 1], color="0.7", linewidth=1)
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_title("Test ROC. The dashed oracle uses the label inputs.")
    axis.legend(loc="lower right", fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def _plot_pr(path: Path, y_true, scores: dict) -> None:
    figure, axis = plt.subplots(figsize=(7, 5))
    prevalence = float(np.mean(y_true))
    axis.hlines(prevalence, 0, 1, color="0.7", linewidth=1, label=f"Prevalence ({prevalence:.3f})")
    for name, score in scores.items():
        precision, recall, _thresholds = precision_recall_curve(y_true, score)
        style = "--" if name == "rfm_oracle_diagnostic" else "-"
        axis.plot(recall, precision, linestyle=style, label=DISPLAY_NAMES.get(name, name))
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title("Test precision-recall")
    axis.legend(loc="best", fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def _plot_calibration(path: Path, y_true, score) -> None:
    figure, axis = plt.subplots(figsize=(7, 5))
    try:
        fraction, mean_score = calibration_curve(y_true, score, n_bins=8, strategy="quantile")
        axis.plot(mean_score, fraction, marker="o", label="Champion")
    except ValueError:
        axis.text(0.5, 0.5, "Not enough rows to bin", ha="center")
    axis.plot([0, 1], [0, 1], color="0.7", linewidth=1)
    axis.set_xlabel("Mean predicted score")
    axis.set_ylabel("Observed segment rate")
    axis.set_title("Reliability. Balanced class weights shift scores off the diagonal.")
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def _plot_ks(path: Path, y_true, score) -> None:
    figure, axis = plt.subplots(figsize=(7, 5))
    if np.unique(score).size < 2:
        axis.text(0.5, 0.5, "Constant score", ha="center")
    else:
        false_positive, true_positive, thresholds = roc_curve(y_true, score)
        # roc_curve thresholds are aligned with the points after the first.
        axis.plot(thresholds[1:], (true_positive - false_positive)[1:])
    axis.set_xlabel("Cutoff")
    axis.set_ylabel("TPR - FPR")
    axis.set_title(f"KS curve (max {ks_statistic(y_true, score):.3f})")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def _plot_cost(path: Path, y_true, score, threshold: float, fp_cost: float, fn_cost: float) -> None:
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score, dtype=float)
    grid = np.round(np.linspace(0.01, 0.99, 99), 2)
    costs = []
    positives = int(y_true.sum())
    for cutoff in grid:
        predicted = score >= cutoff
        false_positive = int(np.sum(predicted & (y_true == 0)))
        false_negative = int(positives - np.sum(predicted & (y_true == 1)))
        costs.append(false_positive * fp_cost + false_negative * fn_cost)
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(grid, costs)
    axis.axvline(threshold, color="0.3", linestyle="--", label=f"Chosen {threshold:.2f}")
    axis.set_xlabel("Cutoff")
    axis.set_ylabel("Validation cost")
    axis.set_title("Cutoff chosen on validation, not on the test cohort")
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def _plot_importance(path: Path, rows: list[dict]) -> None:
    top = list(rows)[:15]
    figure, axis = plt.subplots(figsize=(8, 6))
    if not top:
        axis.text(0.5, 0.5, "No importance computed", ha="center")
    else:
        names = [row["feature"] for row in top][::-1]
        means = [row["mean"] for row in top][::-1]
        axis.barh(names, means)
    axis.set_xlabel("Drop in test ROC-AUC when shuffled")
    axis.set_title("Permutation importance of the served model")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
