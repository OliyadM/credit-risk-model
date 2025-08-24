"""Customer features that are not inputs to the proxy label.

The label is a cluster of recency, frequency rate, and monetary rate.
A column that restates those three would make the holdout AUC a replay
of KMeans. The columns below are mix and shape: what the customer buys,
on which channel, at what time of day, and how regular the gaps are.

Excluded on purpose: recency, frequency, monetary, both rates, the
transaction count, value or amount sums and means, tenure, the mean gap
between transactions (it is essentially 1/frequency), unique-counts
divided by the transaction count (that ratio rewrites frequency), and
FraudResult. FraudResult is a different event, and it is too rare to be
a stable credit feature. Country and currency are constants in this file.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BASE_COLUMNS = [
    "debit_share",
    "night_share",
    "weekend_share",
    "hour_std",
    "interval_cv",
    "category_entropy",
    "channel_entropy",
    "provider_entropy",
    "product_entropy",
    "value_cv",
]

# Exact names. Substring checks would ban value_cv, which is a shape, not a sum.
DENIED_FEATURE_NAMES = {
    "recency",
    "frequency",
    "frequencyrate",
    "monetary",
    "monetaryrate",
    "n_txn",
    "transaction_count",
    "value_sum",
    "value_mean",
    "value_std",
    "value_min",
    "value_max",
    "amount_sum",
    "amount_mean",
    "amount_std",
    "amount_min",
    "amount_max",
    "tenure",
    "tenure_days",
    "interval_days_mean",
    "fraudresult",
    "fraud_share",
    "customerid",
    "is_high_risk",
    "product_diversity",
    "category_diversity",
    "channel_diversity",
    "provider_diversity",
}

_LEVEL_COLUMNS = (
    ("ProductCategory", "cat"),
    ("ChannelId", "channel"),
    ("ProviderId", "provider"),
    ("PricingStrategy", "pricing"),
)


def assert_features_allowed(columns: list[str] | pd.Index) -> None:
    """Raise if a label input, or the label itself, is in the model matrix.

    CustomerId may sit beside the features. It is not a model input, so it
    is ignored here. Every other denied name is a restatement of the proxy.
    """
    found = {str(column).lower() for column in columns if str(column).lower() != "customerid"}
    leaked = found & DENIED_FEATURE_NAMES
    if leaked:
        raise ValueError(f"Refusing features that restate the proxy label: {sorted(leaked)}")


def aggregate_features(
    transactions: pd.DataFrame,
    levels: dict[str, list[str]] | None = None,
    min_level_share: float = 0.01,
    timezone_offset_hours: int = 3,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Aggregate one row per customer.

    `levels` is the vocabulary learned on the fit cohort. Later cohorts
    reuse it so a category that shows up only in the test window cannot
    add a column the model has never seen. Its mass goes to the residual
    share. Pass `levels=None` only on the fit cohort.
    """
    if transactions.empty:
        empty = pd.DataFrame(columns=["CustomerId", *BASE_COLUMNS])
        return empty, {column: [] for column, _prefix in _LEVEL_COLUMNS}

    work = transactions.copy()
    # Uganda is UTC+3 all year. A fixed offset keeps night and weekend
    # identical in a container that has no timezone database. The file
    # itself is stamped in UTC.
    local = work["TransactionStartTime"] + pd.Timedelta(hours=timezone_offset_hours)
    work["hour"] = local.dt.hour
    work["is_night"] = ((work["hour"] < 6) | (work["hour"] >= 22)).astype(float)
    work["is_weekend"] = (local.dt.day_of_week >= 5).astype(float)
    # Positive Amount is a debit from the customer in this file. The sign
    # is the behavior. The magnitude is a monetary total, so it stays out.
    work["is_debit"] = (work["Amount"] > 0).astype(float)
    for column, _prefix in _LEVEL_COLUMNS:
        work[column] = _as_level(work[column])

    grouped = work.groupby("CustomerId", sort=False)
    base = grouped.agg(
        debit_share=("is_debit", "mean"),
        night_share=("is_night", "mean"),
        weekend_share=("is_weekend", "mean"),
        value_mean=("Value", "mean"),
    )
    # Population dispersion (ddof=0). The sample standard deviation still
    # moves when the same mix is repeated, which would smuggle the
    # transaction count back in.
    value_mean = base["value_mean"].to_numpy(dtype=float)
    value_std = grouped["Value"].std(ddof=0).fillna(0.0).to_numpy(dtype=float)
    base["hour_std"] = grouped["hour"].std(ddof=0).fillna(0.0)
    base["value_cv"] = np.where(value_mean > 0, value_std / value_mean, 0.0)
    # Entropy is a function of the mix only. Repeating the same
    # transactions would not change it, so it does not restate the count.
    base["category_entropy"] = _entropy(work, "ProductCategory").reindex(base.index).fillna(0.0)
    base["channel_entropy"] = _entropy(work, "ChannelId").reindex(base.index).fillna(0.0)
    base["provider_entropy"] = _entropy(work, "ProviderId").reindex(base.index).fillna(0.0)
    base["product_entropy"] = _entropy(work, "ProductId").reindex(base.index).fillna(0.0)
    base["interval_cv"] = _interval_cv(work).reindex(base.index).fillna(0.0)

    learned = {} if levels is None else levels
    used: dict[str, list[str]] = {}
    frame = base.loc[:, BASE_COLUMNS].copy()
    for column, prefix in _LEVEL_COLUMNS:
        vocabulary = list(learned.get(column, _learn_levels(work[column], min_level_share)))
        used[column] = vocabulary
        shares = _share_table(work, column, vocabulary, prefix)
        frame = frame.join(shares, how="left")

    frame = frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    frame.index = frame.index.astype(str)
    frame.index.name = "CustomerId"
    result = frame.reset_index()
    assert_features_allowed([column for column in result.columns if column != "CustomerId"])
    return result, used


def _learn_levels(series: pd.Series, min_level_share: float) -> list[str]:
    """Keep levels that are common enough to be a stable share.

    Rare levels move into the residual. A column for a category with a
    handful of rows would be a customer identifier in disguise.
    """
    shares = series.value_counts(normalize=True, dropna=False)
    kept = []
    for value, share in shares.items():
        text = str(value)
        if text == "nan":
            continue
        if float(share) >= min_level_share:
            kept.append(text)
    return kept


def _share_table(
    frame: pd.DataFrame, column: str, levels: list[str], prefix: str
) -> pd.DataFrame:
    """Mean of one-hot columns. Each row of the result sums to 1.

    Levels not in `levels` are residual, including levels that appear
    only in a later cohort.
    """
    known = set(levels)
    mapped = frame[column].where(frame[column].isin(known), other="__residual__")
    dummies = pd.get_dummies(mapped, dtype=float)
    order = [*levels, "__residual__"]
    for name in order:
        if name not in dummies.columns:
            dummies[name] = 0.0
    dummies = dummies.loc[:, order]
    renamed = {}
    for name in order:
        label = "residual" if name == "__residual__" else _slug(name)
        renamed[name] = f"{prefix}_share_{label}"
    dummies = dummies.rename(columns=renamed)
    if dummies.columns.duplicated().any():
        raise ValueError(f"Duplicate {prefix} share columns: {list(dummies.columns)}")
    dummies["CustomerId"] = frame["CustomerId"].astype(str).to_numpy()
    # numeric_only keeps the mean off the id column. The id stays as the index.
    return dummies.groupby("CustomerId", sort=False).mean(numeric_only=True)


def _entropy(frame: pd.DataFrame, column: str) -> pd.Series:
    """Shannon entropy of one customer's mix, in nats.

    Proportions are used, not counts, so a longer history of the same
    mix has the same value. Zero-probability levels never appear in the
    groupby, and a single-level customer has entropy 0.
    """
    counts = frame.groupby(["CustomerId", column], sort=False).size()
    totals = counts.groupby(level=0).transform("sum")
    proportions = counts / totals
    contribution = -(proportions * np.log(proportions.to_numpy(dtype=float)))
    return contribution.groupby(level=0).sum()


def _interval_cv(frame: pd.DataFrame) -> pd.Series:
    """Coefficient of variation of the gaps, not the mean gap.

    The mean gap restates frequency. The coefficient of variation says
    whether the gaps are even or bursty, which the RFM rates do not.
    Population standard deviation is used because many customers have
    only a few gaps and the sample correction is mostly noise.
    """
    ordered = frame.sort_values(["CustomerId", "TransactionStartTime"], kind="mergesort")
    gap_days = (
        ordered.groupby("CustomerId", sort=False)["TransactionStartTime"]
        .diff()
        .dt.total_seconds()
        / 86400.0
    )
    stats = (
        pd.DataFrame(
            {
                "CustomerId": ordered["CustomerId"].astype(str).to_numpy(),
                "gap": gap_days.to_numpy(dtype=float),
            }
        )
        .dropna()
        .groupby("CustomerId", sort=False)["gap"]
        .agg(["count", "mean", "std"])
    )
    count = stats["count"].to_numpy(dtype=float)
    mean = stats["mean"].to_numpy(dtype=float)
    sample_std = stats["std"].fillna(0.0).to_numpy(dtype=float)
    scale = np.sqrt(np.clip(count - 1.0, 0.0, None) / np.clip(count, 1.0, None))
    population_std = sample_std * scale
    cv = np.where((count >= 2) & (mean > 0), population_std / mean, 0.0)
    return pd.Series(cv, index=stats.index, name="interval_cv")


def _as_level(series: pd.Series) -> pd.Series:
    """String levels, with whole numbers rendered as '2' rather than '2.0'."""
    if pd.api.types.is_numeric_dtype(series):
        return series.map(_number_level)
    return series.astype(str)


def _number_level(value: object) -> str:
    if pd.isna(value):
        return "nan"
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return str(value)


def _slug(value: str) -> str:
    cleaned = "".join(character.lower() if character.isalnum() else "_" for character in str(value))
    parts = [part for part in cleaned.split("_") if part]
    return "_".join(parts) or "unknown"
