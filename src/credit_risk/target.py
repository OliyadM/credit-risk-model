"""Build the proxy label from RFM, and freeze the rule.

There is no default flag in the Xente file. The stand-in is a cluster of
engagement: quiet, infrequent, low spend-rate customers versus everyone
else. That cluster is a segment of the same behavior a lender can see,
not a repayment outcome. Training is only honest if the model is not
handed the three numbers the cluster was built from, and if the choice
of which cluster is "high risk" is fixed before the test customers are
scored.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# What the cluster actually sees. Recency is a fraction of the observed
# window, and the two rates are log1p'd. Absolute days and raw UGX-per-day
# are not comparable across cohorts: a later cohort's whole life can be
# shorter than the fit cohort's quiet period, and a few merchants pull the
# monetary rate across orders of magnitude.
RFM_COLUMNS = ["RecencyFraction", "LogFrequencyRate", "LogMonetaryRate"]


@dataclass
class ProxyModel:
    """Fitted segmenter. `high_risk_cluster` is an id, not a recomputed rule."""

    scaler: StandardScaler
    kmeans: KMeans
    high_risk_cluster: int
    risk_scores: list[float]
    n_clusters: int


def customer_rfm(
    transactions: pd.DataFrame, snapshot: pd.Timestamp, min_transactions: int
) -> pd.DataFrame:
    """One RFM row per customer with at least `min_transactions` rows.

    Frequency and monetary value are log1p rates per day of tenure, not
    totals. A customer observed for three weeks would otherwise look
    "low frequency" next to a customer observed for two months, and a
    raw UGX rate would be scaled around a few very large merchants.
    Recency is the fraction of the observed window since the last
    transaction, so a short cohort can still look quiet. Tenure and the
    window are at least one day so a same-day history has a defined rate.

    Customers under the minimum are omitted rather than imputed. A
    one-transaction file does not have a recency story yet.
    """
    if transactions.empty:
        return pd.DataFrame(columns=["CustomerId", *RFM_COLUMNS, "n_txn"])

    grouped = transactions.groupby("CustomerId", sort=False)
    first = grouped["TransactionStartTime"].min()
    last = grouped["TransactionStartTime"].max()
    count = grouped["TransactionId"].count()
    monetary = grouped["Value"].sum()
    tenure_days = ((last - first).dt.total_seconds() / 86400.0).clip(lower=1.0)
    snapshot_ts = pd.Timestamp(snapshot)
    if snapshot_ts.tzinfo is None:
        snapshot_ts = snapshot_ts.tz_localize("UTC")
    window_days = ((snapshot_ts - first).dt.total_seconds() / 86400.0).clip(lower=1.0)
    recency_days = (snapshot_ts - last).dt.total_seconds() / 86400.0
    # Fraction of the window since the last transaction. 0 is "active at
    # the decision time"; 1 is "silent since the window opened." Absolute
    # days would make a short later cohort unable to look high-recency.
    recency_fraction = (recency_days / window_days).clip(lower=0.0, upper=1.0)
    frequency_rate = count / tenure_days
    monetary_rate = monetary / tenure_days
    frame = pd.DataFrame(
        {
            "CustomerId": first.index.astype(str),
            "RecencyFraction": recency_fraction.to_numpy(dtype=float),
            "LogFrequencyRate": np.log1p(frequency_rate.to_numpy(dtype=float)),
            "LogMonetaryRate": np.log1p(monetary_rate.to_numpy(dtype=float)),
            "n_txn": count.to_numpy(dtype=int),
        }
    )
    # The minimum is applied after the rate is defined so the dropped
    # rows are exactly the histories that are too short to score.
    return frame.loc[frame["n_txn"] >= min_transactions].reset_index(drop=True)


def fit_proxy(rfm: pd.DataFrame, n_clusters: int, seed: int) -> ProxyModel:
    """Fit KMeans on standardized RFM and store which center is high risk.

    The risk score is computed on the scaled centers. Raw UGX amounts
    would dominate a sum of cluster means, and the label would collapse
    to "spent the least." The columns are already a fraction and two
    logs; scaling then gives them a comparable unit before the sum.
    """
    if len(rfm) < 2:
        raise ValueError("Need at least two customers to fit the proxy")
    clusters = min(n_clusters, len(rfm))
    if clusters < 2:
        raise ValueError("n_clusters must leave room for two segments")
    matrix = rfm.loc[:, RFM_COLUMNS].to_numpy(dtype=float)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)
    model = KMeans(n_clusters=clusters, n_init=10, random_state=seed)
    model.fit(scaled)
    # Center columns are Recency, FrequencyRate, MonetaryRate, in that order.
    centers = model.cluster_centers_
    risk = centers[:, 0] - centers[:, 1] - centers[:, 2]
    high_risk = int(np.argmax(risk))
    return ProxyModel(
        scaler=scaler,
        kmeans=model,
        high_risk_cluster=high_risk,
        risk_scores=[float(value) for value in risk],
        n_clusters=clusters,
    )


def assign_proxy(rfm: pd.DataFrame, proxy: ProxyModel) -> pd.DataFrame:
    """Label customers with the stored cluster id.

    The high-risk id is not recomputed from this batch. Scoring only
    quiet customers must not relabel the least-quiet of them as high risk.
    """
    if rfm.empty:
        return pd.DataFrame(columns=["CustomerId", "cluster", "is_high_risk"])
    matrix = rfm.loc[:, RFM_COLUMNS].to_numpy(dtype=float)
    scaled = proxy.scaler.transform(matrix)
    cluster = proxy.kmeans.predict(scaled)
    return pd.DataFrame(
        {
            "CustomerId": rfm["CustomerId"].astype(str).to_numpy(),
            "cluster": cluster.astype(int),
            "is_high_risk": (cluster == proxy.high_risk_cluster).astype(int),
        }
    )
