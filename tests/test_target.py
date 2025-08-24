"""The proxy label is a frozen function of standardized RFM rates."""

import numpy as np
import pandas as pd
import pytest

from credit_risk.target import RFM_COLUMNS, assign_proxy, customer_rfm, fit_proxy


def _transactions() -> pd.DataFrame:
    start = pd.Timestamp("2018-11-15", tz="UTC")
    rows = []
    for step in range(4):
        rows.append(
            {
                "TransactionId": f"T{step}",
                "CustomerId": "C1",
                "Value": 100.0,
                "TransactionStartTime": start + pd.Timedelta(days=step * 2),
            }
        )
    return pd.DataFrame(rows)


def test_rates_change_with_value_and_not_with_a_longer_label_name():
    # Arrange
    frame = _transactions()
    scaled = frame.copy()
    scaled["Value"] = scaled["Value"] * 10
    snapshot = pd.Timestamp("2018-12-01", tz="UTC")
    # Act
    base = customer_rfm(frame, snapshot, min_transactions=3)
    bigger = customer_rfm(scaled, snapshot, min_transactions=3)
    # Then
    assert RFM_COLUMNS == ["RecencyFraction", "LogFrequencyRate", "LogMonetaryRate"]
    assert bigger.loc[0, "LogMonetaryRate"] > base.loc[0, "LogMonetaryRate"]
    assert bigger.loc[0, "LogFrequencyRate"] == pytest.approx(base.loc[0, "LogFrequencyRate"])
    assert bigger.loc[0, "RecencyFraction"] == pytest.approx(base.loc[0, "RecencyFraction"])


def test_short_histories_are_not_labeled():
    # Arrange
    frame = _transactions().iloc[:2]
    # Act
    labeled = customer_rfm(frame, pd.Timestamp("2018-12-01", tz="UTC"), min_transactions=3)
    # Then
    assert labeled.empty


def test_high_risk_cluster_is_frozen_on_the_fit_batch():
    # Arrange: one tight active group and one quiet group, far apart in RFM.
    rows = []
    for index in range(30):
        rows.append(
            {
                "CustomerId": f"A{index}",
                "RecencyFraction": 0.05,
                "LogFrequencyRate": float(np.log1p(4.0)),
                "LogMonetaryRate": float(np.log1p(1000.0)),
                "n_txn": 5,
            }
        )
    for index in range(30):
        rows.append(
            {
                "CustomerId": f"B{index}",
                "RecencyFraction": 0.9,
                "LogFrequencyRate": float(np.log1p(0.1)),
                "LogMonetaryRate": float(np.log1p(5.0)),
                "n_txn": 3,
            }
        )
    rfm = pd.DataFrame(rows)
    # Act
    proxy = fit_proxy(rfm, n_clusters=2, seed=0)
    labeled = assign_proxy(rfm, proxy)
    only_active = assign_proxy(rfm[rfm["CustomerId"].str.startswith("A")], proxy)
    # Then
    # High risk is the quiet group (high recency, low rates), and a batch
    # that contains none of them is not relabeled from its own means.
    quiet = labeled["CustomerId"].str.startswith("B")
    assert labeled.loc[quiet, "is_high_risk"].eq(1).all()
    assert labeled.loc[~quiet, "is_high_risk"].eq(0).all()
    assert only_active["is_high_risk"].eq(0).all()
    centers = proxy.kmeans.cluster_centers_
    risk = centers[:, 0] - centers[:, 1] - centers[:, 2]
    assert proxy.high_risk_cluster == int(np.argmax(risk))
