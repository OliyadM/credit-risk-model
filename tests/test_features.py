"""Feature aggregation stays off the columns that define the proxy label."""

import pandas as pd
import pytest

from credit_risk.data import validate_transactions
from credit_risk.features import (
    DENIED_FEATURE_NAMES,
    aggregate_features,
    assert_features_allowed,
)
from credit_risk.woe import WOEEncoder


def _frame() -> pd.DataFrame:
    start = pd.Timestamp("2018-11-15T10:00:00", tz="UTC")
    rows = []
    for step, amount in enumerate((1000.0, -20.0, 1000.0, -20.0)):
        rows.append(
            {
                "TransactionId": f"T{step}",
                "BatchId": "B",
                "AccountId": "A",
                "SubscriptionId": "S",
                "CustomerId": "CustomerId_1",
                "CurrencyCode": "UGX",
                "CountryCode": 256,
                "ProviderId": "ProviderId_4",
                "ProductId": "ProductId_1" if step % 2 == 0 else "ProductId_2",
                "ProductCategory": "airtime" if amount > 0 else "financial_services",
                "ChannelId": "ChannelId_3",
                "Amount": amount,
                "Value": abs(amount),
                "TransactionStartTime": start + pd.Timedelta(days=step),
                "PricingStrategy": 2,
                "FraudResult": 0,
            }
        )
    return validate_transactions(pd.DataFrame(rows))


def test_missing_column_is_rejected():
    # Arrange
    frame = _frame().drop(columns=["Value"])
    # Act / Then
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_transactions(frame)


def test_features_exclude_label_inputs_and_ignore_scale():
    # Arrange
    frame = _frame()
    scaled = frame.copy()
    scaled["Amount"] = scaled["Amount"] * 10
    scaled["Value"] = scaled["Value"] * 10
    # Act
    features, _levels = aggregate_features(frame, min_level_share=0.0)
    scaled_features, _levels = aggregate_features(scaled, min_level_share=0.0)
    # Then
    # Scaling value changes monetary rate and must not change the model matrix.
    columns = [column for column in features.columns if column != "CustomerId"]
    assert_features_allowed(columns)
    assert not ({column.lower() for column in columns} & DENIED_FEATURE_NAMES)
    pd.testing.assert_frame_equal(features, scaled_features)
    assert features.loc[0, "debit_share"] == pytest.approx(0.5)


def test_repeating_the_mix_does_not_change_entropy_or_shares():
    # Arrange
    frame = _frame()
    repeated = pd.concat([frame, frame], ignore_index=True)
    repeated["TransactionId"] = [f"T{i}" for i in range(len(repeated))]
    # Act
    once, _levels = aggregate_features(frame, min_level_share=0.0)
    twice, _levels = aggregate_features(repeated, min_level_share=0.0)
    # Then
    # Counts doubled. Shares and entropy are proportions, so they stay put.
    # interval_cv can move because repeated timestamps add zero-length gaps.
    stable = [
        column
        for column in once.columns
        if column != "CustomerId" and column != "interval_cv"
    ]
    pd.testing.assert_frame_equal(once[stable], twice[stable])
    assert once.loc[0, "category_entropy"] > 0


def test_share_columns_follow_the_fit_vocabulary():
    # Arrange
    frame = _frame()
    later = frame.copy()
    later["ProductCategory"] = "ticket"
    # Act
    fit_features, levels = aggregate_features(frame, min_level_share=0.0)
    later_features, _levels = aggregate_features(later, levels=levels, min_level_share=0.0)
    # Then
    assert list(fit_features.columns) == list(later_features.columns)
    share_columns = [column for column in fit_features.columns if column.startswith("cat_share_")]
    assert fit_features.loc[0, share_columns].sum() == pytest.approx(1.0)
    assert later_features.loc[0, "cat_share_residual"] == pytest.approx(1.0)


def test_woe_reuses_edges_from_fit():
    # Arrange
    rng_values = pd.Series(range(40), dtype=float)
    frame = pd.DataFrame({"a": rng_values})
    target = (frame["a"] > 20).astype(int).to_numpy()
    encoder = WOEEncoder(n_bins=4)
    # Act
    encoder.fit(frame, target)
    encoded = encoder.transform(frame.assign(a=frame["a"] + 100))
    # Then
    # A shift past the training range falls in the stored end bin.
    assert encoded["a"].nunique() == 1
    known = set(encoder.woe_["a"].values())
    assert encoded["a"].iloc[0] in known
