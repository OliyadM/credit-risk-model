"""Load transactions and refuse a file that is not the one this model expects.

Validation happens before any cohort, scaler, or cluster is fit. A silent
coercion here would change the label and the features together, and the
report would not say so.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "TransactionId",
    "BatchId",
    "AccountId",
    "SubscriptionId",
    "CustomerId",
    "CurrencyCode",
    "CountryCode",
    "ProviderId",
    "ProductId",
    "ProductCategory",
    "ChannelId",
    "Amount",
    "Value",
    "TransactionStartTime",
    "PricingStrategy",
    "FraudResult",
]


def load_transactions(path: str | Path) -> pd.DataFrame:
    """Read the Xente training file and return a validated frame."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(
            f"Raw transactions not found at {source}. Save the Xente Fraud "
            "Detection Challenge training file as data/raw/data.csv."
        )
    frame = pd.read_csv(source)
    return validate_transactions(frame)


def validate_transactions(frame: pd.DataFrame) -> pd.DataFrame:
    """Check the schema and drop only rows whose time or amount cannot be read.

    FraudResult has to be 0 or 1 on every remaining row. It is not a model
    feature, but a file that fails this check is not the Xente training
    extract and should not be turned into a proxy label.
    """
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = frame.loc[:, REQUIRED_COLUMNS].copy()
    out["TransactionStartTime"] = pd.to_datetime(
        out["TransactionStartTime"], utc=True, errors="coerce"
    )
    out["Amount"] = pd.to_numeric(out["Amount"], errors="coerce")
    out["Value"] = pd.to_numeric(out["Value"], errors="coerce")
    broken = (
        out["TransactionStartTime"].isna() | out["Amount"].isna() | out["Value"].isna()
    )
    dropped = int(broken.sum())
    if dropped:
        logger.warning("Dropped %s rows with an unparseable time or amount", dropped)
    out = out.loc[~broken].copy()
    if out.empty:
        raise ValueError("No usable transactions after validation")

    fraud = pd.to_numeric(out["FraudResult"], errors="coerce")
    if fraud.isna().any() or not fraud.isin([0, 1]).all():
        raise ValueError("FraudResult must be 0 or 1 on every row")
    out["FraudResult"] = fraud.astype(int)

    duplicate_ids = int(out["TransactionId"].duplicated().sum())
    if duplicate_ids:
        logger.warning("Dropped %s duplicate TransactionId rows", duplicate_ids)
        out = out.drop_duplicates("TransactionId", keep="first")

    out["CustomerId"] = out["CustomerId"].astype(str)
    return out.reset_index(drop=True)
