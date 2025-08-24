"""Split customers by when they first appear.

A random row split puts the same buying habits on both sides of the
score, and it lets the test period leak into the fit through customers
who are active the whole way through. Holding out the latest first-seen
customers is the check this file can actually support: can the score
say something about people who were not around when the segments were
defined?

Feature windows stop at the next cohort's start so a fit customer is
not described with transactions from the test period.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class CohortSplit:
    fit_ids: list[str]
    val_ids: list[str]
    test_ids: list[str]
    val_start: pd.Timestamp
    test_start: pd.Timestamp
    data_end: pd.Timestamp


def assign_cohorts(
    transactions: pd.DataFrame, val_fraction: float, test_fraction: float
) -> CohortSplit:
    """Assign each customer from the order of their first transaction.

    The last `test_fraction` of customers are the test cohort. The
    `val_fraction` just before them are validation. Ties on the timestamp
    break on customer id so the assignment does not depend on file order.
    """
    first = (
        transactions.groupby("CustomerId", sort=False)["TransactionStartTime"]
        .min()
        .rename("first_txn")
        .reset_index()
        .sort_values(["first_txn", "CustomerId"], kind="mergesort")
        .reset_index(drop=True)
    )
    count = len(first)
    n_test = max(1, int(round(count * test_fraction)))
    n_val = max(1, int(round(count * val_fraction)))
    if n_test + n_val >= count:
        raise ValueError(
            f"Not enough customers ({count}) for the requested split "
            f"({n_val} validation, {n_test} test)"
        )
    fit = first.iloc[: count - n_val - n_test]
    val = first.iloc[count - n_val - n_test : count - n_test]
    test = first.iloc[count - n_test :]
    return CohortSplit(
        fit_ids=fit["CustomerId"].astype(str).tolist(),
        val_ids=val["CustomerId"].astype(str).tolist(),
        test_ids=test["CustomerId"].astype(str).tolist(),
        val_start=pd.Timestamp(val["first_txn"].min()),
        test_start=pd.Timestamp(test["first_txn"].min()),
        data_end=pd.Timestamp(transactions["TransactionStartTime"].max()),
    )


def slice_window(
    transactions: pd.DataFrame, customer_ids: list[str], end: pd.Timestamp | None
) -> pd.DataFrame:
    """Keep one cohort, and drop transactions at or after `end`.

    `end` is the moment the next cohort starts. Transactions after that
    belong to a later decision, so they are not part of this cohort's
    features or its RFM.
    """
    mask = transactions["CustomerId"].isin(set(customer_ids))
    if end is not None:
        mask = mask & (transactions["TransactionStartTime"] < end)
    return transactions.loc[mask].copy()
