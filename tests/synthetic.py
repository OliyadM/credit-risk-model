"""Small transaction frames for the smoke tests.

Two kinds of customer are interleaved through time so every cohort has
both an active pattern and a quiet one. Active customers keep transacting
and buy airtime. Quiet customers transact three times and stop, on
financial services. The proxy should be able to separate them, and the
non-RFM features should be able to see the category mix.
"""

from __future__ import annotations

import pandas as pd


def make_transactions(n_customers: int = 60) -> pd.DataFrame:
    start = pd.Timestamp("2018-11-15", tz="UTC")
    rows = []
    for index in range(n_customers):
        first = start + pd.Timedelta(days=index)
        active = index % 2 == 0
        days = list(range(12)) if active else [0, 0, 1]
        value = 5000.0 if active else 100.0
        for step, day in enumerate(days):
            amount = value if step % 2 == 0 else -value
            rows.append(
                {
                    "TransactionId": f"T{index}_{step}",
                    "BatchId": "B",
                    "AccountId": "A",
                    "SubscriptionId": "S",
                    "CustomerId": f"CustomerId_{index:04d}",
                    "CurrencyCode": "UGX",
                    "CountryCode": 256,
                    "ProviderId": "ProviderId_4" if active else "ProviderId_6",
                    "ProductId": "ProductId_1" if active else "ProductId_2",
                    "ProductCategory": "airtime" if active else "financial_services",
                    "ChannelId": "ChannelId_3" if active else "ChannelId_2",
                    "Amount": amount,
                    "Value": value,
                    "TransactionStartTime": first + pd.Timedelta(days=day, hours=step),
                    "PricingStrategy": 2 if active else 4,
                    "FraudResult": 0,
                }
            )
    return pd.DataFrame(rows)
