# Model card

## Summary

Served model: `random_forest` `d7f25e5ccf02`.
Seed 42. Trained at 2026-11-19T20:35:29.546285+00:00.

The label is a KMeans segment of recency fraction and log activity rates. It is not a loan default. Metrics measure whether other behavior tracks that segment on later customers. They are not a probability of default.

## Intended use

Rank Xente customers by how much their non-RFM behavior resembles the high-risk engagement segment, for a research or portfolio demo of a proxy credit score. The audience is a reviewer who can read the label definition.

## Out of scope

- A probability of default, a Basel PD, or a decision to decline credit.
- Customers with fewer than 3 transactions in the scoring window.
- Any population other than this Ugandan eCommerce sample (currency and country are constants).

## Data and label

Public Xente training transactions, 2018-11-15T02:18:49+00:00 to 2019-02-13T10:01:28+00:00, 95662 rows, 3742 customers. The raw file is not in git.

The label is the high-risk KMeans cluster of standardized recency fraction, log frequency rate, and log monetary rate. Cluster id 2 of 3. The segmenter was fit on the fit cohort only and then applied, unchanged, to validation and test customers.

Scored customers: fit 1654, validation 461, test 560.

## Features

Mix and shape only: debit share, local night and weekend share, hour dispersion, gap regularity, entropy of the product and channel mix, ticket-size dispersion, and shares of category, channel, provider, and pricing level. RFM totals and rates, transaction counts, and FraudResult are not inputs.

## Training

Customers are ordered by first transaction. The latest slice is the test cohort, the slice before that is validation, and the rest is the fit cohort. Logistic regression (weight of evidence, class_weight balanced) and a random forest (class_weight balanced_subsample) are compared on validation ROC-AUC. Logistic regression wins a tie within 0.01. The champion is refit on fit plus validation. Parameters: `{'max_depth': 12, 'min_samples_leaf': 5}`.

class_weight='balanced' moves scores away from observed frequencies. Use them to rank and to apply the cutoff. Do not quote them as a PD.

## Holdout metrics

Rounded to 3 decimals. Unrounded values are in `reports/metrics.json`.

| | ROC-AUC | PR-AUC | KS | Brier |
| --- | ---: | ---: | ---: | ---: |
| Constant baseline | 0.500 | 0.218 | 0.000 | 0.170 |
| Champion (random_forest) | 0.785 | 0.581 | 0.433 | 0.146 |
| RFM oracle, not served | 1.000 | 0.999 | 0.998 | 0.014 |

Test customers: 560. Positives: 122.
At cutoff 0.30: precision 0.364, recall 0.713, TP 87, FP 152, TN 286, FN 35.

Assumed cost is 1.0 per false positive and 5.0 per false negative. Champion test cost 327.0, against 610.0 for calling everyone low risk and 438.0 for calling everyone high risk.

## Limitations

- Low engagement is not default. The segment flags quiet files, including customers who have not had long to look active.
- The history is about three months. This is not a stability test across seasons.
- The 1:5 cost ratio is a placeholder.
- Rates reduce the effect of unequal window lengths. They do not remove it.
- K = 3 is a choice, not a discovered number of credit grades.

## Reproduce

`PYTHONPATH=src python -m credit_risk.train --config configs/config.yaml`
