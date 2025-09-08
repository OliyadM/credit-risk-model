# Evaluation

The label is a KMeans segment of recency fraction and log activity rates. It is not a loan default. Metrics measure whether other behavior tracks that segment on later customers. They are not a probability of default.

## Data

- Transactions: 95662
- Customers in the file: 3742
- Window: 2018-11-15T02:18:49+00:00 to 2019-02-13T10:01:28+00:00
- Validation cohort starts: 2019-01-01T22:04:20+00:00
- Test cohort starts: 2019-01-23T09:32:57+00:00
- Minimum transactions inside the cohort window: 3

| Cohort | Assigned | Scored | Positives | Prevalence |
| --- | ---: | ---: | ---: | ---: |
| fit | 2246 | 1654 | 347 | 0.210 |
| validation | 748 | 461 | 123 | 0.267 |
| test | 748 | 560 | 122 | 0.218 |

## Proxy label

KMeans with 3 clusters on standardized RecencyFraction, LogFrequencyRate, LogMonetaryRate. High-risk cluster id: 2.

Scaled centers follow that column order. The risk score of a center is z_recency - z_frequency - z_monetary.

| Cluster | Risk score | Scaled center |
| ---: | ---: | --- |
| 0 | -1.270 | 0.858, 1.050, 1.078 |
| 1 | 0.264 | -0.938, -0.678, -0.524 |
| 2 | 1.389 | 0.744, -0.131, -0.514 |

## Model selection

Highest validation ROC-AUC. Logistic regression is preferred when it is within 0.01 of the best score.

Champion: `random_forest` with parameters `{'max_depth': 12, 'min_samples_leaf': 5}`.

| Model | Best validation ROC-AUC |
| --- | ---: |
| Logistic regression | 0.767 |
| Random forest | 0.831 |

The grid, scored on the validation cohort:

| Model | Parameters | Validation ROC-AUC | Validation PR-AUC |
| --- | --- | ---: | ---: |
| logistic_regression | `{'C': 0.1}` | 0.765 | 0.585 |
| logistic_regression | `{'C': 1.0}` | 0.767 | 0.583 |
| logistic_regression | `{'C': 10.0}` | 0.766 | 0.578 |
| random_forest | `{'max_depth': 6, 'min_samples_leaf': 5}` | 0.814 | 0.668 |
| random_forest | `{'max_depth': 6, 'min_samples_leaf': 20}` | 0.812 | 0.659 |
| random_forest | `{'max_depth': 12, 'min_samples_leaf': 5}` | 0.831 | 0.685 |
| random_forest | `{'max_depth': 12, 'min_samples_leaf': 20}` | 0.821 | 0.667 |

## Threshold

False positive cost 1.0, false negative cost 5.0. Both are assumptions, not a measured loss.
Champion cutoff 0.30, chosen on the validation scores of the pre-refit champion (validation cost 196.0).

## Test cohort

The champion was refit on the fit and validation customers before this table. The RFM oracle is not a candidate. It is the same label predicted from the three columns that define it, so its AUC shows what a leaked feature set can print.

| Model | ROC-AUC | PR-AUC | KS | Brier | Precision | Recall | Cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Constant baseline | 0.500 | 0.218 | 0.000 | 0.170 | 0.000 | 0.000 | 610.0 |
| Logistic regression | 0.652 | 0.479 | 0.308 | 0.218 | 0.299 | 0.672 | 392.0 |
| Random forest | 0.785 | 0.581 | 0.433 | 0.146 | 0.364 | 0.713 | 327.0 |
| RFM oracle (diagnostic) | 1.000 | 0.999 | 0.998 | 0.014 | 0.991 | 0.934 | 41.0 |

Test cost of the champion at its cutoff: 327.0. Call everyone low risk: 610.0. Call everyone high risk: 438.0.

## Information value

Diagnostic only. Columns were not dropped on information value, because that selection would be fit on the same rows as the model.

| Feature | IV |
| --- | ---: |
| provider_share_providerid_6 | 0.5278 |
| category_entropy | 0.4132 |
| cat_share_financial_services | 0.3419 |
| product_entropy | 0.3318 |
| interval_cv | 0.3162 |
| cat_share_airtime | 0.2679 |
| provider_entropy | 0.2620 |
| value_cv | 0.2416 |
| debit_share | 0.2281 |
| channel_entropy | 0.2140 |
| channel_share_channelid_2 | 0.2117 |
| provider_share_providerid_4 | 0.2060 |
| channel_share_channelid_3 | 0.2060 |
| night_share | 0.1809 |
| provider_share_providerid_5 | 0.1278 |
| pricing_share_2 | 0.1095 |
| pricing_share_4 | 0.1067 |
| weekend_share | 0.1054 |
| hour_std | 0.0550 |
| provider_share_providerid_1 | 0.0512 |
| cat_share_utility_bill | 0.0009 |
| cat_share_data_bundles | 0.0009 |
| cat_share_tv | 0.0009 |
| cat_share_residual | 0.0009 |
| channel_share_residual | 0.0009 |
| provider_share_providerid_3 | 0.0009 |
| provider_share_residual | 0.0009 |
| pricing_share_1 | 0.0009 |
| pricing_share_residual | 0.0009 |

## Permutation importance

Mean drop in test ROC-AUC when the column is shuffled. Descriptive. Not used to select features or the champion.

| Feature | Mean drop | Std |
| --- | ---: | ---: |
| hour_std | 0.1036 | 0.0038 |
| value_cv | 0.0275 | 0.0060 |
| category_entropy | 0.0241 | 0.0053 |
| pricing_share_2 | 0.0121 | 0.0023 |
| provider_entropy | 0.0121 | 0.0033 |
| provider_share_providerid_6 | 0.0083 | 0.0020 |
| night_share | 0.0072 | 0.0028 |
| cat_share_airtime | 0.0071 | 0.0003 |
| pricing_share_4 | 0.0056 | 0.0018 |
| cat_share_financial_services | 0.0050 | 0.0032 |
| product_entropy | 0.0046 | 0.0018 |
| weekend_share | 0.0024 | 0.0022 |
| channel_share_channelid_2 | 0.0018 | 0.0003 |
| cat_share_utility_bill | 0.0017 | 0.0004 |
| channel_entropy | 0.0011 | 0.0012 |
| debit_share | 0.0007 | 0.0006 |
| provider_share_providerid_4 | 0.0004 | 0.0009 |
| cat_share_residual | 0.0002 | 0.0002 |
| cat_share_tv | 0.0002 | 0.0001 |
| channel_share_residual | 0.0000 | 0.0001 |
| provider_share_residual | 0.0000 | 0.0000 |
| pricing_share_residual | -0.0000 | 0.0003 |
| pricing_share_1 | -0.0002 | 0.0002 |
| channel_share_channelid_3 | -0.0010 | 0.0007 |
| provider_share_providerid_3 | -0.0011 | 0.0002 |
| cat_share_data_bundles | -0.0011 | 0.0005 |
| provider_share_providerid_1 | -0.0039 | 0.0030 |
| provider_share_providerid_5 | -0.0108 | 0.0032 |
| interval_cv | -0.0299 | 0.0048 |

Figures are in `reports/figures/`.
