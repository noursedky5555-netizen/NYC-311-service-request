# Project Report: Bank Marketing Subscription Analytics

## Problem

Analyze a Portuguese bank's direct-marketing campaign and estimate whether a
contacted client will subscribe to a term deposit (`y`). The intended modeling
use is pre-call prioritization, not post-call evaluation.

## Dataset

The project uses the 45,211-row Bank Marketing dataset from the UCI Machine
Learning Repository (ID 222; DOI 10.24432/C5K306). It includes customer,
campaign, and prior-campaign fields and a binary `yes`/`no` outcome. The
pipeline downloads the official CSV on first run. Source attribution and dataset details are in [README.md](./README.md).
The UCI API metadata does not expose a dataset-specific license field for this
entry, so confirm applicable reuse terms with UCI before redistribution.

## Methodology

The loader applies a fixed schema and interprets `NaN` as missing. QA measures
nulls, exact duplicate rows, invalid values, numeric summaries, and target
balance. Cleaning removes exact duplicates and rows with invalid age, day,
duration, campaign count, prior-contact count, or target. Missing categories
are represented as `unknown`. `pdays = -1` is preserved as the source's
"never previously contacted" sentinel and a separate prior-contact indicator
is derived. Numeric imputation is fitted inside each model using training data
only. `duration` is excluded because it is unavailable before the call and is
explicitly called out by UCI as a leakage risk.

Two Spark ML pipelines are compared: logistic regression and random forest.
An exact-row hash assigns a deterministic, approximately 80/20 train/test
split; removing exact duplicate source rows first prevents identical copies
from straddling the partitions. The selected model maximizes holdout ROC-AUC,
with F1 as tie-breaker. Precision, recall, F1, accuracy, and confusion counts
are also reported. This is a fixed holdout comparison, not a claim of
hyperparameter optimization or causal inference.

## Spark usage

PySpark performs typed CSV ingestion, data validation and filtering, group
aggregations, approximate numeric quantiles, Pearson correlation, Spark SQL
conversion summaries, window-based job ranking, and model fitting/evaluation.
Spark exports cleaned dashboard data and compact EDA/model artifacts for
Streamlit.

## Results

On the downloaded data, the cleaning step retained 45,211 rows; there were no
exact duplicate rows or invalid-domain rows. The target was imbalanced:
39,922 `no` and 5,289 `yes` examples (11.7% positive). The descriptive
highest-rate segments were `student` (28.7%, 938 records) and March (52.0%,
477 records); these are observational rates, not causal findings.

The deterministic holdout contained 36,076 training and 9,135 test rows. At
the default 0.5 classification threshold:

| Model | ROC-AUC | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Logistic regression | 0.760 | 0.897 | 0.759 | 0.147 | 0.247 |
| Random forest (selected) | 0.780 | 0.897 | 0.726 | 0.157 | 0.258 |

The selected random forest has modest positive-class recall at the default
threshold, despite a 0.780 ROC-AUC. The dashboard threshold control allows
precision/recall trade-offs to be inspected; it does not change the reported
full-holdout ROC-AUC. Re-running the pipeline regenerates these values from
the source data.

## Dashboard

The Streamlit app includes job and campaign-month filters, customer and
subscription KPIs, conversion charts, model comparison, feature analysis, and
a test-set confusion matrix recalculated for a selectable probability
threshold. ROC-AUC is shown for the complete holdout; precision, recall, and
F1 are recalculated for the selected segments.

## Conclusion and limitations

The project demonstrates a reproducible Spark analytics and classification
workflow with explicit leakage prevention and an interactive reporting layer.
The source is historical and observational, contains no stable client ID, may
have class imbalance, and supports neither causal claims nor production
deployment decisions without temporal validation, fairness review, calibration,
and monitoring.
