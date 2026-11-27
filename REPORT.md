# Project Report: NYC 311 Service Request Analytics

## Problem

This project analyzes NYC 311 service request volume and closure patterns by
complaint type, agency, borough, channel, ZIP, and creation time. It also
compares classifiers for whether a **closed** request was resolved within
seven days, using only request-time information.

## Dataset

The source is NYC Open Data's 311 Service Requests from 2020 to Present
(Socrata dataset `erm2-nwe9`). The January–March 2024 extraction contains
**794,394 source requests** (287,253 in January, 240,474 in February, and
266,667 in March). The API is queried in date partitions and 50,000-row pages.
See [README.md](./README.md) for schema, source attribution, setup, and
current reuse-term guidance.

## Methodology

Spark measures schema, nulls, duplicate keys, invalid timestamps, and
coordinate bounds. It derives resolution hours and calendar fields, preserves
open requests for descriptive counts, and excludes unresolved cases from
supervised training because their outcomes are right-censored. Daily,
complaint, agency, borough, channel, and ZIP aggregations reveal demand and
response-time variation. Spark SQL and window-ranked aggregates demonstrate
distributed analytical processing.

The target is closed within seven days; this is a project-defined comparison
threshold, not an official NYC SLA. A chronological split separates earlier
training data from later evaluation records. Models use only request-time
fields; closure timestamp and resolution duration are excluded. Numeric
imputation and categorical indexing/encoding are fit on training data only.
Logistic Regression and Random Forest are compared using ROC-AUC, PR-AUC,
precision, recall, F1, accuracy, and confusion counts. Model input is sampled
for local memory limits, while the selected source data are processed for
Spark analytics in full.

## Spark usage

PySpark reads month-partitioned CSV API pages, unions the selected periods,
casts timestamps, filters and derives features, performs group aggregates,
Spark SQL, window ranking, and model training/evaluation. Only summarized
outputs and bounded samples are exported for the dashboard.

## Results

Spark found no duplicate request keys. It retained **794,012** records after
removing **382** records with closed timestamps earlier than creation; 784,757
retained requests were closed and 9,255 were still open in the downloaded
snapshot. The most common complaint categories were Illegal Parking (123,083),
Heat/Hot Water (106,446), and Noise - Residential (68,498). The overall
median closure time was about **12.7 hours**, compared with a mean of about
**317 hours**, indicating a long-tailed distribution. About **81.0%** of
closed requests were resolved within the project-defined seven-day interval.
These are descriptive statistics, not causal rankings or official SLAs.

The models were trained on 62,951 sampled earlier-period records and evaluated
on 30,985 later-created requests (test window starting March 13, 2024). At the
default 0.5 threshold:

| Model | ROC-AUC | PR-AUC | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression (selected) | 0.954 | 0.989 | 0.893 | 0.923 | 0.946 | 0.935 |
| Random Forest | 0.938 | 0.985 | 0.831 | 0.827 | 1.000 | 0.905 |

The target is prevalent (about 81% in the test sample); compare with a
majority-class accuracy baseline of roughly 0.81 and consider precision/recall,
not accuracy alone. The very high classification performance should not be
interpreted as causal or production-ready. Outputs are reproducible from the
run-time API snapshot and saved under `outputs/nyc311/`.

## Dashboard

Streamlit shows date/borough/complaint/hour filters, volume and closure KPIs,
daily trends, top complaint categories, borough resolution patterns, hourly
heatmap, geocoded sample, ZIP and agency summaries, and the model comparison
and feature results.

## Conclusion and limitations

NYC 311 is a genuinely large public dataset suitable for Spark time-partitioned
ingestion and distributed analytics. Records are observational and may change
after extraction. Open cases are censored from resolution-time analysis, and
seven days is not an official SLA. Model outputs are exploratory and must not
be used as service eligibility decisions or causal agency rankings.
