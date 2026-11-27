# Project Report: NYC Taxi Big Data Analytics

## Problem

Analyze demand, pickup/dropoff patterns, fare distributions, and trip duration
in the NYC Taxi & Limousine Commission (TLC) Yellow Taxi records. As a
project-specific ML task, estimate the metered `fare_amount` from trip
distance, duration, time, vendor, rate code, and pickup/dropoff zones.

## Dataset

The pipeline downloads the official TLC monthly Yellow Taxi Parquet records
and TLC taxi-zone lookup. The default month is January 2024; users can select
multiple months and scale the real input to tens of millions of trips. The
Parquet source has a wide, vintage-dependent schema; the pipeline checks the
required fields and records the inferred source schema. No synthetic row
expansion is used. See [README.md](./README.md) for links, setup, source
attribution, and current-terms guidance.

## Methodology

Spark measures source nulls and invalid values, filters records with missing
timestamps/zones, nonpositive or extreme trip duration/distance/fare, and
derives pickup date/hour/day/month/weekend features. The TLC zone dimension is
broadcast-joined for pickup and dropoff labels. Duplicate trips are not blindly
removed because repeated field combinations can represent legitimate records.

The project creates grouped demand, zone, route, and payment summaries, plus
numeric summaries and Pearson correlations. The model uses a deterministic
80/20 source-row hash split; pipeline imputation and category encoding are fit
on the training partition only. Linear Regression and Random Forest Regression
are compared using RMSE, MAE, and R². `total_amount`, tips, tolls, and other
fare components are excluded as target leakage. Bounded samples control model
runtime; the full selected source volume is used for Spark quality checks and
aggregations.

## Spark usage

Spark reads Parquet natively, unions multiple months, filters and transforms
trip records, joins the TLC zone dimension twice with broadcast joins, and
executes SQL and grouped/window aggregations. Correlation analysis and model
training/evaluation use Spark ML. The dashboard consumes compact aggregates
and a bounded prediction sample instead of collecting the full input into
Pandas.

## Results

The January 2024 run processed **2,964,624 source trips**, retaining
**2,825,499** after cleaning. The selected Random Forest Regression model
achieved **$3.68 RMSE**, **$1.28 MAE**, and **0.954 R²** on a held-out sample
of 11,518 trips; its training sample contained 22,787 trips. Linear Regression
achieved $3.82 RMSE, $1.41 MAE, and 0.950 R² on the same evaluation sample.
These results use a deterministic target-independent trip-attribute hash
holdout and bounded samples for laptop memory/runtime. They are generated in
`outputs/taxi/model_metrics.json` and `model_comparison.csv`; selecting other
months will change the results.

## Dashboard

The Streamlit dashboard provides date, borough, and pickup-hour filters;
trip-volume, average-fare, distance, and duration KPIs; daily/hourly demand
charts; pickup-zone and route tables; payment mix; numeric correlations; fare
model comparison; feature importance; and actual-versus-predicted samples.

## Conclusion and limitations

The project demonstrates a real, scalable Spark workflow over published
multi-million-row Parquet inputs. Local Spark is the default, and users can
increase input volume by selecting more months or configure a Spark cluster
where shared input paths are available. Results are descriptive, sensitive to
data-quality filters, and not causal. Fare estimation is retrospective
trip-level estimation, not a customer-facing upfront quote; sampled-model
metrics need later-month validation before operational decisions.
