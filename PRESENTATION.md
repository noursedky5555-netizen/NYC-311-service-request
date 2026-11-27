# Presentation / Demo: NYC Taxi Big Data Analytics

**Team:** `[Add 5–7 student names before submission]`
**Project:** NYC TLC Yellow Taxi Big Data Analytics
**Source:** NYC Taxi & Limousine Commission monthly Parquet trip records

## Suggested 6-slide presentation (5–7 minutes)

### 1. Problem and scale

- Analyze taxi demand, fares, durations, routes, and zones.
- Predict metered fare from trip attributes as a project-specific regression
  task.
- Official TLC Parquet records are published monthly; January 2024 is the
  default example. Selecting more months scales the real data to tens of
  millions of trips.

### 2. Dataset and architecture

- TLC Yellow Taxi Parquet plus the TLC taxi-zone lookup dimension.
- PySpark Parquet ingestion → domain cleaning/time features → two broadcast
  zone joins → SQL and distributed aggregations → Spark ML → compact exports
  → Streamlit.
- Emphasize that the source is real trip data; no artificial row replication.

### 3. Spark big-data operations

- Native Parquet read and multi-month union.
- DataFrame filters, time feature derivation, aggregation and correlation.
- Broadcast joins to pickup/dropoff zones.
- Spark SQL hourly demand, window ranking for pickup zones/routes.
- Compare `--months 2024-01` against a multi-month run; discuss row counts,
  runtime, and partition behavior.

### 4. EDA and modeling

- Show data-quality/null counts and trips retained.
- Explain hour-of-day, borough, zone, route, fare, and distance patterns.
- Compare Linear Regression and Random Forest Regression using RMSE, MAE, R².
- Exclude total amount, tips, tolls, and other fare components from features.
- State that the model estimates fare after the trip; it is not an upfront
  fare quote. Report actual generated scores rather than assumed metrics.

### 5. Interactive dashboard

- Filter by date, pickup borough, and hour.
- Show trip KPIs, daily/hourly demand, top pickup zones/routes, payment mix,
  correlation, and fare-model diagnostics.

### 6. Limitations and takeaways

- Public trip data are observational, not causal.
- Filters and fare estimation bounds affect results.
- Model uses bounded samples for laptop runtime; validate on later months.
- Demonstrate scale-up on a configured cluster with shared storage for
  production-size workloads.

## Live demo steps

1. Run `python -m taxi_analytics.pipeline` for the default real-month download.
2. Inspect `outputs/taxi/data_profile.json` and Spark aggregate/model artifacts.
3. Run a larger real workload:
   `python -m taxi_analytics.pipeline --months 2024-01 2024-02 2024-03`.
4. Launch `python -m streamlit run app.py`.
5. Change date/borough/hour controls and explain the relationship between
   demand, routes, fares, and model errors.
6. Close with the limitations and next step: validate by later-month temporal
   holdout, then run on a distributed Spark cluster.
