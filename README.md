# NYC Taxi Trips: Big Data Analytics with Apache Spark

## Project overview

- **Problem:** Analyze demand, routes, fares, and trip durations across NYC TLC Yellow Taxi records, then estimate metered fare from trip and route attributes.
- **Dataset:** Official NYC Taxi & Limousine Commission (TLC) monthly Parquet trip records and taxi-zone lookup.
- **Big-data approach:** Spark reads real monthly Parquet files directly, processes every selected trip with DataFrame transformations, SQL, joins, aggregations, window functions, correlation, and Spark ML. The dashboard receives only compact aggregate exports and a bounded model-evaluation sample.
- **Dashboard:** Streamlit with date, borough, and hour filters.
- **Default scale:** January 2024 Yellow Taxi Parquet (about 50 MB compressed); add months to scale to tens of millions of trips.

## Architecture

```text
NYC TLC monthly Parquet + zone lookup CSV
                   │
                   ▼
       PySpark distributed ingestion
                   │
   quality filters + derived time features
       broadcast pickup/dropoff joins
                   │
       Spark SQL + grouped analytics
       window-ranked zones and routes
                   │
  sampled train/evaluation Spark ML models
                   │
 compact CSV/JSON artifacts ──► Streamlit
```

The sample month is a slice of a much larger, continuously published TLC
dataset. This project uses actual trip records, not synthetic row expansion.
Choose more source months to increase the real workload. On a cluster, provide
a Spark master and make the input paths accessible to the driver and workers.

## Repository structure

```text
.
├── app.py
├── data/raw/taxi/                  # Downloaded TLC files (git-ignored)
├── outputs/taxi/                   # Generated aggregates and model metrics (git-ignored)
├── PRESENTATION.md
├── README.md
├── REPORT.md
├── pyproject.toml
└── src/taxi_analytics/pipeline.py
```

## Dataset source, schema, and attribution

NYC TLC publishes monthly Yellow Taxi trip records as Parquet. The default
source is
[`yellow_tripdata_2024-01.parquet`](https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet).
The official trip-record landing page is
[NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page).
Zone IDs are enriched using TLC's
[taxi_zone_lookup.csv](https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv).
The pipeline records the actual inferred schema and source-row count at run
time because TLC's fields may change between data vintages.

The model target is `fare_amount` (USD). Useful source fields include pickup
and dropoff timestamps, passenger count, trip distance, rate code, vendor,
pickup/dropoff zone IDs, payment type, and fare. The pipeline computes trip
duration, hour, day of week, month, and weekend indicators. It excludes
`total_amount`, tips, tolls, and other fare components from model inputs to
avoid target leakage. Extreme/invalid durations, distances, fares, missing
timestamps, and missing zone IDs are filtered with explicit counts.

TLC's public data page provides data dictionaries and usage notes; review the
current TLC terms and any applicable notices before redistribution. This
repository downloads source files on demand and does not commit the data.
Attribute the source as **NYC Taxi & Limousine Commission (TLC), Yellow Taxi
Trip Records**.

## Team requirement

The course brief calls for a team of **5–7 students**. Team names were not
provided for this workspace. Add the actual roster to the team line in
[PRESENTATION.md](./PRESENTATION.md) before submitting.

## Requirements and installation

- Python 3.10+
- Java 8+ supported by the installed Spark release (Java 17 recommended)
- Spark 3.5.x (installed by project dependencies)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

## Run the Spark pipeline

Default: download/process the 2024-01 Yellow Taxi file and current zone lookup.

```powershell
python -m taxi_analytics.pipeline
```

Scale the real source workload by selecting additional months:

```powershell
python -m taxi_analytics.pipeline --months 2024-01 2024-02 2024-03
```

The monthly source files are downloaded under `data/raw/taxi/`. To use
pre-downloaded data or a different Spark master:

```powershell
python -m taxi_analytics.pipeline `
  --months 2024-01 2024-02 `
  --raw-dir .\data\raw\taxi `
  --output-dir .\outputs\taxi `
  --master "local[*]"
```

For cluster execution, use a configured master (for example
`--master spark://<host>:7077`) and stage the same Parquet/CSV input paths on
storage visible to the Spark driver and executors. Local mode uses all
available cores. Increase selected months to grow the distributed Spark input.

### Spark transformations and outputs

The pipeline:

1. Reads official monthly Parquet files using Spark; unions month schemas.
2. Measures row counts, nulls, invalid values, and schema; cleans domain
   outliers and derives trip-time features.
3. Broadcast-joins the TLC zone dimension twice (pickup and dropoff).
4. Uses DataFrame aggregations, Spark SQL, and window-ranked zone/route
   summaries; computes summary statistics and Pearson correlations.
5. Creates a deterministic 80/20 hash holdout and uses bounded samples for
   Spark ML training/evaluation to control runtime while all selected trips
   participate in preprocessing and aggregate analysis.
6. Compares Linear Regression with Random Forest Regression using RMSE, MAE,
   and R². Feature imputers, indexers, and encoders are fit only on training
   data.
7. Exports compact dashboard artifacts; it never collects the full raw trip
   dataset to the driver.

Generated files in `outputs/taxi/` include:

- `data_profile.json`, `key_findings.json`, `numeric_summary.csv`
- `daily_trips.csv`, `hourly_demand.csv`, `pickup_zones.csv`
- `popular_routes.csv`, `payment_types.csv`, `numeric_correlations.csv`
- `model_metrics.json`, `model_comparison.csv`, `model_predictions.csv`
- `feature_importance.csv`

## Run the dashboard

After the pipeline completes:

```powershell
python -m streamlit run app.py
```

Use the date, borough, and hour filters to explore demand. The dashboard
shows volume, fare, distance, and duration KPIs, busy zones/routes, payment
mix, correlations, feature importance, model comparison, and a sampled
actual-versus-predicted fare chart.

## Presentation and demo

Follow the 6-slide outline and run checklist in
[PRESENTATION.md](./PRESENTATION.md). Briefly run a single month first; then
show the multi-month command to demonstrate scaling the real input volume.

## Limitations

This is trip-level observational data, not a random sample of all travel.
Cleaning thresholds affect results; borough and route aggregates are not
causal. The fare model estimates a metered fare after the trip using trip
duration/distance and route, not a pre-trip price quote. It uses sampled
training and test partitions to manage laptop runtime; always inspect the
reported sample sizes and validate on a later month before operational use.
