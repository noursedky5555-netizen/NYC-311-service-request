# NYC 311 Service Request Analytics — PySpark Big Data Project

## Project idea

**Question:** How do NYC service-request volume and resolution patterns differ
by complaint type, borough, agency, channel, and time—and can we estimate
whether a closed request is resolved within seven days using only information
available when it was created?

This is a distinct public-sector dataset, not the NYC taxi data used in labs.
It uses the official 311 Service Requests from 2020 to Present dataset, which
contains tens of millions of records. The default reproducible run downloads
January–March 2024 (794,394 source records in the verified API snapshot).
The 2024 full-year API currently reports 3,457,597 requests; expanding the
date range processes millions more real records.

## Dataset, source and attribution

- **Dataset:** [NYC 311 Service Requests from 2020 to Present](https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2020-to-Present/erm2-nwe9)
- **Socrata API ID:** `erm2-nwe9`
- **API:** `https://data.cityofnewyork.us/resource/erm2-nwe9`
- **Owner/source:** NYC Open Data / City of New York.
- **Attribution:** “NYC Open Data, 311 Service Requests from 2020 to Present.”
- **License/terms:** Review the dataset page and NYC Open Data terms before
  redistribution or other reuse; public API availability is not a substitute
  for checking current terms.

The pipeline downloads the request key, created and closed timestamps, agency,
complaint and descriptor, location type, ZIP, borough, intake channel, and
coordinates. Exact schema and data-quality counts are written to
`outputs/nyc311/data_profile.json`. Fields and values can evolve as the city
updates the dataset.

## Architecture

```text
NYC Open Data Socrata API
    │ monthly time filters, 50k-row pages, cached CSV partitions
    ▼
Spark CSV/DataFrame ingestion ── null/schema/domain checks
    │
    ├── date/time feature engineering
    ├── borough / agency / complaint aggregates
    ├── Spark SQL and window-ranked complaint summaries
    └── chronological Spark ML train/test comparison
               │
        compact CSV/JSON exports
               ▼
        Streamlit dashboard
```

The project performs distributed Spark transforms and aggregation over all
selected requests. Only compact aggregates and bounded samples are collected
for dashboard/model outputs. It does not duplicate rows or claim that a
single-machine run is a cluster. Use the `--master` option and shared storage
to run across a configured Spark cluster.

## Repository layout

```text
.
├── app.py
├── data/raw/nyc311/                 # API page partitions (git-ignored)
├── outputs/nyc311/                  # Spark/dashboard artifacts (git-ignored)
├── PRESENTATION.md
├── README.md
├── REPORT.md
├── pyproject.toml
├── src/nyc311/pipeline.py
└── tests/test_pipeline_helpers.py
```

## Setup

- Python 3.10+
- Java 8+ supported by the selected PySpark version (Java 17 recommended)
- Internet access to NYC Open Data for first download

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

Optional: set a Socrata application token in `SOCRATA_APP_TOKEN` to improve
API rate limits. The public dataset is accessible without authentication.

## Run the Spark pipeline

Default run: fetch January–March 2024 into month-partitioned CSV pages, process
them with Spark, and compare classifiers.

```powershell
python -m nyc311.pipeline
```

Scale out to the full 2024 year (the API currently reports over 3.4 million
requests):

```powershell
python -m nyc311.pipeline --start-month 2024-01 --end-month 2024-12
```

Refresh a cached month to include current updates to request statuses:

```powershell
python -m nyc311.pipeline --start-month 2024-01 --end-month 2024-03 --refresh
```

Select a historical time range, output folder, or Spark master:

```powershell
python -m nyc311.pipeline `
  --start-month 2023-01 `
  --end-month 2023-12 `
  --raw-dir .\data\raw\nyc311 `
  --output-dir .\outputs\nyc311 `
  --master "local[*]"
```

For cluster mode, set `--master` to the configured Spark master and ensure
the downloaded CSV partition paths are visible to driver and executors.
Monthly downloads are cached using a row-count manifest. A month partition
whose expected count changes is downloaded again. API pagination uses a
stable unique-key order and 50,000-row pages. Use `--refresh` when updates to
existing open/closed records are needed even if the row count is unchanged.

### Spark processing and modeling

The pipeline:

1. Queries official monthly records from Socrata, using bounded pages rather
   than attempting a single huge API response.
2. Uses Spark to parse timestamps, profile nulls/duplicates/invalid
   coordinates, and derive day, hour, weekday, month, and resolution-time
   features.
3. Aggregates requests by day/borough, complaint, agency, ZIP, channel, and
   time. Uses Spark SQL and window functions for ranked categories.
4. Models the descriptive target **closed within seven days** among already
   closed requests. The target is a project-defined interval, not an official
   city SLA. Open records are excluded from model training to avoid treating
   right-censored cases as failures.
5. Uses only fields known at request creation; `closed_date` and measured
   resolution time are excluded. Categorical/numeric transforms are fit only
   on training data. The test period is chronologically later than training.
6. Compares Logistic Regression with Random Forest using ROC-AUC, PR-AUC,
   precision, recall, F1, accuracy, and confusion counts.

Model fitting samples 10% of the earlier partition (capped at 100,000 rows)
and 20% of the later partition (capped at 50,000 rows) to keep local Java
memory reasonable; all selected records still contribute to cleaning and
Spark aggregates. Report the actual sample sizes beside evaluation metrics.
For a larger cluster, raise these explicit caps in `train_models` deliberately
and monitor memory.

Generated artifacts in `outputs/nyc311/`:

- `data_profile.json`, `key_findings.json`
- `daily_requests.csv`, `complaint_types.csv`, `boroughs.csv`
- `hourly_patterns.csv`, `channels.csv`, `zip_hotspots.csv`
- `agency_complaints.csv`, `map_sample.csv`
- `model_comparison.csv`, `model_metrics.json`, `model_predictions.csv`
- `feature_importance.csv`

## Run the interactive dashboard

After the pipeline finishes:

```powershell
python -m streamlit run app.py
```

The dashboard offers date, borough, complaint, and hour filters; request and
closure KPIs; daily volume; complaint and borough response charts; time
heatmap; geospatial request sample; ZIP/agency tables; model metrics,
comparison, scores, and feature importance.

### Deploy to Streamlit Community Cloud

The repository includes the generated Q1 2024 dashboard artifacts under
`outputs/nyc311/`, so the hosted dashboard can start without downloading the
source dataset or launching Spark. In Streamlit Community Cloud, select
`app.py` as the entry point and the repository's `main` branch. The
`pyproject.toml` package mapping is needed because the Python package lives
under `src/nyc311`. To publish a refreshed data snapshot, rerun the Spark
pipeline and commit the updated `outputs/nyc311/` CSV and JSON files.

## Testing

```powershell
python -m unittest discover -s tests -v
```

## Report and presentation

See [REPORT.md](./REPORT.md) for problem, dataset, methodology, Spark,
measured results, dashboard, and limitations. See [PRESENTATION.md](./PRESENTATION.md)
for a concise slide/demo flow. Add the actual **5–7 student team roster**
before submitting.

## Limitations

This is observational service-request data, not a random sample or causal
study. API snapshots can change; keep the run-time row counts and extraction
period with results. Resolution-time statistics use closed records, so open
cases are right-censored. Seven days is a chosen analysis target, not a
verified agency service-level agreement. Do not interpret the classifier as
an official agency performance score or use it to deny service.
