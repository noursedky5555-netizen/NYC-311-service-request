# Bank Marketing Analytics and Subscription Prediction

## Project at a glance

- **Problem:** Analyze direct-marketing campaigns by a Portuguese banking institution and estimate whether a client will subscribe to a term deposit.
- **Dataset:** UCI Machine Learning Repository, [Bank Marketing (ID 222)](https://archive.ics.uci.edu/dataset/222/bank+marketing).
- **Target:** `y` (`yes` / `no`).
- **Processing and models:** Local Apache Spark / PySpark.
- **Dashboard:** Streamlit.
- **Modeling decision:** `duration` is deliberately excluded. UCI notes that call duration is not known before the call and strongly affects the outcome, so using it for a pre-call model would create target leakage.

## Project layout

```text
.
├── app.py                         # Streamlit dashboard
├── data/
│   └── raw/                        # Downloaded source data (not committed)
├── outputs/                        # Generated Spark exports (not committed)
├── pyproject.toml
├── README.md
├── REPORT.md
├── PRESENTATION.md                 # Slide outline and live-demo checklist
└── src/
    └── bank_marketing/
        └── pipeline.py             # Loading, QA, Spark EDA, ML and exports
```

## Dataset, schema, and attribution

The official UCI CSV contains 45,211 examples and 17 columns: `age`, `job`,
`marital`, `education`, `default`, `balance`, `housing`, `loan`, `contact`,
`day_of_week` (the numeric day of month in the source), `month`, `duration`,
`campaign`, `pdays`, `previous`, `poutcome`, and target `y`. Inputs include
integer-valued measures and categorical strings; source missing values are
represented as `NaN`. `pdays = -1` represents no previous contact. UCI's page
documents each field and the research context.

The dataset is attributed to Sérgio Moro, Paulo Rita, and Paulo Cortez,
"A data-driven approach to predict the success of bank telemarketing,"
*Decision Support Systems* (2014), DOI
[10.1016/j.dss.2014.03.001](https://doi.org/10.1016/j.dss.2014.03.001).
The dataset DOI is [10.24432/C5K306](https://doi.org/10.24432/C5K306).
Review the current UCI dataset page for its applicable use and redistribution
terms. The pipeline downloads data directly from UCI; raw data and generated
outputs are excluded from version control.

The UCI dataset API metadata does not expose a dataset-specific license field
for this entry. Attribution is provided above; treat reuse and redistribution
terms as unspecified until confirmed on the [UCI dataset page](https://archive.ics.uci.edu/dataset/222/bank+marketing)
or with UCI. Do not assume that public availability alone grants redistribution
rights.

## Team requirement

The project brief requires a team of **5–7 students**. Team membership was not
provided for this workspace; add the actual members to the team line in
[PRESENTATION.md](./PRESENTATION.md) before submission. Do not submit the
placeholder as a roster.

## Requirements and setup

- Python 3.10+
- Java 8 or later supported by the selected PySpark release (Java 17 is recommended)
- The project Python dependencies listed in `pyproject.toml`

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

The first pipeline run downloads the official CSV to `data/raw/bank_marketing.csv`.
For an offline/manual setup, download the UCI CSV and specify its path with
`--input`.

## Run the Spark pipeline

From the project root (with the environment activated):

```powershell
python -m bank_marketing.pipeline
```

Optional paths:

```powershell
python -m bank_marketing.pipeline --input .\data\raw\bank_marketing.csv --output-dir .\outputs
```

The pipeline uses Spark DataFrame filters and aggregations, a Spark SQL query,
a window-based ranking, a deterministic 80/20 hash holdout, and Spark ML
pipelines for logistic regression and random forest. Median imputers and
categorical indexers are fitted on the training partition only. Exact duplicate
rows are removed to prevent copies of identical observations crossing the
holdout boundary. `unknown` categories are retained; missing numeric values
are imputed within the training pipeline. The `pdays = -1` convention is kept,
and an additional indicator flags prior contact.

Generated artifacts in `outputs/`:

- `data_profile.json`, `numeric_summary.csv`, `target_balance.csv`
- `numeric_distributions.csv`, `numeric_correlations.csv`
- `conversion_by_job.csv`, `conversion_by_month.csv`
- `dashboard_data.csv`, `insights.json`
- `model_comparison.csv`, `model_metrics.json`, `model_predictions.csv`
- `feature_importance.csv` (random-forest importance or absolute logistic coefficient)

## Run the dashboard

Run the Spark pipeline once, then launch the interactive dashboard:

```powershell
python -m streamlit run app.py
```

Using `python -m streamlit` avoids PowerShell `CommandNotFoundException` when
the Python `Scripts` directory containing the `streamlit` launcher is not on
`PATH`. Run this from the project root using the same Python environment where
the dependencies were installed.

The dashboard offers job/month filters, observed conversion KPIs and plots,
held-out model metrics, a configurable probability threshold, a confusion
matrix, model comparison, and feature analysis. Model precision, recall, and
F1 respond to the filters and threshold; displayed ROC-AUC is for the complete
holdout.

## Presentation / demo flow

1. State the campaign question and the UCI source/target.
2. Show the architecture: UCI CSV → typed Spark load and QA → Spark SQL/window
   summaries → leakage-aware Spark ML → CSV/JSON exports → Streamlit.
3. Explain nulls, duplicate handling, the `pdays = -1` convention, and why
   `duration` is excluded.
4. Run the pipeline and compare logistic regression with random forest using
   ROC-AUC, precision, recall, F1, and confusion counts.
5. Launch Streamlit, filter job/month segments, adjust the threshold, and
   review conversion patterns and feature analysis.
6. Close with the limitations: observational historical data, no client ID,
   class imbalance, and no causal interpretation of feature importance.
