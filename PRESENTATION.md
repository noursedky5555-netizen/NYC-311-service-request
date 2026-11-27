# Presentation / Demo: NYC 311 Service Request Analytics

**Team:** `[Add 5–7 student names before submission]`
**Project:** NYC 311 complaint demand and resolution analytics
**Dataset:** NYC Open Data, 311 Service Requests from 2020 to Present

## Six-slide flow (5–7 minutes)

### 1. Problem

- How do request volume and resolution times differ across complaint types,
  agencies, boroughs, and time?
- Can request-time fields estimate whether a closed request resolves within
  seven days?
- Explain this is exploratory service analytics, not a city SLA audit.

### 2. Dataset and scale

- NYC Open Data Socrata dataset `erm2-nwe9`, published from 2020 onward.
- Default extraction: 794,394 records across Q1 2024; extend the months to
  process millions of real requests.
- Monthly date filters and 50K-row API pages are cached as CSV partitions.

### 3. Spark architecture

- API paging → Spark CSV ingestion → timestamp/coordinate QA → feature
  derivation → Spark aggregations/SQL/windows → chronological Spark ML →
  compact exports → Streamlit.
- Highlight null/domain checks, request-time feature engineering, and the
  full-record aggregate pipeline vs. bounded model samples.

### 4. EDA and data-quality findings

- Show source vs retained rows, duplicate keys, nulls, and coordinate checks.
- Explore complaint types, borough volume, request-hour patterns, ZIP
  hotspots, and closure times.
- Q1 2024 top categories included Illegal Parking (123K), Heat/Hot Water
  (106K), and Residential Noise (68K).
- Median closure time was about 12.7 hours vs a 317-hour mean, showing a
  long-tailed distribution.
- Note open requests are still included in volume, but not closed-time
  statistics.

### 5. ML and dashboard demo

- Target: closed within seven days (a project-selected threshold, not an
  official service commitment).
- Compare logistic regression and random forest with ROC-AUC, PR-AUC,
  precision, recall, and F1 on later-created requests.
- On the Q1 sample, logistic regression reached 0.954 ROC-AUC and 0.946
  recall; explain the roughly 81% positive-class base rate.
- Show date/borough/complaint/hour controls and the map/table/model outputs.
- Exclude closed date and resolution hours from predictors.

### 6. Limitations and next steps

- Data is observational and changes over time; extraction counts are
  snapshot-specific.
- Closure time has right-censoring from open cases; agency rankings are not
  causal.
- Validate on longer future periods and distributed Spark infrastructure
  before any operational use.

## Live demo

1. `python -m nyc311.pipeline` downloads/processes the default quarter.
2. Show per-month API partitions, Spark profile, aggregate outputs, and
   chronological model metrics.
3. Scale source volume with:
   `python -m nyc311.pipeline --start-month 2024-01 --end-month 2024-12`.
4. Launch `python -m streamlit run app.py`.
5. Filter borough/complaint/date/hour and explain findings and model
   limitations.
