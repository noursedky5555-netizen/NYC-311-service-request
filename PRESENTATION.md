# Presentation and Demo: Bank Marketing Analytics

**Team:** `[Add 5–7 student names before submission]`
**Project:** Predicting term-deposit subscriptions from bank marketing data
**Dataset:** [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)

## Suggested 6-slide flow (5–7 minutes)

### 1. Business problem and dataset

- Can historical campaign data help prioritize pre-call outreach?
- Target: `y` (`yes` / `no`); 45,211 observations and 17 columns.
- Source: UCI Machine Learning Repository, dataset DOI 10.24432/C5K306.
- State that the data are observational, not experimental.

### 2. Architecture and Spark pipeline

- UCI CSV → typed PySpark DataFrame → data-quality checks and cleaning.
- Spark DataFrame aggregations, Spark SQL, window ranking, quantiles, and
  Pearson correlation produce EDA artifacts.
- Training-only preprocessing → logistic regression and random forest →
  holdout evaluation → CSV/JSON artifacts → Streamlit.
- Point to `src/bank_marketing/pipeline.py` and generated `outputs/`.

### 3. Data quality and EDA findings

- Source contains 45,211 rows; after cleaning, 45,211 remain.
- No exact duplicates or invalid-domain rows in this source copy.
- 5,289 positive targets (11.7%); missing categorical values are retained as
  an `unknown` category.
- Highest observed segment rates: students, 28.7% (938 records); March, 52.0%
  (477 records). These are descriptive rates, not causal effects.

### 4. Modeling choices and results

- Exclude `duration`: it is only known after the call and would leak outcome
  information into a pre-call prediction.
- Deterministic hash holdout: 36,076 training and 9,135 test examples.
- Compare logistic regression (ROC-AUC 0.760, F1 0.247) and random forest
  (ROC-AUC 0.780, F1 0.258); random forest selected by ROC-AUC.
- At threshold 0.5, selected-model precision is 0.726 and recall is 0.157.
  Discuss the imbalanced target and threshold trade-off; accuracy alone would
  be misleading.

### 5. Interactive dashboard

- Filter by job and campaign month.
- Review customer/subscription KPIs, segment-rate charts, and numeric
  distribution/correlation visuals.
- Adjust the probability threshold and inspect filtered precision, recall,
  F1, and confusion matrix.
- Compare the models and inspect top feature importance.

### 6. Limitations and conclusion

- Historical data, no stable client identifier, class imbalance, and no
  temporal validation.
- Feature importance and segment rates are not causal explanations.
- A production decision would require temporal/fairness validation,
  calibration, and monitoring.

## Live demo checklist

1. From the repository root, run `python -m bank_marketing.pipeline`.
2. Show Spark's generated profile, SQL/window aggregations, model comparison,
   and prediction/EDA exports in `outputs/`.
3. Run `python -m streamlit run app.py` and open the local URL printed by
   Streamlit.
4. Change job/month filters and the prediction threshold; explain that
   filtered confusion metrics change while full-holdout ROC-AUC does not.
5. Close with the limitations above and show the setup steps in `README.md`.
