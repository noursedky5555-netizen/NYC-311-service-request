"""Interactive Streamlit dashboard for the generated Bank Marketing artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

st.set_page_config(page_title="Bank Marketing Analytics", page_icon="📊", layout="wide")
st.title("Bank Marketing: Term Deposit Analytics")
st.caption(
    "Explore the UCI Portuguese bank campaign data and evaluate a pre-call "
    "subscription model. The post-call duration field is excluded from prediction."
)

required = [
    "dashboard_data.csv",
    "numeric_correlations.csv",
    "numeric_distributions.csv",
    "model_comparison.csv",
    "model_predictions.csv",
    "feature_importance.csv",
    "data_profile.json",
    "model_metrics.json",
    "insights.json",
]
missing = [name for name in required if not (OUTPUT_DIR / name).is_file()]
if missing:
    st.error(
        "Pipeline outputs are missing: "
        + ", ".join(missing)
        + ". Run `python -m bank_marketing.pipeline` first."
    )
    st.stop()

data = pd.read_csv(OUTPUT_DIR / "dashboard_data.csv")
comparison = pd.read_csv(OUTPUT_DIR / "model_comparison.csv")
predictions = pd.read_csv(OUTPUT_DIR / "model_predictions.csv")
importance = pd.read_csv(OUTPUT_DIR / "feature_importance.csv")
correlations = pd.read_csv(OUTPUT_DIR / "numeric_correlations.csv")
distributions = pd.read_csv(OUTPUT_DIR / "numeric_distributions.csv")
profile = json.loads((OUTPUT_DIR / "data_profile.json").read_text(encoding="utf-8"))
model_metrics = json.loads(
    (OUTPUT_DIR / "model_metrics.json").read_text(encoding="utf-8")
)
insights = json.loads((OUTPUT_DIR / "insights.json").read_text(encoding="utf-8"))

with st.sidebar:
    st.header("Explore segments")
    jobs = sorted(data["job"].dropna().unique().tolist())
    months = sorted(data["month"].dropna().unique().tolist())
    selected_jobs = st.multiselect("Job", jobs, default=jobs)
    selected_months = st.multiselect("Campaign month", months, default=months)
    threshold = st.slider(
        "Positive prediction threshold", min_value=0.05, max_value=0.95, value=0.50, step=0.05
    )

filtered = data[data["job"].isin(selected_jobs) & data["month"].isin(selected_months)]
filtered_predictions = predictions[
    predictions["job"].isin(selected_jobs) & predictions["month"].isin(selected_months)
].copy()
filtered_predictions["predicted"] = (
    filtered_predictions["subscription_probability"] >= threshold
).astype(int)

total = len(filtered)
subscriptions = int((filtered["y"] == "yes").sum())
conversion_rate = subscriptions / total if total else 0.0
average_age = float(filtered["age"].mean()) if total else 0.0
average_campaign_contacts = float(filtered["campaign"].mean()) if total else 0.0

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Customers in selection", f"{total:,}")
kpi2.metric("Observed subscriptions", f"{subscriptions:,}")
kpi3.metric("Observed subscription rate", f"{conversion_rate:.1%}")
kpi4.metric("Average campaign contacts", f"{average_campaign_contacts:.2f}")

st.subheader("Key findings")
finding_col1, finding_col2, finding_col3 = st.columns(3)
finding_col1.metric(
    "Overall subscription rate",
    f"{insights['overall_subscription_rate']:.1%}",
)
finding_col2.metric(
    "Highest-rate job segment",
    insights["highest_job_subscription_rate"]["job"],
    f"{insights['highest_job_subscription_rate']['subscription_rate']:.1%} · "
    f"{insights['highest_job_subscription_rate']['customers']:,} customers",
)
finding_col3.metric(
    "Highest-rate campaign month",
    insights["highest_month_subscription_rate"]["month"],
    f"{insights['highest_month_subscription_rate']['subscription_rate']:.1%} · "
    f"{insights['highest_month_subscription_rate']['customers']:,} customers",
)
st.caption(insights["interpretation"])

chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    st.subheader("Subscription rate by job")
    if total:
        job_view = (
            filtered.assign(subscribed=filtered["y"].eq("yes").astype(int))
            .groupby("job", as_index=False)
            .agg(customers=("y", "size"), subscription_rate=("subscribed", "mean"))
            .sort_values("subscription_rate", ascending=False)
        )
        st.plotly_chart(
            px.bar(
                job_view,
                x="job",
                y="subscription_rate",
                color="customers",
                hover_data=["customers"],
                labels={"subscription_rate": "Subscription rate"},
            ),
            width="stretch",
        )
    else:
        st.info("No records match the selected filters.")

with chart_col2:
    st.subheader("Subscription rate by campaign month")
    if total:
        month_view = (
            filtered.assign(subscribed=filtered["y"].eq("yes").astype(int))
            .groupby("month", as_index=False)
            .agg(customers=("y", "size"), subscription_rate=("subscribed", "mean"))
        )
        st.plotly_chart(
            px.bar(
                month_view,
                x="month",
                y="subscription_rate",
                color="customers",
                hover_data=["customers"],
                labels={"subscription_rate": "Subscription rate"},
            ),
            width="stretch",
        )
    else:
        st.info("No records match the selected filters.")

st.subheader("Model evaluation on the held-out test set")
st.caption(
    f"Selected model: {model_metrics['selected_model']} · "
    f"threshold is adjustable for the confusion-matrix metrics below."
)
if not filtered_predictions.empty:
    actual = filtered_predictions["label"].astype(int)
    predicted = filtered_predictions["predicted"].astype(int)
    tp = int(((actual == 1) & (predicted == 1)).sum())
    fp = int(((actual == 0) & (predicted == 1)).sum())
    tn = int(((actual == 0) & (predicted == 0)).sum())
    fn = int(((actual == 1) & (predicted == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ROC-AUC (full holdout)", f"{model_metrics['selected_model_metrics']['roc_auc']:.3f}")
    m2.metric("Filtered precision", f"{precision:.3f}")
    m3.metric("Filtered recall", f"{recall:.3f}")
    m4.metric("Filtered F1", f"{f1:.3f}")
    cm_col, importance_col = st.columns(2)
    with cm_col:
        st.markdown("**Filtered confusion matrix**")
        st.dataframe(
            pd.DataFrame(
                [[tn, fp], [fn, tp]],
                index=["Actual no", "Actual yes"],
                columns=["Predicted no", "Predicted yes"],
            ),
            width="stretch",
        )
    with importance_col:
        st.markdown("**Top model features**")
        st.plotly_chart(
            px.bar(
                importance.head(15).sort_values("importance"),
                x="importance",
                y="feature",
                orientation="h",
                labels={"importance": "Importance / absolute coefficient"},
            ),
            width="stretch",
        )
else:
    st.info("No held-out predictions match the selected filters.")

st.subheader("Model comparison")
st.dataframe(comparison, width="stretch", hide_index=True)

st.subheader("Numeric patterns")
pattern_col1, pattern_col2 = st.columns(2)
with pattern_col1:
    selected_numeric = st.selectbox(
        "Distribution to inspect",
        ["age", "balance", "campaign", "duration", "pdays", "previous"],
    )
    if total:
        st.plotly_chart(
            px.histogram(
                filtered,
                x=selected_numeric,
                nbins=30,
                title=f"{selected_numeric} distribution",
            ),
            width="stretch",
        )
    st.caption(
        "Approximate quartiles and range across the cleaned data (Spark "
        "approxQuantile, 1% relative error)."
    )
    st.dataframe(distributions, width="stretch", hide_index=True)
with pattern_col2:
    correlation_pivot = correlations.pivot(
        index="feature", columns="other_feature", values="pearson_correlation"
    )
    st.plotly_chart(
        px.imshow(
            correlation_pivot,
            text_auto=".2f",
            zmin=-1,
            zmax=1,
            color_continuous_scale="RdBu",
            title="Pearson correlation (numeric fields)",
        ),
        width="stretch",
    )

with st.expander("Data quality and methodology"):
    st.write(
        f"Rows before cleaning: {profile['rows_before_cleaning']:,}; "
        f"rows after cleaning: {profile.get('rows_after_cleaning', 0):,}; "
        f"exact duplicates found: {profile['exact_duplicate_rows']:,}."
    )
    st.write(
        "Missing values by source column",
        pd.DataFrame(
            profile["missing_values_by_column"].items(),
            columns=["column", "missing_values"],
        ),
    )
    st.write(
        "Numeric model imputers are fitted within the training pipeline only. "
        "Categorical nulls are retained as an explicit `unknown` category. "
        "The original `pdays = -1` means no prior contact; a separate "
        "`previously_contacted` feature preserves that meaning."
    )

st.caption(
    "Source: UCI Machine Learning Repository, Bank Marketing "
    "(Moro, Rita & Cortez, DOI: 10.24432/C5K306)."
)
