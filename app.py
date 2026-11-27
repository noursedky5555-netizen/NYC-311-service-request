"""Interactive Streamlit dashboard for NYC 311 service request analytics."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "nyc311"
REQUIRED_FILES = [
    "daily_requests.csv",
    "hourly_patterns.csv",
    "channels.csv",
    "zip_hotspots.csv",
    "agency_complaints.csv",
    "map_sample.csv",
    "model_comparison.csv",
    "model_predictions.csv",
    "feature_importance.csv",
    "data_profile.json",
    "key_findings.json",
    "model_metrics.json",
]

st.set_page_config(
    page_title="NYC 311 Service Analytics", page_icon="🏙️", layout="wide"
)
st.title("NYC 311: Service Request & Response Analytics")
st.caption(
    "Explore hundreds of thousands of official NYC Open Data requests with "
    "PySpark. Find complaint hotspots, compare agency response patterns, and "
    "estimate whether a closed request is resolved within seven days."
)

missing = [name for name in REQUIRED_FILES if not (OUTPUT_DIR / name).is_file()]
if missing:
    st.error(
        "Spark outputs are missing: "
        + ", ".join(missing)
        + ". Run `python -m nyc311.pipeline` first."
    )
    st.stop()

daily = pd.read_csv(OUTPUT_DIR / "daily_requests.csv", parse_dates=["created_day"])
hourly = pd.read_csv(OUTPUT_DIR / "hourly_patterns.csv")
channels = pd.read_csv(OUTPUT_DIR / "channels.csv")
zip_hotspots = pd.read_csv(
    OUTPUT_DIR / "zip_hotspots.csv", parse_dates=["created_day"]
)
agency_complaints = pd.read_csv(OUTPUT_DIR / "agency_complaints.csv")
map_sample = pd.read_csv(OUTPUT_DIR / "map_sample.csv", parse_dates=["created_day"])
comparison = pd.read_csv(OUTPUT_DIR / "model_comparison.csv")
predictions = pd.read_csv(OUTPUT_DIR / "model_predictions.csv", parse_dates=["created_ts"])
importance = pd.read_csv(OUTPUT_DIR / "feature_importance.csv")
profile = json.loads((OUTPUT_DIR / "data_profile.json").read_text(encoding="utf-8"))
findings = json.loads((OUTPUT_DIR / "key_findings.json").read_text(encoding="utf-8"))
metrics = json.loads((OUTPUT_DIR / "model_metrics.json").read_text(encoding="utf-8"))

date_min = daily["created_day"].min().date()
date_max = daily["created_day"].max().date()
borough_options = sorted(daily["borough"].dropna().unique().tolist())
complaint_options = sorted(daily["complaint_type"].dropna().unique().tolist())
with st.sidebar:
    st.header("Request filters")
    selected_dates = st.date_input(
        "Created date range",
        value=(date_min, date_max),
        min_value=date_min,
        max_value=date_max,
    )
    selected_boroughs = st.multiselect(
        "Borough", borough_options, default=borough_options
    )
    selected_complaints = st.multiselect(
        "Complaint type", complaint_options, default=complaint_options
    )
    selected_hours = st.slider(
        "Created hour", min_value=0, max_value=23, value=(0, 23)
    )

if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
    start_date, end_date = selected_dates
elif isinstance(selected_dates, tuple) and len(selected_dates) == 1:
    start_date = end_date = selected_dates[0]
elif isinstance(selected_dates, tuple):
    start_date, end_date = date_min, date_max
else:
    start_date = end_date = selected_dates
daily_view = daily[
    daily["created_day"].dt.date.between(start_date, end_date)
    & daily["borough"].isin(selected_boroughs)
    & daily["complaint_type"].isin(selected_complaints)
]
borough_view = daily_view.groupby("borough", as_index=False).agg(
    requests=("requests", "sum"),
    closed_requests=("closed_requests", "sum"),
    total_resolution_hours=("total_resolution_hours", "sum"),
    resolved_within_7d_requests=("resolved_within_7d_requests", "sum"),
)
borough_view["average_resolution_hours"] = (
    borough_view["total_resolution_hours"]
    / borough_view["closed_requests"].replace(0, pd.NA)
)
borough_view["resolved_within_7d_rate"] = (
    borough_view["resolved_within_7d_requests"]
    / borough_view["closed_requests"].replace(0, pd.NA)
)
hourly_view = hourly[
    hourly["created_hour"].between(selected_hours[0], selected_hours[1])
    & hourly["borough"].isin(selected_boroughs)
    & hourly["complaint_type"].isin(selected_complaints)
]
zip_daily = zip_hotspots[
    zip_hotspots["created_day"].between(
        pd.Timestamp(start_date), pd.Timestamp(end_date)
    )
    & zip_hotspots["borough"].isin(selected_boroughs)
    & zip_hotspots["complaint_type"].isin(selected_complaints)
]
zip_view = (
    zip_daily.groupby(["incident_zip", "borough"], as_index=False)
    .agg(requests=("requests", "sum"))
    .sort_values("requests", ascending=False)
)
route_view = agency_complaints[
    agency_complaints["complaint_type"].isin(selected_complaints)
]
map_view = map_sample[
    map_sample["created_day"].dt.date.between(start_date, end_date)
    & map_sample["borough"].isin(selected_boroughs)
    & map_sample["complaint_type"].isin(selected_complaints)
]

requests = int(daily_view["requests"].sum())
closed = int(daily_view["closed_requests"].sum())
resolved_within_7d = int(daily_view["resolved_within_7d_requests"].sum())
total_resolution_hours = float(daily_view["total_resolution_hours"].sum())
resolved_rate = resolved_within_7d / closed if closed else 0
mean_resolution_hours = total_resolution_hours / closed if closed else 0
closed_pct = closed / requests if requests else 0

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Requests in selection", f"{requests:,}")
kpi2.metric("Closed share", f"{closed_pct:.1%}")
kpi3.metric("Mean closure time", f"{mean_resolution_hours:.1f} hours")
kpi4.metric("7-day resolution rate", f"{resolved_rate:.1%}")
st.caption(
    f"Source records: {profile['source_rows']:,} · Quality-filtered requests: "
    f"{profile['cleaned_rows']:,} · Months: {', '.join(metrics['months'])}"
)

st.subheader("Key findings")
find1, find2, find3 = st.columns(3)
find1.metric("Overall requests", f"{findings['requests']:,}")
find2.metric("Still open in selected snapshot", f"{findings['open_requests']:,}")
find3.metric(
    "7-day closure share",
    f"{findings['resolved_within_7d_rate']:.1%}",
    help="Calculated among closed requests; open requests are excluded.",
)
st.caption(findings["interpretation"])

left, right = st.columns(2)
with left:
    st.subheader("Daily request volume")
    if daily_view.empty:
        st.info("No records match the selected date and borough filters.")
    else:
        daily_chart = (
            daily_view.groupby("created_day", as_index=False)["requests"]
            .sum()
            .sort_values("created_day")
        )
        st.plotly_chart(
            px.line(
                daily_chart,
                x="created_day",
                y="requests",
                labels={"created_day": "Created date", "requests": "Requests"},
            ),
            width="stretch",
        )
with right:
    st.subheader("Top complaint categories")
    if daily_view.empty:
        st.info("No complaint categories match the selection.")
    else:
        complaint_chart = daily_view.groupby("complaint_type", as_index=False).agg(
            requests=("requests", "sum"),
            closed_requests=("closed_requests", "sum"),
            resolved_within_7d_requests=("resolved_within_7d_requests", "sum"),
        )
        complaint_chart["resolved_within_7d_rate"] = (
            complaint_chart["resolved_within_7d_requests"]
            / complaint_chart["closed_requests"].replace(0, pd.NA)
        )
        top_complaints = complaint_chart.nlargest(15, "requests").sort_values("requests")
        st.plotly_chart(
            px.bar(
                top_complaints,
                x="requests",
                y="complaint_type",
                color="resolved_within_7d_rate",
                orientation="h",
                labels={
                    "requests": "Requests",
                    "complaint_type": "Complaint",
                    "resolved_within_7d_rate": "Resolved ≤7 days",
                },
                color_continuous_scale="Viridis",
            ),
            width="stretch",
        )

left, right = st.columns(2)
with left:
    st.subheader("Resolution patterns by borough")
    if borough_view.empty:
        st.info("No borough data match the selected filters.")
    else:
        st.plotly_chart(
            px.bar(
                borough_view.sort_values("average_resolution_hours"),
                x="borough",
                y="average_resolution_hours",
                color="requests",
                hover_data=["resolved_within_7d_rate"],
                labels={"average_resolution_hours": "Mean resolution (hours)"},
            ),
            width="stretch",
        )
with right:
    st.subheader("Requests by creation hour and weekday")
    if hourly_view.empty:
        st.info("No hourly data match the selected filters.")
    else:
        hour_pivot = (
            hourly_view.groupby(
                ["created_day_of_week", "created_hour"], as_index=False
            )["requests"]
            .sum()
            .pivot(
                index="created_day_of_week",
                columns="created_hour",
                values="requests",
            )
            .fillna(0)
        )
        st.plotly_chart(
            px.imshow(
                hour_pivot,
                aspect="auto",
                labels={
                    "x": "Hour",
                    "y": "Day of week (1=Sunday)",
                    "color": "Requests",
                },
                color_continuous_scale="Blues",
            ),
            width="stretch",
        )

left, right = st.columns(2)
with left:
    st.subheader("Complaint locations")
    if map_view.empty:
        st.info("No geocoded requests in this filter selection.")
    else:
        st.plotly_chart(
            px.scatter_map(
                map_view.sample(min(8000, len(map_view)), random_state=42),
                lat="latitude_value",
                lon="longitude_value",
                color="borough",
                hover_name="complaint_type",
                zoom=9,
                height=480,
                map_style="carto-positron",
            ),
            width="stretch",
        )
with right:
    st.subheader("Highest-volume ZIP codes")
    st.dataframe(
        zip_view.head(20),
        width="stretch",
        hide_index=True,
    )

st.subheader("Agency / complaint response table")
st.dataframe(
    route_view.sort_values("requests", ascending=False).head(30),
    width="stretch",
    hide_index=True,
)

st.subheader("Intake channel mix")
st.plotly_chart(
    px.pie(
        channels,
        values="requests",
        names="open_data_channel_type",
        hole=0.4,
    ),
    width="stretch",
)

st.subheader("Seven-day resolution model")
metric = metrics["selected_model_metrics"]
model_col, feature_col = st.columns(2)
with model_col:
    st.markdown(
        f"**Selected model:** {metrics['selected_model']} · trained on "
        f"{metrics['train_rows']:,} records and evaluated on "
        f"{metrics['test_rows']:,} later-created records."
    )
    a, b, c, d = st.columns(4)
    a.metric("ROC-AUC", f"{metric['roc_auc']:.3f}")
    b.metric("PR-AUC", f"{metric['pr_auc']:.3f}")
    c.metric("Precision", f"{metric['precision']:.3f}")
    d.metric("Recall", f"{metric['recall']:.3f}")
    st.dataframe(comparison, width="stretch", hide_index=True)
    st.caption(
        "Model applies to closed cases in the selected historical period. "
        "Open requests are excluded from the label to avoid treating them as "
        "failed/late outcomes."
    )
with feature_col:
    st.plotly_chart(
        px.bar(
            importance.head(15).sort_values("importance"),
            x="importance",
            y="feature",
            orientation="h",
            labels={"importance": "Feature importance / |coefficient|"},
        ),
        width="stretch",
    )

st.subheader("Model score sample")
if predictions.empty:
    st.info("No prediction rows exported.")
else:
    st.plotly_chart(
        px.histogram(
            predictions,
            x="resolved_within_7d_probability",
            color="label",
            nbins=30,
            barmode="overlay",
            labels={"label": "Resolved within 7 days"},
        ),
        width="stretch",
    )

with st.expander("Data quality and methodology"):
    st.write(
        f"Invalid rows removed: {profile['rows_removed']:,}; duplicate request "
        f"keys: {profile['duplicate_unique_keys']:,}. Two-class model uses "
        f"closed requests only and a chronological holdout."
    )
    st.write("Domain check counts")
    st.dataframe(
        pd.DataFrame(
            [
                {"check": name, "records": count}
                for name, count in profile["domain_check_counts"].items()
            ]
        ),
        width="stretch",
        hide_index=True,
    )
    st.write("Source null counts")
    st.dataframe(
        pd.DataFrame(profile["null_counts"].items(), columns=["field", "nulls"]),
        width="stretch",
        hide_index=True,
    )
    st.write("Source schema")
    st.dataframe(pd.DataFrame(profile["source_schema"]), width="stretch", hide_index=True)

st.caption(
    "Source: NYC Open Data, 311 Service Requests from 2020 to Present "
    "(Socrata dataset erm2-nwe9). Exploratory analysis, not a city performance audit."
)
