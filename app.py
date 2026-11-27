"""Streamlit dashboard for the NYC TLC Yellow Taxi Spark analytics pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "taxi"
REQUIRED_FILES = [
    "daily_trips.csv",
    "hourly_demand.csv",
    "pickup_zones.csv",
    "popular_routes.csv",
    "payment_types.csv",
    "numeric_correlations.csv",
    "numeric_summary.csv",
    "data_profile.json",
    "key_findings.json",
    "model_comparison.csv",
    "model_metrics.json",
    "model_predictions.csv",
    "feature_importance.csv",
]

st.set_page_config(
    page_title="NYC Taxi Big Data Analytics", page_icon="🚕", layout="wide"
)
st.title("NYC Taxi Trips: Big Data Analytics")
st.caption(
    "Explore millions of official NYC TLC Yellow Taxi trips processed with "
    "PySpark. Analytics below are derived from the selected real trip-data months."
)

missing = [name for name in REQUIRED_FILES if not (OUTPUT_DIR / name).is_file()]
if missing:
    st.error(
        "Spark outputs are missing: "
        + ", ".join(missing)
        + ". Run `python -m taxi_analytics.pipeline` first."
    )
    st.stop()

daily = pd.read_csv(OUTPUT_DIR / "daily_trips.csv", parse_dates=["pickup_date"])
hourly = pd.read_csv(OUTPUT_DIR / "hourly_demand.csv")
zones = pd.read_csv(OUTPUT_DIR / "pickup_zones.csv")
routes = pd.read_csv(OUTPUT_DIR / "popular_routes.csv")
payments = pd.read_csv(OUTPUT_DIR / "payment_types.csv")
correlations = pd.read_csv(OUTPUT_DIR / "numeric_correlations.csv")
numeric_summary = pd.read_csv(OUTPUT_DIR / "numeric_summary.csv")
profile = json.loads((OUTPUT_DIR / "data_profile.json").read_text(encoding="utf-8"))
findings = json.loads((OUTPUT_DIR / "key_findings.json").read_text(encoding="utf-8"))
metrics = json.loads((OUTPUT_DIR / "model_metrics.json").read_text(encoding="utf-8"))
comparison = pd.read_csv(OUTPUT_DIR / "model_comparison.csv")
predictions = pd.read_csv(OUTPUT_DIR / "model_predictions.csv")
importance = pd.read_csv(OUTPUT_DIR / "feature_importance.csv")

date_min = daily["pickup_date"].min().date()
date_max = daily["pickup_date"].max().date()
borough_options = sorted(daily["pickup_borough"].dropna().unique().tolist())
with st.sidebar:
    st.header("Trip filters")
    date_range = st.date_input(
        "Pickup date range",
        value=(date_min, date_max),
        min_value=date_min,
        max_value=date_max,
    )
    boroughs = st.multiselect(
        "Pickup borough", borough_options, default=borough_options
    )
    selected_hours = st.slider("Pickup hour", min_value=0, max_value=23, value=(0, 23))

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = end_date = date_range
daily_filtered = daily[
    daily["pickup_date"].dt.date.between(start_date, end_date)
    & daily["pickup_borough"].isin(boroughs)
]
hourly_filtered = hourly[
    hourly["pickup_borough"].isin(boroughs)
    & hourly["pickup_hour"].between(selected_hours[0], selected_hours[1])
]
zone_filtered = zones[zones["pickup_borough"].isin(boroughs)]
route_filtered = routes[routes["pickup_borough"].isin(boroughs)]

trip_count = int(daily_filtered["trips"].sum())
fare_sum = float(daily_filtered["total_fare_amount"].sum())
distance_sum = float(daily_filtered["total_distance_miles"].sum())
duration_sum = float(daily_filtered["total_duration_minutes"].sum())
average_fare = fare_sum / trip_count if trip_count else 0
average_distance = distance_sum / trip_count if trip_count else 0
average_duration = duration_sum / trip_count if trip_count else 0

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Trips in selection", f"{trip_count:,}")
kpi2.metric("Mean metered fare", f"${average_fare:,.2f}")
kpi3.metric("Mean trip distance", f"{average_distance:.2f} mi")
kpi4.metric("Mean trip duration", f"{average_duration:.1f} min")
st.caption(
    f"Source rows: {profile['source_rows']:,} · Valid trips after filtering: "
    f"{profile['cleaned_rows']:,} · Months: {', '.join(metrics['months'])}"
)

st.subheader("Key findings")
finding1, finding2, finding3 = st.columns(3)
finding1.metric("All cleaned trips", f"{findings['trips']:,}")
finding2.metric("Overall median fare", f"${findings['median_fare']:.2f}")
finding3.metric(
    "Mean trip distance", f"{findings['average_distance_miles']:.2f} miles"
)
st.caption(findings["interpretation"])

left, right = st.columns(2)
with left:
    st.subheader("Daily trip demand")
    if daily_filtered.empty:
        st.info("No trips match the selected date and borough filters.")
    else:
        daily_chart = (
            daily_filtered.groupby("pickup_date", as_index=False)["trips"]
            .sum()
            .sort_values("pickup_date")
        )
        st.plotly_chart(
            px.line(
                daily_chart,
                x="pickup_date",
                y="trips",
                markers=True,
                labels={"pickup_date": "Pickup date", "trips": "Trips"},
            ),
            width="stretch",
        )
with right:
    st.subheader("Pickup demand by hour")
    if hourly_filtered.empty:
        st.info("No trips match the selected borough and hour filters.")
    else:
        hour_chart = hourly_filtered.groupby("pickup_hour", as_index=False)[
            "trips"
        ].sum()
        st.plotly_chart(
            px.bar(
                hour_chart,
                x="pickup_hour",
                y="trips",
                labels={"pickup_hour": "Hour of day", "trips": "Trips"},
            ),
            width="stretch",
        )

left, right = st.columns(2)
with left:
    st.subheader("Busiest pickup zones")
    st.plotly_chart(
        px.bar(
            zone_filtered.head(15).sort_values("pickup_trips"),
            x="pickup_trips",
            y="pickup_zone",
            color="pickup_borough",
            orientation="h",
            labels={"pickup_trips": "Trips", "pickup_zone": "TLC pickup zone"},
        ),
        width="stretch",
    )
with right:
    st.subheader("Most common taxi routes")
    route_display = route_filtered.head(12).copy()
    route_display["route"] = (
        route_display["pickup_zone"].astype(str)
        + " → "
        + route_display["dropoff_zone"].astype(str)
    )
    st.dataframe(
        route_display[
            ["route", "pickup_borough", "dropoff_borough", "trips", "average_fare"]
        ].rename(
            columns={
                "pickup_borough": "from",
                "dropoff_borough": "to",
                "average_fare": "mean fare ($)",
            }
        ),
        width="stretch",
        hide_index=True,
    )

st.subheader("Fare prediction model")
model_col1, model_col2 = st.columns([1, 1])
selected = metrics["selected_model_metrics"]
with model_col1:
    st.markdown(
        f"**Selected:** {metrics['selected_model']} · trained on "
        f"{metrics['training_sample_rows']:,} Spark-sampled trips; evaluated on "
        f"{metrics['evaluation_sample_rows']:,} held-out trips."
    )
    a, b, c = st.columns(3)
    a.metric("RMSE", f"${selected['rmse']:.2f}")
    b.metric("MAE", f"${selected['mae']:.2f}")
    c.metric("R²", f"{selected['r2']:.3f}")
    st.caption(
        "Fare prediction uses trip distance, duration, time and route. Other fare "
        "components (tips, tolls, total amount) are excluded to avoid leakage."
    )
    st.dataframe(comparison, width="stretch", hide_index=True)
with model_col2:
    st.plotly_chart(
        px.bar(
            importance.head(15).sort_values("importance"),
            x="importance",
            y="feature",
            orientation="h",
            labels={"importance": "Random-forest importance"},
        ),
        width="stretch",
    )

st.subheader("Prediction error sample")
st.plotly_chart(
    px.scatter(
        predictions.sample(min(8000, len(predictions)), random_state=42),
        x="fare_amount",
        y="prediction",
        color="trip_distance",
        hover_data=["trip_duration_minutes", "pickup_borough", "dropoff_borough"],
        labels={"fare_amount": "Actual fare ($)", "prediction": "Predicted fare ($)"},
    ).add_shape(
        type="line",
        x0=0,
        x1=float(predictions["fare_amount"].quantile(0.99)),
        y0=0,
        y1=float(predictions["fare_amount"].quantile(0.99)),
        line={"dash": "dash", "color": "gray"},
    ),
    width="stretch",
)

eda_left, eda_right = st.columns(2)
with eda_left:
    st.subheader("Payment type mix")
    payment_labels = {1: "Credit card", 2: "Cash", 3: "No charge", 4: "Dispute", 5: "Unknown", 6: "Voided"}
    payments["payment_label"] = payments["payment_type"].map(payment_labels).fillna(
        payments["payment_type"].astype(str)
    )
    st.plotly_chart(
        px.pie(payments, values="trips", names="payment_label", hole=0.4),
        width="stretch",
    )
with eda_right:
    st.subheader("Numeric feature correlations")
    matrix = correlations.pivot(
        index="feature", columns="other_feature", values="pearson_correlation"
    )
    st.plotly_chart(
        px.imshow(
            matrix,
            text_auto=".2f",
            zmin=-1,
            zmax=1,
            color_continuous_scale="RdBu",
        ),
        width="stretch",
    )

with st.expander("Data quality, schema, and modeling details"):
    st.write(
        f"Invalid trips removed: {profile['rows_removed']:,}. "
        f"Zone dimension was joined twice (pickup and dropoff) using Spark "
        f"broadcast joins. Month files: {', '.join(metrics['months'])}."
    )
    st.write(
        pd.DataFrame(
            [{"check": key, "rows": value} for key, value in profile["domain_check_counts"].items()]
        )
    )
    st.write(pd.DataFrame(profile["null_counts"].items(), columns=["column", "nulls"]))
    st.write("Inferred Parquet source schema", pd.DataFrame(profile["source_schema"]))
    st.write("Spark numeric summary", numeric_summary)

st.caption(
    "Data: NYC Taxi & Limousine Commission (TLC), Yellow Taxi Trip Records. "
    "See README for source links and reuse terms."
)
