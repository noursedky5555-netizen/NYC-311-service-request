"""Scalable PySpark analytics and fare prediction for NYC TLC taxi trips."""

from __future__ import annotations

import argparse
import json
import logging
import re
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import Imputer, OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml.stat import Correlation
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "taxi"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "taxi"
TLC_BASE_URL = "https://d37ci6vzurychx.cloudfront.net"
DEFAULT_MONTHS = ["2024-01"]
RANDOM_SEED = 42

RAW_COLUMNS = [
    "VendorID",
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "passenger_count",
    "trip_distance",
    "RatecodeID",
    "store_and_fwd_flag",
    "PULocationID",
    "DOLocationID",
    "payment_type",
    "fare_amount",
]
NUMERIC_FEATURES = [
    "trip_distance",
    "trip_duration_minutes",
    "passenger_count",
    "pickup_hour",
    "pickup_day_of_week",
    "pickup_month",
]
CATEGORICAL_FEATURES = [
    "VendorID",
    "RatecodeID",
    "PULocationID",
    "DOLocationID",
    "pickup_borough",
    "dropoff_borough",
]
PICKUP_URL = f"{TLC_BASE_URL}/misc/taxi_zone_lookup.csv"
MONTH_PATTERN = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")


def parse_months(values: list[str]) -> list[str]:
    """Validate and de-duplicate YYYY-MM input while retaining user order."""
    months = list(dict.fromkeys(values))
    invalid = [month for month in months if not MONTH_PATTERN.fullmatch(month)]
    if not months or invalid:
        raise ValueError(
            "Provide one or more valid months in YYYY-MM format; "
            f"invalid values: {invalid}"
        )
    return months


def _download(url: str, destination: Path) -> None:
    """Download a TLC file once, writing atomically so partial files are not reused."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size:
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        url, headers={"User-Agent": "nyc-taxi-big-data/1.0"}
    )
    LOGGER.info("Downloading %s", url)
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            with temporary.open("wb") as target:
                while chunk := response.read(1024 * 1024):
                    target.write(chunk)
        if temporary.stat().st_size == 0:
            raise RuntimeError(f"Download from {url} returned an empty file.")
        temporary.replace(destination)
    except (OSError, TimeoutError) as exc:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download {url}: {exc}. Download the TLC file manually "
            f"to {destination} and rerun."
        ) from exc


def download_inputs(months: list[str], raw_dir: Path) -> tuple[list[Path], Path]:
    """Fetch official Yellow Taxi monthly Parquet files and zone lookup."""
    paths = []
    for month in months:
        filename = f"yellow_tripdata_{month}.parquet"
        path = raw_dir / filename
        _download(f"{TLC_BASE_URL}/trip-data/{filename}", path)
        paths.append(path)
    zones_path = raw_dir / "taxi_zone_lookup.csv"
    _download(PICKUP_URL, zones_path)
    return paths, zones_path


def load_source(spark: SparkSession, paths: list[Path]) -> DataFrame:
    """Read one or more monthly Parquet files using Spark's native Parquet reader."""
    frames = [spark.read.parquet(str(path)) for path in paths]
    required = set(RAW_COLUMNS)
    for path, frame in zip(paths, frames):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(
                f"{path.name} is missing required TLC columns: {', '.join(missing)}"
            )
    combined = frames[0]
    for frame in frames[1:]:
        combined = combined.unionByName(frame, allowMissingColumns=True)
    return combined


def load_zones(spark: SparkSession, path: Path) -> DataFrame:
    """Read the TLC taxi-zone dimension table."""
    zones = (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .csv(str(path))
    )
    expected = {"LocationID", "Borough", "Zone"}
    missing = expected - set(zones.columns)
    if missing:
        raise ValueError(f"TLC zone lookup is missing columns: {sorted(missing)}")
    return zones.select(
        F.col("LocationID").cast("int").alias("LocationID"),
        F.coalesce(F.col("Borough"), F.lit("Unknown")).alias("Borough"),
        F.coalesce(F.col("Zone"), F.lit("Unknown")).alias("Zone"),
    ).dropDuplicates(["LocationID"])


def profile_source(raw: DataFrame) -> dict[str, Any]:
    """Collect source row count, null counts, types, and trip-domain violations."""
    expressions = [F.count(F.lit(1)).alias("source_rows")]
    expressions.extend(
        F.sum(F.col(column).isNull().cast("long")).alias(f"null_{column}")
        for column in RAW_COLUMNS
    )
    invalid_tests = {
        "missing_pickup_or_dropoff_time": F.col("tpep_pickup_datetime").isNull()
        | F.col("tpep_dropoff_datetime").isNull(),
        "nonpositive_distance": F.col("trip_distance") <= 0,
        "distance_over_100_miles": F.col("trip_distance") > 100,
        "nonpositive_fare": F.col("fare_amount") <= 0,
        "fare_over_500_dollars": F.col("fare_amount") > 500,
        "missing_zone_id": F.col("PULocationID").isNull()
        | F.col("DOLocationID").isNull(),
    }
    expressions.extend(
        F.sum(condition.cast("long")).alias(f"invalid_{name}")
        for name, condition in invalid_tests.items()
    )
    row = raw.agg(*expressions).first()
    return {
        "source_rows": int(row["source_rows"]),
        "source_schema": [
            {"column": field.name, "type": field.dataType.simpleString()}
            for field in raw.schema.fields
        ],
        "null_counts": {
            column: int(row[f"null_{column}"] or 0) for column in RAW_COLUMNS
        },
        "domain_check_counts": {
            name: int(row[f"invalid_{name}"] or 0) for name in invalid_tests
        },
    }


def clean_and_enrich(raw: DataFrame, zones: DataFrame) -> DataFrame:
    """Filter invalid trips, derive temporal features, and join both zone dimensions."""
    source = raw.withColumn(
        "trip_duration_minutes",
        (
            F.unix_timestamp("tpep_dropoff_datetime")
            - F.unix_timestamp("tpep_pickup_datetime")
        )
        / 60.0,
    )
    valid = source.filter(
        F.col("tpep_pickup_datetime").isNotNull()
        & F.col("tpep_dropoff_datetime").isNotNull()
        & (F.col("tpep_dropoff_datetime") > F.col("tpep_pickup_datetime"))
        & F.col("trip_duration_minutes").between(1, 240)
        & F.col("trip_distance").between(0.1, 100)
        & F.col("fare_amount").between(2.5, 500)
        & F.col("PULocationID").isNotNull()
        & F.col("DOLocationID").isNotNull()
        & (
            F.col("passenger_count").isNull()
            | F.col("passenger_count").between(1, 8)
        )
    )
    pickup_zones = F.broadcast(
        zones.select(
            F.col("LocationID").alias("PULocationID"),
            F.col("Borough").alias("pickup_borough"),
            F.col("Zone").alias("pickup_zone"),
        )
    )
    dropoff_zones = F.broadcast(
        zones.select(
            F.col("LocationID").alias("DOLocationID"),
            F.col("Borough").alias("dropoff_borough"),
            F.col("Zone").alias("dropoff_zone"),
        )
    )
    enriched = valid.join(pickup_zones, "PULocationID", "left").join(
        dropoff_zones, "DOLocationID", "left"
    )
    enriched = enriched.fillna(
        {"pickup_borough": "Unknown", "dropoff_borough": "Unknown"}
    )
    return (
        enriched.withColumn("pickup_date", F.to_date("tpep_pickup_datetime"))
        .withColumn("pickup_hour", F.hour("tpep_pickup_datetime"))
        .withColumn("pickup_day_of_week", F.dayofweek("tpep_pickup_datetime"))
        .withColumn("pickup_month", F.month("tpep_pickup_datetime"))
        .withColumn(
            "pickup_weekend",
            F.when(F.dayofweek("tpep_pickup_datetime").isin(1, 7), 1).otherwise(0),
        )
    )


def _export(
    frame: DataFrame | pd.DataFrame, path: Path, limit: int | None = None
) -> None:
    """Export a bounded Spark result for dashboard consumption."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(frame, DataFrame):
        bounded = frame.limit(limit) if limit else frame
        frame = bounded.toPandas()
    frame.to_csv(path, index=False)


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def export_analytics(
    clean: DataFrame, profile: dict[str, Any], output_dir: Path
) -> None:
    """Generate distributed aggregations, SQL outputs, and dimension summaries."""
    output_dir.mkdir(parents=True, exist_ok=True)
    clean.createOrReplaceTempView("nyc_taxi_clean")
    daily = clean.groupBy("pickup_date", "pickup_borough").agg(
        F.count("*").alias("trips"),
        F.sum("fare_amount").alias("total_fare_amount"),
        F.avg("fare_amount").alias("average_fare"),
        F.sum("trip_distance").alias("total_distance_miles"),
        F.avg("trip_distance").alias("average_distance_miles"),
        F.sum("trip_duration_minutes").alias("total_duration_minutes"),
        F.avg("trip_duration_minutes").alias("average_duration_minutes"),
        F.avg("passenger_count").alias("average_passengers"),
        F.sum(F.when(F.col("payment_type") == 1, F.col("tip_amount")).otherwise(0))
        .alias("total_card_tips"),
    )
    _export(daily.orderBy("pickup_date", "pickup_borough"), output_dir / "daily_trips.csv")

    hourly = clean.sparkSession.sql(
        """
        SELECT pickup_hour, pickup_borough,
               COUNT(*) AS trips,
               AVG(fare_amount) AS average_fare,
               AVG(trip_duration_minutes) AS average_duration_minutes
        FROM nyc_taxi_clean
        GROUP BY pickup_hour, pickup_borough
        ORDER BY pickup_hour, pickup_borough
        """
    )
    _export(hourly, output_dir / "hourly_demand.csv")

    zones_summary = clean.groupBy(
        "PULocationID", "pickup_zone", "pickup_borough"
    ).agg(
        F.count("*").alias("pickup_trips"),
        F.avg("fare_amount").alias("average_fare"),
        F.avg("trip_distance").alias("average_distance_miles"),
    )
    zones_summary = zones_summary.withColumn(
        "pickup_rank",
        F.row_number().over(
            Window.orderBy(F.col("pickup_trips").desc(), F.col("PULocationID"))
        ),
    )
    _export(zones_summary.orderBy("pickup_rank"), output_dir / "pickup_zones.csv")

    routes = clean.groupBy(
        "PULocationID",
        "pickup_zone",
        "pickup_borough",
        "DOLocationID",
        "dropoff_zone",
        "dropoff_borough",
    ).agg(
        F.count("*").alias("trips"),
        F.avg("fare_amount").alias("average_fare"),
        F.avg("trip_duration_minutes").alias("average_duration_minutes"),
    )
    route_window = Window.orderBy(F.col("trips").desc())
    _export(
        routes.withColumn("route_rank", F.row_number().over(route_window))
        .orderBy("route_rank")
        .limit(100),
        output_dir / "popular_routes.csv",
    )
    payment = clean.groupBy("payment_type").agg(
        F.count("*").alias("trips"),
        F.avg("fare_amount").alias("average_fare"),
    )
    _export(payment.orderBy(F.col("trips").desc()), output_dir / "payment_types.csv")

    numeric = [
        "trip_distance",
        "trip_duration_minutes",
        "passenger_count",
        "fare_amount",
    ]
    summary = clean.select(*numeric).summary(
        "count", "mean", "stddev", "min", "25%", "50%", "75%", "max"
    )
    _export(summary, output_dir / "numeric_summary.csv")
    correlation_input = VectorAssembler(
        inputCols=numeric, outputCol="numeric_vector", handleInvalid="skip"
    ).transform(clean)
    matrix = Correlation.corr(correlation_input, "numeric_vector").first()[0].toArray()
    _export(
        pd.DataFrame(
            [
                {
                    "feature": left,
                    "other_feature": right,
                    "pearson_correlation": float(matrix[i, j]),
                }
                for i, left in enumerate(numeric)
                for j, right in enumerate(numeric)
            ]
        ),
        output_dir / "numeric_correlations.csv",
    )

    overall = clean.agg(
        F.count("*").alias("cleaned_rows"),
        F.avg("fare_amount").alias("average_fare"),
        F.expr("percentile_approx(fare_amount, 0.5, 10000)").alias("median_fare"),
        F.avg("trip_distance").alias("average_distance"),
        F.avg("trip_duration_minutes").alias("average_duration"),
    ).first()
    profile["cleaned_rows"] = int(overall["cleaned_rows"])
    profile["rows_removed"] = profile["source_rows"] - profile["cleaned_rows"]
    _write_json(profile, output_dir / "data_profile.json")
    _write_json(
        {
            "trips": int(overall["cleaned_rows"]),
            "average_fare": float(overall["average_fare"]),
            "median_fare": float(overall["median_fare"]),
            "average_distance_miles": float(overall["average_distance"]),
            "average_duration_minutes": float(overall["average_duration"]),
            "interpretation": "Descriptive trip statistics; not causal estimates.",
        },
        output_dir / "key_findings.json",
    )


def _build_model(estimator: Any) -> Pipeline:
    indexers = [
        StringIndexer(
            inputCol=column, outputCol=f"{column}_index", handleInvalid="keep"
        )
        for column in CATEGORICAL_FEATURES
    ]
    encoder = OneHotEncoder(
        inputCols=[f"{column}_index" for column in CATEGORICAL_FEATURES],
        outputCols=[f"{column}_encoded" for column in CATEGORICAL_FEATURES],
        handleInvalid="keep",
    )
    imputed = [f"{column}_imputed" for column in NUMERIC_FEATURES]
    imputer = Imputer(
        inputCols=NUMERIC_FEATURES, outputCols=imputed, strategy="median"
    )
    assembler = VectorAssembler(
        inputCols=imputed + [f"{column}_encoded" for column in CATEGORICAL_FEATURES],
        outputCol="features",
        handleInvalid="keep",
    )
    return Pipeline(stages=[*indexers, encoder, imputer, assembler, estimator])


def _regression_metrics(predictions: DataFrame) -> dict[str, float]:
    metrics = {}
    for name in ("rmse", "mae", "r2"):
        evaluator = RegressionEvaluator(
            labelCol="fare_amount", predictionCol="prediction", metricName=name
        )
        metrics[name] = float(evaluator.evaluate(predictions))
    return metrics


def _feature_names(metadata: dict[str, Any], size: int) -> list[str]:
    names = [f"feature_{index}" for index in range(size)]
    for attributes in metadata.get("ml_attr", {}).get("attrs", {}).values():
        for attribute in attributes:
            index = attribute.get("idx")
            if index is not None and index < size:
                names[index] = attribute.get("name", names[index])
    return names


def train_models(clean: DataFrame, output_dir: Path) -> dict[str, Any]:
    """Compare two regressors on a hash holdout with bounded training/evaluation samples."""
    hashed = F.pmod(
        F.xxhash64(
            "tpep_pickup_datetime",
            "tpep_dropoff_datetime",
            "PULocationID",
            "DOLocationID",
            "trip_distance",
        ),
        F.lit(10),
    )
    training = clean.filter(hashed < 8).sample(
        withReplacement=False, fraction=0.01, seed=RANDOM_SEED
    ).cache()
    test = clean.filter(hashed >= 8).sample(
        withReplacement=False, fraction=0.02, seed=RANDOM_SEED
    ).cache()
    training_count, test_count = training.count(), test.count()
    if training_count < 1000 or test_count < 1000:
        raise ValueError(
            f"Model partitions are too small: train={training_count}, test={test_count}"
        )

    estimators = [
        LinearRegression(
            featuresCol="features",
            labelCol="fare_amount",
            maxIter=40,
            regParam=0.05,
        ),
        RandomForestRegressor(
            featuresCol="features",
            labelCol="fare_amount",
            numTrees=20,
            maxDepth=8,
            seed=RANDOM_SEED,
        ),
    ]
    fitted = {}
    comparison = []
    for estimator in estimators:
        name = estimator.__class__.__name__
        LOGGER.info("Training %s on %s sampled trips", name, training_count)
        model = _build_model(estimator).fit(training)
        predictions = model.transform(test).cache()
        metrics = _regression_metrics(predictions)
        comparison.append({"model": name, **metrics})
        fitted[name] = (model, predictions)

    results = pd.DataFrame(comparison).sort_values(["rmse", "mae"])
    results.to_csv(output_dir / "model_comparison.csv", index=False)
    selected = str(results.iloc[0]["model"])
    model, predictions = fitted[selected]
    selected_metrics = next(item for item in comparison if item["model"] == selected)

    export_predictions = predictions.select(
        "fare_amount",
        "prediction",
        "trip_distance",
        "trip_duration_minutes",
        "pickup_borough",
        "dropoff_borough",
    ).limit(25_000)
    _export(export_predictions, output_dir / "model_predictions.csv")
    fitted_vector = model.transform(training.limit(1))
    metadata = fitted_vector.schema["features"].metadata
    final_estimator = model.stages[-1]
    if not hasattr(final_estimator, "featureImportances"):
        raise TypeError("Selected fare model does not expose feature importances.")
    importance = final_estimator.featureImportances.toArray()
    pd.DataFrame(
        {
            "feature": _feature_names(metadata, len(importance)),
            "importance": importance,
        }
    ).sort_values("importance", ascending=False).head(30).to_csv(
        output_dir / "feature_importance.csv", index=False
    )
    for _, prediction_df in fitted.values():
        prediction_df.unpersist()
    training.unpersist()
    test.unpersist()
    return {
        "selected_model": selected,
        "selected_model_metrics": selected_metrics,
        "comparison": comparison,
        "training_sample_rows": training_count,
        "evaluation_sample_rows": test_count,
        "holdout_design": (
            "deterministic 80/20 trip-attribute hash excluding the target; "
            "bounded samples used for model runtime"
        ),
        "target": "fare_amount in USD",
        "leakage_controls": (
            "total_amount, tip_amount, tolls_amount, and other fare components "
            "are excluded from model features"
        ),
    }


def run_pipeline(
    months: list[str],
    raw_dir: Path,
    output_dir: Path,
    master: str,
) -> None:
    paths, zones_path = download_inputs(months, raw_dir)
    spark = (
        SparkSession.builder.appName("NYCTaxiBigData")
        .master(master)
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    try:
        raw = load_source(spark, paths)
        profile = profile_source(raw)
        zones = load_zones(spark, zones_path)
        clean = clean_and_enrich(raw, zones)
        LOGGER.info("Read %s source trips across %s month(s)", profile["source_rows"], len(months))
        export_analytics(clean, profile, output_dir)
        model_results = train_models(clean, output_dir)
        model_results["months"] = months
        model_results["source_rows"] = profile["source_rows"]
        model_results["cleaned_rows"] = profile["cleaned_rows"]
        _write_json(model_results, output_dir / "model_metrics.json")
        LOGGER.info(
            "Pipeline complete: %s source rows, %s cleaned, selected %s",
            profile["source_rows"],
            profile["cleaned_rows"],
            model_results["selected_model"],
        )
    finally:
        spark.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--months",
        nargs="+",
        default=DEFAULT_MONTHS,
        help="TLC data months in YYYY-MM format; use multiple months to scale out.",
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--master",
        default="local[*]",
        help="Spark master URL; use local[*] or a configured cluster master.",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    run_pipeline(parse_months(args.months), args.raw_dir, args.output_dir, args.master)


if __name__ == "__main__":
    main()
