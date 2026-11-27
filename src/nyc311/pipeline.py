"""Download Socrata NYC 311 records and run scalable Spark analytics and ML."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import Imputer, OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructField, StructType

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "nyc311"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "nyc311"
API_BASE = "https://data.cityofnewyork.us/resource/erm2-nwe9"
PAGE_SIZE = 50_000
DEFAULT_START_MONTH = "2024-01"
DEFAULT_END_MONTH = "2024-03"
MONTH_PATTERN = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")
RANDOM_SEED = 42

SOURCE_COLUMNS = [
    "unique_key",
    "created_date",
    "closed_date",
    "agency",
    "agency_name",
    "complaint_type",
    "descriptor",
    "location_type",
    "incident_zip",
    "borough",
    "open_data_channel_type",
    "latitude",
    "longitude",
]
SOURCE_SCHEMA = StructType(
    [StructField(column, StringType(), True) for column in SOURCE_COLUMNS]
)
NUMERIC_FEATURES = ["created_hour", "created_day_of_week", "created_month"]
CATEGORICAL_FEATURES = [
    "agency",
    "complaint_type",
    "borough",
    "open_data_channel_type",
    "location_type",
]
def parse_month(value: str) -> tuple[int, int]:
    if not MONTH_PATTERN.fullmatch(value):
        raise ValueError(f"Month must use YYYY-MM format: {value!r}")
    year, month = (int(part) for part in value.split("-"))
    return year, month


def month_range(start_month: str, end_month: str) -> list[str]:
    start_year, start_value = parse_month(start_month)
    end_year, end_value = parse_month(end_month)
    if (start_year, start_value) > (end_year, end_value):
        raise ValueError("start-month must be earlier than or equal to end-month.")
    months = []
    year, month = start_year, start_value
    while (year, month) <= (end_year, end_value):
        months.append(f"{year:04d}-{month:02d}")
        month += 1
        if month == 13:
            year, month = year + 1, 1
    return months


def _api_url(params: dict[str, str | int], format: str = "json") -> str:
    return f"{API_BASE}.{format}?{urllib.parse.urlencode(params)}"


def _fetch(url: str, app_token: str | None = None) -> bytes:
    headers = {"User-Agent": "nyc311-big-data-course-project/1.0"}
    if app_token:
        headers["X-App-Token"] = app_token
    request = urllib.request.Request(url, headers=headers)
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code not in {429, 500, 502, 503, 504} or attempt == 4:
                raise RuntimeError(
                    f"NYC Open Data returned HTTP {exc.code} ({url}): {exc}. "
                    "Check query parameters or set SOCRATA_APP_TOKEN."
                ) from exc
            retry_after = exc.headers.get("Retry-After")
            try:
                delay = min(float(retry_after), 60) if retry_after else 2**attempt
            except ValueError:
                delay = 2**attempt
        except (OSError, TimeoutError, urllib.error.URLError) as exc:
            if attempt == 4:
                raise RuntimeError(
                    f"NYC Open Data request failed ({url}): {exc}. "
                    "Check network access or set SOCRATA_APP_TOKEN."
                ) from exc
            delay = 2**attempt
        LOGGER.warning("API request failed; retrying in %s seconds.", delay)
        time.sleep(delay)
    raise RuntimeError(f"NYC Open Data request failed after retries: {url}")


def _month_where(month: str) -> str:
    year, month_number = parse_month(month)
    start = f"{year:04d}-{month_number:02d}-01T00:00:00"
    if month_number == 12:
        end = f"{year + 1:04d}-01-01T00:00:00"
    else:
        end = f"{year:04d}-{month_number + 1:02d}-01T00:00:00"
    return f"created_date >= '{start}' AND created_date < '{end}'"


def _expected_month_rows(month: str, app_token: str | None) -> int:
    payload = _fetch(
        _api_url(
            {
                "$select": "count(*) as request_count",
                "$where": _month_where(month),
            }
        ),
        app_token,
    )
    result = json.loads(payload)
    if not result or "request_count" not in result[0]:
        raise ValueError(f"NYC Open Data returned an invalid count response for {month}.")
    return int(result[0]["request_count"])


def _download_month(
    month: str, raw_dir: Path, app_token: str | None, refresh: bool
) -> tuple[list[Path], int]:
    """Fetch one monthly time partition as independently readable CSV pages."""
    expected = _expected_month_rows(month, app_token)
    month_dir = raw_dir / month
    manifest_path = month_dir / "_manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    existing_pages = sorted(month_dir.glob("part-*.csv"))
    if not refresh and manifest.get("expected_rows") == expected and existing_pages:
        LOGGER.info("Using cached %s partition (%s rows)", month, expected)
        return existing_pages, expected

    month_dir.mkdir(parents=True, exist_ok=True)
    for path in existing_pages:
        path.unlink()
    offset = 0
    paths = []
    while offset < expected:
        page_path = month_dir / f"part-{offset // PAGE_SIZE:05d}.csv"
        params: dict[str, str | int] = {
            "$select": ",".join(SOURCE_COLUMNS),
            "$where": _month_where(month),
            "$order": "unique_key",
            "$limit": PAGE_SIZE,
            "$offset": offset,
        }
        content = _fetch(_api_url(params, format="csv"), app_token)
        if not content.strip():
            raise RuntimeError(
                f"NYC Open Data returned an empty page at offset {offset} "
                f"for {month}; expected {expected} rows."
            )
        page_path.write_bytes(content)
        with page_path.open("r", encoding="utf-8-sig", newline="") as page:
            page_rows = max(sum(1 for _ in csv.reader(page)) - 1, 0)
        if page_rows == 0:
            raise RuntimeError(f"No records parsed from {page_path}.")
        paths.append(page_path)
        offset += page_rows
        LOGGER.info("Downloaded %s: %s / %s requests", month, min(offset, expected), expected)
    if offset != expected:
        raise RuntimeError(
            f"Downloaded {offset} rows for {month}; API count reported {expected}."
        )
    manifest_path.write_text(
        json.dumps({"month": month, "expected_rows": expected, "page_size": PAGE_SIZE}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return paths, expected


def download_months(
    months: list[str], raw_dir: Path, app_token: str | None, refresh: bool = False
) -> tuple[list[Path], dict[str, int]]:
    files: list[Path] = []
    counts = {}
    for month in months:
        month_files, row_count = _download_month(month, raw_dir, app_token, refresh)
        files.extend(month_files)
        counts[month] = row_count
    return files, counts


def load_source(spark: SparkSession, files: list[Path]) -> DataFrame:
    if not files:
        raise ValueError("No NYC 311 source files were downloaded.")
    return (
        spark.read.option("header", True)
        .option("mode", "FAILFAST")
        .option("nullValue", "")
        .schema(SOURCE_SCHEMA)
        .csv([str(path) for path in files])
    )


def profile_source(raw: DataFrame) -> dict[str, Any]:
    conditions = {
        "missing_unique_key": F.col("unique_key").isNull(),
        "missing_created_date": F.col("created_date").isNull(),
        "unparseable_created_date": F.to_timestamp("created_date").isNull(),
        "closed_before_created": F.to_timestamp("closed_date")
        < F.to_timestamp("created_date"),
        "invalid_latitude": F.col("latitude").isNotNull()
        & ~F.col("latitude").cast("double").between(40.4, 41.0),
        "invalid_longitude": F.col("longitude").isNotNull()
        & ~F.col("longitude").cast("double").between(-74.3, -73.6),
    }
    expressions = [F.count("*").alias("source_rows")]
    expressions.extend(
        F.sum(F.col(column).isNull().cast("long")).alias(f"null_{column}")
        for column in SOURCE_COLUMNS
    )
    expressions.extend(
        F.sum(condition.cast("long")).alias(f"invalid_{name}")
        for name, condition in conditions.items()
    )
    row = raw.agg(*expressions).first()
    unique_count = raw.select("unique_key").distinct().count()
    source_rows = int(row["source_rows"])
    return {
        "source_rows": source_rows,
        "duplicate_unique_keys": source_rows - unique_count,
        "source_schema": [
            {"column": field.name, "type": field.dataType.simpleString()}
            for field in raw.schema.fields
        ],
        "null_counts": {
            column: int(row[f"null_{column}"] or 0) for column in SOURCE_COLUMNS
        },
        "domain_check_counts": {
            name: int(row[f"invalid_{name}"] or 0) for name in conditions
        },
    }


def clean_and_enrich(raw: DataFrame) -> DataFrame:
    parsed = (
        raw.withColumn("created_ts", F.to_timestamp("created_date"))
        .withColumn("closed_ts", F.to_timestamp("closed_date"))
        .withColumn("latitude_value", F.col("latitude").cast("double"))
        .withColumn("longitude_value", F.col("longitude").cast("double"))
    )
    valid = parsed.filter(
        F.col("unique_key").isNotNull()
        & F.col("created_ts").isNotNull()
        & (
            F.col("closed_ts").isNull()
            | (F.col("closed_ts") >= F.col("created_ts"))
        )
        & (
            F.col("latitude_value").isNull()
            | F.col("latitude_value").between(40.4, 41.0)
        )
        & (
            F.col("longitude_value").isNull()
            | F.col("longitude_value").between(-74.3, -73.6)
        )
    )
    resolved = valid.withColumn(
        "resolution_hours",
        (
            F.col("closed_ts").cast("long") - F.col("created_ts").cast("long")
        )
        / 3600.0,
    ).withColumn(
        "resolved_within_7d",
        F.when(
            F.col("resolution_hours").isNotNull(),
            (F.col("resolution_hours") <= 168).cast("double"),
        ),
    )
    enriched = (
        resolved.withColumn("created_day", F.to_date("created_ts"))
        .withColumn("created_hour", F.hour("created_ts"))
        .withColumn("created_day_of_week", F.dayofweek("created_ts"))
        .withColumn("created_month", F.month("created_ts"))
        .withColumn(
            "days_open",
            F.when(
                F.col("closed_ts").isNotNull(),
                F.col("resolution_hours") / 24.0,
            ),
        )
    )
    for column in CATEGORICAL_FEATURES:
        enriched = enriched.withColumn(
            column,
            F.when(
                F.col(column).isNull() | (F.trim(F.col(column)) == ""),
                "Unknown",
            ).otherwise(F.trim(F.col(column))),
        )
    return enriched


def _export(frame: DataFrame | pd.DataFrame, path: Path, limit: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(frame, DataFrame):
        frame = (frame.limit(limit) if limit else frame).toPandas()
    frame.to_csv(path, index=False)


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def export_analytics(
    clean: DataFrame,
    profile: dict[str, Any],
    months: list[str],
    monthly_counts: dict[str, int],
    output_dir: Path,
) -> None:
    clean.createOrReplaceTempView("nyc311_clean")
    daily = clean.groupBy("created_day", "borough", "complaint_type").agg(
        F.count("*").alias("requests"),
        F.sum(F.when(F.col("closed_ts").isNotNull(), 1).otherwise(0)).alias(
            "closed_requests"
        ),
        F.sum("resolution_hours").alias("total_resolution_hours"),
        F.sum("resolved_within_7d").alias("resolved_within_7d_requests"),
        F.avg("resolution_hours").alias("average_resolution_hours"),
        F.expr("percentile_approx(resolution_hours, 0.5, 10000)").alias(
            "median_resolution_hours"
        ),
        F.avg("resolved_within_7d").alias("resolved_within_7d_rate"),
    )
    _export(
        daily.orderBy("created_day", "borough"),
        output_dir / "daily_requests.csv",
    )
    complaint_summary = clean.groupBy("complaint_type").agg(
        F.count("*").alias("requests"),
        F.sum(F.when(F.col("closed_ts").isNotNull(), 1).otherwise(0)).alias(
            "closed_requests"
        ),
        F.avg("resolution_hours").alias("average_resolution_hours"),
        F.expr("percentile_approx(resolution_hours, 0.5, 10000)").alias(
            "median_resolution_hours"
        ),
        F.avg("resolved_within_7d").alias("resolved_within_7d_rate"),
    )
    window = Window.orderBy(F.col("requests").desc(), F.col("complaint_type"))
    _export(
        complaint_summary.withColumn("rank", F.row_number().over(window))
        .orderBy("rank")
        .limit(40),
        output_dir / "complaint_types.csv",
    )
    borough_summary = clean.groupBy("borough").agg(
        F.count("*").alias("requests"),
        F.avg("resolution_hours").alias("average_resolution_hours"),
        F.expr("percentile_approx(resolution_hours, 0.5, 10000)").alias(
            "median_resolution_hours"
        ),
        F.avg("resolved_within_7d").alias("resolved_within_7d_rate"),
    )
    _export(borough_summary.orderBy(F.col("requests").desc()), output_dir / "boroughs.csv")
    hourly_summary = clean.groupBy(
        "created_hour", "created_day_of_week", "borough", "complaint_type"
    ).agg(
        F.count("*").alias("requests"),
        F.avg("resolution_hours").alias("average_resolution_hours"),
    )
    _export(
        hourly_summary.orderBy("created_day_of_week", "created_hour"),
        output_dir / "hourly_patterns.csv",
    )
    channel_summary = clean.groupBy("open_data_channel_type").agg(
        F.count("*").alias("requests"),
        F.avg("resolution_hours").alias("average_resolution_hours"),
        F.avg("resolved_within_7d").alias("resolved_within_7d_rate"),
    )
    _export(channel_summary.orderBy(F.col("requests").desc()), output_dir / "channels.csv")
    zip_summary = clean.filter(F.col("incident_zip").isNotNull()).groupBy(
        "created_day", "incident_zip", "borough", "complaint_type"
    ).agg(
        F.count("*").alias("requests"),
        F.avg("resolution_hours").alias("average_resolution_hours"),
    )
    _export(zip_summary, output_dir / "zip_hotspots.csv")
    spark_sql = clean.sparkSession.sql(
        """
        SELECT agency, complaint_type, COUNT(*) AS requests,
               AVG(resolution_hours) AS average_resolution_hours,
               AVG(resolved_within_7d) AS resolved_within_7d_rate
        FROM nyc311_clean
        GROUP BY agency, complaint_type
        HAVING COUNT(*) >= 100
        ORDER BY requests DESC
        """
    )
    _export(spark_sql.limit(500), output_dir / "agency_complaints.csv")
    _export(
        clean.select(
            "created_day",
            "latitude_value",
            "longitude_value",
            "borough",
            "complaint_type",
        )
        .filter(F.col("latitude_value").isNotNull() & F.col("longitude_value").isNotNull())
        .orderBy(F.xxhash64("unique_key"))
        .limit(30_000),
        output_dir / "map_sample.csv",
    )

    counts = clean.agg(
        F.count("*").alias("cleaned_rows"),
        F.sum(F.col("closed_ts").isNotNull().cast("long")).alias("closed_rows"),
        F.sum(F.col("closed_ts").isNull().cast("long")).alias("open_rows"),
        F.avg("resolution_hours").alias("mean_resolution_hours"),
        F.expr("percentile_approx(resolution_hours, 0.5, 10000)").alias(
            "median_resolution_hours"
        ),
        F.avg("resolved_within_7d").alias("resolved_within_7d_rate"),
    ).first()
    profile.update(
        {
            "months": months,
            "monthly_source_rows": monthly_counts,
            "cleaned_rows": int(counts["cleaned_rows"]),
            "rows_removed": profile["source_rows"] - int(counts["cleaned_rows"]),
            "closed_rows": int(counts["closed_rows"] or 0),
            "open_rows": int(counts["open_rows"] or 0),
        }
    )
    _write_json(profile, output_dir / "data_profile.json")
    _write_json(
        {
            "requests": int(counts["cleaned_rows"]),
            "closed_requests": int(counts["closed_rows"] or 0),
            "open_requests": int(counts["open_rows"] or 0),
            "mean_resolution_hours": float(counts["mean_resolution_hours"] or 0),
            "median_resolution_hours": float(
                counts["median_resolution_hours"] or 0
            ),
            "resolved_within_7d_rate": float(counts["resolved_within_7d_rate"] or 0),
            "interpretation": (
                "Resolution rates are descriptive. Open cases are not used in "
                "resolution-time statistics; right-censoring can bias averages."
            ),
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


def _metrics(predictions: DataFrame) -> dict[str, float | int]:
    counts = {
        (int(row["label"]), int(row["prediction"])): int(row["count"])
        for row in predictions.groupBy("label", "prediction").count().collect()
    }
    tp, fp = counts.get((1, 1), 0), counts.get((0, 1), 0)
    tn, fn = counts.get((0, 0), 0), counts.get((1, 0), 0)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    auc = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
    ).evaluate(predictions)
    pr_auc = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderPR"
    ).evaluate(predictions)
    return {
        "roc_auc": float(auc),
        "pr_auc": float(pr_auc),
        "accuracy": float((tp + tn) / max(tp + tn + fp + fn, 1)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
    }


def _feature_names(metadata: dict[str, Any], size: int) -> list[str]:
    names = [f"feature_{index}" for index in range(size)]
    for attributes in metadata.get("ml_attr", {}).get("attrs", {}).values():
        for attribute in attributes:
            index = attribute.get("idx")
            if index is not None and index < size:
                names[index] = attribute.get("name", names[index])
    return names


def train_models(clean: DataFrame, output_dir: Path) -> dict[str, Any]:
    """Compare classifiers for seven-day resolution using a chronological split."""
    resolved = clean.filter(F.col("resolved_within_7d").isNotNull())
    cutoff_epoch = (
        resolved.withColumn("created_epoch", F.unix_timestamp("created_ts"))
        .approxQuantile("created_epoch", [0.8], 0.01)[0]
    )
    cutoff_string = (
        resolved.sparkSession.range(1)
        .select(
            F.from_unixtime(F.lit(int(cutoff_epoch))).alias("cutoff")
        )
        .first()["cutoff"]
    )
    cutoff = F.to_timestamp(F.lit(cutoff_string))
    labelled = resolved.withColumn("label", F.col("resolved_within_7d"))
    temporal_train = labelled.filter(F.col("created_ts") < cutoff)
    temporal_test = labelled.filter(F.col("created_ts") >= cutoff)
    training = temporal_train.sample(
        withReplacement=False, fraction=0.10, seed=RANDOM_SEED
    ).limit(100_000).cache()
    test = temporal_test.sample(
        withReplacement=False, fraction=0.20, seed=RANDOM_SEED
    ).limit(50_000).cache()
    train_count, test_count = training.count(), test.count()
    if train_count < 1000 or test_count < 1000:
        raise ValueError(
            f"Model partitions are too small: train={train_count}, test={test_count}"
        )
    if test.select("label").distinct().count() < 2:
        raise ValueError("Chronological test partition must contain both classes.")
    if training.select("label").distinct().count() < 2:
        raise ValueError("Chronological training partition must contain both classes.")

    classifiers = [
        LogisticRegression(
            featuresCol="features",
            labelCol="label",
            maxIter=50,
            regParam=0.05,
        ),
        RandomForestClassifier(
            featuresCol="features",
            labelCol="label",
            numTrees=30,
            maxDepth=8,
            seed=RANDOM_SEED,
        ),
    ]
    fitted: dict[str, tuple[Any, DataFrame]] = {}
    comparisons = []
    for classifier in classifiers:
        name = classifier.__class__.__name__
        LOGGER.info("Training %s on %s requests", name, train_count)
        model = _build_model(classifier).fit(training)
        predictions = model.transform(test).cache()
        comparisons.append({"model": name, **_metrics(predictions)})
        fitted[name] = (model, predictions)

    comparison_frame = pd.DataFrame(comparisons).sort_values(
        ["roc_auc", "f1"], ascending=False
    )
    comparison_frame.to_csv(output_dir / "model_comparison.csv", index=False)
    selected_name = str(comparison_frame.iloc[0]["model"])
    model, predictions = fitted[selected_name]
    selected_metrics = next(row for row in comparisons if row["model"] == selected_name)
    _export(
        predictions.select(
            "unique_key",
            "complaint_type",
            "borough",
            "created_ts",
            "label",
            "prediction",
            vector_to_array("probability")[1].alias("resolved_within_7d_probability"),
        ),
        output_dir / "model_predictions.csv",
        limit=30_000,
    )
    transformed = model.transform(training.limit(1))
    metadata = transformed.schema["features"].metadata
    final_model = model.stages[-1]
    importances = (
        final_model.featureImportances.toArray()
        if hasattr(final_model, "featureImportances")
        else abs(final_model.coefficients.toArray())
    )
    pd.DataFrame(
        {
            "feature": _feature_names(metadata, len(importances)),
            "importance": importances,
        }
    ).sort_values("importance", ascending=False).head(30).to_csv(
        output_dir / "feature_importance.csv", index=False
    )
    for _, prediction_frame in fitted.values():
        prediction_frame.unpersist()
    training.unpersist()
    test.unpersist()
    return {
        "selected_model": selected_name,
        "selected_model_metrics": selected_metrics,
        "model_comparison": comparisons,
        "train_rows": train_count,
        "test_rows": test_count,
        "train_start": str(temporal_train.agg(F.min("created_ts")).first()[0]),
        "train_end": cutoff_string,
        "test_start": cutoff_string,
        "test_end": str(temporal_test.agg(F.max("created_ts")).first()[0]),
        "split_method": "chronological 80/20 timestamp split; bounded sampling for local model runtime",
        "target": "closed within seven days (closed requests only)",
        "leakage_control": (
            "Only fields known at request creation and creation-time features "
            "are model inputs; closed_date and resolution time are excluded."
        ),
    }


def run_pipeline(
    months: list[str],
    raw_dir: Path,
    output_dir: Path,
    master: str,
    refresh: bool = False,
) -> None:
    app_token = os.environ.get("SOCRATA_APP_TOKEN")
    files, monthly_counts = download_months(months, raw_dir, app_token, refresh)
    spark = (
        SparkSession.builder.appName("NYC311ServiceRequestAnalytics")
        .master(master)
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    try:
        raw = load_source(spark, files)
        profile = profile_source(raw)
        expected_rows = sum(monthly_counts.values())
        if profile["source_rows"] != expected_rows:
            raise ValueError(
                f"Spark read {profile['source_rows']} records but the NYC Open Data "
                f"monthly counts total {expected_rows}."
            )
        clean = clean_and_enrich(raw)
        export_analytics(clean, profile, months, monthly_counts, output_dir)
        model_metrics = train_models(clean, output_dir)
        model_metrics["months"] = months
        model_metrics["source_rows"] = profile["source_rows"]
        model_metrics["cleaned_rows"] = profile["cleaned_rows"]
        _write_json(model_metrics, output_dir / "model_metrics.json")
        LOGGER.info(
            "Pipeline complete: %s rows fetched; %s valid requests",
            profile["source_rows"],
            profile["cleaned_rows"],
        )
    finally:
        spark.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-month", default=DEFAULT_START_MONTH, help="YYYY-MM")
    parser.add_argument("--end-month", default=DEFAULT_END_MONTH, help="YYYY-MM, inclusive")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--master", default="local[*]", help="Spark master URL")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Redownload selected API months, including updated closures.",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    run_pipeline(
        month_range(args.start_month, args.end_month),
        args.raw_dir,
        args.output_dir,
        args.master,
        args.refresh,
    )


if __name__ == "__main__":
    main()
