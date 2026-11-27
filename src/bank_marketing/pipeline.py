"""Download, profile, transform, model, and export the UCI Bank Marketing data."""

from __future__ import annotations

import argparse
import json
import logging
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import Imputer, OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.ml.stat import Correlation
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import (
    IntegerType,
    StringType,
    StructField,
    StructType,
)

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "bank_marketing.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
DATA_URL = "https://archive.ics.uci.edu/static/public/222/data.csv"
RANDOM_SEED = 42

SCHEMA = StructType(
    [
        StructField("age", IntegerType(), True),
        StructField("job", StringType(), True),
        StructField("marital", StringType(), True),
        StructField("education", StringType(), True),
        StructField("default", StringType(), True),
        StructField("balance", IntegerType(), True),
        StructField("housing", StringType(), True),
        StructField("loan", StringType(), True),
        StructField("contact", StringType(), True),
        StructField("day_of_week", IntegerType(), True),
        StructField("month", StringType(), True),
        StructField("duration", IntegerType(), True),
        StructField("campaign", IntegerType(), True),
        StructField("pdays", IntegerType(), True),
        StructField("previous", IntegerType(), True),
        StructField("poutcome", StringType(), True),
        StructField("y", StringType(), True),
    ]
)

NUMERIC_FEATURES = [
    "age",
    "balance",
    "day_of_week",
    "campaign",
    "pdays",
    "previous",
    "previously_contacted",
]
CATEGORICAL_FEATURES = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome",
]
CATEGORICAL_DOMAINS = {
    "job": {
        "admin.",
        "blue-collar",
        "entrepreneur",
        "housemaid",
        "management",
        "retired",
        "self-employed",
        "services",
        "student",
        "technician",
        "unemployed",
        "unknown",
    },
    "marital": {"divorced", "married", "single", "unknown"},
    "education": {"primary", "secondary", "tertiary", "unknown"},
    "default": {"no", "yes", "unknown"},
    "housing": {"no", "yes", "unknown"},
    "loan": {"no", "yes", "unknown"},
    "contact": {"cellular", "telephone", "unknown"},
    "month": {
        "apr",
        "aug",
        "dec",
        "feb",
        "jan",
        "jul",
        "jun",
        "mar",
        "may",
        "nov",
        "oct",
        "sep",
        "unknown",
    },
    "poutcome": {"failure", "other", "success", "unknown"},
}
RAW_COLUMNS = [field.name for field in SCHEMA.fields]


def _download_dataset(path: Path) -> None:
    """Fetch the official UCI CSV when a local copy is not present."""
    path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Downloading dataset from %s", DATA_URL)
    request = urllib.request.Request(
        DATA_URL, headers={"User-Agent": "bank-marketing-analytics/1.0"}
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            path.write_bytes(response.read())
    except (OSError, TimeoutError) as exc:
        raise RuntimeError(
            f"Could not download the UCI dataset from {DATA_URL}: {exc}. "
            "Download it manually and pass --input PATH."
        ) from exc


def load_raw_data(spark: SparkSession, input_path: Path) -> DataFrame:
    """Read the UCI CSV with a stable schema and explicit missing-value handling."""
    if not input_path.is_file():
        _download_dataset(input_path)
    frame = (
        spark.read.option("header", True)
        .option("mode", "FAILFAST")
        .option("nullValue", "NaN")
        .option("emptyValue", "")
        .schema(SCHEMA)
        .csv(str(input_path))
    )
    actual = set(frame.columns)
    expected = set(RAW_COLUMNS)
    if actual != expected:
        raise ValueError(
            "Dataset columns do not match the expected UCI schema. "
            f"Missing: {sorted(expected - actual)}; unexpected: {sorted(actual - expected)}"
        )
    return frame


def _count_missing(frame: DataFrame) -> dict[str, int]:
    expressions = [
        F.sum(F.col(column).isNull().cast("long")).alias(column)
        for column in frame.columns
    ]
    row = frame.agg(*expressions).first()
    return {column: int(row[column] or 0) for column in frame.columns}


def profile_data(raw: DataFrame) -> tuple[dict[str, Any], DataFrame]:
    """Inspect nulls, exact duplicates, invalid-domain values, and class balance."""
    total = raw.count()
    duplicate_count = total - raw.dropDuplicates().count()
    invalid_rules = {
        "age_outside_18_100": (F.col("age") < 18) | (F.col("age") > 100),
        "day_outside_1_31": (F.col("day_of_week") < 1) | (F.col("day_of_week") > 31),
        "duration_negative": F.col("duration") < 0,
        "campaign_below_1": F.col("campaign") < 1,
        "pdays_below_minus_1": F.col("pdays") < -1,
        "previous_negative": F.col("previous") < 0,
        "invalid_target": F.col("y").isNull() | ~F.col("y").isin("yes", "no"),
    }
    for column, allowed_values in CATEGORICAL_DOMAINS.items():
        normalized = F.lower(F.trim(F.col(column)))
        invalid_rules[f"invalid_{column}"] = F.col(column).isNotNull() & ~normalized.isin(
            *sorted(allowed_values)
        )
    invalid_count_row = raw.agg(
        *[
            F.sum(condition.cast("long")).alias(name)
            for name, condition in invalid_rules.items()
        ]
    ).first()
    invalid_counts = {
        name: int(invalid_count_row[name] or 0) for name in invalid_rules
    }
    class_balance = (
        raw.groupBy("y")
        .count()
        .orderBy("y")
        .toPandas()
        .to_dict(orient="records")
    )
    profile = {
        "rows_before_cleaning": total,
        "exact_duplicate_rows": duplicate_count,
        "missing_values_by_column": _count_missing(raw),
        "invalid_rows_by_rule": invalid_counts,
        "class_balance": class_balance,
        "schema": [
            {"column": field.name, "type": field.dataType.simpleString()}
            for field in raw.schema.fields
        ],
    }
    numeric_summary = raw.select(
        "age", "balance", "day_of_week", "duration", "campaign", "pdays", "previous"
    ).summary("count", "mean", "stddev", "min", "max")
    return profile, numeric_summary


def clean_data(raw: DataFrame) -> DataFrame:
    """Remove invalid records, collapse exact duplicates, and normalize categories."""
    clean = raw.dropDuplicates()
    for column in CATEGORICAL_FEATURES:
        clean = clean.withColumn(
            column,
            F.when(
                F.trim(F.col(column)) == "",
                F.lit(None).cast("string"),
            ).otherwise(F.lower(F.trim(F.col(column)))),
        )
    clean = clean.filter(
        F.col("age").between(18, 100)
        & F.col("day_of_week").between(1, 31)
        & (F.col("duration") >= 0)
        & (F.col("campaign") >= 1)
        & (F.col("pdays") >= -1)
        & (F.col("previous") >= 0)
        & F.col("y").isin("yes", "no")
    )
    for column, allowed_values in CATEGORICAL_DOMAINS.items():
        clean = clean.filter(
            F.col(column).isNull() | F.col(column).isin(*sorted(allowed_values))
        )
    clean = clean.fillna(
        {column: "unknown" for column in CATEGORICAL_FEATURES}
    ).withColumn(
        "previously_contacted",
        F.when(F.col("pdays").isNotNull() & (F.col("pdays") >= 0), 1).otherwise(0),
    )
    return clean


def _export_csv(frame: DataFrame | pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    pdf = frame.toPandas() if isinstance(frame, DataFrame) else frame
    pdf.to_csv(destination, index=False)


def _write_json(data: dict[str, Any], destination: Path) -> None:
    destination.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")


def export_eda(clean: DataFrame, profile: dict[str, Any], output_dir: Path) -> None:
    """Run Spark SQL and a window ranking for the dashboard's EDA artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(profile, output_dir / "data_profile.json")
    numeric_columns = [
        "age",
        "balance",
        "day_of_week",
        "duration",
        "campaign",
        "pdays",
        "previous",
    ]
    _export_csv(
        clean.select(*numeric_columns).summary("count", "mean", "stddev", "min", "max"),
        output_dir / "numeric_summary.csv",
    )
    distribution_rows = []
    for column in numeric_columns:
        quantiles = clean.approxQuantile(column, [0.0, 0.25, 0.5, 0.75, 1.0], 0.01)
        distribution_rows.append(
            {
                "feature": column,
                "min": quantiles[0] if quantiles else None,
                "q25": quantiles[1] if quantiles else None,
                "median": quantiles[2] if quantiles else None,
                "q75": quantiles[3] if quantiles else None,
                "max": quantiles[4] if quantiles else None,
            }
        )
    _export_csv(
        pd.DataFrame(distribution_rows), output_dir / "numeric_distributions.csv"
    )
    numeric_vectors = VectorAssembler(
        inputCols=numeric_columns,
        outputCol="numeric_features",
        handleInvalid="skip",
    ).transform(clean)
    correlation_matrix = Correlation.corr(
        numeric_vectors, "numeric_features", method="pearson"
    ).first()[0].toArray()
    correlation_rows = [
        {
            "feature": left,
            "other_feature": right,
            "pearson_correlation": float(correlation_matrix[left_index, right_index]),
        }
        for left_index, left in enumerate(numeric_columns)
        for right_index, right in enumerate(numeric_columns)
    ]
    _export_csv(
        pd.DataFrame(correlation_rows), output_dir / "numeric_correlations.csv"
    )
    clean.createOrReplaceTempView("bank_marketing_clean")
    job_summary = clean.sparkSession.sql(
        """
        SELECT job,
               COUNT(*) AS customers,
               SUM(CASE WHEN y = 'yes' THEN 1 ELSE 0 END) AS subscriptions,
               AVG(CASE WHEN y = 'yes' THEN 1.0 ELSE 0.0 END) AS subscription_rate
        FROM bank_marketing_clean
        GROUP BY job
        """
    )
    job_window = Window.orderBy(
        F.col("subscription_rate").desc(), F.col("customers").desc()
    )
    ranked_jobs = job_summary.withColumn(
        "conversion_rank", F.row_number().over(job_window)
    ).orderBy("conversion_rank")
    _export_csv(
        ranked_jobs,
        output_dir / "conversion_by_job.csv",
    )
    month_summary = (
        clean.groupBy("month")
        .agg(
            F.count("*").alias("customers"),
            F.sum(F.when(F.col("y") == "yes", 1).otherwise(0)).alias("subscriptions"),
            F.avg(F.when(F.col("y") == "yes", 1.0).otherwise(0.0)).alias(
                "subscription_rate"
            ),
        )
        .orderBy("month")
    )
    _export_csv(month_summary, output_dir / "conversion_by_month.csv")
    _export_csv(
        clean.groupBy("y").count().orderBy("y"),
        output_dir / "target_balance.csv",
    )
    _export_csv(clean, output_dir / "dashboard_data.csv")
    highest_job = job_summary.orderBy(
        F.col("subscription_rate").desc(), F.col("customers").desc()
    ).first()
    highest_month = month_summary.orderBy(
        F.col("subscription_rate").desc(), F.col("customers").desc()
    ).first()
    overall_rate = clean.agg(
        F.avg(F.when(F.col("y") == "yes", 1.0).otherwise(0.0)).alias("rate")
    ).first()["rate"]
    _write_json(
        {
            "overall_subscription_rate": float(overall_rate or 0.0),
            "highest_job_subscription_rate": {
                "job": highest_job["job"],
                "customers": int(highest_job["customers"]),
                "subscription_rate": float(highest_job["subscription_rate"]),
            },
            "highest_month_subscription_rate": {
                "month": highest_month["month"],
                "customers": int(highest_month["customers"]),
                "subscription_rate": float(highest_month["subscription_rate"]),
            },
            "interpretation": (
                "These are descriptive rates, not causal estimates; segment "
                "sample sizes should be considered when comparing rates."
            ),
        },
        output_dir / "insights.json",
    )


def _classification_metrics(predictions: DataFrame) -> dict[str, float]:
    counts = {
        (int(row["label"]), int(row["prediction"])): int(row["count"])
        for row in predictions.groupBy("label", "prediction").count().collect()
    }
    true_positive = counts.get((1, 1), 0)
    false_positive = counts.get((0, 1), 0)
    false_negative = counts.get((1, 0), 0)
    true_negative = counts.get((0, 0), 0)
    total = true_positive + false_positive + false_negative + true_negative
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (true_positive + true_negative) / total if total else 0.0
    auc = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
    ).evaluate(predictions)
    return {
        "roc_auc": float(auc),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
    }


def _build_model(classifier: Any) -> Pipeline:
    imputed_columns = [f"{column}_imputed" for column in NUMERIC_FEATURES]
    indexers = [
        StringIndexer(
            inputCol=column,
            outputCol=f"{column}_index",
            handleInvalid="keep",
        )
        for column in CATEGORICAL_FEATURES
    ]
    encoder = OneHotEncoder(
        inputCols=[f"{column}_index" for column in CATEGORICAL_FEATURES],
        outputCols=[f"{column}_encoded" for column in CATEGORICAL_FEATURES],
        handleInvalid="keep",
    )
    imputer = Imputer(
        inputCols=NUMERIC_FEATURES,
        outputCols=imputed_columns,
        strategy="median",
    )
    assembler = VectorAssembler(
        inputCols=imputed_columns
        + [f"{column}_encoded" for column in CATEGORICAL_FEATURES],
        outputCol="features",
        handleInvalid="keep",
    )
    return Pipeline(stages=[*indexers, encoder, imputer, assembler, classifier])


def _feature_names(metadata: dict[str, Any], size: int) -> list[str]:
    attributes = metadata.get("ml_attr", {}).get("attrs", {})
    names = [f"feature_{index}" for index in range(size)]
    for entries in attributes.values():
        for entry in entries:
            index = entry.get("idx")
            if index is not None and index < size:
                names[index] = entry.get("name", names[index])
    return names


def train_and_evaluate(clean: DataFrame, output_dir: Path) -> dict[str, Any]:
    """Make a stable 80/20 hash holdout and compare two Spark ML classifiers."""
    labelled = clean.withColumn(
        "label", F.when(F.col("y") == "yes", 1.0).otherwise(0.0)
    )
    bucket = F.pmod(F.xxhash64(*[F.col(column) for column in RAW_COLUMNS]), F.lit(10))
    train = labelled.filter(bucket < 8).cache()
    test = labelled.filter(bucket >= 8).cache()
    train_count, test_count = train.count(), test.count()
    if train_count == 0 or test_count == 0:
        raise ValueError("The deterministic 80/20 split produced an empty partition.")
    if test.select("label").distinct().count() < 2:
        raise ValueError("The test partition must contain both target classes.")

    classifiers = [
        LogisticRegression(
            featuresCol="features",
            labelCol="label",
            maxIter=60,
            regParam=0.05,
        ),
        RandomForestClassifier(
            featuresCol="features",
            labelCol="label",
            numTrees=100,
            maxDepth=8,
            seed=RANDOM_SEED,
        ),
    ]
    model_results = []
    fitted_models = {}
    for classifier in classifiers:
        name = classifier.__class__.__name__
        LOGGER.info("Training %s", name)
        fitted = _build_model(classifier).fit(train)
        predictions = fitted.transform(test)
        metrics = _classification_metrics(predictions)
        model_results.append({"model": name, **metrics})
        fitted_models[name] = (fitted, predictions)

    comparison = pd.DataFrame(model_results).sort_values(
        ["roc_auc", "f1"], ascending=False
    )
    _export_csv(comparison, output_dir / "model_comparison.csv")
    best_name = str(comparison.iloc[0]["model"])
    best_model, best_predictions = fitted_models[best_name]
    best_metrics = next(row for row in model_results if row["model"] == best_name)

    prediction_export = best_predictions.select(
        "y",
        "job",
        "month",
        "label",
        "prediction",
        vector_to_array("probability")[1].alias("subscription_probability"),
    ).withColumnRenamed("y", "actual")
    _export_csv(prediction_export, output_dir / "model_predictions.csv")

    transformed_train = best_model.transform(train.limit(1))
    features_metadata = transformed_train.schema["features"].metadata
    classifier_model = best_model.stages[-1]
    if hasattr(classifier_model, "featureImportances"):
        importances = classifier_model.featureImportances.toArray()
    else:
        importances = abs(classifier_model.coefficients.toArray())
    feature_names = _feature_names(features_metadata, len(importances))
    importance_frame = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)
    _export_csv(importance_frame.head(25), output_dir / "feature_importance.csv")

    train.unpersist()
    test.unpersist()
    return {
        "selected_model": best_name,
        "selected_model_metrics": best_metrics,
        "train_rows": train_count,
        "test_rows": test_count,
        "split": "deterministic 80/20 hash split, seed-free and grouped by exact source row",
        "excluded_feature": (
            "duration: it is only known after the call and would cause target leakage "
            "for pre-call prediction"
        ),
    }


def run_pipeline(input_path: Path, output_dir: Path) -> None:
    spark = (
        SparkSession.builder.appName("BankMarketingAnalytics")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    try:
        raw = load_raw_data(spark, input_path)
        profile, _ = profile_data(raw)
        clean = clean_data(raw).cache()
        clean_count = clean.count()
        if clean_count == 0:
            raise ValueError("No rows remain after data quality filters.")
        export_eda(clean, profile, output_dir)
        model_results = train_and_evaluate(clean, output_dir)
        profile["rows_after_cleaning"] = clean_count
        profile["rows_removed"] = profile["rows_before_cleaning"] - clean_count
        _write_json(profile, output_dir / "data_profile.json")
        _write_json(model_results, output_dir / "model_metrics.json")
        LOGGER.info(
            "Pipeline complete: %s cleaned rows, selected %s",
            clean_count,
            model_results["selected_model"],
        )
    finally:
        spark.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    run_pipeline(args.input, args.output_dir)


if __name__ == "__main__":
    main()
