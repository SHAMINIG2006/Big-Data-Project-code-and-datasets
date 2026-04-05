"""
Module 4: Feature Engineering
---------------------------------
Runs the full PySpark ML Pipeline on the static cybersecurity_attacks.csv:
  StringIndexer → OneHotEncoder → VectorAssembler → StandardScaler

Saves the fitted PipelineModel to models/feature_pipeline/ for use by
Module 5 (Spark ML training).

Run from project root:
    py -3 modules/module4_features/feature_engineering.py
"""

import sys
import os

# Force UTF-8 stdout on Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler,
)
from pyspark.sql.functions import col, when
try:
    from pyspark.sql.functions import try_cast
    HAS_TRY_CAST = True
except ImportError:
    HAS_TRY_CAST = False

# Resolve project root relative to this file
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

PIPELINE_PATH = os.path.join(PROJECT_ROOT, "models", "feature_pipeline")


def process_features():
    spark = (
        SparkSession.builder
        .appName("CyberSecurityFeatureEngineering")
        # Disable ANSI mode so cast() returns NULL on bad values (not exception)
        .config("spark.sql.ansi.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    csv_path = os.path.join(PROJECT_ROOT, "cybersecurity_attacks.csv")
    print(f"Loading static dataset from: {csv_path}")

    # ── Load static CSV (batch – required so Pipeline.fit() can run) ──
    # Use all-string schema to avoid NumberFormatException on malformed rows,
    # then cast numeric columns explicitly.
    df = spark.read.csv(csv_path, header=True, inferSchema=False)
    total = df.count()
    print(f"  [OK] Loaded {total:,} rows, {len(df.columns)} columns")

    # ── Drop rows with nulls in the columns we need ───────────────────
    numeric_cols     = ["Source Port", "Destination Port", "Packet Length", "Anomaly Scores"]
    categorical_cols = ["Protocol", "Packet Type", "Traffic Type", "Severity Level"]

    df = df.dropna(subset=numeric_cols + categorical_cols + ["Attack Type"])
    print(f"  [OK] Rows after dropping nulls: {df.count():,}")

    # Cast numeric columns explicitly (try_cast → NULL on bad values)
    if HAS_TRY_CAST:
        for c in numeric_cols:
            df = df.withColumn(c, try_cast(col(c), "double"))
    else:
        for c in numeric_cols:
            df = df.withColumn(c, col(c).cast("double"))

    # ── Pipeline stages ───────────────────────────────────────────────
    indexed_cols = [c + "_indexed" for c in categorical_cols]
    encoded_cols = [c + "_encoded" for c in categorical_cols]

    # Stage 1: StringIndexer for each categorical column
    indexers = [
        StringIndexer(
            inputCol=c, outputCol=c + "_indexed", handleInvalid="keep"
        )
        for c in categorical_cols
    ]

    # Stage 2: OneHotEncoder
    encoders = [
        OneHotEncoder(inputCol=c + "_indexed", outputCol=c + "_encoded")
        for c in categorical_cols
    ]

    # Stage 3: Label indexer for target
    label_indexer = StringIndexer(
        inputCol="Attack Type", outputCol="label", handleInvalid="keep"
    )

    # Stage 4: VectorAssembler – combines numeric + encoded features
    feature_cols = numeric_cols + encoded_cols
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features")

    # Stage 5: StandardScaler
    scaler = StandardScaler(
        inputCol="raw_features", outputCol="features",
        withMean=False, withStd=True,
    )

    pipeline = Pipeline(stages=indexers + encoders + [label_indexer, assembler, scaler])

    # ── Fit the Pipeline ──────────────────────────────────────────────
    print("\nFitting pipeline (StringIndexer → OHE → VectorAssembler → StandardScaler)...")
    pipeline_model = pipeline.fit(df)
    print("  [OK] Pipeline fitted.")

    # ── Transform and show sample ─────────────────────────────────────
    transformed = pipeline_model.transform(df)
    print("\nSample of transformed data (label | features):")
    transformed.select("label", "features").show(5, truncate=80)

    # ── Save the fitted PipelineModel ─────────────────────────────────
    os.makedirs(os.path.dirname(PIPELINE_PATH), exist_ok=True)
    pipeline_model.write().overwrite().save(PIPELINE_PATH)
    print(f"\n[SAVED] PipelineModel saved to: {PIPELINE_PATH}")
    print("[DONE] Feature engineering complete. Module 5 can now load this pipeline.")

    spark.stop()


if __name__ == "__main__":
    process_features()
