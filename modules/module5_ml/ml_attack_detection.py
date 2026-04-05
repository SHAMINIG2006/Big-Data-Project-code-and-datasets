"""
Module 5: ML Attack Detection
---------------------------------
Trains TWO classifiers so both sklearn (needed by the dashboard) and
PySpark MLlib (required for the Big Data stack) are covered:

  A) scikit-learn RandomForest  → models/rf_attack_model.joblib  (dashboard uses this)
  B) PySpark MLlib RandomForest → models/spark_rf_model/          (Spark-native ML)

Run from project root:
    py -3 modules/module5_ml/ml_attack_detection.py

Note: Run module4_features/feature_engineering.py first so that the
      fitted PipelineModel exists at models/feature_pipeline/.
"""

import sys
import os

# Force UTF-8 stdout on Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Resolve project root relative to this file
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

CSV_PATH      = os.path.join(PROJECT_ROOT, "cybersecurity_attacks.csv")
MODELS_DIR    = os.path.join(PROJECT_ROOT, "models")
PIPELINE_PATH = os.path.join(MODELS_DIR, "feature_pipeline")


# ══════════════════════════════════════════════════════════════════════════════
# Part A – scikit-learn (dashboard-compatible, fast, no Spark needed)
# ══════════════════════════════════════════════════════════════════════════════

def train_sklearn_model():
    print("=" * 60)
    print("Part A: Training scikit-learn RandomForest")
    print("=" * 60)

    df = pd.read_csv(CSV_PATH)
    print(f"Dataset loaded: {len(df):,} rows, {len(df.columns)} columns")

    categorical_cols = ["Protocol", "Packet Type", "Traffic Type", "Severity Level"]
    numeric_cols     = ["Source Port", "Destination Port", "Packet Length", "Anomaly Scores"]
    target_col       = "Attack Type"

    # Fill nulls
    df[numeric_cols]     = df[numeric_cols].fillna(0)
    df[categorical_cols] = df[categorical_cols].fillna("Unknown")
    df[target_col]       = df[target_col].fillna("Unknown")

    # Encode categorical features
    feat_encoders = {}
    for c in categorical_cols:
        le = LabelEncoder()
        df[c + "_enc"] = le.fit_transform(df[c].astype(str))
        feat_encoders[c] = le

    # Encode target
    label_enc    = LabelEncoder()
    df["label"]  = label_enc.fit_transform(df[target_col].astype(str))

    feature_cols = numeric_cols + [c + "_enc" for c in categorical_cols]
    X = df[feature_cols]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"  Train size: {len(X_train):,}  |  Test size: {len(X_test):,}")
    print("  Training Random Forest (n_estimators=50)...")

    rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    preds = rf.predict(X_test)
    acc   = accuracy_score(y_test, preds)

    print(f"\n  [OK] Accuracy: {acc:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_test, preds, target_names=label_enc.classes_))

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(rf,            os.path.join(MODELS_DIR, "rf_attack_model.joblib"))
    joblib.dump(label_enc,     os.path.join(MODELS_DIR, "label_encoder.joblib"))
    joblib.dump(feat_encoders, os.path.join(MODELS_DIR, "feature_encoders.joblib"))

    print(f"  [SAVED] Models saved to {MODELS_DIR}/")
    print("  -> rf_attack_model.joblib  (RandomForest)")
    print("  -> label_encoder.joblib")
    print("  -> feature_encoders.joblib")


# ══════════════════════════════════════════════════════════════════════════════
# Part B – PySpark MLlib (Spark-native, loads fitted pipeline from Module 4)
# ══════════════════════════════════════════════════════════════════════════════

def train_spark_model():
    print("\n" + "=" * 60)
    print("Part B: Training PySpark MLlib RandomForest")
    print("=" * 60)

    try:
        from pyspark.sql import SparkSession
        from pyspark.ml import PipelineModel
        from pyspark.ml.classification import RandomForestClassifier as SparkRF
        from pyspark.ml.evaluation import MulticlassClassificationEvaluator
    except ImportError as e:
        print(f"  [WARNING] PySpark import failed ({e}). Skipping Part B.")
        return

    spark = (
        SparkSession.builder
        .appName("CyberSecurityMLlib")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    # ── Check that feature pipeline exists ────────────────────────────
    if not os.path.exists(PIPELINE_PATH):
        print(
            f"  [WARNING] PipelineModel not found at {PIPELINE_PATH}.\n"
            "  Run module4_features/feature_engineering.py first. Skipping Part B."
        )
        spark.stop()
        return

    print(f"  Loading PipelineModel from: {PIPELINE_PATH}")
    pipeline_model = PipelineModel.load(PIPELINE_PATH)

    # ── Load and transform data ────────────────────────────────────────
    print(f"  Loading dataset from: {CSV_PATH}")
    df = spark.read.csv(CSV_PATH, header=True, inferSchema=True)

    from pyspark.sql.functions import col as scol
    numeric_cols     = ["Source Port", "Destination Port", "Packet Length", "Anomaly Scores"]
    categorical_cols = ["Protocol", "Packet Type", "Traffic Type", "Severity Level"]

    df = df.dropna(subset=numeric_cols + categorical_cols + ["Attack Type"])
    for c in numeric_cols:
        df = df.withColumn(c, scol(c).cast("double"))

    print("  Transforming features with fitted pipeline...")
    transformed = pipeline_model.transform(df)

    # ── Train / test split (80/20) ─────────────────────────────────────
    train_df, test_df = transformed.randomSplit([0.8, 0.2], seed=42)
    print(f"  Train: {train_df.count():,}  |  Test: {test_df.count():,}")

    # ── Train Spark RandomForest ───────────────────────────────────────
    print("  Training PySpark RandomForestClassifier (numTrees=50)...")
    rf_spark = SparkRF(
        featuresCol="features",
        labelCol="label",
        numTrees=50,
        seed=42,
    )
    spark_model = rf_spark.fit(train_df)

    # ── Evaluate ───────────────────────────────────────────────────────
    predictions = spark_model.transform(test_df)

    evaluator_acc = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy"
    )
    evaluator_f1 = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="f1"
    )

    acc = evaluator_acc.evaluate(predictions)
    f1  = evaluator_f1.evaluate(predictions)

    print(f"\n  [OK] PySpark RF Accuracy : {acc:.4f}")
    print(f"  [OK] PySpark RF F1 Score : {f1:.4f}")

    # ── Save Spark model ───────────────────────────────────────────────
    spark_model_path = os.path.join(MODELS_DIR, "spark_rf_model")
    spark_model.write().overwrite().save(spark_model_path)
    print(f"\n  [SAVED] Spark model saved to: {spark_model_path}/")

    spark.stop()


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    train_sklearn_model()
    train_spark_model()
    print("\n[DONE] Module 5 complete.")
