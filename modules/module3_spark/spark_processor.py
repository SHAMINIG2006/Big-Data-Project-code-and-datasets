"""
Module 3: Spark Batch Processor
--------------------------------
Runs 7 extended batch analyses on cybersecurity_attacks.csv using PySpark.
Reads the CSV with all-string schema first, then casts only the two
numeric columns we need (Anomaly Scores, Packet Length) to avoid
NumberFormatException on malformed rows.

Writes a plain-text summary to metrics_output.txt.

Run from project root:
    py -3 modules/module3_spark/spark_processor.py
"""

import sys
import os

# Force UTF-8 stdout on Windows consoles (cp1252 crashes on non-ASCII)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, avg, desc, hour, to_timestamp
try:
    from pyspark.sql.functions import try_cast
    HAS_TRY_CAST = True
except ImportError:
    HAS_TRY_CAST = False

# Resolve project root relative to this file
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

OUTPUT_FILE = os.path.join(PROJECT_ROOT, "metrics_output.txt")


def run_batch_processing():
    spark = (
        SparkSession.builder
        .appName("CyberSecurityBatchProcessor")
        # Disable ANSI mode so cast() returns NULL instead of raising on bad values
        .config("spark.sql.ansi.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    csv_path = os.path.join(PROJECT_ROOT, "cybersecurity_attacks.csv")
    print(f"Loading dataset from: {csv_path}")

    # Read ALL columns as strings first to avoid NumberFormatException
    # on any malformed numeric cells.
    df_raw = spark.read.csv(csv_path, header=True, inferSchema=False)

    # Cast only the numeric columns we actually aggregate on.
    # Use try_cast (PySpark 3.4+) when available so malformed cells become NULL;
    # fall back to plain cast() which is lenient when ANSI mode is off.
    if HAS_TRY_CAST:
        df = (
            df_raw
            .withColumn("Anomaly Scores", try_cast(col("Anomaly Scores"), "double"))
            .withColumn("Packet Length",  try_cast(col("Packet Length"),  "long"))
            .filter(col("Attack Type").isNotNull())
        )
    else:
        df = (
            df_raw
            .withColumn("Anomaly Scores", col("Anomaly Scores").cast("double"))
            .withColumn("Packet Length",  col("Packet Length").cast("long"))
            .filter(col("Attack Type").isNotNull())
        )

    total = df.count()

    lines = []   # buffer for file output

    def tprint(msg=""):
        print(msg)
        lines.append(msg + "\n")

    def show_section(label, frame, n=10):
        sep = "=" * 60
        tprint(f"\n{sep}")
        tprint(label)
        tprint(sep)
        for row in frame.limit(n).collect():
            tprint("  " + " | ".join(str(v) if v is not None else "NULL" for v in row))

    tprint(f"[OK] Total records loaded: {total:,}")

    # 1. Attack Type Distribution
    show_section(
        "[1] Attack Type Distribution",
        df.groupBy("Attack Type").agg(count("*").alias("count")).orderBy(desc("count")),
    )

    # 2. Severity Level Breakdown
    show_section(
        "[2] Severity Level Breakdown",
        df.groupBy("Severity Level").agg(count("*").alias("count")).orderBy(desc("count")),
    )

    # 3. Protocol Distribution + Avg Packet Length
    show_section(
        "[3] Protocol Distribution (avg packet length)",
        df.groupBy("Protocol").agg(
            count("*").alias("count"),
            avg("Packet Length").alias("avg_packet_length"),
        ).orderBy(desc("count")),
    )

    # 4. Avg Anomaly Score per Attack Type
    show_section(
        "[4] Avg Anomaly Score per Attack Type",
        df.groupBy("Attack Type").agg(
            avg("Anomaly Scores").alias("avg_anomaly_score"),
        ).orderBy(desc("avg_anomaly_score")),
    )

    # 5. Top-10 Source IPs by Event Count
    show_section(
        "[5] Top-10 Source IPs by Event Count",
        df.groupBy("Source IP Address").agg(
            count("*").alias("event_count"),
        ).orderBy(desc("event_count")),
        n=10,
    )

    # 6. Severity x Protocol Cross-Tab
    show_section(
        "[6] Severity x Protocol Cross-Tab",
        df.groupBy("Severity Level", "Protocol").agg(
            count("*").alias("count"),
        ).orderBy("Severity Level", desc("count")),
        n=20,
    )

    # 7. Hourly Event Trend (needs parseable Timestamp column)
    try:
        df_ts = df.withColumn("ts", to_timestamp(col("Timestamp")))
        df_ts = df_ts.filter(col("ts").isNotNull())
        n_ts  = df_ts.count()
        if n_ts > 0:
            show_section(
                "[7] Hourly Event Trend",
                df_ts.withColumn("hour", hour(col("ts")))
                     .groupBy("hour")
                     .agg(count("*").alias("events"))
                     .orderBy("hour"),
                n=24,
            )
        else:
            tprint("\n[7] Hourly Event Trend -- skipped (Timestamp column not parseable as datetime)")
    except Exception as e:
        tprint(f"\n[7] Hourly Event Trend -- skipped: {e}")

    tprint("\n[DONE] Batch processing complete.")

    # Write plain-text report to file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"[OK] Summary written to: {OUTPUT_FILE}")

    spark.stop()


if __name__ == "__main__":
    run_batch_processing()
