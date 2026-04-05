"""
Module 2: Spark Structured Streaming Ingestion
------------------------------------------------
Reads the growing logs/live_logs.csv via Spark Structured Streaming and
writes each micro-batch to logs/parquet_sink/ in Parquet format.
A foreachBatch callback prints the row count for every micro-batch so
progress is visible without opening the parquet files.

Run Module 1 first in a separate terminal, then:
    py -3 modules/module2_ingestion/data_ingestion.py
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import (
    StructType, StructField,
    StringType, IntegerType, DoubleType,
)
import os

# Resolve project root relative to this file
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Paths
LOG_DIR        = os.path.join(PROJECT_ROOT, "logs")
PARQUET_SINK   = os.path.join(PROJECT_ROOT, "logs", "parquet_sink")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "logs", "checkpoints", "ingestion")


def start_ingestion():
    spark = (
        SparkSession.builder
        .appName("CyberSecurityLogIngestion")
        .config("spark.sql.streaming.checkpointLocation", CHECKPOINT_DIR)
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    # Full schema including the new Emit_Timestamp column added by Module 1
    schema = StructType([
        StructField("Timestamp",               StringType(),  True),
        StructField("Source IP Address",        StringType(),  True),
        StructField("Destination IP Address",   StringType(),  True),
        StructField("Source Port",              IntegerType(), True),
        StructField("Destination Port",         IntegerType(), True),
        StructField("Protocol",                 StringType(),  True),
        StructField("Packet Length",            IntegerType(), True),
        StructField("Packet Type",              StringType(),  True),
        StructField("Traffic Type",             StringType(),  True),
        StructField("Payload Data",             StringType(),  True),
        StructField("Malware Indicators",       StringType(),  True),
        StructField("Anomaly Scores",           DoubleType(),  True),
        StructField("Alerts/Warnings",          StringType(),  True),
        StructField("Attack Type",              StringType(),  True),
        StructField("Attack Signature",         StringType(),  True),
        StructField("Action Taken",             StringType(),  True),
        StructField("Severity Level",           StringType(),  True),
        StructField("User Information",         StringType(),  True),
        StructField("Device Information",       StringType(),  True),
        StructField("Network Segment",          StringType(),  True),
        StructField("Geo-location Data",        StringType(),  True),
        StructField("Proxy Information",        StringType(),  True),
        StructField("Firewall Logs",            StringType(),  True),
        StructField("IDS/IPS Alerts",           StringType(),  True),
        StructField("Log Source",               StringType(),  True),
        StructField("Emit_Timestamp",           StringType(),  True),
    ])

    os.makedirs(LOG_DIR,        exist_ok=True)
    os.makedirs(PARQUET_SINK,   exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print(f"Reading streaming logs from : {LOG_DIR}/")
    print(f"Writing Parquet sink to     : {PARQUET_SINK}/")
    print(f"Checkpoint dir              : {CHECKPOINT_DIR}/\n")

    streaming_df = (
        spark.readStream
        .schema(schema)
        .option("header", "true")
        .csv(LOG_DIR + "/")
    )

    batch_counter = [0]   # mutable counter accessible inside closure

    def process_batch(batch_df: DataFrame, batch_id: int):
        count = batch_df.count()
        batch_counter[0] += 1
        print(
            f"[Batch #{batch_counter[0]:04d} | id={batch_id}] "
            f"Rows received: {count:,} — writing to Parquet..."
        )
        (
            batch_df.write
            .mode("append")
            .parquet(PARQUET_SINK)
        )
        print(f"  ✅ Batch #{batch_counter[0]:04d} written.")

    query = (
        streaming_df.writeStream
        .foreachBatch(process_batch)
        .option("checkpointLocation", CHECKPOINT_DIR)
        .start()
    )

    query.awaitTermination()


if __name__ == "__main__":
    start_ingestion()
