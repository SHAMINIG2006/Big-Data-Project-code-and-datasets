"""
Module 1: Log Generator
-----------------------
Streams rows from cybersecurity_attacks.csv into logs/live_logs.csv.
Emits BATCH_SIZE rows per tick (default 5) with a real wall-clock
Emit_Timestamp column so downstream modules can do time-based analysis.

Run from project root:
    py -3 modules/module1_log_generator/log_generator.py
"""

import pandas as pd
import time
import os
from datetime import datetime

# Resolve project root relative to this file so script works from any directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Configuration
INPUT_CSV   = os.path.join(PROJECT_ROOT, "cybersecurity_attacks.csv")
OUTPUT_LOGS = os.path.join(PROJECT_ROOT, "logs", "live_logs.csv")
DELAY       = 1.0   # seconds between each batch
BATCH_SIZE  = 5     # rows written per tick (demonstrates throughput)


def stream_logs():
    print(f"Starting log generator.")
    print(f"  Source : {INPUT_CSV}")
    print(f"  Sink   : {OUTPUT_LOGS}")
    print(f"  Batch  : {BATCH_SIZE} rows / {DELAY}s tick\n")

    os.makedirs(os.path.dirname(OUTPUT_LOGS), exist_ok=True)

    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"Error: {INPUT_CSV} not found.")
        return

    print(f"Dataset loaded: {len(df):,} rows, {len(df.columns)} columns")

    # Initialise output file with header (include extra Emit_Timestamp column)
    header_df = df.head(0).copy()
    header_df["Emit_Timestamp"] = pd.Series(dtype="object")
    header_df.to_csv(OUTPUT_LOGS, index=False)

    batch_num  = 0
    total_sent = 0

    while True:
        df_shuffled = df.sample(frac=1).reset_index(drop=True)

        for start in range(0, len(df_shuffled), BATCH_SIZE):
            batch = df_shuffled.iloc[start : start + BATCH_SIZE].copy()
            now   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            batch["Emit_Timestamp"] = now

            batch.to_csv(OUTPUT_LOGS, mode="a", header=False, index=False)

            batch_num  += 1
            total_sent += len(batch)

            # Print one summary line per batch instead of one line per row
            attack_sample = batch["Attack Type"].iloc[0]
            ip_sample     = batch["Source IP Address"].iloc[0]
            print(
                f"[{now}] Batch #{batch_num:04d} | "
                f"+{len(batch)} rows (total {total_sent:,}) | "
                f"sample: {attack_sample} from {ip_sample}"
            )
            time.sleep(DELAY)


if __name__ == "__main__":
    try:
        stream_logs()
    except KeyboardInterrupt:
        print("\nLog generator stopped.")
