


'''
Here is the complete list of all 12 data quality checks performed by the updated validate_ohlcv_data.py script:
1. File & Schema Integrity Checks
	1.	Required Columns Check: Verifies that the CSV file contains all required OHLCV columns (timestamp, open, high, low, close, volume).
	2.	Empty File Detection: Flags files that are completely empty or contain no data rows.
	3.	Missing Value (NaN / Null) Check: Scans all required columns for missing or null values and reports exact counts per column.
2. Temporal & Timezone Checks (UTC)
	1.	UTC Timestamp Parsing Check: Attempts to parse all timestamps specifically as UTC (pd.to_datetime(..., utc=True)) and flags unparseable (NaT) dates.
	2.	Chronological Sorting Check: Confirms that timestamps are strictly sorted in ascending chronological order from past to present (is_monotonic_increasing).
	3.	Duplicate Timestamp Detection: Flags repeated/duplicate minute timestamps occurring within the same file.
	4.	Future Timestamp Check: Ensures no record extends beyond the current system UTC time.
	5.	NSE Market Hours Check (UTC): Validates that all intraday bar timestamps fall strictly 
    within standard NSE trading hours converted to UTC (03:45 UTC to 09:45 UTC, corresponding 
    to 09:15 to 15:30 IST).
    . Price & High/Low Sanity Checks
	1.	OHLC Price Hierarchy & Positive Value Check:
⚬	High must be greater than or equal to Open, Close, and Low (High >= Open/Close/Low).
⚬	Low must be less than or equal to Open, Close, and High (Low <= Open/Close/High).
⚬	All price values (open, high, low, close) must be strictly positive numbers (‭$> 0$‬).
4. Volume & Microstructure Checks
	1.	Negative Volume Check: Flags any row where volume drops below zero (volume < 0).
	2.	Cumulative Volume Pattern Detection: Checks if raw intraday volume decreases repeatedly across sequential bars (which indicates cumulative day volume was recorded instead of per-minute bar volume).
	3.	Flat Candle Filter: Flags files where more than 20% of rows have zero price movement (high == low) while actively reporting trading volume 
    (volume > 0).
    5. Anomaly & Outlier Checks
⚬	Extreme Price Spike Filter: Flags suspicious 1-minute bars where closing price moves by more than 10% in a single bar.
'''


# validate_ohlcv _data.py
import os
import numpy as np
import pandas as pd

# DATA_ROOT = "data_ohlcv_sample"
DATA_ROOT = os.path.expanduser(
    "~/path/to/my_project/data_ohlcv_sample"
)

def validate_file(path):
    try:
        # Load CSV
        df = pd.read_csv(path)
        issues = []

        # --- Check 1: Required Columns Presence ---
        required_cols = {"timestamp", "open", "high", "low", "close", "volume"}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            return [f"Missing required columns: {missing_cols}"]

        # --- Check 2: Empty File Check ---
        if df.empty:
            return ["File is completely empty"]

        # --- Check 3: Missing (NaN / Null) Values Check ---
        nan_cols = df[list(required_cols)].isna().sum()
        cols_with_nan = nan_cols[nan_cols > 0]
        if not cols_with_nan.empty:
            nan_summary = ", ".join([f"{col}:{cnt}" for col, cnt in cols_with_nan.items()])
            issues.append(f"Contains NaN values ({nan_summary})")

        # Drop NaNs temporarily for mathematical evaluations below
        df_clean = df.dropna(subset=list(required_cols)).copy()
        if df_clean.empty:
            return issues + ["No valid rows remaining after dropping NaNs"]

        # --- Check 4: Parsing and UTC Timezone Validation ---
        df_clean["timestamp"] = pd.to_datetime(df_clean["timestamp"], errors="coerce", utc=True)

        if df_clean["timestamp"].isna().any():
            issues.append("Unparseable (NaT) timestamps detected")

        # --- Check 5: Chronological Ordering ---
        if not df_clean["timestamp"].is_monotonic_increasing:
            issues.append("Timestamps not strictly chronological/sorted")

        # --- Check 6: Duplicate Timestamps Check ---
        duplicate_ts = df_clean["timestamp"].duplicated().sum()
        if duplicate_ts > 0:
            issues.append(f"Found {duplicate_ts} duplicate timestamps")

        # --- Check 7: Future Timestamp Check ---
        now_utc = pd.Timestamp.now(tz="UTC")
        if (df_clean["timestamp"] > now_utc + pd.Timedelta(minutes=1)).any():
            issues.append("Future timestamps detected")

        # --- Check 8: NSE Market Hours Check in UTC (03:45 to 09:45 UTC) ---
        # IST 09:15 to 15:30 maps directly to UTC 03:45 to 09:45
        times_utc = df_clean["timestamp"].dt.time
        utc_start = pd.to_datetime("03:45").time()
        utc_end = pd.to_datetime("09:45").time()

        out_of_hours = df_clean[(times_utc < utc_start) | (times_utc > utc_end)]
        if len(out_of_hours) > 0:
            issues.append(f"{len(out_of_hours)} rows outside normal NSE market hours (03:45-09:45 UTC)")

        # --- Check 9: OHLC Price Hierarchy & Sanity ---
        # High must be >= Open, Close, Low
        if not (df_clean["high"] >= df_clean[["open", "close"]].max(axis=1)).all():
            issues.append("High < Open or Close")

        if not (df_clean["high"] >= df_clean["low"]).all():
            issues.append("High < Low")

        # Low must be <= Open, Close, High
        if not (df_clean["low"] <= df_clean[["open", "close"]].min(axis=1)).all():
            issues.append("Low > Open or Close")

        # Non-positive or zero prices
        if (df_clean[["open", "high", "low", "close"]] <= 0).any().any():
            issues.append("Zero or negative price values detected")

        # --- Check 10: Volume Integrity & Microstructure ---
        if (df_clean["volume"] < 0).any():
            issues.append("Negative volume detected")

        # Detect cumulative volume patterns across intraday minutes
        vol_diff = df_clean["volume"].diff()
        if (vol_diff < 0).sum() > 5:
            issues.append("Volume looks cumulative or corrupted (decreases repeatedly)")

        # --- Check 11: Flat Candles Check ---
        flat = df_clean[(df_clean["high"] == df_clean["low"]) & (df_clean["volume"] > 0)]
        if len(flat) > len(df_clean) * 0.2:
            issues.append("Too many flat candles (High == Low with Volume > 0)")

        # --- Check 12: Extreme Price Outliers / Spike Detection ---
        returns = df_clean["close"].pct_change().abs()
        spikes = (returns > 0.10).sum()
        if spikes > 0:
            issues.append(f"Extreme price spike detected ({spikes} bars moved >10% in 1 minute)")

        return issues

    except Exception as e:
        return [f"Read/Processing error: {str(e)}"]


def scan_all():
    bad_files = []
    total_scanned = 0

    for root, _, files in os.walk(DATA_ROOT):
        for f in files:
            if not f.endswith(".csv"):
                continue

            total_scanned += 1
            path = os.path.join(root, f)
            issues = validate_file(path)

            if issues:
                bad_files.append((path, issues))

    print("\n" + "=" * 50)
    print("           DATA QUALITY CHECK REPORT          ")
    print("=" * 50 + "\n")
    print(f"Total CSV Files Scanned: {total_scanned}")

    if not bad_files:
        print("All files look clean ✅")
        return

    print(f"Problematic Files Found: {len(bad_files)}\n")

    for path, issues in bad_files:
        print(f"❌ {path}")
        for issue in issues:
            print(f"   - {issue}")

    print("\n" + "=" * 50)


if __name__ == "__main__":
    scan_all()


# Results: on data_ohlcv_sample
'''

==================================================
           DATA QUALITY CHECK REPORT          
==================================================

Total CSV Files Scanned: 442
Problematic Files Found: 442

❌ data_ohlcv_sample/instr_05/2026-09-09.csv
   - 10 rows outside normal NSE market hours (03:45-09:45 UTC)
   - Volume looks cumulative or corrupted (decreases repeatedly)
❌ data_ohlcv_sample/instr_05/2026-08-17.csv
   - 9 rows outside normal NSE market hours (03:45-09:45 UTC)
   - Volume looks cumulative or corrupted (decreases repeatedly)
   ...
   ..
   ❌ data_ohlcv_sample/instr_02/2025-12-17.csv
   - Missing required columns: {'timestamp'}
❌ data_ohlcv_sample/instr_02/2025-12-16.csv
   - Missing required columns: {'timestamp'}
❌ data_ohlcv_sample/instr_02/2025-12-15.csv
   - Missing required columns: {'timestamp'}
....
.


'''