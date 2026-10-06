


'''
USAGE:

python batch_feature_generator_v01.py \
    --raw-root data_ohlcv_sample \
    --out-root demo_data/data_feats_v01 \
    --script features_engineering_v01.py \
    --start-date 2026-08-22 \
    --end-date 2026-08-31 \
    --resample 1
'''

# batch_feature_generator_v01.py

import os
import sys
import glob
import subprocess
import argparse
import re
import pandas as pd
from datetime import datetime
import tempfile


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def find_instruments(raw_root):
    return [
        name for name in os.listdir(raw_root)
        if os.path.isdir(os.path.join(raw_root, name))
    ]


def extract_date_from_filename(path):
    """
    Assumes filename contains YYYY-MM-DD
    """
    m = re.search(r"(\d{4}-\d{2}-\d{2})", path)
    if not m:
        return None

    return datetime.strptime(m.group(1), "%Y-%m-%d").date()


def find_previous_file(raw_files, current_file):
    """
    Return the immediately preceding available trading-day file.

    raw_files must already be sorted chronologically.
    """
    try:
        idx = raw_files.index(current_file)
    except ValueError:
        return None

    if idx == 0:
        return None

    return raw_files[idx - 1]


def combine_context_files(previous_file, current_file, temp_dir):
    """
    Combine previous trading day + current trading day into
    one temporary CSV for continuous feature calculation.
    """

    prev_df = pd.read_csv(previous_file)
    curr_df = pd.read_csv(current_file)

    combined = pd.concat(
        [prev_df, curr_df],
        ignore_index=True
    )

    # Normalize timestamp column name if necessary
    if "timestamp" in combined.columns and "ts" not in combined.columns:
        combined = combined.rename(columns={"timestamp": "ts"})

    if "ts" in combined.columns:
        combined["ts"] = pd.to_datetime(
            combined["ts"],
            errors="coerce"
        )

        combined = (
            combined
            .sort_values("ts")
            .drop_duplicates("ts")
            .reset_index(drop=True)
        )

    basename = os.path.splitext(
        os.path.basename(current_file)
    )[0]

    temp_file = os.path.join(
        temp_dir,
        f"{basename}_context.csv"
    )

    combined.to_csv(temp_file, index=False)

    return temp_file


def main():

    p = argparse.ArgumentParser()

    p.add_argument(
        "--raw-root",
        default="data_raw"
    )

    p.add_argument(
        "--out-root",
        default="data_features"
    )

    p.add_argument(
        "--resample",
        type=int,
        default=1,
        help="Resample to N-minute bars before feature engineering. "
             "Default 1 = no resampling."
    )

    p.add_argument(
        "--script",
        required=True,
        help="Feature generation script"
    )

    p.add_argument(
        "--start-date",
        required=True,
        help="YYYY-MM-DD"
    )

    p.add_argument(
        "--end-date",
        required=True,
        help="YYYY-MM-DD"
    )

    p.add_argument(
        "--force",
        action="store_true"
    )

    args = p.parse_args()

    start_date = datetime.strptime(
        args.start_date,
        "%Y-%m-%d"
    ).date()

    end_date = datetime.strptime(
        args.end_date,
        "%Y-%m-%d"
    ).date()

    raw_root = os.path.abspath(
        os.path.expanduser(args.raw_root)
    )

    out_root = os.path.abspath(
        os.path.expanduser(args.out_root)
    )

    if not os.path.isdir(raw_root):
        print("Raw root not found:", raw_root)
        sys.exit(1)

    instruments = find_instruments(raw_root)

    if not instruments:
        print("No instruments found under", raw_root)
        sys.exit(1)

    # Temporary directory for combined previous-day + current-day files
    with tempfile.TemporaryDirectory(
        prefix="feature_context_"
    ) as temp_dir:

        for instr in instruments:

            raw_dir = os.path.join(
                raw_root,
                instr
            )

            out_dir = os.path.join(
                out_root,
                instr
            )

            ensure_dir(out_dir)

            raw_files = sorted(
                glob.glob(
                    os.path.join(
                        raw_dir,
                        "*.csv"
                    )
                )
            )

            if not raw_files:
                continue

            print(
                f"\n========== {instr} =========="
            )

            for raw in raw_files:

                file_date = extract_date_from_filename(
                    raw
                )

                if file_date is None:
                    continue

                if not (
                    start_date
                    <= file_date
                    <= end_date
                ):
                    continue

                basename = os.path.splitext(
                    os.path.basename(raw)
                )[0]

                out_file = os.path.join(
                    out_dir,
                    f"{basename}_features.csv"
                )

                if (
                    os.path.exists(out_file)
                    and not args.force
                ):
                    print(
                        "Skipping existing:",
                        out_file
                    )
                    continue

                previous_file = find_previous_file(
                    raw_files,
                    raw
                )

                # -------------------------------------------------
                # First available day:
                # no previous-day context
                # -------------------------------------------------
                if previous_file is None:

                    input_file = raw

                    print(
                        "Processing:",
                        raw,
                        "(no previous-day context)"
                    )

                # -------------------------------------------------
                # Normal case:
                # previous trading day + current day
                # -------------------------------------------------
                else:

                    input_file = combine_context_files(
                        previous_file,
                        raw,
                        temp_dir
                    )

                    print(
                        "Processing:",
                        raw
                    )

                    print(
                        "  Context:",
                        os.path.basename(previous_file)
                    )

                cmd = [
                    sys.executable,
                    args.script,
                    "--infile",
                    input_file,
                    "--outfile",
                    out_file,
                    "--resample",
                    str(args.resample),
                ]

                r = subprocess.run(cmd)

                if r.returncode != 0:
                    print(
                        "ERROR:",
                        raw
                    )
                    continue

                # -------------------------------------------------
                # IMPORTANT:
                # The feature engine calculated features using
                # previous-day context + current day.
                #
                # Now retain only the current trading day.
                # -------------------------------------------------

                try:

                    features = pd.read_csv(
                        out_file
                    )

                    if "ts" not in features.columns:
                        print(
                            "ERROR: output has no ts column:",
                            out_file
                        )
                        continue

                    features["ts"] = pd.to_datetime(
                        features["ts"],
                        errors="coerce"
                    )

                    features = features[
                        features["ts"].dt.date
                        == file_date
                    ].copy()

                    features.to_csv(
                        out_file,
                        index=False
                    )

                    print(
                        f"  Saved current-day features: "
                        f"{len(features)} rows"
                    )

                except Exception as e:

                    print(
                        "ERROR filtering output:",
                        out_file,
                        e
                    )

    print("\n✅ Batch processing complete.")


if __name__ == "__main__":
    main()