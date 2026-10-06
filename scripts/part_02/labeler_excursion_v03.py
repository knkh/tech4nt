

'''
python labeler_excursion_v03.py \
  --infile data_ohlcv_sample/instr_15/2026-05-14.csv \
  --outfile "test/excursion_labeled.csv" \
  --horizon 12 \
  --mfe-threshold-atr 1.2 \
  --mae-threshold-atr 0.7 \
  --end-ret-threshold-atr 0.2 \
  --persistence-threshold 0.55

'''
# How to tune it
'''
How to tune it
If too many rows become FLAT:

lower mfe-threshold-atr to 1.0
raise mae-threshold-atr to 0.8
lower persistence-threshold to 0.50
If labels are too noisy:

raise mfe-threshold-atr
lower mae-threshold-atr
raise persistence-threshold
'''

# labeler_excursion_quality.py
import argparse
import os

import numpy as np
import pandas as pd


LABEL_DOWN = 0
LABEL_FLAT = 1
LABEL_UP = 2

REQUIRED = {"ts", "open", "high", "low", "close"}


def normalize_df(df):
    if "timestamp" in df.columns and "ts" not in df.columns:
        df = df.rename(columns={"timestamp": "ts"})

    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
    return df


def compute_atr_series(df, period=14):
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    return atr


def classify_window(
    entry_price,
    atr_value,
    future_window,
    mfe_threshold_atr,
    mae_threshold_atr,
    end_ret_threshold_atr,
    persistence_threshold,
):
    if not np.isfinite(entry_price) or not np.isfinite(atr_value) or atr_value <= 0:
        return None

    highs = future_window["high"].to_numpy(dtype=np.float64)
    lows = future_window["low"].to_numpy(dtype=np.float64)
    closes = future_window["close"].to_numpy(dtype=np.float64)

    if len(closes) == 0:
        return None

    mfe_up = (np.max(highs) - entry_price) / (atr_value + 1e-9)
    mae_down = (entry_price - np.min(lows)) / (atr_value + 1e-9)
    end_ret = (closes[-1] - entry_price) / (atr_value + 1e-9)
    up_persistence = float(np.mean(closes > entry_price))
    down_persistence = float(np.mean(closes < entry_price))

    up_valid = (
        mfe_up >= mfe_threshold_atr
        and mae_down <= mae_threshold_atr
        and end_ret >= end_ret_threshold_atr
        and up_persistence >= persistence_threshold
    )

    down_valid = (
        mae_down >= mfe_threshold_atr
        and mfe_up <= mae_threshold_atr
        and end_ret <= -end_ret_threshold_atr
        and down_persistence >= persistence_threshold
    )

    if up_valid and not down_valid:
        return LABEL_UP
    if down_valid and not up_valid:
        return LABEL_DOWN
    return LABEL_FLAT


def build_labels(
    df,
    horizon,
    mfe_threshold_atr,
    mae_threshold_atr,
    end_ret_threshold_atr,
    persistence_threshold,
):
    rows = []

    for i in range(len(df) - horizon - 1):
        signal_bar = df.iloc[i]
        future_window = df.iloc[i + 1 : i + horizon + 1]

        if future_window.empty:
            continue

        entry_price = float(signal_bar["close"])
        atr_value = float(signal_bar["atr_14"]) if pd.notna(signal_bar["atr_14"]) else np.nan

        label_dir = classify_window(
            entry_price=entry_price,
            atr_value=atr_value,
            future_window=future_window,
            mfe_threshold_atr=mfe_threshold_atr,
            mae_threshold_atr=mae_threshold_atr,
            end_ret_threshold_atr=end_ret_threshold_atr,
            persistence_threshold=persistence_threshold,
        )

        if label_dir is None:
            continue

        rows.append(
            {
                "ts": signal_bar["ts"],
                "label_dir": int(label_dir),
            }
        )

    return pd.DataFrame(rows, columns=["ts", "label_dir"])


def main():
    parser = argparse.ArgumentParser(
        description="Generate 3-class labels from excursion quality and persistence."
    )
    parser.add_argument("--infile", required=True)
    parser.add_argument("--outfile", required=True)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--mfe-threshold-atr", type=float, default=1.2)
    parser.add_argument("--mae-threshold-atr", type=float, default=0.7)
    parser.add_argument("--end-ret-threshold-atr", type=float, default=0.2)
    parser.add_argument("--persistence-threshold", type=float, default=0.55)
    args = parser.parse_args()

    df = pd.read_csv(args.infile)
    df = normalize_df(df)
    df["atr_14"] = compute_atr_series(df, period=14)

    labeled = build_labels(
        df=df,
        horizon=args.horizon,
        mfe_threshold_atr=args.mfe_threshold_atr,
        mae_threshold_atr=args.mae_threshold_atr,
        end_ret_threshold_atr=args.end_ret_threshold_atr,
        persistence_threshold=args.persistence_threshold,
    )

    os.makedirs(os.path.dirname(args.outfile) or ".", exist_ok=True)
    labeled.to_csv(args.outfile, index=False)

    if labeled.empty:
        print("No labels were generated.")
        return

    print(f"Saved {len(labeled)} labeled rows -> {args.outfile}")
    print(labeled["label_dir"].value_counts(dropna=False).sort_index())


if __name__ == "__main__":
    main()
