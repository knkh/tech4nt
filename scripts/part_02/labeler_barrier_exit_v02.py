
# 👉 Usage:
'''
python labeler_barrier_exit_v02.py \
    --infile  data_ohlcv_sample/instr_15/2026-05-14.csv \
    --outfile "test/barexit_labeled.csv" \
    --horizon 12 \
    --sl_atr 1.2 \
    --tp_atr 1.2 \
    --sl_pct 0.01 \
    --tp_pct 0.01 \
    --min_range 0.003

'''
# labeler_barrier_exit_v3.py

import argparse
import os
import numpy as np
import pandas as pd


LABEL_DOWN = 0
LABEL_FLAT = 1
LABEL_UP = 2


# ==========================================================
# Helpers
# ==========================================================

def normalize_df(df):

    if "timestamp" in df.columns and "ts" not in df.columns:
        df = df.rename(columns={"timestamp": "ts"})

    required = {"ts","open","high","low","close"}

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.copy()

    df["ts"] = pd.to_datetime(df["ts"])

    df = (
        df.sort_values("ts")
          .drop_duplicates("ts")
          .reset_index(drop=True)
    )

    return df


def compute_atr(df, period=14):

    prev_close = df["close"].shift(1)

    tr = pd.concat(
        [
            df["high"]-df["low"],
            (df["high"]-prev_close).abs(),
            (df["low"]-prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.ewm(alpha=1/period, adjust=False).mean()

    return atr


# ==========================================================
# Label builder
# ==========================================================



def compute_trend_label(close_arr, short=9, long=21):
    """
    Returns a per-bar trend integer:
        +1  = bullish  (EMA_short > EMA_long)
         0  = flat / unclear
        -1  = bearish  (EMA_short < EMA_long)
    Uses only past data — no lookahead.
    """
    s = pd.Series(close_arr)
    ema_short = s.ewm(span=short, adjust=False).mean().values
    ema_long  = s.ewm(span=long,  adjust=False).mean().values
    gap_pct   = (ema_short - ema_long) / ema_long          # normalised gap
    trend     = np.where(gap_pct >  0.001,  1,
                np.where(gap_pct < -0.001, -1, 0))
    return trend

def build_barrier_exit_labels(
    df,
    horizon              = 12,
    tp_pct               = 0.010,
    sl_pct               = 0.007,
    tp_atr               = 1.50,
    sl_atr               = 1.00,
    min_range            = 0.005,
    trend_short          = 9,
    trend_long           = 21,
    both_hit_min_ret     = 0.010,
    suppress_on_flat_trend = False,
    # trend_filter REMOVED — it is always applied
):
    
    df    = df.copy().reset_index(drop=True)
    atr   = compute_atr(df).values
    close = df["close"].values
    high  = df["high"].values
    low   = df["low"].values
    n     = len(df)

    trend = compute_trend_label(close, trend_short, trend_long)

    labels = np.full(n, LABEL_FLAT, dtype=np.int8)

    for i in range(n - horizon):

        entry   = close[i]
        atr_now = atr[i]

        if not np.isfinite(atr_now) or atr_now <= 0:
            labels[i] = LABEL_FLAT
            continue

        tp_move  = max(tp_pct * entry, tp_atr * atr_now)
        sl_move  = max(sl_pct * entry, sl_atr * atr_now)
        tp_price = entry + tp_move
        sl_price = entry - sl_move

        label       = LABEL_FLAT
        barrier_hit = False
        t           = trend[i]

        for step in range(1, horizon + 1):
            idx    = i + step
            tp_hit = high[idx] >= tp_price
            sl_hit = low[idx]  <= sl_price

            if tp_hit and sl_hit:
                # ── Both barriers — trend + return must agree ─────────────
                candle_ret = (close[idx] - entry) / entry
                
                if t == 1 and candle_ret >= both_hit_min_ret:
                    label = LABEL_UP
                elif t == -1 and candle_ret <= -both_hit_min_ret:
                    label = LABEL_DOWN
                else:
                    label = LABEL_FLAT   # ambiguous or counter-trend
                barrier_hit = True    # <── without this, noise floor runs on both-hit bars
                break
            elif tp_hit:
                if t == 1:
                    label = LABEL_UP
                elif t == -1:
                    label = LABEL_FLAT
                else:
                    label = LABEL_FLAT if suppress_on_flat_trend else LABEL_UP
                barrier_hit = True
                break

            elif sl_hit:
                if t == -1:
                    label = LABEL_DOWN
                elif t == 1:
                    label = LABEL_FLAT
                else:
                    label = LABEL_FLAT if suppress_on_flat_trend else LABEL_DOWN
                barrier_hit = True
                break


        # ── Noise floor (no barrier hit only) ────────────────────────────
        if not barrier_hit:
            future_high = np.max(high[i + 1 : i + horizon + 1])
            future_low  = np.min(low[i  + 1 : i + horizon + 1])
            excursion   = max(
                (future_high - entry) / entry,
                (entry - future_low)  / entry,
            )
            if excursion < min_range:
                label = LABEL_FLAT

        labels[i] = label

    df = df.iloc[: n - horizon].copy()
    df["label"] = labels[: n - horizon]

    return df[["ts", "label"]]


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--infile",required=True)

    parser.add_argument("--outfile",required=True)

    parser.add_argument("--horizon",type=int,default=12)

    parser.add_argument("--tp_pct",type=float,default=0.010)

    parser.add_argument("--sl_pct",type=float,default=0.007)

    parser.add_argument("--tp_atr",type=float,default=1.5)

    parser.add_argument("--sl_atr",type=float,default=1.0)

    parser.add_argument("--min_range",type=float,default=0.005)

    return parser.parse_args()
def main():

    args = parse_args()

    df = pd.read_csv(os.path.expanduser(args.infile))

    df = normalize_df(df)

    labeled = build_barrier_exit_labels(

        df=df,

        horizon=args.horizon,

        tp_pct=args.tp_pct,

        sl_pct=args.sl_pct,

        tp_atr=args.tp_atr,

        sl_atr=args.sl_atr,

        min_range=args.min_range,

    )

    os.makedirs(
        os.path.dirname(args.outfile),
        exist_ok=True,
    )

    labeled.to_csv(args.outfile,index=False)

    print(f"Saved {len(labeled)} rows")

    print(labeled["label"].value_counts().sort_index())


if __name__=="__main__":
    main()


