
'''
python labeler_macd_v01.py \
  --infile data_ohlcv_sample/instr_15/2026-05-14.csv \
  --outfile test_labels/2026-05-14_labeled.csv \
  --confirm-bars 8 \
  --horizon 12 \
  --weak_thresh 0.02 \
  --spread-threshold-atr 0.1 \
  --growth-threshold-atr 0.1 \
  --resample 1
'''
# labeler_macd_v01.py
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

def add_indicators(df):
    """Calculates all essential baseline data and the new high/low MACD 
    derivatives to handle structural validation layers.
    """
    out = df.copy()
    eps = 1e-9

    prev_close = out["close"].shift(1)
    tr = pd.concat(
        [
            out["high"] - out["low"],
            (out["high"] - prev_close).abs(),
            (out["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["atr_14"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    atr_safe = out["atr_14"].replace(0, np.nan)

    # Standard MACD (Close-based)
    ema_fast = out["close"].ewm(span=12, adjust=False).mean()
    ema_slow = out["close"].ewm(span=26, adjust=False).mean()
    out["macd_line"] = ema_fast - ema_slow
    out["macd_signal"] = out["macd_line"].ewm(span=9, adjust=False).mean()
    out["macd_spread"] = out["macd_line"] - out["macd_signal"]
    out["macd_spread_atr"] = out["macd_spread"] / (atr_safe + eps)

    # New Structural Filter Derivatives: MACD derived from High and Low streams
    ema_fast_hi = out["high"].ewm(span=12, adjust=False).mean()
    ema_slow_hi = out["high"].ewm(span=26, adjust=False).mean()
    out["macd_line_hi"] = (ema_fast_hi - ema_slow_hi)/(atr_safe + eps)

    ema_fast_lo = out["low"].ewm(span=12, adjust=False).mean()
    ema_slow_lo = out["low"].ewm(span=26, adjust=False).mean()
    out["macd_line_lo"] = (ema_fast_lo - ema_slow_lo)/(atr_safe + eps)

    return out


def classify_future_macd(
    signal_bar,
    future_bars,
    confirm_bars,
    weak_thresh,
    spread_threshold_atr,
    growth_threshold_atr,
):
    """Classify future window sequences while enforcing baseline strategy logic 
    and underlying momentum boundaries via high/low structural lines.
    """
    atr_value = float(signal_bar["atr_14"])
    if not np.isfinite(atr_value) or atr_value <= 0:
        return None

    window = future_bars.iloc[:confirm_bars].copy()
    if len(window) < confirm_bars:
        return None

    spread_atr = window["macd_spread_atr"].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(spread_atr)):
        return None
        
    # --------------------------------------------------
    # Strategy Metric Computations
    # --------------------------------------------------
    first_spread_atr = float(spread_atr[0])
    last_spread_atr = float(spread_atr[-1])
    spread_growth_atr = last_spread_atr - first_spread_atr

    # 1. Base Strategy Confirmations
    up_spread_ok = np.mean(spread_atr >= spread_threshold_atr) >= 0.65
    down_spread_ok = np.mean(spread_atr <= -spread_threshold_atr) >= 0.65

    up_growth_ok = spread_growth_atr >= growth_threshold_atr
    down_growth_ok = spread_growth_atr <= -growth_threshold_atr

    weak_spread_up = spread_atr < weak_thresh
    weak_spread_down = spread_atr > -weak_thresh

    # 2. Strict Risk Filters
    is_weak_up = weak_spread_up.any()
    is_weak_down = weak_spread_down.any()

    # 3. New Momentum Regime Layer (Evaluated at the Signal Entry Bar)
    # Ensures absolute structural alignment before validating directional classification
    macd_hi_now = float(signal_bar["macd_line_hi"])
    macd_lo_now = float(signal_bar["macd_line_lo"])

    regime_up_ok = (macd_hi_now > 2.5) and (macd_lo_now > 2)
    regime_down_ok = (macd_hi_now < -1) and (macd_lo_now < -1)

    # 4. Integrated Validation Gates
    # baseline
    up_valid = (up_spread_ok or up_growth_ok or regime_up_ok) and (not is_weak_up) 
    down_valid = (down_spread_ok or down_growth_ok or regime_down_ok) and (not is_weak_down) 

    

    # 5. Label Assignment
    if up_valid and not down_valid:
        return LABEL_UP
    if down_valid and not up_valid:
        return LABEL_DOWN
        
    return LABEL_FLAT


def build_labels(df, confirm_bars, weak_thresh, spread_threshold_atr, growth_threshold_atr, horizon):
    rows = []

    # Pre-compute the updated indicators map containing the structural parameters
    df_calculated = add_indicators(df)

    for i in range(len(df_calculated) - horizon - 1):
        signal_bar = df_calculated.iloc[i]
        future_bars = df_calculated.iloc[i + 1 : i + horizon + 1]
        
        if future_bars.empty:
            continue

        label = classify_future_macd(
            signal_bar=signal_bar,
            future_bars=future_bars,
            confirm_bars=confirm_bars,
            weak_thresh=weak_thresh,
            spread_threshold_atr=spread_threshold_atr,
            growth_threshold_atr=growth_threshold_atr,
        )
        
        if label is None:
            continue

        rows.append(
            {
                "ts": signal_bar["ts"],
                "label": int(label),
            }
        )

    return pd.DataFrame(rows, columns=["ts", "label"])

def resample_bars(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """
    Resample 1-minute OHLCV data into N-minute bars.

    Parameters
    ----------
    minutes : int
        Size of each bar (2, 3, 5, ...)

    Bars are aligned to the NSE market open (09:15 IST = 03:45 UTC).

    Examples
    --------
    minutes=2
        03:45
        03:47
        03:49
        ...

    minutes=3
        03:45
        03:48
        03:51
        ...

    minutes=5
        03:45
        03:50
        03:55
        ...
    """

    df = df.copy()

    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values("ts")
    df = df.set_index("ts")

    rule = f"{minutes}min"

    resampled = (
        df.resample(
            rule,
            origin="start_day",
            offset="3h45min",
            label="left",
            closed="left",
        )
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
    )

    # Remove incomplete bars
    resampled = resampled.dropna(
        subset=["open", "high", "low", "close"]
    )

    # Keep only complete N-minute bars
    counts = (
        df["close"]
        .resample(
            rule,
            origin="start_day",
            offset="3h45min",
            label="left",
            closed="left",
        )
        .count()
    )

    resampled = resampled[counts == minutes]

    return resampled.reset_index()


def main():
    parser = argparse.ArgumentParser(
        description="Generate 3-class labels from future MACD spread confirmation."
    )
    parser.add_argument("--infile", required=True)
    parser.add_argument("--outfile", required=True)
    parser.add_argument("--resample",  type=int, default=1,
                        help="Resample to N-minute bars before feature engineering. "
                             "Default 1 = no resampling (original 1-minute bars).")
    parser.add_argument("--confirm-bars", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument(
        "--weak_thresh",
        type=float,
        default=0.02
    )

    parser.add_argument(
        "--spread-threshold-atr",
        type=float,
        default=0.02,
        help="Future MACD spread threshold normalized by per-bar ATR.",
    )
    parser.add_argument(
        "--growth-threshold-atr",
        type=float,
        default=0.01,
        help="Required MACD spread expansion/contraction normalized by per-bar ATR.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.infile)
    df = normalize_df(df)
    df = add_indicators(df)
    
    # ── Optional resampling ──────────────────────────────────────────────
    if args.resample > 1:
        original_bars = len(df)
        df = resample_bars(df, args.resample)
        
        print(f"[v4_2] Resampled {original_bars} × 1-min bars "
              f"→ {len(df)} × {args.resample}-min bars")

    

    df = add_indicators(df)
    

    labeled = build_labels(
        df=df,
        confirm_bars=args.confirm_bars,
        weak_thresh=args.weak_thresh,
        spread_threshold_atr=args.spread_threshold_atr,
        growth_threshold_atr=args.growth_threshold_atr,
        horizon=args.horizon,
    )

    os.makedirs(os.path.dirname(args.outfile) or ".", exist_ok=True)
    labeled.to_csv(args.outfile, index=False)

    if labeled.empty:
        print("No labels were generated.")
        return

    print(f"Saved {len(labeled)} labeled rows -> {args.outfile}")
    print(labeled["label"].value_counts(dropna=False).sort_index())



if __name__ == "__main__":
    main()
