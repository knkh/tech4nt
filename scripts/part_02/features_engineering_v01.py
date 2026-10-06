


'''
USAGE:

python features_engineering_v01.py \
    --infile  data_ohlcv_sample/instr_15/2026-05-14.csv \
    --outfile data_feats_v01/2026-05-14_features.csv \
    --resample 1
'''

# 👉 custom features for mixed type - futures and cash shares

# featur_engineering_v01.py

import pandas as pd
import numpy as np
import argparse
import os


MIN_LOOKBACK = 21  # drop initial rows
REQUIRED = {"ts", "open", "high", "low", "close", "volume"}
MARKET_OPEN_UTC  = pd.Timedelta(hours=3, minutes=45)   # 09:15 IST = 03:45 UTC
MARKET_CLOSE_UTC = pd.Timedelta(hours=10, minutes=0)   # 15:30 IST = 10:00 UTC
SESSION_MINUTES  = (MARKET_CLOSE_UTC - MARKET_OPEN_UTC).seconds // 60  # 390


# -------------------------
# Input validation
# -------------------------
def normalize_df(df):
    if "timestamp" in df.columns:
        df = df.rename(columns={"timestamp": "ts"})

    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError("Missing: " + str(missing))

    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    df = df.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
    return df


# -------------------------
# ATR (Wilder's smoothing, alpha = 1/period)
# -------------------------
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

# -------------------------
# Katz Fractal Dimension
# -------------------------
def katz_fd(x):
    """
    Katz Fractal Dimension

    Interpretation
    --------------
    ~1.0  -> smooth, efficient trend
    ~1.5  -> noisy market
    ~2.0  -> highly chaotic movement
    """

    x = np.asarray(x, dtype=float)

    n = len(x)

    if n < 2:
        return np.nan

    # Total path length
    L = np.sum(np.abs(np.diff(x)))

    # Maximum distance from first point
    d = np.max(np.abs(x - x[0]))

    if L <= 1e-12 or d <= 1e-12:
        return 1.0

    return np.log10(n) / (
        np.log10(n) + np.log10(d / L)
    )

# -------------------------
# 👉-FEATURE SET
# -------------------------
def add_features(df):
    df = df.copy()
    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    open_  = df["open"]
    volume = df["volume"]
    eps    = 1e-9

    atr_series = compute_atr_series(df, period=14)
    df["atr_14"] = atr_series
    atr_safe = atr_series.replace(0, np.nan)

    # ── 1. RETURNS ────────────────────────────────────────────────────────────
    df["ret_1"]  = close.pct_change()
    df["ret_5"]  = close.pct_change(5)
    # df["ret_hi_12"] = high.pct_change(12)
    # df["ret_lo_12"] = low.pct_change(12)
    df["ret_12"] = close.pct_change(12)   # matches horizon exactly; replaces ret_20

    # ── 2. VOLATILITY ─────────────────────────────────────────────────────────
    df["atr_pct"]        = atr_series / (close + eps)
    df["atr_change"]     = df["atr_pct"] - df["atr_pct"].shift(5)
    df["atr_regime"]     = df["atr_pct"] / (df["atr_pct"].rolling(20).mean() + eps)
    df["atr_percentile"] = df["atr_pct"].rolling(20).rank(pct=True)

    # ── 3. RSI & MACD ─────────────────────────────────────────────────────────
    delta    = close.diff()
    up       = delta.clip(lower=0)
    down     = -delta.clip(upper=0)
    rs       = up.rolling(14).mean() / (down.rolling(14).mean() + eps)
    df["rsi_centered"]        = 100 - (100 / (1 + rs)) - 50

    macd_fast   = close.ewm(span=12, adjust=False).mean()
    macd_slow   = close.ewm(span=26, adjust=False).mean()
    macd_spread = (macd_fast - macd_slow) - (macd_fast - macd_slow).ewm(span=9, adjust=False).mean()

    df["macd_spread_atr"]      = macd_spread / (atr_safe + eps)
    df["macd_spread_velocity"] = df["macd_spread_atr"] - df["macd_spread_atr"].shift(3)  # 8→3: faster for 12-bar horizon

    # ── 4. TREND ──────────────────────────────────────────────────────────────
    ema9  = close.ewm(span=9,  adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()

    # ema9_hi  = high.ewm(span=9,  adjust=False).mean()
    # ema9_lo  = low.ewm(span=9,  adjust=False).mean()

    df["ema_9_slope_pct"]  = (ema9  - ema9.shift(1))  / (ema9.shift(1)  + eps)
    df["close_ema9_ratio"] = close / (ema9 + eps)
    df["trend_strength_21"]= (ema21 - ema21.shift(1)) / (atr_series + eps)
    df["dist_ema21_atr"]   = (close - ema21)           / (atr_series + eps)
    df["trend_persistence"]= (ema9 > ema9.shift(1)).rolling(5).mean()  # 10→5: more responsive
    # df["hi_trend_persistence"]= (ema9_hi > ema9_hi.shift(1)).rolling(5).mean()
    # df["lo_trend_persistence"]= (ema9_lo > ema9_lo.shift(1)).rolling(5).mean()

    # ── 5. MICROSTRUCTURE ─────────────────────────────────────────────────────
    hl = high - low + eps
    df["body_ratio"]     = (close - open_) / hl
    upper_wick           = high - df[["close", "open"]].max(axis=1)
    lower_wick           = df[["close", "open"]].min(axis=1) - low
    df["wick_ratio"]     = (upper_wick - lower_wick) / hl
    df["close_location"] = (close - low) / hl
    df["last_bar_strength"] = (close - open_) / (atr_series + eps)

    # ── 6. RANGE & BREAKOUT (deduplicated) ────────────────────────────────────
    rolling_high20 = high.rolling(20).max()
    rolling_low20  = low.rolling(20).min()

    # dist_breakout and dist_to_high20 were identical — keep one
    df["dist_to_high20"]  = (rolling_high20 - close) / (atr_series + eps)
    df["dist_to_low20"]   = (close - rolling_low20)  / (atr_series + eps)
    df["day_range_pos"]   = (close - df.groupby(df["ts"].dt.date)["low"].cummin()) / \
                            (df.groupby(df["ts"].dt.date)["high"].cummax() -
                             df.groupby(df["ts"].dt.date)["low"].cummin() + eps)

    # ── 7. MEAN REVERSION ─────────────────────────────────────────────────────
    df["zscore_20"] = (close - close.rolling(20).mean()) / (close.rolling(20).std() + eps)

    # ── 8. VOLUME ─────────────────────────────────────────────────────────────
    # df["volume_ratio"] = volume / (volume.rolling(20).mean() + eps)

    # ── 9. SESSION / TIME ─────────────────────────────────────────────────────
    # Compute time-of-day in UTC; no IST hardcoding needed
    time_of_day_utc = (
        df["ts"].dt.hour * 60 + df["ts"].dt.minute
    ).apply(pd.Timedelta, unit="min")

    elapsed = (time_of_day_utc - MARKET_OPEN_UTC).dt.total_seconds() / 60
    df["session_progress"] = (elapsed / SESSION_MINUTES).clip(0, 1)

    #day_open = df.groupby(df["ts"].dt.date)["open"].transform("first")
    #df["gap_from_day_open"] = (close - day_open) / (atr_series + eps)

    # =========== 10. Volatility Compression ⭐⭐⭐⭐⭐ ============
    # =====  < 1 means volatality contracting , > 1 means expanding ======
    atr_short = atr_series.rolling(14).mean()
    atr_long  = atr_series.rolling(30).mean()

    df["volatility_compression"] = atr_short / (atr_long + eps)

    # =========== 11. Bollinger Band Width ⭐⭐⭐⭐ ========
    # ==== measures market compression ====================
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()

    upper = bb_mid + 2 * bb_std
    lower = bb_mid - 2 * bb_std

    df["bb_width"] = (upper - lower) / (bb_mid + eps)

    # ========== 12. Kaufman Efficiency Ratio ⭐⭐⭐⭐⭐ =====
    #==== 1.0 means perfect trend, 0.0 means pure noise =====
    window = 20

    direction = (
        close - close.shift(window)
    ).abs()

    volatility = (
        close.diff().abs()
        .rolling(window)
        .sum()
    )

    df["efficiency_ratio"] = (
        direction /
        (volatility + eps)
    )

    #======= 13. Katz Fractal Dimension. ⭐⭐⭐⭐⭐ ==============
    #------- measures path complexity---------------------------
    
    window = 20

    df["fractal_dimension"] = (
        close
        .rolling(window)
        .apply(katz_fd, raw=False)
    )
        

    #==== 14. Rolling Entropy ⭐⭐⭐⭐ ===========
    #----use return signs instead of price ------
    # --- high entropy - chaotic market, low entropy - structured market
    signs = np.sign(close.diff())

    entropy = (
        signs
        .rolling(20)
        .apply(
            lambda x: (
                -(pd.Series(x)
                .value_counts(normalize=True)
                * np.log2(
                    pd.Series(x)
                    .value_counts(normalize=True)
                )).sum()
            ),
            raw=False,
        )
    )

    df["rolling_entropy"] = entropy

    #===== 15. Trend Curvature ⭐⭐⭐⭐ =======
    #------ it measures trend acceleartion -----------
    slope = ema9.diff()

    df["trend_curvature"] = (
        slope -
        slope.shift(1)
    ) / (atr_series + eps)

    #=======16. ATR Shock ⭐⭐⭐⭐ =========
    # -------- useful around breakouts ---------
    df["atr_shock"] = (
        atr_series -
        atr_series.shift(1)
    ) / (atr_series.shift(1) + eps)

    # ── 17. CLEAN ─────────────────────────────────────────────────────────────
    
    float_cols = df.select_dtypes(include=["float64", "float32"]).columns
    df[float_cols] = df[float_cols].replace([np.inf, -np.inf], np.nan)
    df = df.dropna().copy()
    df[float_cols] = df[float_cols].round(6)

    return df



def get_feature_cols(df):
    # 'ts' represents timestamps, while OHLCV are standard structural exclusions.
    # atr_14 is retained for label-generation consistency but excluded from
    # the model feature set (the model sees atr_pct / macd_spread_atr instead).
    exclude = {"ts", "open", "high", "low", "close", "volume", "label", "atr_14"}
    return [c for c in df.columns if c not in exclude]


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

    # resampled = resampled[counts == minutes]
    resampled = resampled.loc[counts == minutes]

    return resampled.reset_index()

# -------------------------
# MAIN()
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", required=True)
    parser.add_argument("--outfile", required=True)
    parser.add_argument("--resample",  type=int, default=1,
                            help="Resample to N-minute bars before feature engineering. "
                                 "Default 1 = no resampling (original 1-minute bars).")
    parser.add_argument(
        "--keep-atr14",
        action="store_true",
        help="Keep the atr_14 column in the output (needed if this file's "
             "output is consumed directly by the labeler for ATR-consistent "
             "normalization). If not set, atr_14 is dropped from the output.",
    )

    args = parser.parse_args()

    print("[v4_2] Reading", args.infile)
    df = pd.read_csv(args.infile)
    df = normalize_df(df)

    # ── Optional resampling ──────────────────────────────────────────────
    if args.resample > 1:
        original_bars = len(df)
        df = resample_bars(df, args.resample)
        print(f"[v4_2] Resampled {original_bars} × 1-min bars "
                f"→ {len(df)} × {args.resample}-min bars")
    

    # -------------------------
    # FEATURE ENGINEERING
    # -------------------------
    df_feat = add_features(df)

    # ----- drop initial rows ------
    df_feat = df_feat.iloc[MIN_LOOKBACK:].copy()

    print(f"[v4_2] Built features: {df_feat.shape}")

    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)

    # ------- DROP OHLCV columns --------
    drop_cols = [ "rsi_14", "atr_pct",  "open", "high", "low", "close", "volume"]
    df_feat = df_feat.drop(columns=[c for c in drop_cols if c in df_feat.columns])

    if not args.keep_atr14:
        df_feat = df_feat.drop(columns=["atr_14"], errors="ignore")

    # --------- DROP DATE from the cols ----------
    df_feat = df_feat.drop(columns=["date"], errors="ignore")

    # --------- saving csv file --------------
    df_feat.to_csv(args.outfile, index=False)

    print("[v4_2] Saved ->", args.outfile)

    # validation
    # print(df_feat["rsi_14"].describe())


# -------------------------
# Execute only when run directly
# -------------------------
if __name__ == "__main__":
    main()
