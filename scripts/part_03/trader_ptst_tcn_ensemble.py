

# trader_ensemble_genZ01.py
import argparse
import json
import os
import time
from collections import deque
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import torch

from kiteconnect import KiteConnect
from patchtst_model import PatchTST
from tcn_model import TCN
# from helper_files.feature_engineering_40_clean import add_features
from features_engineering_v01 import add_features

# ================= CONFIG =================
DRY_RUN = True # change it to 'False' for ssh
ALLOW_SHORTS = False

WINDOW_SIZE = 64
ENTRY_THRESHOLD = 0.60
STRONG_THRESHOLD = 0.70
ENTRY_MIN_EDGE = 0.30
STRONG_MIN_EDGE = 0.50
# One add is sufficient for paper validation.  The previous three-add setting
# made a single trend dominate the day's P&L and obscured base-entry quality.
MAX_PYRAMID = 3
# VOLATILE_MIN_UP_PROB = 0.68
# VOLATILE_MIN_EDGE = 0.25
# VOLATILE_MIN_BAR_MOVE_ATR = 0.75
# VOLATILE_MIN_BAR_RANGE_ATR = 1.00
# VOLATILE_MIN_CLOSE_LOCATION = 0.75
PYRAMID_MIN_UP_PROB = 0.80
PYRAMID_MIN_EDGE = 0.60
PYRAMID_MAX_DOWN_PROB = 0.10
PYRAMID_MIN_BARS_BETWEEN_ADDS = 5
PYRAMID_MIN_PROFIT_ATR = 1.0
SL_MULT = 5.0

COOLDOWN = 5
MIN_HOLD_BARS = 12
EXTREME_DOWN_EXIT_THRESHOLD = 0.80

# ================= CONFIG =================
MAX_STRETCH_ATR       = 3.0   # max close-to-EMA9 distance in ATR units for entry
MAX_ENTRY_RANGE_POS   = 0.85  # range position is normalised to [0, 1]


POST_STOP_COOLDOWN = 6
POST_SIGNAL_COOLDOWN = 5
COOLDOWN_OVERRIDE_MIN_UP_PROB = 0.90
COOLDOWN_OVERRIDE_MIN_EDGE = 0.70
SIGNAL_EXIT_MIN_RET5 = -0.001
SIGNAL_EXIT_MAX_BAR_MOVE_ATR = -0.35
SIGNAL_EXIT_MAX_CLOSE_LOCATION = 0.35

DEVICE = "cpu"

# `tcn` is the safe default.  Its checkpoint reproduces its held-out test
# result using its own scaler.  Do not use `blend` until the PatchTST
# checkpoint/dataset provenance is reconciled and the blend is validated.
MODEL_MODE = "tcn"  # allowed: "tcn", "blend"
PTST_WEIGHT = 0.60
TCN_WEIGHT = 0.40



DATA_ROOT = os.path.expanduser("~/Documents/my_project/record_data_main/data_ohlcv")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.expanduser("~/Documents/my_project/clone_files/data/part_2/demo_data/ptst_cm_demo_validation")
MODEL_DIR_TCN = os.path.expanduser("~/Documents/my_project/clone_files/data/part_2/demo_data/tcn_cm_demo_validation")
WINDOWED_DIR = os.path.expanduser("~/Documents/my_project/clone_files/data/part_2/demo_data/dataset_v01r1")

LOG_DIR = os.path.join(SCRIPT_DIR, "logs_demo_ens")

RAW_USECOLS_TS = ["ts", "open", "high", "low", "close", "volume"]
RAW_USECOLS_TIMESTAMP = ["timestamp", "open", "high", "low", "close", "volume"]
FEATURE_LOOKBACK_BARS = 50
PRELOAD_EXTRA_BARS = FEATURE_LOOKBACK_BARS
MIN_RAW_MODEL_BARS = WINDOW_SIZE + FEATURE_LOOKBACK_BARS
LIVE_TAIL_ROWS = 200
BUFFER_SIZE = WINDOW_SIZE + FEATURE_LOOKBACK_BARS

ENTRY_CUTOFF_UTC = (9, 30)
PYRAMID_CUTOFF_UTC = (9, 20)
MARKET_CLOSE_UTC = (9, 40)
MAX_OPEN_POSITIONS = 1  # CE and PE are correlated account-level exposure

BUY = "BUY"
SELL = "SELL"
FLAT, LONG, SHORT = 0, 1, -1


os.makedirs(LOG_DIR, exist_ok=True)

INSTRUMENTS = {

    "NIFTY26OCT22500CE": {
            "exchange": "NFO",
            "lot_size": 65,
            "lots": 1,
            "execution": "LIMIT",
            "protective_exit_execution": "LIMIT",
            "tick_size": 0.20,
            "limit_buffer_points": 2.0,
            "product": "MIS",
    },
    
    "SBIN": {
            "exchange": "NSE",
            "qty": 5,                    # ← number of shares to trade
            "execution": "LIMIT",
            "protective_exit_execution": "LIMIT",
            "tick_size": 1.0,
            "limit_buffer_points": 5.0,
            "product": "MIS",
        },
}


def open_position_count(states):
    return sum(st.position == LONG for st in states.values())


# ================= BROKER =================

def init_kite():
    if DRY_RUN:
        return None

    api_key = os.getenv("KITE_API_KEY")
    access_token = os.getenv("KITE_ACCESS_TOKEN")

    if not api_key or not access_token:
        raise SystemExit("KITE_API_KEY and KITE_ACCESS_TOKEN must be set when DRY_RUN=False.")

    kite_client = KiteConnect(api_key=api_key)
    kite_client.set_access_token(access_token)
    return kite_client

kite = init_kite()
# ================= STATE =================

class State:
    def __init__(self):
        self.buffer = deque(maxlen=BUFFER_SIZE)

        self.position = FLAT
        self.entry_price = None
        self.entry_atr = None
        self.peak_price = None
        self.net_qty = 0
        self.last_file_mtime = None
        self.cooldown = 0
        self.last_ts = None
        self.last_action_ts = None
        self.last_realized_pnl = None
        self.last_price = None
        self.trailing_stop = None
        self.bars_in_trade = 0
        self.pyramid_count = 0
        self.last_add_price = None
        self.last_add_atr = None
        self.last_add_bar = None
        self.pyramid_peak_price = None
        self.pyramid_trailing_stop = None

        self.pending_buy = False
        self.pending_buy_ts = None
        self.pending_buy_reason = None
        


    def reset_position(self):
        self.position = FLAT
        self.entry_price = None
        self.entry_atr = None
        self.peak_price = None
        self.net_qty = 0
        self.cooldown = COOLDOWN
        self.bars_in_trade = 0
        self.pyramid_count = 0
        self.last_add_price = None
        self.last_add_atr = None
        self.last_add_bar = None
        self.pyramid_peak_price = None
        self.pyramid_trailing_stop = None
        self.trailing_stop = None

        self.pending_buy = False
        self.pending_buy_ts = None
        self.pending_buy_reason = None
        
# ================= UTIL =================

def now_utc():
    return datetime.now(timezone.utc)

def to_utc(ts):
    ts = pd.Timestamp(ts)

    if ts.tzinfo is None:
        return ts.tz_localize(timezone.utc)

    return ts.tz_convert(timezone.utc)

def normalize_df(df):
    if "timestamp" in df.columns and "ts" not in df.columns:
        df = df.rename(columns={"timestamp": "ts"})

    required = {"ts", "open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    out = df.copy()
    # out["ts"] = pd.to_datetime(out["ts"], errors="coerce")
    out["ts"] = pd.to_datetime(out["ts"], errors="coerce", utc=True)
    out = out.dropna(subset=["ts"]).sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
    return out

def read_market_csv(path, tail_rows=None):
    try:
        df = pd.read_csv(path, usecols=RAW_USECOLS_TS)
    except ValueError as exc:
        if "Usecols do not match columns" not in str(exc):
            raise
        df = pd.read_csv(path, usecols=RAW_USECOLS_TIMESTAMP)

    if tail_rows is not None and len(df) > tail_rows:
        df = df.tail(tail_rows).copy()

    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float32)

    return normalize_df(df)

def compute_atr(df, period=14):
    if len(df) < period + 1:
        return None

    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.ewm(alpha=1 / period, adjust=False).mean().iloc[-1]
    return float(atr) if pd.notna(atr) else None


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
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def round_to_tick(price, tick_size=0.05):
    return round(float(price) / tick_size) * tick_size


def get_limit_price(side, last_price, tick_size=0.05, buffer_points=0.10):
    if side == BUY:
        return round_to_tick(float(last_price) + buffer_points, tick_size)
    return round_to_tick(float(last_price) - buffer_points, tick_size)

# --------- SECTION-III: TRADING CONCEPT ----------
def preload_state(sym, st):
    sym_dir = os.path.join(DATA_ROOT, sym)
    if not os.path.isdir(sym_dir):
        print(f"WARN | Missing instrument folder: {sym_dir}")
        return

    files = sorted(
        os.path.join(sym_dir, name)
        for name in os.listdir(sym_dir)
        if name.endswith(".csv")
    )

    if not files:
        return

    frames = []
    rows_needed = WINDOW_SIZE + PRELOAD_EXTRA_BARS
    rows_collected = 0

    for path in reversed(files):
        try:
            df_part = read_market_csv(path, tail_rows=rows_needed)
            if len(df_part):
                frames.insert(0, df_part)
                rows_collected += len(df_part)
            if rows_collected >= rows_needed:
                break
        except Exception as exc:
            print(f"WARN | preload failed {path}: {exc}")

    if not frames:
        return

    hist = pd.concat(frames, ignore_index=True).sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
    # preload_df = hist.iloc[-min(WINDOW_SIZE, len(hist)) :]
    preload_df = hist.iloc[-min(BUFFER_SIZE, len(hist)):]
    for _, row in preload_df.iterrows():
        st.buffer.append(row.to_dict())

    if st.buffer:
        st.last_ts = pd.Timestamp(st.buffer[-1]["ts"])

    print(f"PRELOAD | {sym} | loaded {len(st.buffer)}/{MIN_RAW_MODEL_BARS} raw bars")

# ─────────────────────────────────────────────
# TRADE HELPERS
# ─────────────────────────────────────────────

def get_trade_qty(cfg, price=None):
    """
    Returns trade quantity based on instrument config.
    Derivative : lot_size × lots
    Cash value : trade_value / price
    Cash fixed : qty
    """
    if cfg.get("lot_size") is not None:
        # Derivative — lot_size and lots must both be present
        return cfg["lot_size"] * cfg["lots"]
    
    # ----- python -0 optimisation -------
    '''
    # assert statements are disabled when Python runs with the -O optimisation flag. 
    For a trading system where a missing price could silently default to 1 share, 
    a proper check is safer:
    '''

    if cfg.get("trade_value") is not None:
        # Cash — fixed rupee value, quantity depends on current price
        if price is None or price <= 0:
            raise ValueError(
                f"trade_value mode requires a valid price but got price={price}"
            )
        return max(1, int(cfg["trade_value"] / price))
    '''
    if cfg.get("trade_value") is not None:
        assert price is not None, "price required for trade_value mode"
        return max(1, int(cfg["trade_value"] / price))
    '''
    # Cash — fixed share count
    return cfg["qty"]

# ============= ORDER PLACEMENT --------------
def place_order(
    sym,
    side,
    qty,
    cfg,
    last_price,
    execution_override=None,
    limit_buffer_override=None,
):
    if qty <= 0:
        return False

    if DRY_RUN:
        print(f"DRY | {side} {sym} x{qty} @ ~{last_price}")
        return True

    try:
        execution = execution_override or cfg.get("execution", "LIMIT")
        product = cfg.get("product", "MIS")

        if execution == "LIMIT":
            tick_size = cfg.get("tick_size", 0.05)
            buffer_points = (
                cfg.get("limit_buffer_points", 0.10)
                if limit_buffer_override is None
                else limit_buffer_override
            )
            limit_price = get_limit_price(side, last_price, tick_size, buffer_points)

            order_id = kite.place_order(
                variety="regular",
                exchange=cfg["exchange"],
                tradingsymbol=sym,
                transaction_type=side,
                quantity=qty,
                order_type="LIMIT",
                price=limit_price,
                product=product,
            )

            print(f"ORDER OK | {side} {sym} x{qty} LIMIT {limit_price} | id={order_id}")
            return True

        order_id = kite.place_order(
            variety="regular",
            exchange=cfg["exchange"],
            tradingsymbol=sym,
            transaction_type=side,
            quantity=qty,
            order_type="MARKET",
            product=product,
        )

        print(f"ORDER OK | {side} {sym} x{qty} MARKET | id={order_id}")
        return True

    except Exception as exc:
        print(f"ORDER ERROR | {sym} | {side} | qty={qty} | price={last_price} | {exc}")
        return False


# ================= MODEL LOAD =================

device = torch.device("cpu")
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
torch.set_grad_enabled(False)

def load_scaler(model_dir):
    """Load the scaler saved with one model; models must not share scalers."""
    with open(os.path.join(model_dir, "scaler.json")) as f:
        scaler = json.load(f)
    mean = np.array(scaler["mean"], dtype=np.float32)
    std = np.array(scaler["std"], dtype=np.float32)
    return mean, np.where(std == 0, 1.0, std)


if MODEL_MODE not in {"tcn", "blend"}:
    raise ValueError("MODEL_MODE must be 'tcn' or 'blend'")
if MODEL_MODE == "blend" and not np.isclose(PTST_WEIGHT + TCN_WEIGHT, 1.0):
    raise ValueError("PTST_WEIGHT and TCN_WEIGHT must sum to 1.0 in blend mode")

ptst_mean, ptst_std = load_scaler(MODEL_DIR)
tcn_mean, tcn_std = load_scaler(MODEL_DIR_TCN)
if len(ptst_mean) != len(tcn_mean):
    raise ValueError("PatchTST and TCN use incompatible feature dimensions")

# ---- PatchTST ----
patchtst = None
if MODEL_MODE == "blend":
    patchtst = PatchTST(
        c_in       = len(ptst_mean),
        patch_len  = 8,
        stride     = 4,
        max_len    = WINDOW_SIZE,
        num_layers = 2,
        num_heads  = 2,
        d_model    = 64,
        d_ff       = 128,
        dropout    = 0.1,
        n_classes  = 3,
    )
    state_dict = torch.load(
        os.path.join(MODEL_DIR, "best_mpdel.pth"),
        map_location="cpu",
        weights_only=True,
    )
    patchtst.load_state_dict(state_dict)
    del state_dict
    patchtst.to(device)
    patchtst.eval()

# ---- TCN ----
tcn = TCN(
    num_features = len(tcn_mean),
    num_classes  = 3,
    channels     = (32, 64, 64, 64, 64),
    kernel_size  = 4,
    dropout      = 0.2,
)
state_dict = torch.load(
    os.path.join(MODEL_DIR_TCN, "best_model.pth"),
    map_location="cpu",
    weights_only=True,
)
tcn.load_state_dict(state_dict)
del state_dict
tcn.to(device)
tcn.eval()

# ================= MODEL HELPERS =================
with open(os.path.join(WINDOWED_DIR, "meta.json")) as f:
    meta = json.load(f)

feature_cols = meta["features"]

# ----------- FEATURE BUILDING ----------
def build_model_feature_window(df_buf):
    feat_df = add_features(df_buf.copy())
    feat_df = feat_df.dropna(subset=feature_cols)

    if len(feat_df) < WINDOW_SIZE + 1:
        raise ValueError("Not enough clean feature rows")

    x = feat_df[feature_cols].iloc[-(WINDOW_SIZE + 1):-1].to_numpy(dtype=np.float32)
    return x


# -------------- ENSEMBLE PREDICTIONS ---------------
def predict_ensemble(buffer_np):
    """Return TCN probabilities, or an explicitly selected probability blend.

    Each model is standardised with the scaler saved at its own training time.
    In blend mode probabilities (not raw logits) are averaged, making the result
    easier to audit and avoiding an arbitrary logit-scale interaction.
    """
    if buffer_np.shape[0] < WINDOW_SIZE:
        return None, None, None, None

    x_tcn = (buffer_np - tcn_mean[None, :]) / tcn_std[None, :]
    x_tcn = np.nan_to_num(x_tcn, nan=0.0, posinf=0.0, neginf=0.0)
    xb_tcn = torch.from_numpy(x_tcn).unsqueeze(0)

    with torch.inference_mode():
        tcn_probs = torch.softmax(tcn(xb_tcn)[0], dim=0).cpu().numpy()

        ptst_probs = None
        if MODEL_MODE == "blend":
            x_ptst = (buffer_np - ptst_mean[None, :]) / ptst_std[None, :]
            x_ptst = np.nan_to_num(x_ptst, nan=0.0, posinf=0.0, neginf=0.0)
            xb_ptst = torch.from_numpy(x_ptst).unsqueeze(0)
            ptst_probs = torch.softmax(patchtst(xb_ptst)[0], dim=0).cpu().numpy()
            probs = PTST_WEIGHT * ptst_probs + TCN_WEIGHT * tcn_probs
        else:
            probs = tcn_probs

    pred = int(np.argmax(probs))
    conf = float(probs[pred])
    model_probs = {"tcn": tcn_probs, "patchtst": ptst_probs}
    return pred, conf, probs, model_probs

# ================= INDICATORS =================

def compute_executor_indicators(df):
    close = df["close"]
    price_now = close.iloc[-1]
    open_now = df["open"].iloc[-1]
    high_now = df["high"].iloc[-1]
    low_now = df["low"].iloc[-1]

    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()

    ret_5 = close.pct_change(5)
    ret_2 = close.pct_change(2)
    

    macd_line = ema9 - ema21
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_spread = macd_line - macd_signal

    recent_high = df["high"].rolling(12).max().iloc[-1]
    recent_low  = df["low"].rolling(12).min().iloc[-1]

    pos_in_range = (price_now - recent_low) / (recent_high - recent_low + 1e-6)
    pos_in_range = np.clip(pos_in_range, 0.0, 1.0)

    atr = compute_atr(df)
    if atr is None or atr <= 0:
        macd_spread_atr_now = None
        macd_spread_atr_prev = None
        stretch_atr = None
        bar_move_atr = None
        bar_range_atr = None
    else:
        macd_spread_atr = macd_spread / (atr + 1e-9)
        macd_spread_atr_now = float(macd_spread_atr.iloc[-1])
        macd_spread_atr_prev = float(macd_spread_atr.iloc[-2]) if len(macd_spread_atr) >= 2 else None
        stretch_atr = float((close.iloc[-1] - ema9.iloc[-1]) / (atr + 1e-9))
        bar_move_atr = float((price_now - open_now) / (atr + 1e-9))
        bar_range_atr = float((high_now - low_now) / (atr + 1e-9))

    close_location = float((price_now - low_now) / (high_now - low_now + 1e-9))

    return {
         "price_now": float(price_now),
        "ema9_now": float(ema9.iloc[-1]),
        "ema21_now": float(ema21.iloc[-1]),
        "ret_5_now": float(ret_5.iloc[-1]) if pd.notna(ret_5.iloc[-1]) else None,
        "ret_2_now": float(ret_2.iloc[-1]) if pd.notna(ret_2.iloc[-1]) else None,  # ← add
        "macd_spread_atr_now": macd_spread_atr_now,
        "macd_spread_atr_prev": macd_spread_atr_prev,
        "stretch_atr": stretch_atr,
        "bar_move_atr": bar_move_atr,
        "bar_range_atr": bar_range_atr,
        "close_location": float(np.clip(close_location, 0.0, 1.0)),
        "atr": atr,
        "recent_high": recent_high,
        "recent_low": recent_low,
        "pos_in_range": float(pos_in_range),
    }

    '''
def is_volatile_bullish_entry(probs, ind):
    p_down, p_flat, p_up = map(float, probs)
    edge = p_up - max(p_down, p_flat)
    ret_5 = ind.get("ret_5_now")
    ret_2 = ind.get("ret_2_now")
    bar_move_atr = ind.get("bar_move_atr")
    bar_range_atr = ind.get("bar_range_atr")
    close_location = ind.get("close_location")

    ready = (
        ret_5 is not None
        and ret_5 > 0
        and bar_move_atr is not None
        #and bar_move_atr >= VOLATILE_MIN_BAR_MOVE_ATR
        and bar_range_atr is not None
        # and bar_range_atr >= VOLATILE_MIN_BAR_RANGE_ATR
        and close_location is not None
        # and close_location >= VOLATILE_MIN_CLOSE_LOCATION
        # and p_up >= VOLATILE_MIN_UP_PROB
        and edge >= VOLATILE_MIN_EDGE
    )
    '''
    metrics = [
        f"BAR_MOVE_ATR={bar_move_atr:.3f}" if bar_move_atr is not None else "BAR_MOVE_ATR=NA",
        f"BAR_RANGE_ATR={bar_range_atr:.3f}" if bar_range_atr is not None else "BAR_RANGE_ATR=NA",
        f"CLOSE_LOC={close_location:.3f}" if close_location is not None else "CLOSE_LOC=NA",
    ]
    return ready, metrics


def has_confirmed_downside(ind):
    ret_5 = ind.get("ret_5_now")
    bar_move_atr = ind.get("bar_move_atr")
    close_location = ind.get("close_location")
    return (
        ret_5 is not None
        and ret_5 <= SIGNAL_EXIT_MIN_RET5
    ) or (
        bar_move_atr is not None
        and bar_move_atr <= SIGNAL_EXIT_MAX_BAR_MOVE_ATR
        and close_location is not None
        and close_location <= SIGNAL_EXIT_MAX_CLOSE_LOCATION
    )


def can_override_cooldown(p_up, edge, ret_5, ind):
    bar_move_atr = ind.get("bar_move_atr")
    close_location = ind.get("close_location")
    return (
        p_up >= COOLDOWN_OVERRIDE_MIN_UP_PROB
        and edge >= COOLDOWN_OVERRIDE_MIN_EDGE
        and ret_5 is not None
        and ret_5 > 0
        and (
            bar_move_atr is None
            or bar_move_atr >= SIGNAL_EXIT_MAX_BAR_MOVE_ATR
        )
        and (
            close_location is None
            or close_location >= SIGNAL_EXIT_MAX_CLOSE_LOCATION
        )
    )


# ================= DECISION =================

def decide_action_v3(pred, probs, df_buf, st):
    if len(df_buf) < 21:
        return "HOLD", ["NOT_ENOUGH_BARS"], {}

    ind = compute_executor_indicators(df_buf)
    if ind["atr"] is None:
        return "HOLD", ["NO_ATR"], ind

    p_down, p_flat, p_up = map(float, probs)

    edge    = p_up - max(p_down, p_flat)
    ret_5   = ind["ret_5_now"]
    ret_2   = ind["ret_2_now"]
    stretch = ind["stretch_atr"]

    ts_utc   = to_utc(df_buf["ts"].iloc[-1])
    hhmm_utc = (ts_utc.hour, ts_utc.minute)

    reasons = [
        f"P_UP={p_up:.3f}",
        f"P_DOWN={p_down:.3f}",
        f"P_FLAT={p_flat:.3f}",
        f"EDGE={edge:.3f}",
        f"T_UTC={ts_utc.strftime('%H:%M')}",
    ]
    if ret_5   is not None: reasons.append(f"RET5={ret_5:.4f}")
    if stretch is not None: reasons.append(f"STRETCH={stretch:.3f}")
    # ── compute once, used in both entry paths ──────────────────────────
    momentum_ok = ret_5 is not None and ret_5 > 0

    # ── EXIT ─────────────────────────────────────────────────────────────────

    
    if st.position == LONG:

        # During minimum hold period — only exit on extreme confirmed down signal
        if st.bars_in_trade < MIN_HOLD_BARS:
            if p_down >= EXTREME_DOWN_EXIT_THRESHOLD and has_confirmed_downside(ind):
                return "EXIT_LONG_SIGNAL", reasons + ["EARLY_STRONG_DOWN_CONFIRMED"], ind
            if p_down >= EXTREME_DOWN_EXIT_THRESHOLD:
                return "HOLD", reasons + ["EARLY_DOWN_UNCONFIRMED", "HOLD_HORIZON"], ind
            return "HOLD", reasons + ["HOLD_HORIZON"], ind

        # After minimum hold — exit on strong down signal
        if p_down >= EXTREME_DOWN_EXIT_THRESHOLD and has_confirmed_downside(ind):
            return "EXIT_LONG_SIGNAL", reasons + ["STRONG_DOWN_CONFIRMED"], ind

        # Exit on model weakening: all three must agree direction has reversed
        if p_down > p_up and edge < 0 and ret_5 is not None and ret_5 < 0:
            return "EXIT_LONG_MODEL", reasons + ["MODEL_WEAKENING"], ind

        # Pyramid — only attempted if MAX_PYRAMID > 0
        if (
            MAX_PYRAMID > 0
            and st.pyramid_count < MAX_PYRAMID
            and hhmm_utc < PYRAMID_CUTOFF_UTC
            and st.bars_in_trade >= 6
            and p_up >= PYRAMID_MIN_UP_PROB
            and edge >= PYRAMID_MIN_EDGE
            and p_down <= PYRAMID_MAX_DOWN_PROB
            and ret_5 is not None and ret_5 > 0
            and ind["price_now"] >= float(st.entry_price) + PYRAMID_MIN_PROFIT_ATR * ind["atr"]
        ):
            too_soon = (
                st.last_add_bar is not None
                and st.bars_in_trade - st.last_add_bar < PYRAMID_MIN_BARS_BETWEEN_ADDS
            )
            ref_price = st.last_add_price if st.last_add_price is not None else st.entry_price
            if not too_soon and ref_price is not None and ind["price_now"] > ref_price * 1.003:
                return "PYRAMID", reasons + ["PYRAMID_OK"], ind

        return "HOLD", reasons, ind

    # ── ENTRY ────────────────────────────────────────────────────────────────
    
    if st.position == FLAT:

        if st.cooldown > 0:
            return "HOLD", reasons + [f"COOLDOWN={st.cooldown}"], ind

        if hhmm_utc >= ENTRY_CUTOFF_UTC:
            return "HOLD", reasons + ["LATE_ENTRY_BLOCK"], ind

        if stretch is not None and stretch > MAX_STRETCH_ATR:
            return "HOLD", reasons + ["OVERSTRETCH_BLOCK"], ind

        pos = ind.get("pos_in_range")
        if pos is not None and pos > MAX_ENTRY_RANGE_POS:
            return "HOLD", reasons + ["RANGE_BLOCK"], ind

        # Strong signal — execute immediately
        if (
            p_up >= STRONG_THRESHOLD
            and edge >= STRONG_MIN_EDGE
            and momentum_ok
        ):
            return "STRONG_BUY", reasons + ["STRONG"], ind

        # Weaker signal — store for confirmation
        if (
            p_up >= ENTRY_THRESHOLD
            and edge >= ENTRY_MIN_EDGE
            and momentum_ok
        ):
            return "BUY_CANDIDATE", reasons + ["ENTRY_PENDING"], ind

    return "HOLD", reasons, ind
 
# ================= STOP LOSS / TRAILING STOP =================

def check_exit(st, df_buf, candle, sl_mult=SL_MULT):
    """
    Protective exit based on price action alone — independent of model signal.

    Uses entry ATR (fixed at trade open) so stop distance does not change
    as intraday volatility shifts.

    Early bars (< 4): fixed stop below entry price.
    Later bars      : trailing stop from peak price.
    """
    atr = st.entry_atr
    if atr is None or atr <= 0:
        return None

    close_now = float(candle["close"])
    high_now  = float(candle["high"])

    if st.position != LONG:
        return None

    # Update peak price on every bar
    st.peak_price = high_now if st.peak_price is None else max(float(st.peak_price), high_now)

    # Early bars: fixed stop anchored to entry price
    if st.bars_in_trade < 4:
        entry_stop = float(st.entry_price) - sl_mult * atr
        if close_now <= entry_stop:
            return "EXIT_LONG_STOP"
        return None

    # Later bars: trailing stop from peak price
    candidate_stop = float(st.peak_price) - sl_mult * atr
    st.trailing_stop = (
        candidate_stop
        if st.trailing_stop is None
        else max(float(st.trailing_stop), candidate_stop)
    )

    if close_now <= float(st.trailing_stop):
        return "EXIT_LONG_STOP"

    return None


# =================SECTION-IV: LOGGING =================

def log_trade(
    sym,
    candle,
    pred,
    probs,
    st,
    action,
    ind,
    reasons,
    model_probs=None,
):
    out_dir = os.path.join(LOG_DIR, sym)
    os.makedirs(out_dir, exist_ok=True)
    ts_utc = to_utc(candle["ts"])
    if probs is None:
        probs = [np.nan, np.nan, np.nan]
    if ind is None:
        ind = {}
    model_probs = model_probs or {}
    tcn_probs = model_probs.get("tcn")
    ptst_probs = model_probs.get("patchtst")


    row = {
        "ts": ts_utc,
        "price": float(candle["close"]),
        "entry_atr": st.entry_atr,
        "pred": None if pred is None else int(pred),
        "prob_down": float(probs[0]),
        "prob_flat": float(probs[1]),
        "prob_up": float(probs[2]),
        "model_mode": MODEL_MODE,
        "prob_down_tcn": np.nan if tcn_probs is None else float(tcn_probs[0]),
        "prob_flat_tcn": np.nan if tcn_probs is None else float(tcn_probs[1]),
        "prob_up_tcn": np.nan if tcn_probs is None else float(tcn_probs[2]),
        "prob_down_patchtst": np.nan if ptst_probs is None else float(ptst_probs[0]),
        "prob_flat_patchtst": np.nan if ptst_probs is None else float(ptst_probs[1]),
        "prob_up_patchtst": np.nan if ptst_probs is None else float(ptst_probs[2]),
        "ret_5": ind.get("ret_5_now"),
        "ret_2": ind.get("ret_2_now"),
        "ema9": ind.get("ema9_now"),
        "ema21": ind.get("ema21_now"),
        "macd_spread_atr_now": ind.get("macd_spread_atr_now"),
        "macd_spread_atr_prev": ind.get("macd_spread_atr_prev"),
        "stretch_atr": ind.get("stretch_atr"),
        "bar_move_atr": ind.get("bar_move_atr"),
        "bar_range_atr": ind.get("bar_range_atr"),
        "close_location": ind.get("close_location"),
        "action": action,
        "position": st.position,
        "entry_price": st.entry_price,
        "peak_price": st.peak_price,
        "trailing_stop": st.trailing_stop,
        "net_qty": st.net_qty,
        "cooldown": st.cooldown,
        "bars_in_trade": st.bars_in_trade,
        "pyramid_count": st.pyramid_count,
        "last_add_price": st.last_add_price,
        "last_add_atr": st.last_add_atr,
        "pyramid_peak_price": st.pyramid_peak_price,
        "pyramid_trailing_stop": st.pyramid_trailing_stop,
        "last_realized_pnl": st.last_realized_pnl,
        "reasons": "|".join(reasons),
        
    }
    #========== SAVE TRADE LOGS ==========================
    fpath = os.path.join(out_dir, f"{pd.Timestamp(ts_utc).date()}.csv")
    pd.DataFrame([row]).to_csv(fpath, mode="a", header=not os.path.exists(fpath), index=False)


def flatten_all(states):
    print("Flattening all open positions")
    for sym, cfg in INSTRUMENTS.items():
        st = states[sym]
        if st.position == LONG and st.net_qty > 0:
            price = st.last_price if st.last_price is not None else st.entry_price
            price = price if price is not None else 0.0
            if place_order(
                sym,
                SELL,
                st.net_qty,
                cfg,
                price,
                execution_override=cfg.get("protective_exit_execution", "MARKET"),
            ):
                st.reset_position()


# =================SECTION-V:  MAIN =================
def main():

    states = {sym: State() for sym in INSTRUMENTS}

    for sym, st in states.items():
        preload_state(sym, st)

    print(f"{MODEL_MODE.upper()} live executor started | 1-minute bars | "
          f"entry={ENTRY_THRESHOLD:.2f}/{ENTRY_MIN_EDGE:.2f} | max positions={MAX_OPEN_POSITIONS}")

    while True:
        now = now_utc()

        if (now.hour, now.minute) >= MARKET_CLOSE_UTC:
            flatten_all(states)
            break

        for sym, cfg in INSTRUMENTS.items():
            fpath = os.path.join(DATA_ROOT, sym, f"{now.date()}.csv")

            if not os.path.exists(fpath):
                continue

            try:
                file_mtime = os.path.getmtime(fpath)
            except OSError:
                continue

            st = states[sym]
            if st.last_file_mtime is not None and file_mtime == st.last_file_mtime:
                continue

            try:
                df = read_market_csv(fpath, tail_rows=LIVE_TAIL_ROWS)
            except Exception as exc:
                print(f"READ ERROR | {sym} | {exc}")
                continue

            if len(df) < 2:
                continue

            # Use the latest completed candle — last bar is still forming
            candle = df.iloc[-2]

            if st.last_ts is not None and pd.Timestamp(candle["ts"]) <= pd.Timestamp(st.last_ts):
                continue

            st.last_file_mtime = file_mtime
            st.last_ts         = candle["ts"]
            st.last_realized_pnl = None

            price        = float(candle["close"])
            st.last_price = price

            if st.cooldown > 0:
                st.cooldown -= 1

            if st.position != FLAT:
                st.bars_in_trade += 1

            # ── Illiquid bar handling ─────────────────────────────────────────
            price_range_pct = (
                (float(candle["high"]) - float(candle["low"]))
                / (float(candle["close"]) + 1e-9)
            )

            if price_range_pct < 0.0001:
                ts_utc  = to_utc(candle["ts"])
                action  = "ILLIQUID_SKIP"
                reasons = [f"ILLIQUID_SKIP", f"RANGE_PCT={price_range_pct:.6f}"]
                print(f"{ts_utc} | {sym} | {price:.2f} | ILLIQUID_SKIP")

                # Still run stop-loss check even on illiquid bars
                if st.position == LONG and len(st.buffer) >= 2:
                    df_buf      = pd.DataFrame.from_records(st.buffer)
                    exit_action = check_exit(st, df_buf, candle)
                    if exit_action is not None:
                        action = exit_action
                        reasons.append(exit_action)
                        realized_pnl = (
                            (price - st.entry_price) * st.net_qty
                            if st.entry_price is not None else None
                        )
                        execution_override = (
                            cfg.get("protective_exit_execution")
                            if action in {"EXIT_LONG_STOP", "EXIT_LONG_SIGNAL"}
                            else None
                        )
                        if place_order(sym, SELL, st.net_qty, cfg, price,
                                       execution_override=execution_override):
                            st.last_realized_pnl = realized_pnl
                            st.reset_position()
                            if action == "EXIT_LONG_STOP":
                                st.cooldown = POST_STOP_COOLDOWN
                                reasons.append(f"POST_STOP_CD={POST_STOP_COOLDOWN}")
                            elif action == "EXIT_LONG_SIGNAL":
                                st.cooldown = POST_SIGNAL_COOLDOWN
                                reasons.append(f"POST_SIGNAL_CD={POST_SIGNAL_COOLDOWN}")
                            st.last_action_ts = candle["ts"]
                            reasons.append("ORDER_EXIT_OK")
                        else:
                            action = f"{action}_FAILED"
                            reasons.append("ORDER_EXIT_FAILED")

                log_trade(sym, candle, None, None, st, action, {}, reasons)
                continue

            # ── Liquid bar: add to buffer and build features ──────────────────
            st.buffer.append(candle.to_dict())

            if len(st.buffer) < MIN_RAW_MODEL_BARS:
                print(f"WARMUP | {sym} | {len(st.buffer)}/{MIN_RAW_MODEL_BARS}")
                continue

            df_buf = pd.DataFrame.from_records(st.buffer)

            try:
                buffer_np = build_model_feature_window(df_buf)
            except Exception as exc:
                print(f"FEATURE BUILD ERROR | {sym} | {exc}")
                continue

            pred, conf, probs, model_probs = predict_ensemble(buffer_np)
            if pred is None:
                print(f"PREDICT SKIP | {sym} | not enough feature rows")
                continue

            # ── Decision ─────────────────────────────────────────────────────
            action, reasons, ind = decide_action_v3(pred, probs, df_buf, st)

            p_down, p_flat, p_up = map(float, probs)
            edge  = p_up - max(p_down, p_flat)
            ret_5 = ind.get("ret_5_now")
            ret_2 = ind.get("ret_2_now")

            # ── BUY_CANDIDATE: one-bar confirmation ───────────────────────────
            if st.position == FLAT and st.pending_buy:
                ts_utc   = to_utc(candle["ts"])
                hhmm_utc = (ts_utc.hour, ts_utc.minute)

                pending_age      = pd.Timestamp(candle["ts"]) - pd.Timestamp(st.pending_buy_ts)
                confirm_blockers = []

                if pending_age > pd.Timedelta(minutes=2):
                    confirm_blockers.append("BUY_CONFIRM_STALE")
                if st.cooldown > 0:
                    confirm_blockers.append(f"BUY_CONFIRM_COOLDOWN={st.cooldown}")
                if hhmm_utc >= ENTRY_CUTOFF_UTC:
                    confirm_blockers.append("BUY_CONFIRM_LATE_ENTRY_BLOCK")

                if (
                    not confirm_blockers
                    and ret_5 is not None and ret_5 > 0
                    and p_up >= ENTRY_THRESHOLD
                    and edge >= ENTRY_MIN_EDGE
                ):
                    action = "BUY"
                    reasons.append("BUY_CONFIRMED")
                else:
                    action = "HOLD"
                    reasons.extend(confirm_blockers)
                    reasons.append("BUY_CONFIRM_FAILED")

                st.pending_buy        = False
                st.pending_buy_ts     = None
                st.pending_buy_reason = None

            elif action == "BUY_CANDIDATE" and st.position == FLAT:
                st.pending_buy        = True
                st.pending_buy_ts     = candle["ts"]
                st.pending_buy_reason = "|".join(reasons)
                action  = "HOLD"
                reasons.append("BUY_STORED_PENDING")

            if (
                action in {"BUY", "STRONG_BUY"}
                and st.position == FLAT
                and open_position_count(states) >= MAX_OPEN_POSITIONS
            ):
                action = "HOLD"
                reasons.append(f"ACCOUNT_POSITION_CAP={MAX_OPEN_POSITIONS}")

            # ── Protective stop-loss overrides everything ─────────────────────
            if st.position == LONG:
                exit_action = check_exit(st, df_buf, candle)
                if exit_action is not None:
                    action = exit_action
                    reasons.append(exit_action)

            # ── Execute entries ───────────────────────────────────────────────
            if action in {"BUY", "STRONG_BUY"} and st.position == FLAT:
                # Entry block — NEW (single line, works for both):
                qty = get_trade_qty(cfg, price)
                # ----- old code line --------
                # qty = cfg["lot_size"] * cfg["lots"]
                # qty = cfg["qty"]

                if place_order(sym, BUY, qty, cfg, price):
                    st.position       = LONG
                    st.entry_price    = price
                    st.entry_atr      = ind.get("atr")
                    st.peak_price     = price
                    st.trailing_stop  = None
                    st.net_qty        = qty
                    st.cooldown       = COOLDOWN
                    st.last_action_ts = candle["ts"]
                    st.bars_in_trade  = 0
                    st.pyramid_count  = 0
                    st.last_add_price = price
                    st.last_add_bar   = 0
                    reasons.append("ORDER_BUY_OK")
                else:
                    action = f"{action}_FAILED"
                    reasons.append("ORDER_BUY_FAILED")

            # ── Execute pyramid ───────────────────────────────────────────────
            elif action == "PYRAMID" and st.position == LONG:

                # ---- Pyramid block — NEW:----- 
                add_qty = get_trade_qty(cfg, price)

                # ------ Pyramid block — OLD:-----
                # add_qty      = cfg["qty"] 
                # add_qty = cfg["lot_size"] * cfg["lots"]
                prev_qty     = st.net_qty
                total_qty    = prev_qty + add_qty
                weighted_entry = (
                    ((st.entry_price or price) * prev_qty + price * add_qty) / total_qty
                    if total_qty > 0 else price
                )

                if place_order(sym, BUY, add_qty, cfg, price):
                    st.entry_price    = weighted_entry
                    st.peak_price     = max(float(st.peak_price or price), price)
                    st.net_qty        = total_qty
                    st.pyramid_count += 1
                    st.last_add_price = price
                    st.last_add_atr   = ind.get("atr")
                    st.last_add_bar   = st.bars_in_trade
                    st.last_action_ts = candle["ts"]
                    reasons.append("ORDER_PYRAMID_OK")
                else:
                    action = f"{action}_FAILED"
                    reasons.append("ORDER_PYRAMID_FAILED")

            # ── Execute exits ─────────────────────────────────────────────────
            elif action in {"EXIT_LONG_MODEL", "EXIT_LONG_STOP", "EXIT_LONG_SIGNAL"} \
                    and st.position == LONG:

                realized_pnl = (
                    (price - st.entry_price) * st.net_qty
                    if st.entry_price is not None else None
                )
                execution_override = (
                    cfg.get("protective_exit_execution")
                    if action in {"EXIT_LONG_STOP", "EXIT_LONG_SIGNAL"}
                    else None
                )

                if place_order(sym, SELL, st.net_qty, cfg, price,
                               execution_override=execution_override):
                    st.last_realized_pnl = realized_pnl
                    st.reset_position()

                    if action == "EXIT_LONG_STOP":
                        st.cooldown = POST_STOP_COOLDOWN
                        reasons.append(f"POST_STOP_CD={POST_STOP_COOLDOWN}")
                    elif action == "EXIT_LONG_SIGNAL":
                        st.cooldown = POST_SIGNAL_COOLDOWN
                        reasons.append(f"POST_SIGNAL_CD={POST_SIGNAL_COOLDOWN}")

                    st.last_action_ts = candle["ts"]
                    reasons.append("ORDER_EXIT_OK")
                else:
                    action = f"{action}_FAILED"
                    reasons.append("ORDER_EXIT_FAILED")

            # ── Log and print ─────────────────────────────────────────────────
            log_trade(sym, candle, pred, probs, st, action, ind, reasons, model_probs)

            ts_utc = to_utc(candle["ts"])
            print(
                f"{ts_utc} | {sym} | {price:.2f} | "
                f"PRED={pred} CONF={conf:.3f} | "
                f"D={probs[0]:.3f} F={probs[1]:.3f} U={probs[2]:.3f} | "
                f"POS={st.position} | CD={st.cooldown} | "
                f"{action} | {';'.join(reasons[-4:])}"
            )

        time.sleep(1)


if __name__ == "__main__":
    main()
