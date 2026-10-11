
"""Collect live ticks from Kite and save broker-neutral one-minute OHLCV files.

This is a *collection adapter*, not a trading strategy.  Kite Connect supplies
individual price updates ("ticks").  The `TickBarAggregator` below turns them
into completed one-minute candles and writes files whose columns are always:

    timestamp, open, high, low, close, volume

The rest of the book should read those CSV files, not the Kite API.  A reader
who uses another broker normally replaces the small Kite-specific section in
`main()` and keeps the timestamp conversion, aggregation, validation, and file
writing code unchanged.  See `BROKER_ADAPTATION_GUIDE.md` in this folder.

Important: test with paper/sandbox data first.  Market-data permissions,
instrument identifiers, timestamp conventions, and volume fields differ by
broker.
"""

# Kite Connect is used only at the boundary where data enters this program.

import os
import time
import logging
import threading
from datetime import datetime, timezone, timedelta

import pandas as pd
from kiteconnect import KiteTicker

# ============================================================
# CONFIGURATION -- the values a reader normally changes for their own setup.
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

# Secrets belong in environment variables, never in this source file or a
# public repository.  A different broker will use different credential names.
API_KEY = os.getenv("KITE_API_KEY")
ACCESS_TOKEN = os.getenv("KITE_ACCESS_TOKEN")

# One candle represents this many seconds.  60 means one-minute OHLCV bars.
BAR_SEC = 60

# How much history (in seconds) to retain in tick_buffer beyond the last
# completed bar boundary. Only needs to cover ticks belonging to the
# in-progress (not-yet-finalized) bucket, plus a small safety margin.
BUFFER_RETENTION_SEC = BAR_SEC * 3

# Human-friendly name -> Kite's numeric instrument token.  These numeric
# tokens are *broker-specific*.  With another broker, replace this mapping
# with that broker's symbol/id convention; the friendly names can stay.
INSTRUMENTS = {
    "HDFCBANK": 341249,
    "TCS": 2953217,
    "NIFTY26SEPFUT": 17512194,
    "NIFTY26SEP23500CE": 18940162,
    "NIFTY26SEP23300PE": 18924546,

   
}

# Each symbol gets its own subfolder.  The CSV format within it is deliberately
# broker-neutral, so downstream ML code does not need to know the broker.
DATA_DIR = os.path.expanduser(
    "~/Documents/my_project/record_data_main/data_ohlcv"
)
os.makedirs(DATA_DIR, exist_ok=True)


# ============================================================
# MARKET HOURS (CHECK IN IST, BUT STORE DATA IN UTC)
# ============================================================

IST = timezone(timedelta(hours=5, minutes=30))


def market_session_bounds(now_ist=None):
    """Return today's collection window in exchange-local time.

    Adjust this function for another exchange's trading hours, holidays, and
    special sessions.  The rest of the program keeps timestamps in UTC.
    """
    now_ist = now_ist or datetime.now(timezone.utc).astimezone(IST)
    start = now_ist.replace(hour=9, minute=15, second=0, microsecond=0)
    end = now_ist.replace(hour=15, minute=25, second=0, microsecond=0)
    return start, end


def is_market_open():
    now_ist = datetime.now(timezone.utc).astimezone(IST)
    start, end = market_session_bounds(now_ist)
    return start <= now_ist < end


def is_after_market_close():
    now_ist = datetime.now(timezone.utc).astimezone(IST)
    _, end = market_session_bounds(now_ist)
    return now_ist >= end


# ============================================================
# NORMALIZE EVERY INPUT TIMESTAMP TO UTC
# ============================================================
def to_utc(ts):
    """Make a broker timestamp timezone-aware and express it in UTC.

    Kite may send a timezone-naive exchange timestamp; this script interprets
    it as Asia/Kolkata.  This assumption is one of the items another-broker
    user must verify rather than copy blindly.
    """
    ts = pd.Timestamp(ts)

    if ts.tzinfo is None:
        # Kite timestamp = IST naive → convert properly
        ts = ts.tz_localize("Asia/Kolkata").tz_convert("UTC")
    else:
        ts = ts.tz_convert("UTC")

    return ts


# ============================================================
# AGGREGATOR -- TURN MANY TICKS INTO ONE COMPLETED OHLCV BAR (PURE UTC)
# ============================================================
class TickBarAggregator:
    """Accumulate ticks for one symbol, then emit only finished time bars.

    For a one-minute bucket: open is the first tick price, high/low are the
    largest/smallest tick prices, close is the final tick price, and volume is
    the volume traded during that minute.  It is intentionally independent of
    Kite and can be retained when changing brokers.
    """
    def __init__(self, symbol, bar_sec=60):
        self.symbol = symbol
        self.bar_sec = bar_sec
        self.tick_buffer = []
        self.last_completed_end = None
        self.last_cum_vol = None
        self.last_tick_ts = None
        self.lock = threading.Lock()

        # Diagnostics counters, logged periodically / on shutdown.
        self.n_ticks = 0
        self.n_zero_or_missing_vol = 0
        self.n_decreasing_vol = 0

    def add_tick(self, ts, price, cum_vol):
        """Accept one normalized tick.

        `cum_vol` is cumulative volume for the trading day, as supplied by
        Kite's `volume_traded` field.  The program converts it into the volume
        traded *since the preceding tick*.  If a broker supplies per-tick
        volume instead, pass that value through a broker adapter or revise
        this conversion deliberately.
        """
        ts = to_utc(ts)

        try:
            cum_vol = float(cum_vol)
        except (TypeError, ValueError):
            cum_vol = 0.0

        with self.lock:
            self.n_ticks += 1

            # ---- GUARD AGAINST BAD cum_vol READINGS ----
            # Some tick payloads carry volume_traded == 0 or missing. If we
            # treat that as "last_cum_vol = 0", the NEXT valid tick computes
            # vol = cum_vol_next - 0 = cum_vol_next (the entire day's
            # cumulative volume so far) and dumps it into a single bar.
            #
            # Fix: ignore non-positive readings entirely (don't update
            # last_cum_vol). Also ignore readings that DECREASE from the
            # last known value (cumulative day volume should not decrease
            # intraday) -- treat as an out-of-order / duplicate packet.
            if cum_vol <= 0:
                vol = 0.0
                self.n_zero_or_missing_vol += 1
                logging.debug(
                    "%s: ignoring non-positive cum_vol=%s (last=%s)",
                    self.symbol, cum_vol, self.last_cum_vol,
                )
            elif self.last_cum_vol is None:
                vol = 0.0
                self.last_cum_vol = cum_vol
            elif cum_vol < self.last_cum_vol:
                vol = 0.0
                self.n_decreasing_vol += 1
                logging.debug(
                    "%s: ignoring decreasing cum_vol=%s (last=%s)",
                    self.symbol, cum_vol, self.last_cum_vol,
                )
            else:
                vol = cum_vol - self.last_cum_vol
                self.last_cum_vol = cum_vol

            self.last_tick_ts = ts
            self.tick_buffer.append((ts, float(price), vol))

    def _trim_buffer(self, cutoff_ts):
        """Drop ticks older than cutoff_ts; their bars are already finalized."""
        if not self.tick_buffer:
            return
        self.tick_buffer = [t for t in self.tick_buffer if t[0] >= cutoff_ts]

    def get_new_completed_bars(self):
        """Return bars that ended before the current time bucket.

        The currently forming bar is withheld.  That prevents a downstream
        model from seeing a candle whose high, low, close, and volume may
        still change.
        """
        with self.lock:
            if not self.tick_buffer:
                return pd.DataFrame()
            tick_snapshot = list(self.tick_buffer)
            last_completed_end = self.last_completed_end

        df = pd.DataFrame(
            tick_snapshot,
            columns=["timestamp", "price", "volume"]
        ).set_index("timestamp")

        # ---- FORCE UTC ----
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

        # ---- RESAMPLE IN UTC ----
        ohlc = df["price"].resample(f"{self.bar_sec}s").ohlc()
        vol = df["volume"].resample(f"{self.bar_sec}s").sum()

        bars = pd.concat([ohlc, vol], axis=1).dropna()
        bars.columns = ["open", "high", "low", "close", "volume"]

        if bars.empty:
            return bars

        # Keep only fully completed minute bars.
        current_bucket_start = pd.Timestamp.now(tz="UTC").floor(f"{self.bar_sec}s")
        bars = bars[bars.index < current_bucket_start].copy()

        if bars.empty:
            return bars

        # ---- SANITY CHECK ----
        assert (bars["high"] >= bars["open"]).all()
        assert (bars["high"] >= bars["close"]).all()
        assert (bars["low"] <= bars["open"]).all()
        assert (bars["low"] <= bars["close"]).all()

        new_last = bars.index[-1]

        if last_completed_end is None:
            with self.lock:
                if self.last_completed_end is None:
                    self.last_completed_end = new_last
            result = bars
        else:
            new_bars = bars[bars.index > last_completed_end].copy()

            # ---- GAP DETECTION ----
            # If the new bars don't start immediately after the last
            # completed bar, we have a gap (e.g. WS disconnect, process
            # paused). Log it so gaps are visible in the data, rather than
            # silently producing a discontinuous timestamp series.
            if len(new_bars):
                expected_next = last_completed_end + pd.Timedelta(seconds=self.bar_sec)
                actual_first = new_bars.index[0]
                if actual_first > expected_next:
                    gap_bars = int((actual_first - expected_next).total_seconds() // self.bar_sec) + 1
                    logging.warning(
                        "%s: gap detected -- missing ~%d bar(s) between %s and %s",
                        self.symbol, gap_bars, expected_next, actual_first,
                    )

                with self.lock:
                    if self.last_completed_end is None or new_bars.index[-1] > self.last_completed_end:
                        self.last_completed_end = new_bars.index[-1]

            result = new_bars

        # ---- TRIM BUFFER ----
        # Safe to drop everything older than (current_bucket_start - retention),
        # since those ticks can no longer affect any not-yet-finalized bucket.
        cutoff = current_bucket_start - pd.Timedelta(seconds=BUFFER_RETENTION_SEC)
        self._trim_buffer(cutoff)

        return result

    def log_diagnostics(self):
        logging.info(
            "%s: diagnostics -- ticks=%d, zero/missing_cum_vol=%d, decreasing_cum_vol=%d, "
            "buffer_size=%d, last_cum_vol=%s",
            self.symbol, self.n_ticks, self.n_zero_or_missing_vol,
            self.n_decreasing_vol, len(self.tick_buffer), self.last_cum_vol,
        )


# ============================================================
# SAVE -- THE PORTABLE DATA CONTRACT USED BY THE REST OF THE BOOK
# ============================================================
def record_bars(symbol, bars):
    """Append completed bars using the standard timestamp-first OHLCV schema."""
    bars = bars.copy()

    # ---- VALIDATION ----
    assert str(bars.index.tz) == "UTC"
    assert bars.index.max() < pd.Timestamp.now(tz="UTC") + pd.Timedelta(minutes=1)

    # Store timestamps as timezone-aware UTC values.  Put `timestamp` first so
    # every file follows the published portable schema exactly.
    bars["timestamp"] = bars.index
    bars = bars[["timestamp", "open", "high", "low", "close", "volume"]]

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    folder = os.path.join(DATA_DIR, symbol)
    os.makedirs(folder, exist_ok=True)
    fpath = os.path.join(folder, f"{today}.csv")

    if os.path.exists(fpath):
        bars.to_csv(fpath, mode="a", header=False, index=False)
    else:
        bars.to_csv(fpath, index=False)

    # ---- LOG IN IST (for readability only) ----
    last_ts_ist = bars.index[-1].tz_convert("Asia/Kolkata")
    logging.info("%s: wrote %d bars @ %s IST -> %s",
                  symbol, len(bars), last_ts_ist, fpath)


# ============================================================
# MAIN -- KITE-SPECIFIC COLLECTION LAYER
# ============================================================
def main():
    """Connect to Kite, translate its tick payloads, and feed the generic core.

    When adapting the script, this function is the main place to work.  The
    broker-specific concepts are: client construction, subscription ids,
    callback registration, and the names of token/price/timestamp/volume
    fields.  Do not copy those field names to a different broker.
    """
    kite_ws = KiteTicker(API_KEY, ACCESS_TOKEN)

    aggregators = {sym: TickBarAggregator(sym, BAR_SEC) for sym in INSTRUMENTS}

    def on_ticks(_ws, ticks):
        if not is_market_open():
            return

        for tk in ticks:
            token = tk.get("instrument_token")
            if token not in INSTRUMENTS.values():
                continue

            symbol = next(k for k, v in INSTRUMENTS.items() if v == token)

            # This is the broker-payload translation boundary.  A different
            # broker should produce the same four values: symbol, price,
            # timestamp, and volume (with its volume convention understood).
            price = tk.get("last_price") or tk.get("ltp")
            ts = tk.get("exchange_timestamp") or datetime.now(timezone.utc)
            vol = tk.get("volume_traded", 0)

            aggregators[symbol].add_tick(ts, price, vol)

    def on_connect(ws, _):
        tokens = list(INSTRUMENTS.values())
        ws.subscribe(tokens)
        ws.set_mode(ws.MODE_FULL, tokens)
        logging.info("Subscribed to %d instruments.", len(tokens))

    def on_close(_ws, code, reason):
        logging.warning("WebSocket closed: code=%s reason=%s", code, reason)

    def on_reconnect(_ws, attempts_count):
        logging.warning("WebSocket reconnecting (attempt %d)...", attempts_count)

    def on_noreconnect(_ws):
        logging.error("WebSocket gave up reconnecting.")

    kite_ws.on_ticks = on_ticks
    kite_ws.on_connect = on_connect
    kite_ws.on_close = on_close
    kite_ws.on_reconnect = on_reconnect
    kite_ws.on_noreconnect = on_noreconnect

    logging.info("Starting recorder (UTC mode)...")
    kite_ws.connect(threaded=True)

    market_close_flushed = False
    last_diag_log = time.monotonic()
    DIAG_LOG_INTERVAL_SEC = 300  # log per-symbol diagnostics every 5 minutes

    while True:
        if is_market_open():
            for symbol, agg in aggregators.items():
                try:
                    new_bars = agg.get_new_completed_bars()
                except Exception as exc:
                    logging.error("%s: bar build failed: %s", symbol, exc)
                    continue

                if len(new_bars):
                    record_bars(symbol, new_bars)

            if time.monotonic() - last_diag_log >= DIAG_LOG_INTERVAL_SEC:
                for agg in aggregators.values():
                    agg.log_diagnostics()
                last_diag_log = time.monotonic()

        elif is_after_market_close():
            if not market_close_flushed:
                logging.info("Market closed. Final flush and shutdown.")
                for symbol, agg in aggregators.items():
                    try:
                        new_bars = agg.get_new_completed_bars()
                    except Exception as exc:
                        logging.error("%s: final bar build failed: %s", symbol, exc)
                        continue

                    if len(new_bars):
                        record_bars(symbol, new_bars)
                    agg.log_diagnostics()
                market_close_flushed = True
                kite_ws.close()
                break
        time.sleep(1)


if __name__ == "__main__":
    main()

    
    
