
# feature engineering

A) Important Tyoes of features invisaged:

1. Returns (ret_1, ret_5, ret_12)

Simple percentage price change over different lookback windows (1, 5, 12 bars). These give the model a sense of recent momentum at multiple time-scales — short (1 bar), medium (5 bars), and a window matched to the model's prediction horizon (12 bars, per the comment "matches horizon exactly").

2. Volatility (atr_pct, atr_change, atr_regime, atr_percentile)

Built on ATR (Average True Range), which measures how much an instrument typically moves per bar. These features tell the model:

atr_pct: how large is volatility relative to price right now
atr_change: is volatility rising or falling vs. 5 bars ago
atr_regime: is current volatility unusually high or low vs. its own 20-bar average
atr_percentile: where does today's volatility rank within the recent 20-bar history

Purpose: volatility context matters because the same price move means something different in a calm market vs. a turbulent one.

3. Momentum oscillators (rsi_centered, macd_spread_atr, macd_spread_velocity)

Classic technical-analysis momentum indicators, but normalized:

RSI is re-centered around 0 instead of the usual 0–100 scale (easier for ML models to use — centered features tend to train better than skewed ones).
MACD is expressed as a spread scaled by ATR (macd_spread_atr) rather than in raw price units, so it's comparable across instruments with different price levels. macd_spread_velocity captures whether that momentum is accelerating or decelerating.

Purpose: capture overbought/oversold and momentum-shift signals in a scale-invariant way.

4. Trend (ema_9_slope_pct, close_ema9_ratio, trend_strength_21, dist_ema21_atr, trend_persistence)

Based on exponential moving averages (EMA-9 fast, EMA-21 slow):

Slope and ratio features tell the model whether price is above/below its short-term trend and how steep that trend is.
trend_strength_21 and dist_ema21_atr measure trend strength/distance in ATR-normalized units (so "how far from trend" is comparable across volatility regimes).
trend_persistence measures how consistently the trend has held direction over the last 5 bars (a rolling win-rate of "is EMA rising").

Purpose: distinguish trending regimes from choppy/rangebound ones, and gauge how "committed" the current trend is.

5. Microstructure / candle shape (body_ratio, wick_ratio, close_location, last_bar_strength)

These describe the shape of individual candles rather than multi-bar patterns:

body_ratio: how much of the candle's range is "body" (close-open) vs. wick
wick_ratio: asymmetry between upper and lower wicks (rejection patterns)
close_location: where the close sits within the bar's high-low range (near top = bullish pressure, near bottom = bearish)
last_bar_strength: how strong was the last bar's move, in ATR units

Purpose: capture short-term buyer/seller pressure and rejection signals often used in price-action trading.

6. Range & breakout (dist_to_high20, dist_to_low20, day_range_pos)

Distance (in ATR units) from the current price to the recent 20-bar high/low, plus where price sits within the current day's range so far.

Purpose: proximity to breakout levels — the model can learn how price behaves near support/resistance zones.

7. Mean reversion (zscore_20)

Standard z-score: how many standard deviations is the current price from its 20-bar mean.

Purpose: high positive/negative z-scores can signal "stretched" prices likely to revert — the opposite logic from the trend features above. Including both gives the model information to decide which regime applies.

8. Session/time (session_progress)

Normalized position within the trading day (0 = market open, 1 = market close), computed from IST session hours.

Purpose: market behavior is not time-of-day-neutral — volatility and liquidity typically differ at open, midday, and close. This lets the model learn session-specific patterns.

9. Volatility structure — the "advanced" block (flagged ⭐ in the code)

This is really a second layer of volatility/complexity features, more sophisticated than plain ATR:


Important: The file also contains several commented-out features (ret_hi_12, hi_trend_persistence, volume_ratio, gap_from_day_open, etc.) and an explicit deduplication note ("dist_breakout and dist_to_high20 were identical — keep one"). This is itself a nice concrete illustration of the "researcher degrees of freedom" problem. 


A) Two-Stage Feature Pipeline:

1. Stage 1: The Core Feature Calculation Script (Single-Day Unit)
The foundational building block is a pure Python script (e.g., generate_daily_features.py). Its single responsibility is to process one isolated unit of data—typically a single day's market data CSV.
⚬	Inputs: Raw single-day market data (OHLCV bars).
⚬	Processing: Computes technical indicators, moving averages, relative strength indexes, and custom target labels.
⚬	Output: A standardized daily feature-and-label DataFrame.
Why start small? Testing and debugging feature logic on a small 1-day dataset takes milliseconds, allowing rapid development without waiting for years of historical data to process.
2. Stage 2: The Batch Driver Script (Multi-Period Orchestrator)
Once the 1-day feature logic is tested and validated, a Batch Runner script (e.g., batch_generate_features.py) takes over.
⚬	Job: Accepts parameters such as start_date, end_date, or a list of raw data file paths.
⚬	Mechanism: Iterates through dates sequentially, passes each day's raw data to the Stage 1 engine, and stacks or saves the resulting feature sets into a master historical dataset.


    