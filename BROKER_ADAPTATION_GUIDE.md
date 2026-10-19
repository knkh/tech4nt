# Using another broker or market-data source

The collector supplied with this book uses Kite Connect because it is a
concrete, working example. It is not a recommendation of Kite or of any other
broker. A genuinely generic collector cannot know how every broker authenticates
users, identifies instruments, streams live prices, or names fields in its
messages.

The portable part of this project is therefore the **market-data output**, not
the API call. Whatever broker, exchange feed, CSV download, or data vendor you
use, make it produce the same completed OHLCV files:

```text
timestamp,open,high,low,close,volume
2026-09-13 03:45:00+00:00,100.0,101.2,99.8,100.7,15420
```

Rules for this contract:

- `timestamp` is timezone-aware UTC, and denotes the **start** of the bar.
- `open`, `high`, `low`, and `close` are numeric prices for one fixed interval.
- `volume` is the volume traded during that interval, not cumulative daily
  volume.
- Write only completed bars. Do not overwrite a bar still being formed.
- Keep the column names and order exactly as shown. Downstream feature,
  training, and trading programs then do not need broker-specific changes.

## What the Kite example does

The source file has two layers.

| Layer | Job | Keep it when changing broker? |
|---|---|---|
| Broker adapter in `main()` | Connects, subscribes, and translates a broker tick into symbol, price, timestamp, and volume. | Replace or modify. |
| Generic core | Converts timestamps to UTC, makes OHLCV bars, detects gaps, validates bars, and writes CSV files. | Usually keep unchanged. |

A *tick* is simply a small live message such as “the last traded price is now
100.70.” A one-minute candle is built from all ticks arriving in that minute:

- Open: first price in the minute
- High: highest price in the minute
- Low: lowest price in the minute
- Close: final price in the minute
- Volume: total traded volume in the minute

## Adaptation checklist

1. Read your broker's official market-data documentation. Confirm that live
   streaming is permitted for your account and that you have the necessary
   exchange-data entitlement.
2. Find the broker's stable instrument identifier for each symbol. Replace the
   numeric Kite tokens in `INSTRUMENTS`; do not assume another broker uses the
   same symbols or expiry notation.
3. Replace `KiteTicker(...)`, its connection call, and its subscription code
   with the equivalent client in your broker's SDK or WebSocket API.
4. In that API's tick callback, translate each message into four values:
   `symbol`, `price`, `timestamp`, and `volume`.
5. Verify timestamps. If a timestamp has no timezone attached, identify the
   exchange timezone from the documentation before converting it to UTC. Never
   assume a naive timestamp is UTC.
6. Verify volume semantics. Kite provides cumulative intraday volume in
   `volume_traded`, which the example converts to per-tick volume. A provider
   may instead give per-trade volume, per-tick volume, no volume, or delayed
   volume. Treating one of these as another can create absurdly large bars.
7. Adjust exchange hours, holidays, and special sessions in
   `market_session_bounds()`. This example uses Indian-market hours in IST.
8. Run the collector during a short test session. Open the CSV and check UTC
   timestamps, OHLC ordering, non-negative volume, no duplicate bars, and
   visible gap warnings after a deliberate disconnect.

## The four fields every adapter must supply

| Meaning needed by the generic core | Kite example | What to find in another API |
|---|---|---|
| Instrument identity | `instrument_token` | Symbol, contract id, token, or subscription id |
| Last traded price | `last_price` / `ltp` | Last price, trade price, or quote price |
| Event time | `exchange_timestamp` | Exchange/trade timestamp; receive time only as a documented fallback |
| Volume | `volume_traded` | Whether it is cumulative daily or an incremental amount |

## A minimal adapter pattern

The generic aggregator does not need to know the broker. Conceptually, every
broker callback should do this:

```python
for message in broker_messages:
    symbol = id_to_symbol[message_instrument_id(message)]
    price = message_last_price(message)
    timestamp = message_exchange_time(message)
    volume = message_volume(message)
    aggregators[symbol].add_tick(timestamp, price, volume)
```

Those `message_...` names are deliberately placeholders. Replace them only
after consulting the documentation for the selected provider. If that provider
supplies **incremental** volume rather than cumulative daily volume, adjust the
volume handling in `TickBarAggregator.add_tick()` rather than silently passing
it as cumulative volume.

## If the source already supplies candles

Some APIs return completed one-minute candles rather than ticks. In that case,
you may skip `TickBarAggregator`, but still validate that each candle is
complete, timezone-convert its timestamp to UTC, select the six standard
columns, and use `record_bars()` to save it. Historical downloads should be
written in the same schema, preferably into the same per-symbol folder layout.

## Common mistakes

- Embedding API keys or access tokens in the script. Use environment variables
  or your broker's approved credential mechanism.
- Treating a timezone-naive broker timestamp as UTC without verification.
- Saving daily cumulative volume in every one-minute row.
- Writing a still-forming candle, then training or trading on a close price
  that later changes.
- Filling a data outage with made-up candles. Preserve the gap and investigate
  it.
- Assuming an instrument id remains valid after an option expiry or contract
  rollover.

Changing the collection adapter changes neither the learning ideas nor the
downstream file format. It simply makes a different data source speak the same
small, reliable OHLCV language.
