# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run production bots (root wrappers delegate to bots/)
python3 bitcoin_bot.py
python3 gold_bot.py
python3 forex_bot.py
python3 gold_mr_bot.py

# Run dry-run live validator directly
python3 bots/dry_run_live_donchian.py

# Run backtests
python3 backtest_multi_tf_lorentzian_andean.py --mode regime_sweep
python3 validate_donchian_strategy.py
python3 backtests/run_donchian_final.py

# Install dependencies
pip install -r requirements.txt             # crypto bots
pip install -r requirements-forex.txt       # forex/gold MT5 bots (adds MetaTrader5)

# Deploy (Railway)
# railway.json configures: startCommand=python bitcoin_bot.py, restartPolicy=ALWAYS
```

Stop a running bot by creating its kill file:
```bash
touch logs/hybrid_STOP    # bitcoin bot
touch logs/gold_STOP      # gold bot
touch logs/forex_STOP     # forex bot
touch logs/gold_mr_STOP   # gold MR bot
```

## Deployment topology

| Bot | Runs on | Mode |
|---|---|---|
| `bitcoin_bot.py` | Railway (cloud, always-on) | Live — `HYBRID_DRY_RUN=false` |
| `forex_bot.py` | Windows PC (`C:\Users\watch\`) via scheduled task, starts 14:00 UTC daily | Live — `FOREX_DRY_RUN=false` |
| `gold_bot.py` | Windows PC, same scheduled task mechanism | Live |
| `gold_mr_bot.py` | Windows PC | Live — `GOLD_MR_DRY_RUN` defaults `true`; set `false` for live |

Logs are git-committed from Windows/Railway and pulled to Mac for review. The local `logs/` directory is a snapshot, not live. To see truly current forex/gold bot activity, check the Windows runtime log at `C:\Users\watch\auto-bot-trade\logs\forex_runtime.log`.

`hybrid_bot.log` being 0 bytes locally is normal — Railway writes to its own container filesystem, not here.

## Architecture

### Two separate trading ecosystems

**Crypto (Binance Futures)** — `bots/bitcoin_bot.py`
- Calls Binance REST API directly (HMAC-signed requests, multi-endpoint rotation for rate-limit resilience)
- Timeframes: M1 (execution confirmation) / M5 (signal) / M15 (trend filter)
- Signals sourced from `bots/donchian_core.py` → `latest_signal()`
- Position management: algo orders (`STOP_MARKET` + `TAKE_PROFIT_MARKET`) placed immediately after entry via `/fapi/v1/algoOrder`
- Dry-run mode controlled by `HYBRID_DRY_RUN=true`; all order calls are no-ops but state/CSV is still written

**Forex/Gold (MetaTrader 5)** — `bots/forex_bot.py`, `bots/gold_bot.py`, `bots/gold_mr_bot.py`
- Connects via `MetaTrader5` Python library (**Windows only** — requires MT5 terminal running)
- Same M1/M5/M15 multi-timeframe logic as crypto, but data pulled from MT5
- Credentials: `MT5_LOGIN`, `MT5_PASSWORD`, `MT5_SERVER`
- Dry-run controlled by `FOREX_DRY_RUN=true` / `GOLD_MR_DRY_RUN=true`
- stdout/stderr emoji → ASCII fallback because Windows console can't print emoji

### Signal pipeline (shared by all Donchian bots)

```
donchian_core.latest_signal()
  ├─ prepare_timeframes()       # align M1/M5/M15, compute indicators via backtest module
  ├─ signal_from_closed_row()   # Donchian breakout + ADX + BOS check
  ├─ apply_micro_edge_filters() # session hours, ADX ceiling, ATR percentile, volume ratio
  └─ bt.confirm_m1()            # require M1 candle close in breakout direction
```

`donchian_core.py` imports `backtest_multi_tf_lorentzian_andean` as `bt` — the same indicator engine drives both live and backtest. **Changing indicator logic in `backtests/backtest_multi_tf_lorentzian_andean.py` affects live bots.**

`_REJECT_STATS` (Counter in `donchian_core.py`) accumulates per-reason rejection counts. Bots log hourly summaries — check these first when diagnosing why signals aren't firing.

### gold_mr_bot strategy (different from others)

`bots/gold_mr_bot.py` uses **Keltner Channel mean-reversion** (not Donchian breakout):
- BUY when price dips below KC lower + RSI < 30, confirmation candle closes back inside
- SELL when price spikes above KC upper + RSI > 70, confirmation candle closes back inside
- D1 ADX filter is **inverted** vs momentum bots — blocks when D1 ADX ≥ 30 (trending is bad for MR)
- Has AI news gate (DeepSeek) that blocks on high-impact events; fails open
- Middle-band exit: closes early when price reaches EMA20

### Strategy tiers (Donchian bots)

Signals are classified **Tier A** (standard, 0.25% risk) or **Tier B** (high-quality breakout, 1–2% risk). `bt.high_quality_breakout()` in the backtest module defines Tier B criteria.

### State and persistence

Each bot maintains:
- `logs/<bot>_state.json` — daily counters, position memory, connection state
- `logs/<bot>_trades.csv` — trade log (schema-checked; auto-rotated on schema change)
- `logs/<bot>_runtime.log` — rotating daily log

`connection_down: true` in `hybrid_state.json` is often a stale artifact: `reset_day()` calls `save_state()` before `mark_connected()` resets the flag in memory. It does not mean the bot is actually disconnected.

### Notifications

`bots/notifier.py` wraps Telegram Bot API. All bots import it for order-opened/closed/error alerts. Requires `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID`. Missing credentials silently disable notifications.

`notify_error()` = red alert (⚠️). `notify_order_result()` = raw dump of MT5/Binance result. Always call `notify_error()` when an order fails — do not rely on `notify_order_result()` alone to signal failure.

### Root-level wrapper files

`bitcoin_bot.py`, `gold_bot.py`, `forex_bot.py`, `gold_mr_bot.py` at root are one-liners that `runpy.run_path()` into the real bot in `bots/`. These exist so Railway and Windows batch scripts can reference a stable root path.

### Backtests vs strategies

`backtests/` — standalone research scripts. Many write CSV reports to `reports/` or `outputs/`.
`strategies/` — reusable strategy cores (`gold_v2_core.py`, `gold_mr_core.py`) imported by both backtests and live bots.

## Known bugs

### Bitcoin bot — orphan algo orders (unresolved)

After a Binance Futures position closes, the SL+TP algo orders placed via `/fapi/v1/algoOrder` remain and cannot be cancelled. `cleanup_orphan_orders()` uses `DELETE /fapi/v1/allOpenOrders` (with `conditional=true` param) but this does **not** cancel algo/conditional orders — they need individual cancellation via `DELETE /fapi/v1/algoOrder`. The bot detects the orphans but the cancel silently fails every 30s for 10–20+ minutes until a new trade is entered. This happened on May 22 and May 25.

Fix needed in `bots/bitcoin_bot.py` `cleanup_orphan_orders()`: iterate `open_algo_orders()` and cancel each individually.

### gold_mr_bot — order failure notification (fixed 2026-05-28)

`send_order()` previously called `notify_order_result()` regardless of retcode, making failed orders look like normal notifications. Fixed: now checks `retcode != TRADE_RETCODE_DONE` first, calls `notify_error()` with specific message (retcode=10027 → "Enable AutoTrading in MT5 terminal"), and returns early. `notify_order_result` + `notify_order_opened` only fire on success.

## Diagnosing why a bot isn't trading

1. Check hourly `[STATS]` lines in the runtime log — they show rejection reason counts
2. For forex/gold, also grep `[D1_REGIME]` — this filter runs after M5 and doesn't appear in STATS
3. Common causes by bot:

| Bot | Most common block | Typical fix |
|---|---|---|
| bitcoin | `no_donchian_breakout` (91%) | Market ranging — wait or add symbols |
| forex | `no_donchian_breakout` + `D1_REGIME` D1 ADX < 20 | Lower `FOREX_D1_ADX_MIN=15` in `.env` |
| gold_mr | `no_mr_signal` (99%) | Market trending — D1 ADX OK for MR when < 30 |

## Environment variables

Copy `.env.example` to `.env`. Key variables:

| Variable | Bot | Purpose |
|---|---|---|
| `BINANCE_API_KEY` / `BINANCE_SECRET` | bitcoin | Binance Futures credentials |
| `BINANCE_BASE_URL` | bitcoin | Use `https://demo-fapi.binance.com` for testnet |
| `HYBRID_DRY_RUN` | bitcoin | `true` = no real orders |
| `HYBRID_SYMBOLS` | bitcoin | Comma-separated, e.g. `BTCUSDT,ETHUSDT` |
| `HYBRID_MAX_TRADES_PER_DAY` | bitcoin | Default 5; lower to 2 in choppy markets |
| `MT5_LOGIN` / `MT5_PASSWORD` / `MT5_SERVER` | forex/gold | MT5 account |
| `FOREX_DRY_RUN` | forex | `true` = no real orders |
| `FOREX_D1_ADX_MIN` | forex | Default 20; lower to 15 when market is ranging |
| `GOLD_MR_DRY_RUN` | gold_mr | Default `true`; set `false` for live |
| `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` | all | Notification channel |
| `DEEPSEEK_API_KEY` | bitcoin, gold_mr | AI signal validator / news gate |
