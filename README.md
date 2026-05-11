# auto-bot-trade

Trading bot workspace for crypto/forex experiments, live dry-run validation, and backtests.

## Project Layout

- `bots/` - live and dry-run trading bots.
- `backtests/` - backtest, validation, and strategy research scripts.
- `data/` - historical CSV inputs.
- `reports/` - generated backtest/validation CSV outputs.
- `scripts/` - helper scripts and Windows scheduled-task launchers.
- `logs/` - runtime logs, state files, and dry-run CSV logs.

Root-level Python files are lightweight wrappers for the current production bot entrypoints.

## Common Commands

Run the Bitcoin Donchian bot:

```bash
python3 bitcoin_bot.py
```

Run the Gold bot:

```bash
python3 gold_bot.py
```

Run the Forex bot:

```bash
python3 forex_bot.py
```

Run the Donchian live dry-run validator directly from its module:

```bash
python3 bots/dry_run_live_donchian.py
```
```

Run the multi-timeframe backtest:

```bash
python3 backtest_multi_tf_lorentzian_andean.py --mode regime_sweep
```

Run Donchian robustness validation:

```bash
python3 validate_donchian_strategy.py
```

Run forex bot on Windows through the existing wrapper:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_forex_bot.ps1
```

## Current Validated Crypto Strategy

- Donchian breakout: 20
- M15 trend with ADX > 20 and rising
- M5 entry, M1 confirmation
- Fixed RR: 1:2
- Risk: 0.5%
- DRY_RUN validator: `bots/dry_run_live_donchian.py`

Generated validation reports live in `reports/`.
