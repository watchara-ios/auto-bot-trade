"""
Load OHLCV history from MT5 and save to data/ for backtesting.
Supports multiple symbols and timeframes in one run.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import MetaTrader5 as mt5
import pandas as pd

# ===== CONFIG =====
SYMBOLS = ["EURUSDm", "GBPUSDm", "XAUUSDm", "USDJPYm", "EURJPYm"]

TIMEFRAMES = {
    "M1":  mt5.TIMEFRAME_M1,
    "M5":  mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
}

UTC_FROM = datetime(2025, 1, 1, tzinfo=timezone.utc)
UTC_TO   = datetime(2026, 5, 7, tzinfo=timezone.utc)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "mt5_history"

# ===== CONNECT =====
if not mt5.initialize():
    print(f"❌ MT5 initialize failed: {mt5.last_error()}")
    sys.exit(1)

acc = mt5.account_info()
if acc is None:
    print(f"❌ Not logged in: {mt5.last_error()}")
    mt5.shutdown()
    sys.exit(1)

print(f"✅ MT5 connected — account={acc.login}  server={acc.server}  currency={acc.currency}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ===== FETCH =====
results = []
for sym in SYMBOLS:
    # Check symbol exists on broker
    info = mt5.symbol_info(sym)
    if info is None:
        print(f"  ⚠️  {sym}: not found on broker — skipping")
        continue
    if not info.visible:
        mt5.symbol_select(sym, True)

    for tf_name, tf_const in TIMEFRAMES.items():
        rates = mt5.copy_rates_range(sym, tf_const, UTC_FROM, UTC_TO)
        if rates is None or len(rates) == 0:
            err = mt5.last_error()
            print(f"  ⚠️  {sym} {tf_name}: no data returned  last_error={err}")
            continue

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.rename(columns={"tick_volume": "volume"})[
            ["time", "open", "high", "low", "close", "volume"]
        ]

        out_path = OUTPUT_DIR / f"{sym}_{tf_name}.csv"
        df.to_csv(out_path, index=False, encoding="utf-8-sig")

        print(f"  ✅ {sym} {tf_name}: {len(df):,} bars → {out_path.name}")
        results.append({"symbol": sym, "tf": tf_name, "bars": len(df), "file": str(out_path)})

mt5.shutdown()

# ===== SUMMARY =====
print(f"\n{'='*50}")
print(f"Done. {len(results)} files saved to {OUTPUT_DIR}")
if results:
    summary = pd.DataFrame(results)
    print(summary.to_string(index=False))
