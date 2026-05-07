"""
run_forex_donchian.py
=====================
Backtest using EXACTLY the same parameters as forex_bot.py.
Data source: data/mt5_history/{SYMBOL}_{TF}.csv
             (generate with: python load_history_md5.py)

Usage:
    python backtests/run_forex_donchian.py
    python backtests/run_forex_donchian.py --start 2025-01-01 --end 2025-12-31
"""

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backtests"))

from realistic_donchian_backtest import (
    Config,
    load_csv,
    metrics,
    monthly_performance,
    prepare,
    run_backtest,
    write_csv,
)

# ── Output ────────────────────────────────────────────────────────────────────
REPORT_DIR = ROOT / "reports" / "forex_donchian"
DATA_DIR   = ROOT / "data" / "mt5_history"

# ── Symbols traded by forex_bot.py ────────────────────────────────────────────
SYMBOLS = ["EURUSDm", "GBPUSDm", "XAUUSDm", "USDJPYm", "EURJPYm"]

# ── Session filter: forex_bot TRADE_START_HOUR=14, TRADE_END_HOUR=23 (Thai UTC+7)
#    → UTC 07:00–16:00
SESSION_UTC = tuple(range(7, 16))

# ── Base config — mirrors forex_bot.py Config exactly ─────────────────────────
FOREX_CONFIG = Config(
    # Account / risk
    initial_balance   = 10_000.0,     # USD — adjust to your real account size
    fixed_risk_base   = True,
    tier_a_risk       = 0.0025,        # FOREX_TIER_A_RISK = 0.25%
    tier_b_risk       = 0.010,         # FOREX_TIER_B_RISK = 1.0%

    # Strategy
    donchian_n        = 20,            # FOREX_DONCHIAN_N
    rr                = 2.5,           # FOREX_RR
    adx_min           = 18.0,          # FOREX_ADX_MIN
    adx_max           = 50.0,          # FOREX_ADX_MAX
    atr_percentile_min= 30.0,          # FOREX_ATR_PCT_MIN

    # Volume filter (loosened: 0.8x average)
    use_volume_filter = True,
    volume_mult       = 0.8,           # FOREX_VOLUME_MULT

    # ATR expansion: off by default (FOREX_REQUIRE_ATR_EXPANSION=false)
    use_atr_expansion = False,

    # Breakout quality thresholds (loosened from defaults)
    breakout_body_mult   = 1.2,        # DonchianCoreConfig.breakout_body_mult
    breakout_atr_mult    = 0.5,
    max_wick_pct         = 0.5,
    close_quality_min    = 0.5,        # DonchianCoreConfig.close_quality_min

    # ADX acceleration (FOREX_ADX_BARS_RISING = 1 → off)
    adx_bars_rising   = 1,

    # Pro trade management
    breakeven_r       = 0.5,           # FOREX_BREAKEVEN_R

    # Session filter (Thai 14-23 = UTC 07-15)
    include_utc_hours = SESSION_UTC,

    # Limits
    max_trades_per_day = 5,            # FOREX_MAX_TRADES_PER_DAY
    daily_loss_limit_pct = 0.03,       # FOREX_MAX_DAILY_LOSS_PCT = 3%

    # Simulation realism
    fee_rate             = 0.0001,     # ~1 pip commission typical
    spread_atr_mult      = 0.03,
    slippage_min_spread  = 0.3,
    slippage_max_spread  = 1.0,
)


def load_symbol(sym: str) -> dict | None:
    paths = {
        "m1":  DATA_DIR / f"{sym}_M1.csv",
        "m5":  DATA_DIR / f"{sym}_M5.csv",
        "m15": DATA_DIR / f"{sym}_M15.csv",
    }
    missing = [k for k, p in paths.items() if not p.exists()]
    if missing:
        print(f"  ⚠️  {sym}: missing {missing} — run load_history_md5.py first")
        return None
    return {k: load_csv(p) for k, p in paths.items()}


def run(start: str, end: str):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"Forex Donchian Backtest — {start} to {end}")
    print(f"Symbols : {SYMBOLS}")
    print(f"Session : UTC {min(SESSION_UTC):02d}:00–{max(SESSION_UTC)+1:02d}:00 (Thai 14:00–23:00)")
    print(f"RR={FOREX_CONFIG.rr}  ADX={FOREX_CONFIG.adx_min}-{FOREX_CONFIG.adx_max}"
          f"  ATRpct≥{FOREX_CONFIG.atr_percentile_min}  Vol≥{FOREX_CONFIG.volume_mult}x")
    print(f"{'='*60}\n")

    # Test a few variants to understand sensitivity
    variants: list[tuple[str, Config]] = [
        ("forex_exact",        FOREX_CONFIG),
        ("no_session",         replace(FOREX_CONFIG, include_utc_hours=None)),
        ("no_volume",          replace(FOREX_CONFIG, use_volume_filter=False)),
        ("rr2",                replace(FOREX_CONFIG, rr=2.0)),
        ("adx20",              replace(FOREX_CONFIG, adx_min=20.0)),
        ("atr_pct50",          replace(FOREX_CONFIG, atr_percentile_min=50.0)),
        ("adx_accel2",         replace(FOREX_CONFIG, adx_bars_rising=2)),
    ]

    summary_rows = []
    trade_frames = []
    monthly_frames = []

    for sym in SYMBOLS:
        data = load_symbol(sym)
        if data is None:
            continue

        # Use 6-month warm-up window for indicator accuracy
        warmup_start = str(pd.Timestamp(start) - pd.DateOffset(months=6))[:10]

        m1_full  = data["m1"]
        m5_full  = data["m5"]
        m15_full = data["m15"]

        m1  = m1_full.loc[start:end]
        m5r = m5_full.loc[start:end]

        if len(m5r) < 200:
            print(f"  ⚠️  {sym}: insufficient M5 bars ({len(m5r)}) — need ≥200")
            continue

        days = max((m5r.index[-1].date() - m5r.index[0].date()).days + 1, 1)

        for name, cfg in variants:
            m5_warmup  = m5_full.loc[warmup_start:end]
            m15_warmup = m15_full.loc[warmup_start:end]
            m5_prep    = prepare(m5_warmup, m15_warmup, cfg)
            m5_prep    = m5_prep.loc[start:end]

            trades, equity = run_backtest(sym, m1, m5_prep, cfg, name, start[:4])
            row = {"symbol": sym, "variant": name, "days": days,
                   **metrics(trades, equity, days)}
            if not trades.empty:
                monthly = monthly_performance(trades)
                row["neg_months"] = int((monthly["pnl"] < 0).sum())
                trade_frames.append(trades)
                monthly_frames.append(monthly)
            else:
                row["neg_months"] = 0
            summary_rows.append(row)

    if not summary_rows:
        print("No results — check data files in data/mt5_history/")
        return

    summary = pd.DataFrame(summary_rows)

    # ── Print per-symbol table for "forex_exact" variant ───────────────────────
    exact = summary[summary["variant"] == "forex_exact"]
    cols = ["symbol", "total_trades", "winrate", "profit_factor",
            "profit_pct", "max_drawdown", "expectancy", "neg_months"]
    print("─── forex_exact config (per symbol) ───")
    print(exact[cols].to_string(index=False))

    # ── Print variant comparison (avg across symbols) ──────────────────────────
    agg = (
        summary.groupby("variant")[["total_trades", "winrate", "profit_factor",
                                     "profit_pct", "max_drawdown", "expectancy"]]
        .mean()
        .round(3)
        .sort_values("profit_factor", ascending=False)
    )
    print("\n─── Variant comparison (mean across symbols) ───")
    print(agg.to_string())

    # ── Save outputs ───────────────────────────────────────────────────────────
    write_csv(REPORT_DIR / "summary.csv", summary)
    if trade_frames:
        trades_all = pd.concat(trade_frames, ignore_index=True)
        write_csv(REPORT_DIR / "trades.csv", trades_all)
    if monthly_frames:
        monthly_all = pd.concat(monthly_frames, ignore_index=True)
        write_csv(REPORT_DIR / "monthly.csv", monthly_all)

    print(f"\nOutputs saved → {REPORT_DIR}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end",   default="2026-05-07")
    args = parser.parse_args()
    run(args.start, args.end)


if __name__ == "__main__":
    main()
