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

from __future__ import annotations

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
#    → UTC hours 07:00–16:59 inclusive
SESSION_UTC = tuple(range(7, 17))

# ── Base config — mirrors forex_bot.py Config exactly ─────────────────────────
FOREX_CONFIG = Config(
    # Account / risk
    initial_balance   = 10_000.0,     # USD — adjust to your real account size
    fixed_risk_base   = True,
    tier_a_risk       = 0.0025,        # FOREX_TIER_A_RISK = 0.25%
    tier_b_risk       = 0.010,         # FOREX_TIER_B_RISK = 1.0%

    # Strategy — EMA matches forex_bot.py (EMA_FAST=20, EMA_SLOW=50, EMA_BIG=200)
    # Live bot uses EMA(20) vs EMA(50) for M15 trend — NOT the backtest default 50/200
    ema_fast          = 20,            # Config.EMA_FAST
    ema_slow          = 50,            # Config.EMA_SLOW
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
    m5_path  = DATA_DIR / f"{sym}_M5.csv"
    m15_path = DATA_DIR / f"{sym}_M15.csv"
    m1_path  = DATA_DIR / f"{sym}_M1.csv"

    for p, name in [(m5_path, "M5"), (m15_path, "M15")]:
        if not p.exists():
            print(f"  ⚠️  {sym}: missing {name} — run load_history_md5.py first")
            return None

    data = {
        "m5":  load_csv(m5_path),
        "m15": load_csv(m15_path),
    }
    # M1 optional — broker keeps ~3 months only; fall back to M5 for exit simulation
    if m1_path.exists():
        data["m1"] = load_csv(m1_path)
    else:
        print(f"  ℹ️  {sym}: M1 not found — using M5 for exit simulation")
        data["m1"] = data["m5"]
    return data


def _make_variants() -> list[tuple[str, Config]]:
    # EMA(50/200) = live bot config after fix
    ema_50_200 = replace(FOREX_CONFIG, ema_fast=50, ema_slow=200)
    return [
        # ── Live bot exact config (what's deployed right now) ─────────────────
        ("live_bot",          replace(ema_50_200,
                                      use_regime_filter=True, regime_adx_min=20.0)),

        # ── Baselines for comparison ──────────────────────────────────────────
        ("ema20_50_no_d1",    FOREX_CONFIG),          # old live bot (EMA 20/50, no D1)
        ("ema50_200_no_d1",   ema_50_200),            # EMA fix only, no D1

        # ── Regime combinations ───────────────────────────────────────────────
        ("d1_regime",         replace(FOREX_CONFIG,
                                      use_regime_filter=True, regime_adx_min=20.0)),
        ("live_weekly",       replace(ema_50_200,
                                      use_regime_filter=True, regime_adx_min=20.0,
                                      use_weekly_regime=True, weekly_adx_min=20.0)),

        # ── Tighter filters on live_bot ───────────────────────────────────────
        ("live_fresh",        replace(ema_50_200,
                                      use_regime_filter=True, regime_adx_min=20.0,
                                      require_fresh_breakout=True)),
        ("live_best",         replace(ema_50_200,
                                      use_regime_filter=True, regime_adx_min=20.0,
                                      require_fresh_breakout=True,
                                      atr_percentile_min=50.0, adx_min=20.0)),

        # ── Gold without session filter ───────────────────────────────────────
        ("gold_nosession",    replace(ema_50_200,
                                      include_utc_hours=None,
                                      use_regime_filter=True, regime_adx_min=20.0)),
    ]


def _build_exit_bars(data: dict, start: str, end: str) -> tuple[pd.DataFrame, str]:
    m1_full = data["m1"]
    m5_full = data["m5"]
    m1_range = m1_full.loc[start:end]
    m5_range = m5_full.loc[start:end]
    m1_start = m1_range.index[0] if not m1_range.empty else pd.Timestamp(end)
    start_ts = pd.Timestamp(start)
    if getattr(m1_start, "tzinfo", None) is not None and start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize(m1_start.tzinfo)
    exit_bars = pd.concat([m5_range[m5_range.index < m1_start], m1_range]).sort_index()
    note = ("M1" if m1_range.empty or m1_start <= start_ts
            else f"M5 until {m1_start.date()} then M1")
    return exit_bars, note


def _run_symbol(sym: str, data: dict, variants: list, start: str, end: str) -> tuple:
    m5_full  = data["m5"]
    m15_full = data["m15"]
    warmup_start = str(pd.Timestamp(start) - pd.DateOffset(months=6))[:10]

    m5r = m5_full.loc[start:end]
    if len(m5r) < 200:
        print(f"  skip {sym}: only {len(m5r)} M5 bars")
        return [], [], []

    days = max((m5r.index[-1].date() - m5r.index[0].date()).days + 1, 1)
    exit_bars, note = _build_exit_bars(data, start, end)
    print(f"  {sym}: exit={note}")

    rows, trades_list, monthly_list = [], [], []
    for name, cfg in variants:
        m5_prep = prepare(m5_full.loc[warmup_start:end], m15_full.loc[warmup_start:end], cfg)
        m5_prep = m5_prep.loc[start:end]
        trades, equity = run_backtest(sym, exit_bars, m5_prep, cfg, name, start[:4])
        row = {"symbol": sym, "variant": name, "days": days,
               **metrics(trades, equity, days, cfg.initial_balance)}
        if not trades.empty:
            monthly = monthly_performance(trades)
            row["neg_months"] = int((monthly["pnl"] < 0).sum())
            trades_list.append(trades)
            monthly_list.append(monthly)
        else:
            row["neg_months"] = 0
        rows.append(row)
    return rows, trades_list, monthly_list


def _print_summary(summary: pd.DataFrame) -> None:
    cols = ["symbol", "total_trades", "winrate", "profit_factor",
            "profit_pct", "max_drawdown", "expectancy", "neg_months"]
    live = summary[summary["variant"] == "live_bot"]
    print("─── live_bot — EMA(50/200) + D1 regime (deployed config) ───")
    print(live[cols].to_string(index=False))

    agg = (
        summary.groupby("variant")[["total_trades", "winrate", "profit_factor",
                                    "profit_pct", "max_drawdown", "expectancy"]]
        .mean().round(3).sort_values("profit_factor", ascending=False)
    )
    print("\n─── Variant mean across symbols (sorted by PF) ───")
    print(agg.to_string())


def _save_outputs(summary: pd.DataFrame, trade_frames: list, monthly_frames: list) -> None:
    write_csv(REPORT_DIR / "summary.csv", summary)
    write_csv(REPORT_DIR / "trades.csv",
              pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame())
    write_csv(REPORT_DIR / "monthly.csv",
              pd.concat(monthly_frames, ignore_index=True) if monthly_frames else pd.DataFrame())
    print(f"\nOutputs saved → {REPORT_DIR}")


def run(start: str, end: str) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"Forex Donchian Backtest — {start} to {end}")
    print(f"Symbols : {SYMBOLS}")
    print(f"Session : UTC {min(SESSION_UTC):02d}:00-{max(SESSION_UTC)+1:02d}:00  "
          f"EMA({FOREX_CONFIG.ema_fast}/{FOREX_CONFIG.ema_slow})  "
          f"RR={FOREX_CONFIG.rr}  ADX={FOREX_CONFIG.adx_min}-{FOREX_CONFIG.adx_max}")
    print(f"{'='*60}\n")

    variants = _make_variants()
    all_rows, all_trades, all_monthly = [], [], []

    for sym in SYMBOLS:
        data = load_symbol(sym)
        if data is None:
            continue
        rows, trades_list, monthly_list = _run_symbol(sym, data, variants, start, end)
        all_rows.extend(rows)
        all_trades.extend(trades_list)
        all_monthly.extend(monthly_list)

    if not all_rows:
        print("No results — check data files in data/mt5_history/")
        return

    summary = pd.DataFrame(all_rows)
    _print_summary(summary)
    _save_outputs(summary, all_trades, all_monthly)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end",   default="2026-05-07")
    args = parser.parse_args()
    run(args.start, args.end)


if __name__ == "__main__":
    main()
