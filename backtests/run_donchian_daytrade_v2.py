"""
Donchian Daytrade Optimization v2
==================================
Key insight from ADX sensitivity study:
  - Tier B only: 45% winrate, PF=1.29, +19%
  - Tier A only: 29% winrate, PF=0.64, -7%  (net loser)
  - Best ADX range: adx_min=20, adx_max=28

This script tests combinations of:
  1. require_tier_b=True  (lock out losing Tier A entries)
  2. Session filters     (London 07-12 UTC, NY 13-20 UTC, combined)
  3. RR variations       (1.5, 2.0, 2.5)
  4. Max hold time       (120 min, 180 min, 240 min — true daytrade)
  5. max_trades_per_day  (2 vs 3)
  6. Side isolation      (SELL-only, BUY-only — SELL side is stronger)
"""

from dataclasses import replace

import numpy as np
import pandas as pd

from realistic_donchian_backtest import (
    Config,
    REPORT_DIR,
    find_symbol_files,
    load_csv,
    metrics,
    monthly_performance,
    prepare,
    run_backtest,
    write_csv,
)

OUTPUT_DIR = REPORT_DIR / "daytrade_v2"

LONDON_HOURS = tuple(range(7, 13))    # 07-12 UTC
NY_HOURS = tuple(range(13, 21))       # 13-20 UTC
LONDON_NY = tuple(range(7, 21))       # 07-20 UTC combined
ACTIVE_HOURS = (0, 1, 2) + LONDON_NY  # crypto 00-02 + London/NY


def base_tierb() -> Config:
    """Best config from ADX sensitivity + require Tier B."""
    return Config(
        adx_min=20.0,
        adx_max=28.0,
        require_tier_b=True,
        use_volume_filter=True,
        volume_mult=1.2,
        use_atr_expansion=True,
    )


def variants() -> list[tuple[str, Config]]:
    base = base_tierb()
    return [
        # --- Reference: best prior result WITHOUT require_tier_b ---
        ("ref_adx20_28_all_tiers", replace(base, require_tier_b=False)),

        # --- Tier B baseline (no session filter, no hold limit) ---
        ("tierb_baseline", base),

        # --- Session filters ---
        ("tierb_london_only", replace(base, include_utc_hours=LONDON_HOURS)),
        ("tierb_ny_only", replace(base, include_utc_hours=NY_HOURS)),
        ("tierb_london_ny", replace(base, include_utc_hours=LONDON_NY)),
        ("tierb_active_24h", replace(base, include_utc_hours=ACTIVE_HOURS)),

        # --- RR optimization on Tier B baseline ---
        ("tierb_rr15", replace(base, rr=1.5)),
        ("tierb_rr18", replace(base, rr=1.8)),
        ("tierb_rr25", replace(base, rr=2.5)),

        # --- Max hold time (true daytrade: close within N hours) ---
        ("tierb_hold2h", replace(base, max_hold_minutes=120)),
        ("tierb_hold3h", replace(base, max_hold_minutes=180)),
        ("tierb_hold4h", replace(base, max_hold_minutes=240)),

        # --- Session + RR combos (London/NY is best session candidate) ---
        ("tierb_lnny_rr15", replace(base, include_utc_hours=LONDON_NY, rr=1.5)),
        ("tierb_lnny_rr18", replace(base, include_utc_hours=LONDON_NY, rr=1.8)),
        ("tierb_lnny_rr25", replace(base, include_utc_hours=LONDON_NY, rr=2.5)),

        # --- Session + hold time ---
        ("tierb_lnny_hold2h", replace(base, include_utc_hours=LONDON_NY, max_hold_minutes=120)),
        ("tierb_lnny_hold3h", replace(base, include_utc_hours=LONDON_NY, max_hold_minutes=180)),

        # --- Session + RR + hold (triple combo) ---
        ("tierb_lnny_rr15_hold3h", replace(base, include_utc_hours=LONDON_NY, rr=1.5, max_hold_minutes=180)),
        ("tierb_lnny_rr18_hold3h", replace(base, include_utc_hours=LONDON_NY, rr=1.8, max_hold_minutes=180)),

        # --- More trades per day ---
        ("tierb_lnny_max3", replace(base, include_utc_hours=LONDON_NY, max_trades_per_day=3)),

        # --- Side isolation: SELL outperforms BUY historically ---
        ("tierb_sell_only", replace(base, allowed_side="SELL")),
        ("tierb_buy_only", replace(base, allowed_side="BUY")),
        ("tierb_lnny_sell_only", replace(base, include_utc_hours=LONDON_NY, allowed_side="SELL")),
        ("tierb_lnny_buy_only", replace(base, include_utc_hours=LONDON_NY, allowed_side="BUY")),

        # --- Tighter ADX window on Tier B ---
        ("tierb_adx22_28", replace(base, adx_min=22.0, adx_max=28.0)),
        ("tierb_adx20_30", replace(base, adx_max=30.0)),
    ]


def profit_factor(pnl: pd.Series) -> float:
    gross_profit = pnl[pnl > 0].sum()
    gross_loss = -pnl[pnl <= 0].sum()
    return float(gross_profit / gross_loss) if gross_loss else 0.0


def enrich(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades
    t = trades.copy()
    risk = (t["entry"] - t["sl"]).abs() * t["qty"]
    t["r_multiple"] = t["pnl"] / risk.replace(0, np.nan)
    if "entry_time" in t.columns and "exit_time" in t.columns:
        t["hold_minutes"] = (
            pd.to_datetime(t["exit_time"]) - pd.to_datetime(t["entry_time"])
        ).dt.total_seconds() / 60
    return t


def negative_months(trades: pd.DataFrame) -> int:
    monthly = monthly_performance(trades)
    if monthly.empty:
        return 0
    return int((monthly["pnl"] < 0).sum())


def result_breakdown(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows = []
    for (variant, result), grp in trades.groupby(["variant", "result"]):
        rows.append({
            "variant": variant,
            "result": result,
            "count": len(grp),
            "pnl": grp["pnl"].sum(),
            "avg_pnl": grp["pnl"].mean(),
        })
    return pd.DataFrame(rows).sort_values(["variant", "result"])


def load_365d_files(symbol: str) -> dict | None:
    """Explicitly load bitcoin_365d files (most recent 1 year)."""
    from realistic_donchian_backtest import DATA_DIR
    prefix_map = {"BTCUSDT": "bitcoin", "ETHUSDT": "ethereum", "SOLUSDT": "solana"}
    prefix = prefix_map.get(symbol, symbol.lower())
    paths = {
        "m1": DATA_DIR / f"{prefix}_365d_1m.csv",
        "m5": DATA_DIR / f"{prefix}_365d_5m.csv",
        "m15": DATA_DIR / f"{prefix}_365d_15m.csv",
    }
    if not all(p.exists() for p in paths.values()):
        return None
    return paths


def main():
    symbol = "BTCUSDT"
    # Use 365d (most recent year) — same dataset as adx_sensitivity for fair comparison.
    # The 2022-2025 dataset includes the 2022 bear market which skews results.
    files = load_365d_files(symbol) or find_symbol_files(symbol)
    if not files:
        raise RuntimeError("Missing BTCUSDT M1/M5/M15 data files.")

    m1 = load_csv(files["m1"])
    m5_raw = load_csv(files["m5"])
    m15 = load_csv(files["m15"])
    days = max((m5_raw.index[-1].date() - m5_raw.index[0].date()).days + 1, 1)

    summary_rows = []
    trade_frames = []
    monthly_frames = []

    for name, cfg in variants():
        m5 = prepare(m5_raw, m15, cfg)
        trades, equity = run_backtest(symbol, m1, m5, cfg, name, "available")
        trades = enrich(trades)
        row = {
            "symbol": symbol,
            "variant": name,
            "split": "available",
            "days": days,
            **metrics(trades, equity, days),
        }
        row["negative_months"] = negative_months(trades)
        row["avg_r"] = float(trades["r_multiple"].mean()) if not trades.empty else 0.0
        row["avg_hold_min"] = float(trades["hold_minutes"].mean()) if "hold_minutes" in trades.columns and not trades.empty else 0.0
        row["timeout_pct"] = (
            float((trades["result"] == "TIMEOUT").mean() * 100) if not trades.empty and "result" in trades.columns else 0.0
        )
        summary_rows.append(row)
        if not trades.empty:
            trade_frames.append(trades)
            monthly_frames.append(monthly_performance(trades))

    summary = pd.DataFrame(summary_rows).sort_values(
        ["profit_factor", "expectancy"], ascending=[False, False]
    )
    trades_all = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    monthly_all = pd.concat(monthly_frames, ignore_index=True) if monthly_frames else pd.DataFrame()
    result_bdown = result_breakdown(trades_all)

    write_csv(OUTPUT_DIR / "summary.csv", summary)
    write_csv(OUTPUT_DIR / "trades.csv", trades_all)
    write_csv(OUTPUT_DIR / "monthly_performance.csv", monthly_all)
    write_csv(OUTPUT_DIR / "result_breakdown.csv", result_bdown)

    cols = [
        "variant",
        "total_trades",
        "winrate",
        "profit_factor",
        "profit_pct",
        "max_drawdown",
        "expectancy",
        "avg_r",
        "avg_hold_min",
        "timeout_pct",
        "negative_months",
    ]
    print("=== Donchian Daytrade Optimization v2 ===")
    print(summary[cols].to_string(index=False))
    print(f"\nOutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
