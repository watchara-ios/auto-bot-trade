from dataclasses import replace
from pathlib import Path

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
    split_data,
    write_csv,
)


ABLATION_DIR = REPORT_DIR / "ablation"


def base_config() -> Config:
    return Config(use_volume_filter=True, volume_mult=1.2, use_atr_expansion=True)


def ablation_variants() -> list[tuple[str, Config]]:
    base = base_config()
    return [
        ("both_vol1_2_atr", base),
        ("no_tier_b", replace(base, disable_tier_b=True)),
        ("sell_only", replace(base, allowed_side="SELL")),
        ("buy_only", replace(base, allowed_side="BUY")),
        ("exclude_17_19_utc", replace(base, exclude_utc_hours=(17, 18, 19))),
        ("adx_cap_20_30", replace(base, adx_max=30.0)),
        ("atr_pct_gt_30", replace(base, atr_percentile_min=30.0)),
        ("exclude_saturday", replace(base, exclude_weekdays=("Saturday",))),
        (
            "combined_defensive",
            replace(
                base,
                disable_tier_b=True,
                exclude_weekdays=("Saturday",),
                exclude_utc_hours=(17, 18, 19),
                adx_max=30.0,
                atr_percentile_min=30.0,
            ),
        ),
    ]


def enrich_trade_metrics(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades
    out = trades.copy()
    initial_risk = (out["entry"] - out["sl"]).abs() * out["qty"]
    out["r_multiple"] = out["pnl"] / initial_risk.replace(0, np.nan)
    return out


def negative_months(trades: pd.DataFrame) -> int:
    monthly = monthly_performance(trades)
    if monthly.empty:
        return 0
    return int((monthly["pnl"] < 0).sum())


def run_symbol(symbol: str, split: str) -> tuple[list[dict], list[pd.DataFrame], list[pd.DataFrame]]:
    files = find_symbol_files(symbol)
    if not files:
        print(f"skip {symbol}: missing files")
        return [], [], []

    m1_raw = load_csv(files["m1"])
    m5_raw = load_csv(files["m5"])
    m15_raw = load_csv(files["m15"])
    m1_s, m5_s, m15_s = split_data(m1_raw, m5_raw, m15_raw, split)
    if len(m5_s) < 300 or len(m15_s) < 220 or len(m1_s) < 1000:
        print(f"skip {symbol} {split}: insufficient data m1={len(m1_s)} m5={len(m5_s)} m15={len(m15_s)}")
        return [], [], []

    days = max((m5_s.index[-1].date() - m5_s.index[0].date()).days + 1, 1)
    rows = []
    trade_frames = []
    monthly_frames = []
    for name, cfg in ablation_variants():
        m5 = prepare(m5_s, m15_s, cfg)
        trades, equity = run_backtest(symbol, m1_s, m5, cfg, name, split)
        trades = enrich_trade_metrics(trades)
        metric = metrics(trades, equity, days)
        metric["avg_r"] = float(trades["r_multiple"].mean()) if not trades.empty else 0.0
        metric["negative_months"] = negative_months(trades)
        rows.append({"symbol": symbol, "variant": name, "split": split, "days": days, **metric})
        if not trades.empty:
            trade_frames.append(trades)
            monthly_frames.append(monthly_performance(trades))
    return rows, trade_frames, monthly_frames


def main():
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    split = "available"
    summary_rows = []
    trade_frames = []
    monthly_frames = []

    for symbol in symbols:
        rows, trades, monthly = run_symbol(symbol, split)
        summary_rows.extend(rows)
        trade_frames.extend(trades)
        monthly_frames.extend(monthly)

    summary = pd.DataFrame(summary_rows)
    all_trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    all_monthly = pd.concat(monthly_frames, ignore_index=True) if monthly_frames else pd.DataFrame()

    if not summary.empty:
        summary = summary.sort_values(["profit_factor", "expectancy"], ascending=[False, False])

    write_csv(ABLATION_DIR / "summary.csv", summary)
    write_csv(ABLATION_DIR / "trades.csv", all_trades)
    write_csv(ABLATION_DIR / "monthly_performance.csv", all_monthly)

    cols = [
        "symbol",
        "variant",
        "total_trades",
        "winrate",
        "profit_factor",
        "profit_pct",
        "max_drawdown",
        "expectancy",
        "avg_r",
        "negative_months",
    ]
    print("=== Donchian Ablation Diagnostics ===")
    if summary.empty:
        print("No runnable data found.")
        return
    print(summary[cols].to_string(index=False))
    print(f"Outputs: {ABLATION_DIR}")


if __name__ == "__main__":
    main()
