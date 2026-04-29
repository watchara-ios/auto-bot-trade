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


OUTPUT_DIR = REPORT_DIR / "adx_sensitivity"


def base_config() -> Config:
    return Config(use_volume_filter=True, volume_mult=1.2, use_atr_expansion=True)


def variants() -> list[tuple[str, Config]]:
    base = base_config()
    return [
        ("adx_20_25", replace(base, adx_min=20.0, adx_max=25.0)),
        ("adx_20_28", replace(base, adx_min=20.0, adx_max=28.0)),
        ("adx_20_30", replace(base, adx_min=20.0, adx_max=30.0)),
        ("adx_20_32", replace(base, adx_min=20.0, adx_max=32.0)),
        ("adx_22_30", replace(base, adx_min=22.0, adx_max=30.0)),
        ("adx_25_35", replace(base, adx_min=25.0, adx_max=35.0)),
        ("adx_gt_20_baseline", base),
    ]


def enrich_trades(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades
    out = trades.copy()
    risk = (out["entry"] - out["sl"]).abs() * out["qty"]
    out["r_multiple"] = out["pnl"] / risk.replace(0, np.nan)
    return out


def negative_months(trades: pd.DataFrame) -> int:
    monthly = monthly_performance(trades)
    if monthly.empty:
        return 0
    return int((monthly["pnl"] < 0).sum())


def profit_factor(pnl: pd.Series) -> float:
    gross_profit = pnl[pnl > 0].sum()
    gross_loss = -pnl[pnl <= 0].sum()
    return float(gross_profit / gross_loss) if gross_loss else 0.0


def group_breakdown(trades: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows = []
    for keys, group in trades.groupby(["variant", group_col], dropna=False):
        variant, bucket = keys
        wins = group[group["pnl"] > 0]
        rows.append(
            {
                "variant": variant,
                group_col: bucket,
                "trades": len(group),
                "winrate": len(wins) / len(group) * 100 if len(group) else 0.0,
                "profit_factor": profit_factor(group["pnl"]),
                "profit_pct": group["pnl"].sum() / 1000 * 100,
                "expectancy": group["pnl"].mean() if len(group) else 0.0,
                "avg_r": group["r_multiple"].mean() if "r_multiple" in group else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values(["variant", "profit_factor"], ascending=[True, False])


def main():
    symbol = "BTCUSDT"
    files = find_symbol_files(symbol)
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
        trades = enrich_trades(trades)
        row = {"symbol": symbol, "variant": name, "split": "available", "days": days, **metrics(trades, equity, days)}
        row["negative_months"] = negative_months(trades)
        row["avg_r"] = float(trades["r_multiple"].mean()) if not trades.empty else 0.0
        summary_rows.append(row)
        if not trades.empty:
            trade_frames.append(trades)
            monthly_frames.append(monthly_performance(trades))

    summary = pd.DataFrame(summary_rows).sort_values(["profit_factor", "expectancy"], ascending=[False, False])
    trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    monthly = pd.concat(monthly_frames, ignore_index=True) if monthly_frames else pd.DataFrame()
    tier_breakdown = group_breakdown(trades, "tier")
    side_breakdown = group_breakdown(trades, "side")

    write_csv(OUTPUT_DIR / "summary.csv", summary)
    write_csv(OUTPUT_DIR / "trades.csv", trades)
    write_csv(OUTPUT_DIR / "monthly_performance.csv", monthly)
    write_csv(OUTPUT_DIR / "tier_breakdown.csv", tier_breakdown)
    write_csv(OUTPUT_DIR / "side_breakdown.csv", side_breakdown)

    cols = [
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
    print("=== Donchian ADX Sensitivity ===")
    print(summary[cols].to_string(index=False))
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
