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
    split_data,
    write_csv,
)


OUTPUT_DIR = REPORT_DIR / "micro_edge_validation"


def base_config() -> Config:
    return Config(
        use_volume_filter=True,
        volume_mult=1.2,
        use_atr_expansion=True,
        allowed_side="BUY",
        adx_min=20.0,
        adx_max=30.0,
        atr_percentile_min=70.0,
    )


def variants() -> list[tuple[str, Config]]:
    base = base_config()
    return [
        ("buy_atr70_session03", replace(base, include_utc_hours=(3,))),
        ("buy_atr70_session11", replace(base, include_utc_hours=(11,))),
        ("buy_atr70_session03_11", replace(base, include_utc_hours=(3, 11))),
    ]


def enrich_trades(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades
    out = trades.copy()
    risk = (out["entry"] - out["sl"]).abs() * out["qty"]
    out["r_multiple"] = out["pnl"] / risk.replace(0, np.nan)
    out["gross_before_fee"] = out["pnl"] + out["fee"]
    return out


def negative_months(trades: pd.DataFrame) -> int:
    monthly = monthly_performance(trades)
    if monthly.empty:
        return 0
    return int((monthly["pnl"] < 0).sum())


def dominant_month_share(trades: pd.DataFrame) -> float:
    monthly = monthly_performance(trades)
    if monthly.empty:
        return 0.0
    positive = monthly[monthly["pnl"] > 0]["pnl"]
    total_positive = positive.sum()
    if total_positive <= 0:
        return 1.0
    return float(positive.nlargest(2).sum() / total_positive)


def cost_breakdown(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows = []
    for keys, group in trades.groupby(["variant", "split"], dropna=False):
        variant, split = keys
        rows.append(
            {
                "variant": variant,
                "split": split,
                "trades": len(group),
                "gross_before_fee": group["gross_before_fee"].sum(),
                "fee": group["fee"].sum(),
                "spread_cost_est": (group["spread"] * group["qty"]).sum(),
                "slippage_cost_est": (group["slippage"] * group["qty"]).sum(),
                "net_pnl": group["pnl"].sum(),
                "fee_per_trade": group["fee"].mean(),
            }
        )
    return pd.DataFrame(rows)


def validity(row: dict) -> tuple[bool, str]:
    reasons = []
    if row["profit_factor"] < 1.1:
        reasons.append("PF < 1.1")
    if row["expectancy"] <= 0:
        reasons.append("expectancy <= 0")
    if row["total_trades"] < 120:
        reasons.append("trades < 120")
    if row["max_drawdown"] < -25:
        reasons.append("max DD > 25%")
    if row["dominant_2_month_share"] > 0.5:
        reasons.append("dominated by top 2 months")
    return not reasons, "; ".join(reasons)


def run_split(symbol: str, split: str, m1_raw: pd.DataFrame, m5_raw: pd.DataFrame, m15_raw: pd.DataFrame):
    m1_s, m5_s, m15_s = split_data(m1_raw, m5_raw, m15_raw, split)
    if len(m5_s) < 300 or len(m15_s) < 220 or len(m1_s) < 1000:
        return [], [], []
    days = max((m5_s.index[-1].date() - m5_s.index[0].date()).days + 1, 1)
    rows = []
    trade_frames = []
    monthly_frames = []
    for name, cfg in variants():
        m5 = prepare(m5_s, m15_s, cfg)
        trades, equity = run_backtest(symbol, m1_s, m5, cfg, name, split)
        trades = enrich_trades(trades)
        row = {"symbol": symbol, "variant": name, "split": split, "days": days, **metrics(trades, equity, days)}
        row["avg_r"] = float(trades["r_multiple"].mean()) if not trades.empty else 0.0
        row["negative_months"] = negative_months(trades)
        row["dominant_2_month_share"] = dominant_month_share(trades)
        valid, reason = validity(row)
        row["valid_edge"] = valid
        row["invalid_reason"] = reason
        rows.append(row)
        if not trades.empty:
            trade_frames.append(trades)
            monthly_frames.append(monthly_performance(trades))
    return rows, trade_frames, monthly_frames


def main():
    symbol = "BTCUSDT"
    files = find_symbol_files(symbol)
    if not files:
        raise RuntimeError("Missing BTCUSDT M1/M5/M15 files")

    m1_raw = load_csv(files["m1"])
    m5_raw = load_csv(files["m5"])
    m15_raw = load_csv(files["m15"])
    splits = ["available", "train_2022_2023", "test_2024_2025"]

    summary_rows = []
    trade_frames = []
    monthly_frames = []
    for split in splits:
        rows, trades, monthly = run_split(symbol, split, m1_raw, m5_raw, m15_raw)
        summary_rows.extend(rows)
        trade_frames.extend(trades)
        monthly_frames.extend(monthly)

    summary = pd.DataFrame(summary_rows).sort_values(["split", "profit_factor", "expectancy"], ascending=[True, False, False])
    trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    monthly = pd.concat(monthly_frames, ignore_index=True) if monthly_frames else pd.DataFrame()

    write_csv(OUTPUT_DIR / "summary.csv", summary)
    write_csv(OUTPUT_DIR / "trades.csv", trades)
    write_csv(OUTPUT_DIR / "monthly_performance.csv", monthly)
    write_csv(OUTPUT_DIR / "cost_breakdown.csv", cost_breakdown(trades))

    cols = [
        "split",
        "variant",
        "total_trades",
        "winrate",
        "profit_factor",
        "profit_pct",
        "max_drawdown",
        "expectancy",
        "trades_per_day",
        "negative_months",
        "dominant_2_month_share",
        "valid_edge",
        "invalid_reason",
    ]
    print("=== Donchian Micro-Edge Validation ===")
    print(summary[cols].to_string(index=False))
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
