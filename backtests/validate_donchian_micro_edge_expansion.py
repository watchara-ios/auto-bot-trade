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


OUTPUT_DIR = REPORT_DIR / "micro_edge_expansion"


SESSION_WINDOWS = {
    "session_11": (11,),
    "session_10_12": (10, 11, 12),
    "session_09_13": (9, 10, 11, 12, 13),
    "session_08_14": (8, 9, 10, 11, 12, 13, 14),
}


def base_config() -> Config:
    return Config(
        use_volume_filter=True,
        volume_mult=1.2,
        use_atr_expansion=True,
        allowed_side="BUY",
        adx_min=20.0,
        adx_max=30.0,
    )


def enrich_trades(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades
    out = trades.copy()
    risk = (out["entry"] - out["sl"]).abs() * out["qty"]
    out["r_multiple"] = out["pnl"] / risk.replace(0, np.nan)
    out["gross_before_fee"] = out["pnl"] + out["fee"]
    return out


def profit_factor(pnl: pd.Series) -> float:
    gross_profit = pnl[pnl > 0].sum()
    gross_loss = -pnl[pnl <= 0].sum()
    return float(gross_profit / gross_loss) if gross_loss else 0.0


def gross_profit_factor(trades: pd.DataFrame) -> float:
    if trades.empty:
        return 0.0
    return profit_factor(trades["gross_before_fee"])


def max_drawdown(trades: pd.DataFrame, initial_balance: float = 1000.0) -> float:
    if trades.empty:
        return 0.0
    equity = initial_balance + trades["pnl"].cumsum()
    peak = equity.cummax()
    dd = (equity - peak) / peak.replace(0, np.nan) * 100
    return float(dd.min())


def top_2_month_contribution(trades: pd.DataFrame) -> float:
    monthly = monthly_performance(trades)
    if monthly.empty:
        return 0.0
    positive = monthly[monthly["pnl"] > 0]["pnl"]
    total_positive = positive.sum()
    if total_positive <= 0:
        return 1.0
    return float(positive.nlargest(2).sum() / total_positive)


def split_is_stable(summary: pd.DataFrame, variant: str) -> bool:
    checks = summary[(summary["variant"] == variant) & (summary["split"].isin(["train_2022_2023", "test_2024_2025"]))]
    if len(checks) < 2:
        return False
    return bool(
        (checks["profit_factor_net"] >= 1.1).all()
        and (checks["expectancy"] > 0).all()
        and (checks["max_drawdown"] >= -25).all()
        and (checks["top_2_month_contribution"] <= 0.40).all()
    )


def validity(row: pd.Series, stable: bool) -> tuple[bool, str]:
    reasons = []
    if row["profit_factor_net"] < 1.1:
        reasons.append("PF < 1.1")
    if row["expectancy"] <= 0:
        reasons.append("expectancy <= 0")
    if row["total_trades"] < 120:
        reasons.append("trades < 120")
    if row["max_drawdown"] < -25:
        reasons.append("max DD > 25%")
    if row["top_2_month_contribution"] > 0.40:
        reasons.append("top 2 months > 40%")
    if not stable:
        reasons.append("not stable across splits")
    return not reasons, "; ".join(reasons)


def run_one(
    symbol: str,
    variant: str,
    split: str,
    cfg: Config,
    m1_raw: pd.DataFrame,
    m5_raw: pd.DataFrame,
    m15_raw: pd.DataFrame,
):
    m1_s, m5_s, m15_s = split_data(m1_raw, m5_raw, m15_raw, split)
    if len(m5_s) < 300 or len(m15_s) < 220 or len(m1_s) < 1000:
        return None, pd.DataFrame(), pd.DataFrame()
    days = max((m5_s.index[-1].date() - m5_s.index[0].date()).days + 1, 1)
    m5 = prepare(m5_s, m15_s, cfg)
    trades, equity = run_backtest(symbol, m1_s, m5, cfg, variant, split)
    trades = enrich_trades(trades)
    metric = metrics(trades, equity, days)
    row = {
        "symbol": symbol,
        "variant": variant,
        "split": split,
        "days": days,
        **metric,
        "profit_factor_net": metric["profit_factor"],
        "profit_factor_gross": gross_profit_factor(trades),
        "top_2_month_contribution": top_2_month_contribution(trades),
        "avg_r": float(trades["r_multiple"].mean()) if not trades.empty else 0.0,
        "fee": float(trades["fee"].sum()) if not trades.empty else 0.0,
        "gross_before_fee": float(trades["gross_before_fee"].sum()) if not trades.empty else 0.0,
        "net_pnl": float(trades["pnl"].sum()) if not trades.empty else 0.0,
    }
    return row, trades, monthly_performance(trades)


def run_variants(symbol: str, variants: list[tuple[str, Config]], splits: list[str], m1_raw, m5_raw, m15_raw):
    rows = []
    trade_frames = []
    monthly_frames = []
    for name, cfg in variants:
        for split in splits:
            row, trades, monthly = run_one(symbol, name, split, cfg, m1_raw, m5_raw, m15_raw)
            if row is None:
                continue
            rows.append(row)
            if not trades.empty:
                trade_frames.append(trades)
                monthly_frames.append(monthly)
    return rows, trade_frames, monthly_frames


def phase1_variants() -> list[tuple[str, Config]]:
    base = base_config()
    return [
        (f"p1_{name}_atr70", replace(base, include_utc_hours=hours, atr_percentile_min=70.0))
        for name, hours in SESSION_WINDOWS.items()
    ]


def phase2_variants(best_session: str, hours: tuple[int, ...]) -> list[tuple[str, Config]]:
    base = base_config()
    return [
        (f"p2_{best_session}_atr{threshold:g}", replace(base, include_utc_hours=hours, atr_percentile_min=float(threshold)))
        for threshold in [70, 65, 60, 55]
    ]


def phase3_variants(top_configs: pd.DataFrame, variant_cfg: dict[str, Config]) -> list[tuple[str, Config]]:
    out = []
    for variant in top_configs["variant"].head(2):
        cfg = variant_cfg[variant]
        for fee in [0.0005, 0.0007, 0.0010]:
            for slip_min, slip_max, label in [(0.5, 1.5, "slip_base"), (1.0, 2.0, "slip_2x")]:
                out.append(
                    (
                        f"p3_{variant}_fee{fee:.4f}_{label}",
                        replace(cfg, fee_rate=fee, slippage_min_spread=slip_min, slippage_max_spread=slip_max),
                    )
                )
    return out


def fee_impact(trades: pd.DataFrame) -> pd.DataFrame:
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


def main():
    symbol = "BTCUSDT"
    files = find_symbol_files(symbol)
    if not files:
        raise RuntimeError("Missing BTCUSDT M1/M5/M15 files")

    m1_raw = load_csv(files["m1"])
    m5_raw = load_csv(files["m5"])
    m15_raw = load_csv(files["m15"])
    splits = ["available", "train_2022_2023", "test_2024_2025"]

    all_rows = []
    all_trades = []
    all_monthly = []

    p1 = phase1_variants()
    rows, trades, monthly = run_variants(symbol, p1, splits, m1_raw, m5_raw, m15_raw)
    all_rows.extend(rows)
    all_trades.extend(trades)
    all_monthly.extend(monthly)
    p1_summary = pd.DataFrame(rows)
    p1_available = p1_summary[p1_summary["split"] == "available"].sort_values(
        ["profit_factor_net", "expectancy"], ascending=[False, False]
    )
    best_p1_variant = str(p1_available.iloc[0]["variant"])
    best_session_key = best_p1_variant.replace("p1_", "").replace("_atr70", "")
    best_hours = SESSION_WINDOWS[best_session_key]

    p2 = phase2_variants(best_session_key, best_hours)
    p2_cfg_map = {name: cfg for name, cfg in p2}
    rows, trades, monthly = run_variants(symbol, p2, splits, m1_raw, m5_raw, m15_raw)
    all_rows.extend(rows)
    all_trades.extend(trades)
    all_monthly.extend(monthly)
    p2_summary = pd.DataFrame(rows)
    p2_available = p2_summary[p2_summary["split"] == "available"].sort_values(
        ["profit_factor_net", "expectancy"], ascending=[False, False]
    )

    p3 = phase3_variants(p2_available, p2_cfg_map)
    rows, trades, monthly = run_variants(symbol, p3, splits, m1_raw, m5_raw, m15_raw)
    all_rows.extend(rows)
    all_trades.extend(trades)
    all_monthly.extend(monthly)

    summary = pd.DataFrame(all_rows)
    for variant in summary["variant"].unique():
        stable = split_is_stable(summary, variant)
        mask = summary["variant"] == variant
        for idx in summary[mask].index:
            valid, reason = validity(summary.loc[idx], stable)
            summary.loc[idx, "valid_edge"] = valid
            summary.loc[idx, "invalid_reason"] = reason

    summary = summary.sort_values(["split", "profit_factor_net", "expectancy"], ascending=[True, False, False])
    trade_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    monthly_df = pd.concat(all_monthly, ignore_index=True) if all_monthly else pd.DataFrame()

    write_csv(OUTPUT_DIR / "summary.csv", summary)
    write_csv(OUTPUT_DIR / "trades.csv", trade_df)
    write_csv(OUTPUT_DIR / "monthly_performance.csv", monthly_df)
    write_csv(OUTPUT_DIR / "fee_impact.csv", fee_impact(trade_df))

    cols = [
        "split",
        "variant",
        "total_trades",
        "trades_per_day",
        "winrate",
        "profit_factor_gross",
        "profit_factor_net",
        "profit_pct",
        "max_drawdown",
        "expectancy",
        "top_2_month_contribution",
        "valid_edge",
        "invalid_reason",
    ]
    print("=== Donchian Micro-Edge Expansion ===")
    print(summary[summary["split"] == "available"][cols].head(20).to_string(index=False))
    print(f"Best Phase 1 session: {best_session_key}")
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
