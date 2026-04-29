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


OUTPUT_DIR = REPORT_DIR / "micro_edge_final_validation"

SESSION_WINDOWS = {
    "session_09_13": (9, 10, 11, 12, 13),
    "session_08_13": (8, 9, 10, 11, 12, 13),
    "session_09_14": (9, 10, 11, 12, 13, 14),
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


def top_2_month_contribution(trades: pd.DataFrame) -> float:
    monthly = monthly_performance(trades)
    if monthly.empty:
        return 0.0
    positive = monthly[monthly["pnl"] > 0]["pnl"]
    total_positive = positive.sum()
    if total_positive <= 0:
        return 1.0
    return float(positive.nlargest(2).sum() / total_positive)


def run_one(symbol, variant, split, cfg, m1_raw, m5_raw, m15_raw):
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
        "profit_factor_gross": profit_factor(trades["gross_before_fee"]) if not trades.empty else 0.0,
        "top_2_month_contribution": top_2_month_contribution(trades),
        "avg_r": float(trades["r_multiple"].mean()) if not trades.empty else 0.0,
        "fee": float(trades["fee"].sum()) if not trades.empty else 0.0,
        "gross_before_fee": float(trades["gross_before_fee"].sum()) if not trades.empty else 0.0,
        "net_pnl": float(trades["pnl"].sum()) if not trades.empty else 0.0,
    }
    return row, trades, monthly_performance(trades)


def run_variants(symbol, variants, splits, m1_raw, m5_raw, m15_raw):
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


def phase_a_variants():
    base = base_config()
    return [
        (f"phase_a_{name}_atr70", replace(base, include_utc_hours=hours, atr_percentile_min=70.0))
        for name, hours in SESSION_WINDOWS.items()
    ]


def phase_b_variants(best_session, best_hours):
    base = base_config()
    return [
        (
            f"phase_b_{best_session}_atr{atr_threshold:g}",
            replace(base, include_utc_hours=best_hours, atr_percentile_min=float(atr_threshold)),
        )
        for atr_threshold in [70, 65, 60]
    ]


def stress_variants(best_configs, cfg_map):
    out = []
    for variant in best_configs["variant"].head(2):
        cfg = cfg_map[variant]
        for fee in [0.0005, 0.0007, 0.0010]:
            for slip_min, slip_max, slip_label in [(0.5, 1.5, "slip_base"), (1.0, 2.0, "slip_2x")]:
                out.append(
                    (
                        f"stress_{variant}_fee{fee:.4f}_{slip_label}",
                        replace(cfg, fee_rate=fee, slippage_min_spread=slip_min, slippage_max_spread=slip_max),
                    )
                )
        out.append((f"stress_{variant}_delay_1m", replace(cfg, entry_delay_minutes=1)))
    return out


def split_stability(summary: pd.DataFrame, variant: str) -> bool:
    checks = summary[(summary["variant"] == variant) & (summary["split"].isin(["train_2022_2023", "test_2024_2025"]))]
    if len(checks) < 2:
        return False
    return bool((checks["profit_factor_net"] >= 1.15).all())


def stress_ok(summary: pd.DataFrame, variant: str) -> bool:
    if not variant.startswith("stress_"):
        return True
    row = summary[(summary["variant"] == variant) & (summary["split"] == "available")]
    if row.empty:
        return False
    row = row.iloc[0]
    return bool(row["profit_factor_net"] >= 1.10 and row["max_drawdown"] >= -15)


def valid_edge(row: pd.Series, stable: bool) -> tuple[bool, str]:
    reasons = []
    if row["profit_factor_net"] < 1.20:
        reasons.append("PF net < 1.20")
    if row["max_drawdown"] < -10:
        reasons.append("DD > 10%")
    if row["total_trades"] < 150:
        reasons.append("trades < 150")
    if row["top_2_month_contribution"] > 0.35:
        reasons.append("top 2 months > 35%")
    if not stable:
        reasons.append("split PF < 1.15")
    return not reasons, "; ".join(reasons)


def rolling_6m_metrics(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    out = trades.copy()
    out["exit_time"] = pd.to_datetime(out["exit_time"])
    rows = []
    for variant, group in out.groupby("variant"):
        group = group.sort_values("exit_time").set_index("exit_time")
        start = group.index.min().to_period("M").to_timestamp()
        end = group.index.max().to_period("M").to_timestamp()
        window_start = start
        while window_start <= end:
            window_end = window_start + pd.DateOffset(months=6) - pd.Timedelta(seconds=1)
            w = group[(group.index >= window_start) & (group.index <= window_end)]
            if len(w):
                rows.append(
                    {
                        "variant": variant,
                        "window_start": window_start,
                        "window_end": window_end,
                        "trades": len(w),
                        "profit_factor_net": profit_factor(w["pnl"]),
                        "pnl": w["pnl"].sum(),
                        "expectancy": w["pnl"].mean(),
                        "winrate": (w["pnl"] > 0).mean() * 100,
                    }
                )
            window_start += pd.DateOffset(months=1)
    return pd.DataFrame(rows)


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

    rows_all = []
    trades_all = []
    monthly_all = []

    phase_a = phase_a_variants()
    rows, trades, monthly = run_variants(symbol, phase_a, splits, m1_raw, m5_raw, m15_raw)
    rows_all.extend(rows)
    trades_all.extend(trades)
    monthly_all.extend(monthly)
    phase_a_summary = pd.DataFrame(rows)
    phase_a_available = phase_a_summary[phase_a_summary["split"] == "available"].sort_values(
        ["profit_factor_net", "expectancy"], ascending=[False, False]
    )
    best_a_variant = str(phase_a_available.iloc[0]["variant"])
    best_session = best_a_variant.replace("phase_a_", "").replace("_atr70", "")
    best_hours = SESSION_WINDOWS[best_session]

    phase_b = phase_b_variants(best_session, best_hours)
    phase_b_cfg = {name: cfg for name, cfg in phase_b}
    rows, trades, monthly = run_variants(symbol, phase_b, splits, m1_raw, m5_raw, m15_raw)
    rows_all.extend(rows)
    trades_all.extend(trades)
    monthly_all.extend(monthly)
    phase_b_summary = pd.DataFrame(rows)
    phase_b_available = phase_b_summary[phase_b_summary["split"] == "available"].sort_values(
        ["profit_factor_net", "expectancy"], ascending=[False, False]
    )

    stress = stress_variants(phase_b_available, phase_b_cfg)
    rows, trades, monthly = run_variants(symbol, stress, splits, m1_raw, m5_raw, m15_raw)
    rows_all.extend(rows)
    trades_all.extend(trades)
    monthly_all.extend(monthly)

    summary = pd.DataFrame(rows_all)
    for variant in summary["variant"].unique():
        stable = split_stability(summary, variant)
        for idx in summary[summary["variant"] == variant].index:
            valid, reason = valid_edge(summary.loc[idx], stable)
            if str(summary.loc[idx, "variant"]).startswith("stress_") and not stress_ok(summary, summary.loc[idx, "variant"]):
                valid = False
                reason = (reason + "; " if reason else "") + "stress failed"
            summary.loc[idx, "valid_edge"] = valid
            summary.loc[idx, "invalid_reason"] = reason

    trades_df = pd.concat(trades_all, ignore_index=True) if trades_all else pd.DataFrame()
    monthly_df = pd.concat(monthly_all, ignore_index=True) if monthly_all else pd.DataFrame()
    rolling_df = rolling_6m_metrics(trades_df[trades_df["split"] == "available"]) if not trades_df.empty else pd.DataFrame()

    summary = summary.sort_values(["split", "profit_factor_net", "expectancy"], ascending=[True, False, False])
    write_csv(OUTPUT_DIR / "summary.csv", summary)
    write_csv(OUTPUT_DIR / "trades.csv", trades_df)
    write_csv(OUTPUT_DIR / "monthly_performance.csv", monthly_df)
    write_csv(OUTPUT_DIR / "fee_impact.csv", fee_impact(trades_df))
    write_csv(OUTPUT_DIR / "rolling_6m_metrics.csv", rolling_df)

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
    print("=== Donchian Micro-Edge Final Validation ===")
    print(summary[summary["split"] == "available"][cols].head(24).to_string(index=False))
    print(f"Best Phase A session: {best_session}")
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
