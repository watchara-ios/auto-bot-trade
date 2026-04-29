import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from realistic_donchian_backtest import Config, DATA_DIR, REPORT_DIR, find_symbol_files, load_csv, prepare


OUTPUT_DIR = REPORT_DIR / "diagnostics"


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0 or pd.isna(denominator):
        return 0.0
    return numerator / denominator


def profit_factor(pnl: pd.Series) -> float:
    gross_profit = pnl[pnl > 0].sum()
    gross_loss = -pnl[pnl <= 0].sum()
    return safe_div(gross_profit, gross_loss)


def max_drawdown_from_pnl(pnl: pd.Series, initial_balance: float = 1000.0) -> float:
    equity = initial_balance + pnl.cumsum()
    peak = equity.cummax()
    dd = (equity - peak) / peak.replace(0, np.nan) * 100
    return float(dd.min()) if len(dd) else 0.0


def stats(group: pd.DataFrame) -> pd.Series:
    wins = group[group["pnl"] > 0]
    losses = group[group["pnl"] <= 0]
    return pd.Series(
        {
            "trades": len(group),
            "winrate": safe_div(len(wins), len(group)) * 100,
            "profit_factor": profit_factor(group["pnl"]),
            "profit_pct": group["pnl"].sum() / 1000.0 * 100,
            "max_drawdown": max_drawdown_from_pnl(group["pnl"]),
            "expectancy": group["pnl"].mean() if len(group) else 0.0,
            "avg_r": group["r_multiple"].mean() if "r_multiple" in group else 0.0,
            "avg_win": wins["pnl"].mean() if len(wins) else 0.0,
            "avg_loss": losses["pnl"].mean() if len(losses) else 0.0,
            "gross_pnl": group["gross_pnl"].sum() if "gross_pnl" in group else group["pnl"].sum(),
            "fee": group["fee"].sum() if "fee" in group else 0.0,
            "spread_cost_est": group["spread_cost_est"].sum() if "spread_cost_est" in group else 0.0,
            "slippage_cost_est": group["slippage_cost_est"].sum() if "slippage_cost_est" in group else 0.0,
            "net_pnl": group["pnl"].sum(),
        }
    )


def load_reports(report_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(report_dir / "summary.csv")
    trades = pd.read_csv(report_dir / "trades.csv")
    equity = pd.read_csv(report_dir / "equity_curve.csv")
    monthly = pd.read_csv(report_dir / "monthly_performance.csv")
    return summary, trades, equity, monthly


def enrich_with_trade_math(trades: pd.DataFrame) -> pd.DataFrame:
    trades = trades.copy()
    for col in ["timestamp", "entry_time", "exit_time"]:
        trades[col] = pd.to_datetime(trades[col], errors="coerce")

    risk_distance = (trades["entry"] - trades["sl"]).abs()
    initial_risk = risk_distance * trades["qty"]
    trades["initial_risk"] = initial_risk
    trades["r_multiple"] = trades["pnl"] / initial_risk.replace(0, np.nan)
    trades["gross_pnl"] = trades["pnl"] + trades["fee"]
    trades["spread_cost_est"] = trades["spread"] * trades["qty"]
    trades["slippage_cost_est"] = trades["slippage"] * trades["qty"]
    trades["session_utc"] = trades["timestamp"].dt.strftime("%H:00")
    trades["weekday"] = trades["timestamp"].dt.day_name()
    return trades


def feature_files_for_symbol(symbol: str) -> Optional[dict]:
    files = find_symbol_files(symbol)
    if files:
        return files

    prefix_map = {"BTCUSDT": "bitcoin", "ETHUSDT": "ethereum", "SOLUSDT": "solana"}
    prefix = prefix_map.get(symbol, symbol.replace("USDT", "").lower())
    candidates = sorted(DATA_DIR.glob(f"{prefix}_*_5m.csv"))
    for m5_path in candidates:
        m15_path = Path(str(m5_path).replace("_5m.csv", "_15m.csv"))
        if m15_path.exists():
            return {"m5": m5_path, "m15": m15_path}
    return None


def build_feature_table(symbols: list[str]) -> pd.DataFrame:
    frames = []
    for symbol in symbols:
        files = feature_files_for_symbol(symbol)
        if not files:
            continue
        m5 = load_csv(files["m5"])
        m15 = load_csv(files["m15"])
        prepared = prepare(m5, m15, Config())
        features = prepared[
            [
                "m15_adx",
                "m15_adx_rising",
                "atr",
                "volume_ratio",
                "donchian_high",
                "donchian_low",
                "swing_high",
                "swing_low",
            ]
        ].copy()
        features["symbol"] = symbol
        features["timestamp"] = features.index
        frames.append(features.reset_index(drop=True))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def enrich_with_market_features(trades: pd.DataFrame) -> pd.DataFrame:
    features = build_feature_table(sorted(trades["symbol"].unique()))
    if features.empty:
        trades["adx_bucket"] = "UNKNOWN"
        trades["atr_pct_bucket"] = "UNKNOWN"
        trades["volume_mult_bucket"] = "UNKNOWN"
        return trades

    out = trades.merge(features, on=["symbol", "timestamp"], how="left")
    out["adx_bucket"] = pd.cut(
        out["m15_adx"],
        bins=[-np.inf, 25, 30, np.inf],
        labels=["20-25", "25-30", "30+"],
    ).astype("object").fillna("UNKNOWN")

    out["atr_percentile"] = out.groupby("symbol")["atr"].rank(pct=True) * 100
    out["atr_pct_bucket"] = pd.cut(
        out["atr_percentile"],
        bins=[-np.inf, 30, 70, np.inf],
        labels=["0-30", "30-70", "70-100"],
    ).astype("object").fillna("UNKNOWN")

    out["volume_mult_bucket"] = pd.cut(
        out["volume_ratio"],
        bins=[-np.inf, 1.0, 1.2, 1.5, 2.0, np.inf],
        labels=["<1.0", "1.0-1.2", "1.2-1.5", "1.5-2.0", "2.0+"],
    ).astype("object").fillna("UNKNOWN")
    return out


def write_breakdown(df: pd.DataFrame, group_cols: list[str], path: Path):
    if df.empty:
        pd.DataFrame().to_csv(path, index=False)
        return
    result = df.groupby(group_cols, dropna=False).apply(stats, include_groups=False).reset_index()
    result = result.sort_values(["profit_factor", "expectancy"], ascending=[False, False])
    result.to_csv(path, index=False)


def cost_impact(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["symbol", "variant", "split"], dropna=False).agg(
        trades=("pnl", "count"),
        gross_before_fee=("gross_pnl", "sum"),
        fees=("fee", "sum"),
        spread_cost_est=("spread_cost_est", "sum"),
        slippage_cost_est=("slippage_cost_est", "sum"),
        net_pnl=("pnl", "sum"),
        avg_fee_per_trade=("fee", "mean"),
        avg_spread_cost_per_trade=("spread_cost_est", "mean"),
        avg_slippage_cost_per_trade=("slippage_cost_est", "mean"),
        avg_r=("r_multiple", "mean"),
    ).reset_index()
    grouped["fee_pct_of_abs_gross"] = grouped["fees"] / grouped["gross_before_fee"].abs().replace(0, np.nan) * 100
    grouped["spread_pct_of_abs_gross"] = grouped["spread_cost_est"] / grouped["gross_before_fee"].abs().replace(0, np.nan) * 100
    grouped["slippage_pct_of_abs_gross"] = grouped["slippage_cost_est"] / grouped["gross_before_fee"].abs().replace(0, np.nan) * 100
    return grouped


def diagnostic_summary(summary: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    trade_stats = trades.groupby(["symbol", "variant", "split"], dropna=False).apply(stats, include_groups=False).reset_index()
    out = summary.merge(
        trade_stats[["symbol", "variant", "split", "avg_r", "avg_win", "avg_loss", "gross_pnl", "fee", "spread_cost_est", "slippage_cost_est", "net_pnl"]],
        on=["symbol", "variant", "split"],
        how="left",
    )
    return out.sort_values(["profit_factor", "expectancy"], ascending=[False, False])


def print_edge_loss_hint(trades: pd.DataFrame):
    checks = {
        "tier": trades.groupby("tier")["pnl"].mean().sort_values() if "tier" in trades else pd.Series(dtype=float),
        "side": trades.groupby("side")["pnl"].mean().sort_values() if "side" in trades else pd.Series(dtype=float),
        "session_utc": trades.groupby("session_utc")["pnl"].mean().sort_values() if "session_utc" in trades else pd.Series(dtype=float),
        "adx_bucket": trades.groupby("adx_bucket")["pnl"].mean().sort_values() if "adx_bucket" in trades else pd.Series(dtype=float),
        "atr_pct_bucket": trades.groupby("atr_pct_bucket")["pnl"].mean().sort_values() if "atr_pct_bucket" in trades else pd.Series(dtype=float),
        "volume_mult_bucket": trades.groupby("volume_mult_bucket")["pnl"].mean().sort_values() if "volume_mult_bucket" in trades else pd.Series(dtype=float),
        "weekday": trades.groupby("weekday")["pnl"].mean().sort_values() if "weekday" in trades else pd.Series(dtype=float),
    }
    print("=== Edge Loss Quick Hints ===")
    for name, series in checks.items():
        if series.empty:
            continue
        worst = series.index[0]
        print(f"{name}: weakest={worst} avg_pnl={series.iloc[0]:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", default=str(REPORT_DIR))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()

    report_dir = Path(args.report_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary, trades, equity, monthly = load_reports(report_dir)
    trades = enrich_with_trade_math(trades)
    trades = enrich_with_market_features(trades)

    diagnostic_summary(summary, trades).to_csv(output_dir / "diagnostic_summary.csv", index=False)
    write_breakdown(trades, ["symbol", "variant", "split", "tier"], output_dir / "tier_breakdown.csv")
    write_breakdown(trades, ["symbol", "variant", "split", "side"], output_dir / "side_breakdown.csv")
    write_breakdown(trades, ["symbol", "variant", "split", "session_utc"], output_dir / "session_breakdown.csv")
    write_breakdown(trades, ["symbol", "variant", "split", "adx_bucket"], output_dir / "adx_bucket_breakdown.csv")
    write_breakdown(trades, ["symbol", "variant", "split", "atr_pct_bucket"], output_dir / "atr_bucket_breakdown.csv")
    write_breakdown(trades, ["symbol", "variant", "split", "volume_mult_bucket"], output_dir / "volume_bucket_breakdown.csv")
    write_breakdown(trades, ["symbol", "variant", "split", "weekday"], output_dir / "weekday_breakdown.csv")
    cost_impact(trades).to_csv(output_dir / "cost_impact.csv", index=False)

    # Keep copies of source-level summaries in the same diagnostics folder for easy sharing.
    equity.tail(1).to_csv(output_dir / "equity_last_points.csv", index=False)
    monthly.to_csv(output_dir / "monthly_performance_copy.csv", index=False)

    print(f"Diagnostics exported to: {output_dir}")
    print(f"trades analyzed: {len(trades)}")
    print_edge_loss_hint(trades)


if __name__ == "__main__":
    main()
