from __future__ import annotations

import argparse
import itertools
import sys
import warnings
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from strategies.gold_v2_core import (  # noqa: E402
    GoldV2Config,
    load_csv,
    metrics,
    monthly_performance,
    prepare,
    profit_factor,
    run_backtest,
)


SYMBOL = "XAUUSDm"
DATA_DIR = ROOT / "data" / "mt5_history"
OUTPUT_DIR = ROOT / "outputs" / "gold_v2_directional"
SESSION_19_21 = ((12 * 60, 14 * 60),)
RECENT_START = "2026-04-01"
RECENT_END = "2026-05-07"


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        load_csv(DATA_DIR / f"{SYMBOL}_M1.csv"),
        load_csv(DATA_DIR / f"{SYMBOL}_M5.csv"),
        load_csv(DATA_DIR / f"{SYMBOL}_M15.csv"),
    )


def _exit_bars(m1: pd.DataFrame, m5: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    m1_range = m1.loc[start:end]
    m5_range = m5.loc[start:end]
    if m1_range.empty:
        return m5_range
    m1_start = m1_range.index[0]
    return pd.concat([m5_range[m5_range.index < m1_start], m1_range]).sort_index()


def base_config() -> GoldV2Config:
    return GoldV2Config(
        strategy="momentum",
        variant="base_asym",
        session_ranges_utc_minutes=SESSION_19_21,
        adx_min=18.0,
        momentum_rr=1.8,
        use_atr_expansion=False,
        body_mult=2.0,
        max_wick_pct=0.40,
        momentum_atr_percentile_min=40.0,
        max_trades_per_day=3,
        max_hold_minutes=720,
    )


def configs() -> list[GoldV2Config]:
    out = [
        replace(base_config(), variant="candidate_current"),
        replace(base_config(), variant="sell_only", allowed_side="SELL"),
        replace(base_config(), variant="buy_only", allowed_side="BUY"),
    ]
    for body, wick, atr_min, close in itertools.product(
        [2.2, 2.5],
        [0.35, 0.30],
        [45.0, 50.0],
        [None, 0.20],
    ):
        close_name = "close20" if close is not None else "closeOff"
        out.append(
            replace(
                base_config(),
                variant=f"buy_strict_body{body:g}_wick{wick:g}_atr{atr_min:g}_{close_name}",
                allowed_side="BUY",
                body_mult=body,
                max_wick_pct=wick,
                momentum_atr_percentile_min=atr_min,
                close_extreme_pct=close,
            )
        )
        out.append(
            replace(
                base_config(),
                variant=f"asym_sellBase_buy_body{body:g}_wick{wick:g}_atr{atr_min:g}_{close_name}",
                buy_body_mult=body,
                buy_max_wick_pct=wick,
                buy_atr_percentile_min=atr_min,
                buy_close_extreme_pct=close,
            )
        )
    return out


def _days(df: pd.DataFrame) -> int:
    if df.empty:
        return 1
    return max((df.index[-1].date() - df.index[0].date()).days + 1, 1)


def _max_consecutive_losses(trades: pd.DataFrame) -> int:
    run = 0
    max_run = 0
    for pnl in trades["pnl"] if not trades.empty else []:
        if pnl <= 0:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    return max_run


def _month_drop(trades: pd.DataFrame, drop_n: int) -> tuple[float, float]:
    if trades.empty:
        return 0.0, 0.0
    df = trades.copy()
    df["month"] = pd.to_datetime(df["exit_time"]).dt.to_period("M").astype(str)
    drop_months = df.groupby("month")["pnl"].sum().sort_values(ascending=False).head(drop_n).index
    keep = df[~df["month"].isin(drop_months)] if drop_n else df
    return profit_factor(keep["pnl"]), keep["pnl"].sum() / base_config().initial_balance * 100


def _side_pf(trades: pd.DataFrame, side: str) -> tuple[float, float, int]:
    g = trades[trades["side"] == side] if not trades.empty else pd.DataFrame()
    if g.empty:
        return 0.0, 0.0, 0
    return profit_factor(g["pnl"]), g["pnl"].sum(), len(g)


def _recent_reason(row: dict) -> str:
    if row["recent_trades"] < 5:
        return "low_trade_count"
    if row["recent_pf"] < 1.0 and row["recent_avg_r"] < 0:
        return "broken_recent_edge"
    return "non_catastrophic"


def _evaluate(
    config: GoldV2Config,
    m1: pd.DataFrame,
    m5_prepared: pd.DataFrame,
    exit_bars: pd.DataFrame,
    recent_prepared: pd.DataFrame,
    recent_exit: pd.DataFrame,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    trades, equity = run_backtest(SYMBOL, m1, m5_prepared, exit_bars, config)
    monthly = monthly_performance(trades)
    row = {
        "variant": config.variant,
        **metrics(trades, equity, _days(m5_prepared), config.initial_balance),
        "positive_months": int((monthly["pnl"] > 0).sum()) if not monthly.empty else 0,
        "worst_month": float(monthly["pnl"].min()) if not monthly.empty else 0.0,
        "max_consecutive_losses": _max_consecutive_losses(trades),
        "drop_best_1_pf": _month_drop(trades, 1)[0],
        "drop_best_1_profit_pct": _month_drop(trades, 1)[1],
        "drop_best_2_pf": _month_drop(trades, 2)[0],
        "drop_best_2_profit_pct": _month_drop(trades, 2)[1],
        "config": str(asdict(config)),
    }
    buy_pf, buy_pnl, buy_trades = _side_pf(trades, "BUY")
    sell_pf, sell_pnl, sell_trades = _side_pf(trades, "SELL")
    row.update(
        {
            "buy_pf": buy_pf,
            "buy_pnl": buy_pnl,
            "buy_trades": buy_trades,
            "sell_pf": sell_pf,
            "sell_pnl": sell_pnl,
            "sell_trades": sell_trades,
        }
    )
    recent_trades, recent_equity = run_backtest(SYMBOL, m1, recent_prepared, recent_exit, config)
    recent_metrics = metrics(recent_trades, recent_equity, _days(recent_prepared), config.initial_balance)
    row.update(
        {
            "recent_pf": recent_metrics["profit_factor"],
            "recent_profit_pct": recent_metrics["profit_pct"],
            "recent_max_drawdown": recent_metrics["max_drawdown"],
            "recent_trades": recent_metrics["total_trades"],
            "recent_avg_r": recent_metrics["avg_r"],
        }
    )
    row["recent_issue"] = _recent_reason(row)
    return row, trades, monthly


def _stress_pf(config: GoldV2Config, m1: pd.DataFrame, prepared: pd.DataFrame, exit_bars: pd.DataFrame) -> float:
    worst = float("inf")
    for spread_mult, slip_mult in itertools.product([1.0, 1.5, 2.0], [0.0, 0.5, 1.0]):
        stress = replace(
            config,
            spread_atr_mult=config.spread_atr_mult * spread_mult,
            slippage_min_spread=slip_mult,
            slippage_max_spread=slip_mult,
        )
        trades, _ = run_backtest(SYMBOL, m1, prepared, exit_bars, stress)
        worst = min(worst, profit_factor(trades["pnl"]) if not trades.empty else 0.0)
    return 0.0 if worst == float("inf") else worst


def _rolling_monthly(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    df = trades.copy()
    df["month"] = pd.to_datetime(df["exit_time"]).dt.to_period("M").astype(str)
    rows = []
    for month, group in df.groupby("month"):
        rows.append(
            {
                "month": month,
                "trades": len(group),
                "pf": profit_factor(group["pnl"]),
                "pnl": group["pnl"].sum(),
                "avg_r": group["r_multiple"].mean(),
                "buy_pf": profit_factor(group[group["side"] == "BUY"]["pnl"]) if (group["side"] == "BUY").any() else 0.0,
                "sell_pf": profit_factor(group[group["side"] == "SELL"]["pnl"]) if (group["side"] == "SELL").any() else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _verdict(row: pd.Series) -> str:
    total_ok = row["profit_factor"] > 1.3
    stress_ok = row["stress_pf"] > 1.15
    drop_ok = row["drop_best_2_pf"] > 1.05
    recent_ok = row["recent_issue"] != "broken_recent_edge" and row["recent_max_drawdown"] > -3
    direction_ok = row["buy_pnl"] >= 0 and row["sell_pnl"] >= 0
    if total_ok and stress_ok and drop_ok and recent_ok and direction_ok:
        return "PASS_DRY_RUN_CANDIDATE"
    if total_ok and drop_ok and row["recent_issue"] != "broken_recent_edge":
        return "PASS_RESEARCH"
    return "FAIL"


def run(start: str, end: str) -> str:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    m1, m5, m15 = _load_data()
    warm = str(pd.Timestamp(start) - pd.DateOffset(months=6))[:10]
    prepared = prepare(m5.loc[warm:end], m15.loc[warm:end], base_config()).loc[start:end]
    exit_bars = _exit_bars(m1, m5, start, end)
    recent_warm = str(pd.Timestamp(RECENT_START) - pd.DateOffset(months=6))[:10]
    recent_prepared = prepare(m5.loc[recent_warm:RECENT_END], m15.loc[recent_warm:RECENT_END], base_config()).loc[RECENT_START:RECENT_END]
    recent_exit = _exit_bars(m1, m5, RECENT_START, RECENT_END)

    rows = []
    trade_frames = []
    monthly_frames = []
    config_map = {}
    all_configs = configs()
    print(f"\nGold V2 Directional Robustness — configs={len(all_configs)}", flush=True)
    for config in all_configs:
        row, trades, monthly = _evaluate(config, m1, prepared, exit_bars, recent_prepared, recent_exit)
        rows.append(row)
        config_map[config.variant] = config
        if not trades.empty:
            trades["variant"] = config.variant
            trade_frames.append(trades)
        if not monthly.empty:
            monthly_frames.append(monthly)

    summary = pd.DataFrame(rows)
    summary["stress_pf"] = [
        _stress_pf(config_map[v], m1, prepared, exit_bars)
        for v in summary["variant"]
    ]
    summary["final_verdict"] = summary.apply(_verdict, axis=1)
    summary = summary.sort_values(
        ["profit_factor", "stress_pf", "drop_best_2_pf", "recent_pf"],
        ascending=[False, False, False, False],
    )
    best = summary.iloc[0]
    best_variant = str(best["variant"])
    rolling = _rolling_monthly(pd.concat(trade_frames, ignore_index=True).query("variant == @best_variant"))

    _write_csv(OUTPUT_DIR / "summary.csv", summary)
    _write_csv(OUTPUT_DIR / "trades.csv", pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame())
    _write_csv(OUTPUT_DIR / "monthly.csv", pd.concat(monthly_frames, ignore_index=True) if monthly_frames else pd.DataFrame())
    _write_csv(OUTPUT_DIR / "rolling_monthly_best.csv", rolling)

    cols = [
        "variant",
        "profit_factor",
        "stress_pf",
        "drop_best_2_pf",
        "recent_pf",
        "buy_pf",
        "sell_pf",
        "total_trades",
        "max_drawdown",
        "avg_r",
        "recent_trades",
        "recent_issue",
        "final_verdict",
    ]
    print("\nRanked directional configs")
    print(summary[cols].round(3).to_string(index=False))
    print("\nRolling monthly for best")
    print(rolling.round(3).to_string(index=False))
    final = str(best["final_verdict"])
    print(f"\nFinal verdict: {final}")
    return final


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message="Converting to PeriodArray/Index representation")
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2026-05-07")
    args = parser.parse_args()
    run(args.start, args.end)


if __name__ == "__main__":
    main()

