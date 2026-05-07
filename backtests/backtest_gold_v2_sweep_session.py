from __future__ import annotations

import argparse
import itertools
import sys
from dataclasses import asdict, replace
from pathlib import Path
import warnings

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from strategies.gold_v2_core import (  # noqa: E402
    GoldV2Config,
    load_csv,
    metrics,
    monthly_performance,
    prepare,
    run_backtest,
)


SYMBOL = "XAUUSDm"
DATA_DIR = ROOT / "data" / "mt5_history"
OUTPUT_DIR = ROOT / "outputs" / "gold_v2"
BASELINE_SUMMARY = ROOT / "reports" / "gold_donchian" / "summary.csv"


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        load_csv(DATA_DIR / f"{SYMBOL}_M1.csv"),
        load_csv(DATA_DIR / f"{SYMBOL}_M5.csv"),
        load_csv(DATA_DIR / f"{SYMBOL}_M15.csv"),
    )


def _exit_bars(m1: pd.DataFrame, m5: pd.DataFrame, start: str, end: str) -> tuple[pd.DataFrame, str]:
    m1_range = m1.loc[start:end]
    m5_range = m5.loc[start:end]
    if m1_range.empty:
        return m5_range, "M5 only"
    m1_start = m1_range.index[0]
    exit_bars = pd.concat([m5_range[m5_range.index < m1_start], m1_range]).sort_index()
    return exit_bars, f"M5 until {m1_start.date()} then M1"


def sweep_variants() -> list[GoldV2Config]:
    configs = []
    for lookback, wick_body, rr, atr_pct in itertools.product(
        [10, 20, 30],
        [1.2, 1.5, 2.0],
        [1.5, 2.0],
        [20.0, 30.0, 40.0],
    ):
        variant = f"A_sweep_lb{lookback}_wick{wick_body:g}_rr{rr:g}_atr{atr_pct:g}"
        configs.append(
            GoldV2Config(
                strategy="sweep",
                variant=variant,
                sweep_lookback=lookback,
                wick_body_min=wick_body,
                sweep_rr=rr,
                atr_percentile_min=atr_pct,
                max_trades_per_day=3,
                max_hold_minutes=720,
            )
        )
    return configs


def momentum_variants() -> list[GoldV2Config]:
    configs = []
    for session, adx_min, rr, atr_expansion in itertools.product(
        ["london", "ny", "london_ny"],
        [18.0, 20.0, 25.0],
        [1.5, 2.0, 2.5],
        [False, True],
    ):
        variant = (
            f"B_momo_{session}_adx{adx_min:g}_rr{rr:g}_"
            f"atrexp{int(atr_expansion)}"
        )
        configs.append(
            GoldV2Config(
                strategy="momentum",
                variant=variant,
                session=session,
                adx_min=adx_min,
                momentum_rr=rr,
                use_atr_expansion=atr_expansion,
                max_trades_per_day=3,
                max_hold_minutes=720,
            )
        )
    return configs


def _baseline_rows() -> pd.DataFrame:
    if not BASELINE_SUMMARY.exists():
        return pd.DataFrame()
    baseline_names = {"gold_nosession", "live_best", "live_bot", "d1_regime"}
    base = pd.read_csv(BASELINE_SUMMARY)
    base = base[(base["symbol"] == SYMBOL) & (base["variant"].isin(baseline_names))].copy()
    if base.empty:
        return base
    base["strategy"] = "baseline_donchian"
    base["avg_r"] = pd.NA
    base["worst_month"] = pd.NA
    base["passes_success"] = False
    base["config"] = "{}"
    wanted = [
        "symbol",
        "strategy",
        "variant",
        "days",
        "total_trades",
        "winrate",
        "profit_factor",
        "profit_pct",
        "max_drawdown",
        "expectancy",
        "avg_r",
        "trades_per_day",
        "worst_month",
        "passes_success",
        "config",
    ]
    return base[wanted]


def _passes_success(row: dict) -> bool:
    return (
        row["profit_factor"] > 1.15
        and row["max_drawdown"] > -12.0
        and row["total_trades"] >= 80
    )


def _lucky_month_flag(monthly: pd.DataFrame, total_pnl: float) -> bool:
    if monthly.empty or total_pnl <= 0:
        return False
    best = float(monthly["pnl"].max())
    return best / total_pnl > 0.60


def run(start: str, end: str, top: int) -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    m1_raw, m5_raw, m15_raw = _load_data()
    m5_range = m5_raw.loc[start:end]
    m15_range = m15_raw.loc[start:end]
    if len(m5_range) < 300 or len(m15_range) < 100:
        raise RuntimeError("Insufficient XAUUSDm history for requested range")

    warmup_start = str(pd.Timestamp(start) - pd.DateOffset(months=6))[:10]
    exit_bars, exit_note = _exit_bars(m1_raw, m5_raw, start, end)
    days = max((m5_range.index[-1].date() - m5_range.index[0].date()).days + 1, 1)
    variants = sweep_variants() + momentum_variants()

    print(f"\nGold Bot V2 Backtest — {start} to {end}", flush=True)
    print(f"Symbol: {SYMBOL} | variants={len(variants)} | exit={exit_note}", flush=True)
    print(f"Outputs: {OUTPUT_DIR}\n", flush=True)

    summary_rows = []
    trade_frames = []
    equity_frames = []
    monthly_frames = []
    m5_prepared = prepare(
        m5_raw.loc[warmup_start:end],
        m15_raw.loc[warmup_start:end],
        variants[0],
    ).loc[start:end]

    for n, config in enumerate(variants, start=1):
        trades, equity = run_backtest(SYMBOL, m1_raw, m5_prepared, exit_bars, config)
        row = {
            "symbol": SYMBOL,
            "strategy": config.strategy,
            "variant": config.variant,
            "days": days,
            **metrics(trades, equity, days, config.initial_balance),
        }
        monthly = monthly_performance(trades)
        row["lucky_month_risk"] = _lucky_month_flag(monthly, float(trades["pnl"].sum()) if not trades.empty else 0.0)
        row["passes_success"] = _passes_success(row) and not row["lucky_month_risk"]
        row["config"] = str(asdict(config))
        summary_rows.append(row)
        if not trades.empty:
            trade_frames.append(trades)
            monthly_frames.append(monthly)
        if not equity.empty:
            equity_frames.append(equity)
        if n % 25 == 0 or n == len(variants):
            print(f"  completed {n}/{len(variants)} variants", flush=True)

    summary = pd.DataFrame(summary_rows)
    baselines = _baseline_rows()
    ranked = summary.sort_values(
        ["profit_factor", "profit_pct", "max_drawdown"],
        ascending=[False, False, False],
    )
    combined = pd.concat([ranked, baselines], ignore_index=True, sort=False)
    trades_out = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    equity_out = pd.concat(equity_frames, ignore_index=True) if equity_frames else pd.DataFrame()
    monthly_out = pd.concat(monthly_frames, ignore_index=True) if monthly_frames else pd.DataFrame()

    _write_csv(OUTPUT_DIR / "summary.csv", ranked)
    _write_csv(OUTPUT_DIR / "summary_with_baselines.csv", combined)
    _write_csv(OUTPUT_DIR / "trades.csv", trades_out)
    _write_csv(OUTPUT_DIR / "equity.csv", equity_out)
    _write_csv(OUTPUT_DIR / "monthly.csv", monthly_out)

    cols = [
        "strategy",
        "variant",
        "total_trades",
        "winrate",
        "profit_factor",
        "profit_pct",
        "max_drawdown",
        "expectancy",
        "avg_r",
        "trades_per_day",
        "worst_month",
        "lucky_month_risk",
        "passes_success",
    ]
    print("\nRanked Gold V2 results")
    print(ranked[cols].head(top).round(3).to_string(index=False))

    if not baselines.empty:
        print("\nExisting Donchian baselines")
        baseline_cols = [
            "variant",
            "total_trades",
            "winrate",
            "profit_factor",
            "profit_pct",
            "max_drawdown",
            "expectancy",
            "trades_per_day",
        ]
        print(
            baselines[baseline_cols]
            .sort_values(["profit_factor", "profit_pct", "max_drawdown"], ascending=[False, False, False])
            .round(3)
            .to_string(index=False)
        )

    best = ranked.iloc[0]
    print("\nBest config")
    print(best["variant"])
    print(best["config"])
    print(
        "Passes success criteria: "
        f"{bool(best['passes_success'])} "
        f"(PF>1.15, DD<12%, trades>=80, no single-month dependency)"
    )
    return ranked


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message="Converting to PeriodArray/Index representation")
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2026-05-07")
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()
    run(args.start, args.end, args.top)


if __name__ == "__main__":
    main()
