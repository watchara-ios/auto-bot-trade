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
    run_backtest,
)


SYMBOL = "XAUUSDm"
DATA_DIR = ROOT / "data" / "mt5_history"
OUTPUT_DIR = ROOT / "outputs" / "gold_v2_refine"

SESSION_WINDOWS = {
    "thai_19_21": (12, 13),
    "thai_20_22": (13, 14),
    "thai_19_23": (12, 13, 14, 15),
    "thai_21_23": (14, 15),
}


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
    return pd.concat([m5_range[m5_range.index < m1_start], m1_range]).sort_index(), (
        f"M5 until {m1_start.date()} then M1"
    )


def _base_config() -> GoldV2Config:
    return GoldV2Config(
        strategy="momentum",
        variant="base_B_momo_ny_adx18_rr2_atrexp0",
        session="ny",
        session_hours_utc=SESSION_WINDOWS["thai_19_23"],
        adx_min=18.0,
        momentum_rr=2.0,
        use_atr_expansion=False,
        body_mult=1.5,
        max_wick_pct=0.50,
        max_trades_per_day=3,
        max_hold_minutes=720,
    )


def _trend_mode(config: GoldV2Config, mode: str) -> GoldV2Config:
    if mode == "ema20_50":
        return replace(config, use_m15_ema_trend=True, use_m15_ema200_filter=False)
    if mode == "ema20_50_200":
        return replace(config, use_m15_ema_trend=True, use_m15_ema200_filter=True)
    return replace(config, use_m15_ema_trend=False, use_m15_ema200_filter=False)


def _name(config: GoldV2Config, prefix: str, window_name: str, trend_mode: str) -> str:
    close = "close20" if config.close_extreme_pct is not None else "closeOff"
    ext = f"ext{config.ema_extension_atr_max:g}" if config.ema_extension_atr_max is not None else "extOff"
    atr_min = f"atrMin{config.momentum_atr_percentile_min:g}" if config.momentum_atr_percentile_min is not None else "atrMinOff"
    atr_max = f"atrMax{config.momentum_atr_percentile_max:g}" if config.momentum_atr_percentile_max is not None else "atrMaxOff"
    return (
        f"{prefix}_{window_name}_rr{config.momentum_rr:g}_body{config.body_mult:g}_"
        f"wick{config.max_wick_pct:g}_{close}_{ext}_{trend_mode}_{atr_min}_{atr_max}"
    )


def ablation_variants() -> list[GoldV2Config]:
    base = _base_config()
    out = [base]
    for body in [1.8, 2.0, 2.5]:
        out.append(replace(base, variant=f"ablate_body{body:g}", body_mult=body))
    out.append(replace(base, variant="ablate_close_top_bottom_20", close_extreme_pct=0.20))
    for wick in [0.40, 0.30, 0.25]:
        out.append(replace(base, variant=f"ablate_wick{wick:g}", max_wick_pct=wick))
    for ext in [1.5, 2.0]:
        out.append(replace(base, variant=f"ablate_ema20_ext{ext:g}", ema_extension_atr_max=ext))
    out.append(_trend_mode(replace(base, variant="ablate_trend_ema20_50"), "ema20_50"))
    out.append(_trend_mode(replace(base, variant="ablate_trend_ema20_50_200"), "ema20_50_200"))
    for atr_min in [30.0, 40.0, 50.0, 60.0]:
        out.append(replace(base, variant=f"ablate_atr_min{atr_min:g}", momentum_atr_percentile_min=atr_min))
    out.append(replace(base, variant="ablate_atr_max90", momentum_atr_percentile_max=90.0))
    for name, hours in SESSION_WINDOWS.items():
        out.append(replace(base, variant=f"ablate_window_{name}", session_hours_utc=hours))
    for rr in [1.5, 1.8, 2.2]:
        out.append(replace(base, variant=f"ablate_rr{rr:g}", momentum_rr=rr))
    return out


def focused_grid() -> list[GoldV2Config]:
    configs = []
    for (
        window_name,
        rr,
        body,
        wick,
        close_extreme,
        extension,
        trend,
        atr_min,
        atr_max,
    ) in itertools.product(
        SESSION_WINDOWS.keys(),
        [1.5, 1.8, 2.0, 2.2],
        [1.5, 1.8, 2.0],
        [0.50, 0.40, 0.30],
        [None, 0.20],
        [None, 2.0],
        ["none", "ema20_50"],
        [None, 30.0, 40.0, 50.0],
        [None, 90.0],
    ):
        config = replace(
            _base_config(),
            session_hours_utc=SESSION_WINDOWS[window_name],
            momentum_rr=rr,
            body_mult=body,
            max_wick_pct=wick,
            close_extreme_pct=close_extreme,
            ema_extension_atr_max=extension,
            momentum_atr_percentile_min=atr_min,
            momentum_atr_percentile_max=atr_max,
        )
        config = _trend_mode(config, trend)
        config = replace(config, variant=_name(config, "grid", window_name, trend))
        configs.append(config)
    return configs


def variants() -> list[GoldV2Config]:
    seen = {}
    for config in ablation_variants() + focused_grid():
        seen[config.variant] = config
    return list(seen.values())


def _month_stats(monthly: pd.DataFrame, trades: pd.DataFrame) -> dict:
    if monthly.empty:
        return {
            "months": 0,
            "positive_months": 0,
            "positive_month_pct": 0.0,
            "worst_month": 0.0,
            "best_month": 0.0,
            "lucky_month_risk": False,
        }
    pnl = monthly["pnl"]
    gross_positive = pnl[pnl > 0].sum()
    best_month = float(pnl.max())
    total_pnl = float(trades["pnl"].sum()) if not trades.empty else 0.0
    lucky = bool(total_pnl > 0 and gross_positive > 0 and best_month / gross_positive > 0.60)
    return {
        "months": int(len(monthly)),
        "positive_months": int((pnl > 0).sum()),
        "positive_month_pct": float((pnl > 0).mean() * 100),
        "worst_month": float(pnl.min()),
        "best_month": best_month,
        "lucky_month_risk": lucky,
    }


def _passes(row: dict) -> bool:
    return (
        row["profit_factor"] > 1.15
        and row["max_drawdown"] > -8.0
        and row["total_trades"] >= 100
        and row["avg_r"] > 0.05
        and not row["lucky_month_risk"]
        and row["positive_month_pct"] >= 50.0
    )


def _evaluate(
    config: GoldV2Config,
    m1: pd.DataFrame,
    m5_prepared: pd.DataFrame,
    exit_bars: pd.DataFrame,
    days: int,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    trades, equity = run_backtest(SYMBOL, m1, m5_prepared, exit_bars, config)
    monthly = monthly_performance(trades)
    row = {
        "symbol": SYMBOL,
        "strategy": config.strategy,
        "variant": config.variant,
        **metrics(trades, equity, days, config.initial_balance),
        **_month_stats(monthly, trades),
        "config": str(asdict(config)),
    }
    row["passes_success"] = _passes(row)
    return row, trades, equity


def run(start: str, end: str, top: int, save_top: int) -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    m1_raw, m5_raw, m15_raw = _load_data()
    m5_range = m5_raw.loc[start:end]
    warmup_start = str(pd.Timestamp(start) - pd.DateOffset(months=6))[:10]
    exit_bars, exit_note = _exit_bars(m1_raw, m5_raw, start, end)
    days = max((m5_range.index[-1].date() - m5_range.index[0].date()).days + 1, 1)
    all_variants = variants()

    print(f"\nGold V2 Momentum Refine — {start} to {end}", flush=True)
    print(f"Symbol: {SYMBOL} | variants={len(all_variants)} | exit={exit_note}", flush=True)
    print(f"Outputs: {OUTPUT_DIR}\n", flush=True)

    m5_prepared = prepare(
        m5_raw.loc[warmup_start:end],
        m15_raw.loc[warmup_start:end],
        _base_config(),
    ).loc[start:end]

    summary_rows = []
    config_by_variant = {}
    for n, config in enumerate(all_variants, start=1):
        row, _, _ = _evaluate(config, m1_raw, m5_prepared, exit_bars, days)
        summary_rows.append(row)
        config_by_variant[config.variant] = config
        if n % 500 == 0 or n == len(all_variants):
            print(f"  evaluated {n}/{len(all_variants)} configs", flush=True)

    summary = pd.DataFrame(summary_rows).sort_values(
        ["profit_factor", "profit_pct", "max_drawdown"],
        ascending=[False, False, False],
    )
    _write_csv(OUTPUT_DIR / "summary.csv", summary)

    # Save details for both raw top-PF configs and genuinely viable configs.
    trade_frames = []
    equity_frames = []
    monthly_frames = []
    viable = summary[
        (summary["total_trades"] >= 100)
        & (summary["avg_r"] > 0.05)
        & (~summary["lucky_month_risk"])
    ].sort_values(["profit_factor", "profit_pct", "max_drawdown"], ascending=[False, False, False])
    detail_variants = pd.concat([summary["variant"].head(save_top), viable["variant"].head(save_top)]).drop_duplicates()
    for variant in detail_variants:
        config = config_by_variant[variant]
        _, trades, equity = _evaluate(config, m1_raw, m5_prepared, exit_bars, days)
        monthly = monthly_performance(trades)
        if not trades.empty:
            trade_frames.append(trades)
            monthly_frames.append(monthly)
        if not equity.empty:
            equity_frames.append(equity)

    _write_csv(OUTPUT_DIR / "trades.csv", pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame())
    _write_csv(OUTPUT_DIR / "equity.csv", pd.concat(equity_frames, ignore_index=True) if equity_frames else pd.DataFrame())
    _write_csv(OUTPUT_DIR / "monthly.csv", pd.concat(monthly_frames, ignore_index=True) if monthly_frames else pd.DataFrame())

    cols = [
        "variant",
        "profit_factor",
        "profit_pct",
        "max_drawdown",
        "total_trades",
        "winrate",
        "avg_r",
        "positive_months",
        "positive_month_pct",
        "worst_month",
        "lucky_month_risk",
        "passes_success",
    ]
    print("\nRanked momentum refine results")
    print(summary[cols].head(top).round(3).to_string(index=False))
    best = summary.iloc[0]
    print("\nBest config")
    print(best["variant"])
    print(best["config"])
    print(
        "Passes success criteria: "
        f"{bool(best['passes_success'])} "
        "(PF>1.15, DD<8%, trades>=100, avgR>0.05, no lucky month, >=50% positive months)"
    )
    return summary


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message="Converting to PeriodArray/Index representation")
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2026-05-07")
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--save-top", type=int, default=50)
    args = parser.parse_args()
    run(args.start, args.end, args.top, args.save_top)


if __name__ == "__main__":
    main()
