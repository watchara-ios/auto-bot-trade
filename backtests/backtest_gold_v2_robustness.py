from __future__ import annotations

import argparse
import ast
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
OUTPUT_DIR = ROOT / "outputs" / "gold_v2_robustness"

SESSION_RANGES = {
    "thai_19_2030": ((12 * 60, 13 * 60 + 30),),
    "thai_19_21": ((12 * 60, 14 * 60),),
    "thai_1930_21": ((12 * 60 + 30, 14 * 60),),
    "thai_20_2130": ((13 * 60, 14 * 60 + 30),),
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


def _exit_bars(m1: pd.DataFrame, m5: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    m1_range = m1.loc[start:end]
    m5_range = m5.loc[start:end]
    if m1_range.empty:
        return m5_range
    m1_start = m1_range.index[0]
    return pd.concat([m5_range[m5_range.index < m1_start], m1_range]).sort_index()


def candidate_config() -> GoldV2Config:
    return GoldV2Config(
        strategy="momentum",
        variant="candidate_thai_19_21_rr1.8_body2_wick0.4_atrMin40",
        session="ny",
        session_ranges_utc_minutes=SESSION_RANGES["thai_19_21"],
        adx_min=18.0,
        momentum_rr=1.8,
        use_atr_expansion=False,
        body_mult=2.0,
        max_wick_pct=0.40,
        momentum_atr_percentile_min=40.0,
        momentum_atr_percentile_max=None,
        close_extreme_pct=None,
        ema_extension_atr_max=None,
        use_m15_ema_trend=False,
        use_m15_ema200_filter=False,
        max_trades_per_day=3,
        max_hold_minutes=720,
    )


def sensitivity_configs() -> list[GoldV2Config]:
    configs = []
    for rr, body, wick, atr_min, session_name in itertools.product(
        [1.6, 1.7, 1.8, 1.9, 2.0],
        [1.8, 2.0, 2.2],
        [0.35, 0.40, 0.45],
        [35.0, 40.0, 45.0, 50.0],
        SESSION_RANGES.keys(),
    ):
        variant = (
            f"sens_{session_name}_rr{rr:g}_body{body:g}_"
            f"wick{wick:g}_atrMin{atr_min:g}"
        )
        configs.append(
            replace(
                candidate_config(),
                variant=variant,
                momentum_rr=rr,
                body_mult=body,
                max_wick_pct=wick,
                momentum_atr_percentile_min=atr_min,
                session_ranges_utc_minutes=SESSION_RANGES[session_name],
            )
        )
    return configs


def _days(m5: pd.DataFrame) -> int:
    if m5.empty:
        return 1
    return max((m5.index[-1].date() - m5.index[0].date()).days + 1, 1)


def _month_stats(monthly: pd.DataFrame) -> dict:
    if monthly.empty:
        return {
            "months": 0,
            "positive_months": 0,
            "positive_month_pct": 0.0,
            "worst_month": 0.0,
            "best_month": 0.0,
        }
    return {
        "months": int(len(monthly)),
        "positive_months": int((monthly["pnl"] > 0).sum()),
        "positive_month_pct": float((monthly["pnl"] > 0).mean() * 100),
        "worst_month": float(monthly["pnl"].min()),
        "best_month": float(monthly["pnl"].max()),
    }


def _max_consecutive_losses(trades: pd.DataFrame) -> int:
    max_run = 0
    run = 0
    for pnl in trades["pnl"] if not trades.empty else []:
        if pnl <= 0:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    return max_run


def _evaluate(
    config: GoldV2Config,
    m1: pd.DataFrame,
    m5_prepared: pd.DataFrame,
    exit_bars: pd.DataFrame,
    label: str,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trades, equity = run_backtest(SYMBOL, m1, m5_prepared, exit_bars, config)
    monthly = monthly_performance(trades)
    row = {
        "label": label,
        "symbol": SYMBOL,
        "variant": config.variant,
        **metrics(trades, equity, _days(m5_prepared), config.initial_balance),
        **_month_stats(monthly),
        "max_consecutive_losses": _max_consecutive_losses(trades),
        "config": str(asdict(config)),
    }
    if not trades.empty:
        row["avg_sl_distance"] = float((trades["entry"] - trades["sl"]).abs().mean())
        row["avg_spread"] = float(trades["spread"].mean())
        row["spread_cost_pct_of_pnl"] = float(trades["spread"].sum() / abs(trades["pnl"].sum())) if trades["pnl"].sum() else 0.0
    else:
        row["avg_sl_distance"] = 0.0
        row["avg_spread"] = 0.0
        row["spread_cost_pct_of_pnl"] = 0.0
    return row, trades, equity, monthly


def run_sensitivity(
    m1: pd.DataFrame,
    m5_prepared: pd.DataFrame,
    exit_bars: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, GoldV2Config]]:
    rows = []
    configs = {}
    all_configs = [candidate_config()] + sensitivity_configs()
    for n, config in enumerate(all_configs, start=1):
        row, _, _, _ = _evaluate(config, m1, m5_prepared, exit_bars, "sensitivity")
        rows.append(row)
        configs[config.variant] = config
        if n % 200 == 0 or n == len(all_configs):
            print(f"  sensitivity {n}/{len(all_configs)}", flush=True)
    summary = pd.DataFrame(rows).sort_values(
        ["profit_factor", "profit_pct", "max_drawdown"],
        ascending=[False, False, False],
    )
    return summary, configs


def run_stress(
    m1: pd.DataFrame,
    m5_prepared: pd.DataFrame,
    exit_bars: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    base = candidate_config()
    for spread_mult, slip_mult, entry_mode in itertools.product(
        [1.0, 1.5, 2.0],
        [0.0, 0.5, 1.0],
        ["next_m1", "next_m5"],
    ):
        config = replace(
            base,
            variant=f"stress_spread{spread_mult:g}_slip{slip_mult:g}_{entry_mode}",
            spread_atr_mult=base.spread_atr_mult * spread_mult,
            slippage_min_spread=slip_mult,
            slippage_max_spread=slip_mult,
            entry_mode=entry_mode,
        )
        row, _, _, _ = _evaluate(config, m1, m5_prepared, exit_bars, "stress")
        row["spread_mult"] = spread_mult
        row["slippage_spread_mult"] = slip_mult
        row["entry_mode"] = entry_mode
        row["stress_pass"] = (
            row["profit_factor"] > 1.15
            and row["max_drawdown"] > -6.0
            and row["avg_r"] > 0.08
            and row["total_trades"] >= 70
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["spread_mult", "slippage_spread_mult", "entry_mode"]
    )


def _best_train_config(train_summary: pd.DataFrame, configs: dict[str, GoldV2Config]) -> GoldV2Config:
    viable = train_summary[
        (train_summary["total_trades"] >= 40)
        & (train_summary["avg_r"] > 0.03)
        & (train_summary["profit_factor"] > 1.05)
    ]
    if viable.empty:
        viable = train_summary
    return configs[str(viable.iloc[0]["variant"])]


def run_walkforward(m1: pd.DataFrame, m5: pd.DataFrame, m15: pd.DataFrame) -> pd.DataFrame:
    rows = []
    full_configs = [candidate_config()] + sensitivity_configs()

    def eval_split(train_start: str, train_end: str, val_start: str, val_end: str, split: str) -> None:
        warm = str(pd.Timestamp(train_start) - pd.DateOffset(months=6))[:10]
        train_prepared = prepare(m5.loc[warm:train_end], m15.loc[warm:train_end], candidate_config()).loc[train_start:train_end]
        train_exit = _exit_bars(m1, m5, train_start, train_end)
        train_rows = []
        cfg_map = {}
        for config in full_configs:
            row, _, _, _ = _evaluate(config, m1, train_prepared, train_exit, f"{split}_train")
            train_rows.append(row)
            cfg_map[config.variant] = config
        train_summary = pd.DataFrame(train_rows).sort_values(
            ["profit_factor", "profit_pct", "max_drawdown"],
            ascending=[False, False, False],
        )
        best = _best_train_config(train_summary, cfg_map)

        val_warm = str(pd.Timestamp(val_start) - pd.DateOffset(months=6))[:10]
        val_prepared = prepare(m5.loc[val_warm:val_end], m15.loc[val_warm:val_end], best).loc[val_start:val_end]
        val_exit = _exit_bars(m1, m5, val_start, val_end)
        train_best = train_summary.iloc[0].to_dict()
        rows.append({"split": split, "phase": "train_best", "start": train_start, "end": train_end, **train_best})
        val_row, _, _, _ = _evaluate(best, m1, val_prepared, val_exit, f"{split}_validate")
        rows.append({"split": split, "phase": "validate", "start": val_start, "end": val_end, **val_row})

    eval_split("2025-01-01", "2025-12-31", "2026-01-01", "2026-05-07", "train2025_validate2026")
    rolling = [
        ("2025-01-01", "2025-03-31", "2025-04-01", "2025-06-30", "roll_q1_to_q2_2025"),
        ("2025-04-01", "2025-06-30", "2025-07-01", "2025-09-30", "roll_q2_to_q3_2025"),
        ("2025-07-01", "2025-09-30", "2025-10-01", "2025-12-31", "roll_q3_to_q4_2025"),
        ("2025-10-01", "2025-12-31", "2026-01-01", "2026-03-31", "roll_q4_to_q1_2026"),
        ("2026-01-01", "2026-03-31", "2026-04-01", "2026-05-07", "roll_q1_to_q2_2026"),
    ]
    for args in rolling:
        print(f"  walkforward {args[-1]}", flush=True)
        eval_split(*args)
    return pd.DataFrame(rows)


def monthly_drop_tests(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    df = trades.copy()
    df["month"] = pd.to_datetime(df["exit_time"]).dt.to_period("M").astype(str)
    month_pnl = df.groupby("month")["pnl"].sum().sort_values(ascending=False)
    rows = []
    for drop_n in [0, 1, 2]:
        keep = df
        dropped = []
        if drop_n:
            dropped = list(month_pnl.head(drop_n).index)
            keep = df[~df["month"].isin(dropped)]
        rows.append(
            {
                "drop_best_months": drop_n,
                "dropped_months": ",".join(dropped),
                "trades": len(keep),
                "profit_factor": profit_factor(keep["pnl"]) if not keep.empty else 0.0,
                "profit_pct": keep["pnl"].sum() / candidate_config().initial_balance * 100,
                "avg_r": keep["r_multiple"].mean() if not keep.empty else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _breakdown(trades: pd.DataFrame, column: str) -> pd.DataFrame:
    rows = []
    for key, group in trades.groupby(column, dropna=False):
        rows.append(
            {
                "bucket_type": column,
                "bucket": key,
                "trades": len(group),
                "pnl": group["pnl"].sum(),
                "profit_factor": profit_factor(group["pnl"]),
                "winrate": (group["pnl"] > 0).mean() * 100,
                "avg_r": group["r_multiple"].mean(),
            }
        )
    return pd.DataFrame(rows)


def distribution_tests(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    df = trades.copy()
    df["weekday"] = pd.to_datetime(df["timestamp"]).dt.day_name()
    df["hour_bucket"] = pd.to_datetime(df["timestamp"]).dt.strftime("%H:%M")
    return pd.concat(
        [
            _breakdown(df, "side"),
            _breakdown(df, "weekday"),
            _breakdown(df, "hour_bucket"),
        ],
        ignore_index=True,
    )


def verdict(
    candidate_row: dict,
    stress: pd.DataFrame,
    walkforward: pd.DataFrame,
    monthly_drop: pd.DataFrame,
    distribution: pd.DataFrame,
) -> str:
    base_ok = (
        candidate_row["profit_factor"] > 1.15
        and candidate_row["max_drawdown"] > -6.0
        and candidate_row["avg_r"] > 0.08
        and candidate_row["total_trades"] >= 70
    )
    stress_ok = bool(stress["stress_pass"].all())
    wf_val = walkforward[walkforward["phase"] == "validate"]
    wf_ok = bool((wf_val["profit_factor"] > 1.0).all() and (wf_val["avg_r"] > 0).mean() >= 0.5)
    drop1 = monthly_drop[monthly_drop["drop_best_months"] == 1]
    drop_ok = not drop1.empty and float(drop1.iloc[0]["profit_factor"]) >= 1.0
    side = distribution[distribution["bucket_type"] == "side"]
    side_ok = len(side) >= 2 and (side["pnl"] > 0).all()
    if base_ok and stress_ok and wf_ok and drop_ok and side_ok:
        return "PASS_DRY_RUN"
    if base_ok and drop_ok and side_ok:
        return "PASS_RESEARCH"
    return "FAIL"


def run(start: str, end: str) -> str:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    m1, m5, m15 = _load_data()
    warm = str(pd.Timestamp(start) - pd.DateOffset(months=6))[:10]
    prepared = prepare(m5.loc[warm:end], m15.loc[warm:end], candidate_config()).loc[start:end]
    exit_bars = _exit_bars(m1, m5, start, end)

    print(f"\nGold V2 Robustness — {start} to {end}", flush=True)
    sensitivity, configs = run_sensitivity(m1, prepared, exit_bars)
    candidate_row, candidate_trades, candidate_equity, candidate_monthly = _evaluate(
        candidate_config(), m1, prepared, exit_bars, "candidate"
    )
    print("  stress tests", flush=True)
    stress = run_stress(m1, prepared, exit_bars)
    print("  walk-forward tests", flush=True)
    walkforward = run_walkforward(m1, m5, m15)
    drop = monthly_drop_tests(candidate_trades)
    distribution = distribution_tests(candidate_trades)

    result = verdict(candidate_row, stress, walkforward, drop, distribution)
    sensitivity["passes_base_criteria"] = (
        (sensitivity["profit_factor"] > 1.15)
        & (sensitivity["max_drawdown"] > -6.0)
        & (sensitivity["avg_r"] > 0.08)
        & (sensitivity["total_trades"] >= 70)
    )
    _write_csv(OUTPUT_DIR / "summary.csv", sensitivity)
    _write_csv(OUTPUT_DIR / "stress.csv", stress)
    _write_csv(OUTPUT_DIR / "walkforward.csv", walkforward)
    _write_csv(OUTPUT_DIR / "monthly_drop_test.csv", drop)
    _write_csv(OUTPUT_DIR / "trade_distribution.csv", distribution)
    _write_csv(OUTPUT_DIR / "candidate_trades.csv", candidate_trades)
    _write_csv(OUTPUT_DIR / "candidate_monthly.csv", candidate_monthly)
    _write_csv(OUTPUT_DIR / "candidate_equity.csv", candidate_equity)

    cols = [
        "variant",
        "profit_factor",
        "profit_pct",
        "max_drawdown",
        "total_trades",
        "winrate",
        "avg_r",
        "positive_months",
        "worst_month",
        "passes_base_criteria",
    ]
    print("\nTop sensitivity results")
    print(sensitivity[cols].head(20).round(3).to_string(index=False))
    print("\nStress summary")
    print(
        stress[
            [
                "variant",
                "profit_factor",
                "profit_pct",
                "max_drawdown",
                "total_trades",
                "avg_r",
                "stress_pass",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )
    print("\nMonthly drop test")
    print(drop.round(3).to_string(index=False))
    print(f"\nFinal verdict: {result}")
    return result


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

