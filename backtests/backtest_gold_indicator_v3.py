from __future__ import annotations

import argparse
import itertools
import sys
import warnings
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from strategies.gold_v2_core import (  # noqa: E402
    GoldV2Config,
    candidate_positions,
    load_csv,
    metrics,
    momentum_signal,
    monthly_performance,
    prepare,
    profit_factor,
    simulate_exit,
    trade_levels,
)

SYMBOL = "XAUUSDm"
DATA_DIR = ROOT / "data" / "mt5_history"
OUTPUT_DIR = ROOT / "outputs" / "gold_indicator_v3"
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


# ─── Indicator helpers ────────────────────────────────────────────────────────

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _macd_hist(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    macd_line = series.ewm(span=fast, adjust=False).mean() - series.ewm(span=slow, adjust=False).mean()
    return macd_line - macd_line.ewm(span=signal, adjust=False).mean()


def _bb_pct(series: pd.Series, period: int = 20, std_mult: float = 2.0) -> pd.Series:
    ma = series.rolling(period).mean()
    std = series.rolling(period).std(ddof=0)
    lower = ma - std_mult * std
    band = (2 * std_mult * std).replace(0, np.nan)
    return (series - lower) / band


def prepare_v3(m5: pd.DataFrame, m15: pd.DataFrame, base_cfg: GoldV2Config) -> pd.DataFrame:
    aligned = prepare(m5, m15, base_cfg)
    m15c = m15.copy()
    m15c["rsi14"] = _rsi(m15c["close"])
    m15c["macd_hist"] = _macd_hist(m15c["close"])
    m15c["bb_pct"] = _bb_pct(m15c["close"])
    aligned["m15_rsi14"] = m15c["rsi14"].shift(1).reindex(aligned.index, method="ffill")
    aligned["m15_macd_hist"] = m15c["macd_hist"].shift(1).reindex(aligned.index, method="ffill")
    aligned["m15_bb_pct"] = m15c["bb_pct"].shift(1).reindex(aligned.index, method="ffill")
    return aligned


# ─── Indicator filter config ──────────────────────────────────────────────────

@dataclass(frozen=True)
class IndConfig:
    variant: str
    allowed_side: str | None = None
    rsi_buy_min: float | None = None
    rsi_sell_max: float | None = None
    macd_buy: bool = False
    macd_sell: bool = False
    bb_buy_max: float | None = None   # BB%B < threshold → near lower band → BUY zone
    bb_sell_min: float | None = None  # BB%B > threshold → near upper band → SELL zone


def _ind_filter(row: pd.Series, sig: dict, ind: IndConfig) -> bool:
    side = sig["side"]
    if side == "BUY":
        if ind.rsi_buy_min is not None and row.get("m15_rsi14", 50) < ind.rsi_buy_min:
            return False
        if ind.macd_buy and row.get("m15_macd_hist", 0) <= 0:
            return False
        if ind.bb_buy_max is not None and row.get("m15_bb_pct", 0.5) > ind.bb_buy_max:
            return False
    else:
        if ind.rsi_sell_max is not None and row.get("m15_rsi14", 50) > ind.rsi_sell_max:
            return False
        if ind.macd_sell and row.get("m15_macd_hist", 0) >= 0:
            return False
        if ind.bb_sell_min is not None and row.get("m15_bb_pct", 0.5) < ind.bb_sell_min:
            return False
    return True


# ─── Base config and variant list ─────────────────────────────────────────────

def base_config() -> GoldV2Config:
    return GoldV2Config(
        strategy="momentum",
        variant="base",
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


def ind_configs() -> list[IndConfig]:
    s = dict(allowed_side="SELL")
    b = dict(allowed_side="BUY")
    both = dict(allowed_side=None)
    return [
        # SELL variants (control + indicator filters)
        IndConfig(variant="sell_baseline", **s),
        IndConfig(variant="sell_rsi_lt50", rsi_sell_max=50, **s),
        IndConfig(variant="sell_macd_neg", macd_sell=True, **s),
        IndConfig(variant="sell_rsi_macd", rsi_sell_max=50, macd_sell=True, **s),
        IndConfig(variant="sell_bb_top65", bb_sell_min=0.65, **s),
        IndConfig(variant="sell_rsi_bb", rsi_sell_max=50, bb_sell_min=0.65, **s),
        IndConfig(variant="sell_macd_bb", macd_sell=True, bb_sell_min=0.65, **s),
        IndConfig(variant="sell_all", rsi_sell_max=50, macd_sell=True, bb_sell_min=0.65, **s),
        # BUY variants (control + indicator filters)
        IndConfig(variant="buy_baseline", **b),
        IndConfig(variant="buy_rsi_gt50", rsi_buy_min=50, **b),
        IndConfig(variant="buy_macd_pos", macd_buy=True, **b),
        IndConfig(variant="buy_rsi_macd", rsi_buy_min=50, macd_buy=True, **b),
        IndConfig(variant="buy_bb_bot35", bb_buy_max=0.35, **b),
        IndConfig(variant="buy_rsi_bb", rsi_buy_min=50, bb_buy_max=0.35, **b),
        IndConfig(variant="buy_macd_bb", macd_buy=True, bb_buy_max=0.35, **b),
        IndConfig(variant="buy_all", rsi_buy_min=50, macd_buy=True, bb_buy_max=0.35, **b),
        # Both-side variants
        IndConfig(variant="both_baseline", **both),
        IndConfig(variant="both_rsi_macd", rsi_buy_min=50, rsi_sell_max=50, macd_buy=True, macd_sell=True, **both),
        IndConfig(
            variant="both_rsi_macd_bb",
            rsi_buy_min=50, rsi_sell_max=50,
            macd_buy=True, macd_sell=True,
            bb_buy_max=0.35, bb_sell_min=0.65,
            **both,
        ),
    ]


# ─── Backtest loop extended with indicator filter ─────────────────────────────

def run_backtest_ind(
    symbol: str,
    m1: pd.DataFrame,
    m5: pd.DataFrame,
    exit_bars: pd.DataFrame,
    base_cfg: GoldV2Config,
    ind: IndConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = replace(base_cfg, variant=ind.variant, allowed_side=ind.allowed_side)
    rng = np.random.default_rng(cfg.seed)
    balance = cfg.initial_balance
    peak = balance
    max_dd = 0.0
    unavailable_until: pd.Timestamp | None = None
    daily_trades: dict = {}
    daily_pnl: dict = {}
    trades: list[dict] = []
    equity: list[dict] = []

    for i in candidate_positions(m5, cfg):
        ts = m5.index[i]
        row = m5.iloc[i]
        peak = max(peak, balance)
        max_dd = min(max_dd, (balance - peak) / peak)
        equity.append({
            "time": ts, "symbol": symbol, "variant": ind.variant,
            "strategy": cfg.strategy, "balance": balance, "drawdown_pct": max_dd * 100,
        })
        if unavailable_until is not None and ts <= unavailable_until:
            continue
        day = ts.date()
        if daily_trades.get(day, 0) >= cfg.max_trades_per_day:
            continue
        if daily_pnl.get(day, 0.0) <= -(cfg.initial_balance * cfg.daily_loss_limit_pct):
            continue

        sig = momentum_signal(row, cfg)
        if sig is None:
            continue
        if not _ind_filter(row, sig, ind):
            continue

        entry_time = m5.index[i + 1]
        if entry_time in m1.index:
            entry_base = float(m1.loc[entry_time, "open"])
        else:
            entry_base = float(m5.iloc[i + 1]["open"])

        spread = max(float(row["atr"]) * cfg.spread_atr_mult, float(row["close"]) * 0.00001)
        slippage = rng.uniform(cfg.slippage_min_spread, cfg.slippage_max_spread) * spread
        if sig["side"] == "BUY":
            entry = entry_base + spread / 2 + slippage
        else:
            entry = entry_base - spread / 2 - slippage

        levels = trade_levels(row, entry, sig, cfg)
        if levels is None:
            continue
        sl, tp, risk_dist = levels
        qty = (cfg.initial_balance * cfg.risk_pct) / risk_dist
        if qty <= 0 or not np.isfinite(qty):
            continue

        exit_time, exit_price, result = simulate_exit(
            exit_bars, entry_time, sig["side"], sl, tp, cfg.max_hold_minutes
        )
        unavailable_until = exit_time
        if result == "OPEN" or not np.isfinite(exit_price):
            continue

        gross = (exit_price - entry) * qty
        if sig["side"] == "SELL":
            gross = -gross
        fee = (entry * qty + exit_price * qty) * cfg.fee_rate
        pnl = gross - fee
        balance += pnl
        daily_trades[day] = daily_trades.get(day, 0) + 1
        daily_pnl[day] = daily_pnl.get(day, 0.0) + pnl
        initial_risk = risk_dist * qty
        trades.append({
            "symbol": symbol, "variant": ind.variant, "strategy": cfg.strategy,
            "timestamp": ts, "entry_time": entry_time, "exit_time": exit_time,
            "side": sig["side"], "pattern": sig.get("pattern", ""),
            "entry": entry, "sl": sl, "tp": tp, "rr": cfg.momentum_rr, "qty": qty,
            "spread": spread, "slippage": slippage, "fee": fee,
            "gross_pnl": gross, "pnl": pnl,
            "r_multiple": pnl / initial_risk if initial_risk else np.nan,
            "result": result, "balance": balance,
            "atr": row.get("atr"), "atr_percentile_100": row.get("atr_percentile_100"),
            "m15_adx": row.get("m15_adx"), "m15_rsi14": row.get("m15_rsi14"),
            "m15_macd_hist": row.get("m15_macd_hist"), "m15_bb_pct": row.get("m15_bb_pct"),
        })
    return pd.DataFrame(trades), pd.DataFrame(equity)


# ─── Analysis helpers ─────────────────────────────────────────────────────────

def _days(df: pd.DataFrame) -> int:
    if df.empty:
        return 1
    return max((df.index[-1].date() - df.index[0].date()).days + 1, 1)


def _max_consecutive_losses(trades: pd.DataFrame) -> int:
    run = max_run = 0
    for pnl in (trades["pnl"] if not trades.empty else []):
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


def _rolling_monthly(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    df = trades.copy()
    df["month"] = pd.to_datetime(df["exit_time"]).dt.to_period("M").astype(str)
    rows = []
    for month, group in df.groupby("month"):
        rows.append({
            "month": month, "trades": len(group),
            "pf": profit_factor(group["pnl"]),
            "pnl": group["pnl"].sum(),
            "avg_r": group["r_multiple"].mean(),
            "buy_pf": profit_factor(group[group["side"] == "BUY"]["pnl"]) if (group["side"] == "BUY").any() else 0.0,
            "sell_pf": profit_factor(group[group["side"] == "SELL"]["pnl"]) if (group["side"] == "SELL").any() else 0.0,
        })
    return pd.DataFrame(rows)


def _evaluate(
    ind: IndConfig,
    m1: pd.DataFrame,
    m5_prepared: pd.DataFrame,
    exit_bars: pd.DataFrame,
    recent_prepared: pd.DataFrame,
    recent_exit: pd.DataFrame,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    bcfg = base_config()
    trades, equity = run_backtest_ind(SYMBOL, m1, m5_prepared, exit_bars, bcfg, ind)
    monthly = monthly_performance(trades)
    row: dict = {
        "variant": ind.variant,
        **metrics(trades, equity, _days(m5_prepared), bcfg.initial_balance),
        "positive_months": int((monthly["pnl"] > 0).sum()) if not monthly.empty else 0,
        "worst_month": float(monthly["pnl"].min()) if not monthly.empty else 0.0,
        "max_consecutive_losses": _max_consecutive_losses(trades),
        "drop_best_1_pf": _month_drop(trades, 1)[0],
        "drop_best_1_profit_pct": _month_drop(trades, 1)[1],
        "drop_best_2_pf": _month_drop(trades, 2)[0],
        "drop_best_2_profit_pct": _month_drop(trades, 2)[1],
    }
    buy_pf, buy_pnl, buy_trades = _side_pf(trades, "BUY")
    sell_pf, sell_pnl, sell_trades = _side_pf(trades, "SELL")
    row.update({
        "buy_pf": buy_pf, "buy_pnl": buy_pnl, "buy_trades": buy_trades,
        "sell_pf": sell_pf, "sell_pnl": sell_pnl, "sell_trades": sell_trades,
    })
    recent_trades, recent_equity = run_backtest_ind(SYMBOL, m1, recent_prepared, recent_exit, bcfg, ind)
    recent_m = metrics(recent_trades, recent_equity, _days(recent_prepared), bcfg.initial_balance)
    row.update({
        "recent_pf": recent_m["profit_factor"], "recent_profit_pct": recent_m["profit_pct"],
        "recent_max_drawdown": recent_m["max_drawdown"], "recent_trades": recent_m["total_trades"],
        "recent_avg_r": recent_m["avg_r"],
    })
    row["recent_issue"] = _recent_reason(row)
    return row, trades, monthly


def _stress_pf(ind: IndConfig, m1: pd.DataFrame, prepared: pd.DataFrame, exit_bars: pd.DataFrame) -> float:
    worst = float("inf")
    bcfg = base_config()
    for spread_mult, slip_mult in itertools.product([1.0, 1.5, 2.0], [0.0, 0.5, 1.0]):
        stress_cfg = replace(
            bcfg,
            spread_atr_mult=bcfg.spread_atr_mult * spread_mult,
            slippage_min_spread=slip_mult,
            slippage_max_spread=slip_mult,
        )
        trades, _ = run_backtest_ind(SYMBOL, m1, prepared, exit_bars, stress_cfg, ind)
        worst = min(worst, profit_factor(trades["pnl"]) if not trades.empty else 0.0)
    return 0.0 if worst == float("inf") else worst


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
    prepared = prepare_v3(m5.loc[warm:end], m15.loc[warm:end], base_config()).loc[start:end]
    exit_bars = _exit_bars(m1, m5, start, end)
    recent_warm = str(pd.Timestamp(RECENT_START) - pd.DateOffset(months=6))[:10]
    recent_prepared = prepare_v3(
        m5.loc[recent_warm:RECENT_END], m15.loc[recent_warm:RECENT_END], base_config()
    ).loc[RECENT_START:RECENT_END]
    recent_exit = _exit_bars(m1, m5, RECENT_START, RECENT_END)

    rows = []
    trade_frames = []
    monthly_frames = []
    all_inds = ind_configs()
    ind_map = {ind.variant: ind for ind in all_inds}
    print(f"\nGold Indicator V3 — configs={len(all_inds)}", flush=True)
    for ind in all_inds:
        print(f"  {ind.variant}", flush=True)
        row, trades, monthly = _evaluate(ind, m1, prepared, exit_bars, recent_prepared, recent_exit)
        rows.append(row)
        if not trades.empty:
            trade_frames.append(trades)
        if not monthly.empty:
            monthly_frames.append(monthly)

    summary = pd.DataFrame(rows)
    summary["stress_pf"] = [
        _stress_pf(ind_map[v], m1, prepared, exit_bars)
        for v in summary["variant"]
    ]
    summary["final_verdict"] = summary.apply(_verdict, axis=1)
    summary = summary.sort_values(
        ["profit_factor", "stress_pf", "drop_best_2_pf", "recent_pf"],
        ascending=[False, False, False, False],
    )
    best_variant = str(summary.iloc[0]["variant"])
    all_trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    best_trades = all_trades[all_trades["variant"] == best_variant] if not all_trades.empty else pd.DataFrame()
    rolling = _rolling_monthly(best_trades)

    _write_csv(OUTPUT_DIR / "summary.csv", summary)
    _write_csv(OUTPUT_DIR / "trades.csv", all_trades)
    _write_csv(OUTPUT_DIR / "monthly.csv", pd.concat(monthly_frames, ignore_index=True) if monthly_frames else pd.DataFrame())
    _write_csv(OUTPUT_DIR / "rolling_monthly_best.csv", rolling)

    cols = [
        "variant", "profit_factor", "stress_pf", "drop_best_2_pf", "recent_pf",
        "buy_pf", "sell_pf", "total_trades", "max_drawdown", "avg_r",
        "recent_trades", "recent_issue", "final_verdict",
    ]
    print("\nRanked indicator configs")
    print(summary[cols].round(3).to_string(index=False))
    print("\nRolling monthly for best variant:", best_variant)
    print(rolling.round(3).to_string(index=False))
    final = str(summary.iloc[0]["final_verdict"])
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
