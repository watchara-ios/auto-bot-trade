"""
Donchian Multi-Symbol Backtest (BTC + ETH + SOL)
=================================================
Tests the top configs from daytrade_v2 across 3 symbols.
Data period: Jan 2025 – Dec 2025 (available for all symbols).

Top configs from v2 (on BTC 365d):
  - ref_adx20_28_all_tiers  : PF=1.141, profit=+12%, WR=37.6%, DD=-10.4%
  - tierb_sell_only          : PF=1.168, profit=+8.0%, WR=42.3%, 4 neg months
  - tierb_rr15               : PF=1.096, profit=+8.6%, WR=49.0%
  - tierb_london_ny          : PF=1.099, profit=+6.8%, WR=41.0%
"""

from dataclasses import replace

import numpy as np
import pandas as pd

from realistic_donchian_backtest import (
    Config,
    DATA_DIR,
    REPORT_DIR,
    load_csv,
    metrics,
    monthly_performance,
    prepare,
    run_backtest,
    write_csv,
)

OUTPUT_DIR = REPORT_DIR / "multi_symbol"

SYMBOLS = {
    "BTCUSDT": "bitcoin",
    "ETHUSDT": "ethereum",
    "SOLUSDT": "solana",
}

PERIOD_START = "2025-01-01"
PERIOD_END = "2025-12-31 23:59:59"

LONDON_NY = tuple(range(7, 21))


def load_symbol_files(prefix: str) -> dict | None:
    paths = {
        "m1": DATA_DIR / f"{prefix}_2022_2025_1m.csv",
        "m5": DATA_DIR / f"{prefix}_2022_2025_5m.csv",
        "m15": DATA_DIR / f"{prefix}_2022_2025_15m.csv",
    }
    if not all(p.exists() for p in paths.values()):
        return None
    return paths


def slice_period(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[PERIOD_START:PERIOD_END]


def profit_factor(pnl: pd.Series) -> float:
    gp = pnl[pnl > 0].sum()
    gl = -pnl[pnl <= 0].sum()
    return float(gp / gl) if gl else 0.0


def enrich(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades
    t = trades.copy()
    risk = (t["entry"] - t["sl"]).abs() * t["qty"]
    t["r_multiple"] = t["pnl"] / risk.replace(0, np.nan)
    return t


def negative_months(trades: pd.DataFrame) -> int:
    monthly = monthly_performance(trades)
    return int((monthly["pnl"] < 0).sum()) if not monthly.empty else 0


def variants() -> list[tuple[str, Config]]:
    base = Config(
        adx_min=20.0,
        adx_max=28.0,
        use_volume_filter=True,
        volume_mult=1.2,
        use_atr_expansion=True,
    )
    return [
        # Current best baseline
        ("all_tiers_baseline", base),
        # Tier B only variants
        ("tierb_sell_only", replace(base, require_tier_b=True, allowed_side="SELL")),
        ("tierb_rr15", replace(base, require_tier_b=True, rr=1.5)),
        ("tierb_london_ny", replace(base, require_tier_b=True, include_utc_hours=LONDON_NY)),
        # Combined: session + RR
        ("tierb_lnny_rr15", replace(base, require_tier_b=True, include_utc_hours=LONDON_NY, rr=1.5)),
        # High RR for conviction trades
        ("tierb_lnny_rr25", replace(base, require_tier_b=True, include_utc_hours=LONDON_NY, rr=2.5)),
    ]


def portfolio_metrics(all_trades: pd.DataFrame, initial_balance: float = 1000.0) -> pd.DataFrame:
    """Combined portfolio equity per variant (equal allocation across symbols)."""
    if all_trades.empty:
        return pd.DataFrame()
    rows = []
    for variant, grp in all_trades.groupby("variant"):
        symbols = grp["symbol"].nunique()
        total_pnl = grp["pnl"].sum()
        portfolio_profit_pct = total_pnl / (initial_balance * symbols) * 100
        wins = grp[grp["pnl"] > 0]
        rows.append({
            "variant": variant,
            "symbols": symbols,
            "total_trades": len(grp),
            "trades_per_symbol": len(grp) / symbols,
            "winrate": len(wins) / len(grp) * 100 if grp.shape[0] else 0,
            "profit_factor": profit_factor(grp["pnl"]),
            "portfolio_profit_pct": portfolio_profit_pct,
            "total_pnl": total_pnl,
            "negative_months_total": int(
                grp.assign(month=pd.to_datetime(grp["exit_time"]).dt.to_period("M").astype(str))
                .groupby(["symbol", "month"])["pnl"].sum().lt(0).sum()
            ),
        })
    return pd.DataFrame(rows).sort_values("profit_factor", ascending=False)


def main():
    summary_rows = []
    trade_frames = []
    monthly_frames = []
    missing = []

    var_list = variants()

    for symbol, prefix in SYMBOLS.items():
        files = load_symbol_files(prefix)
        if not files:
            missing.append(symbol)
            continue

        m1_full = load_csv(files["m1"])
        m5_full = load_csv(files["m5"])
        m15_full = load_csv(files["m15"])

        m1 = slice_period(m1_full)
        m5_raw = slice_period(m5_full)
        m15 = slice_period(m15_full)

        if len(m5_raw) < 300 or len(m15) < 220 or len(m1) < 1000:
            missing.append(f"{symbol} (insufficient data)")
            continue

        # Warm up indicators on full history, but backtest only on 2025 slice.
        # Use the m5_raw with 6-month warm-up window for indicator accuracy.
        warmup_start = "2024-07-01"
        m5_warmup = m5_full.loc[warmup_start:PERIOD_END]
        m15_warmup = m15_full.loc[warmup_start:PERIOD_END]

        days = max((m5_raw.index[-1].date() - m5_raw.index[0].date()).days + 1, 1)

        for name, cfg in var_list:
            m5_prep = prepare(m5_warmup, m15_warmup, cfg)
            # Trim back to 2025 after warmup
            m5_prep = m5_prep.loc[PERIOD_START:PERIOD_END]

            trades, equity = run_backtest(symbol, m1, m5_prep, cfg, name, "2025")
            trades = enrich(trades)
            row = {
                "symbol": symbol,
                "variant": name,
                "split": "2025",
                "days": days,
                **metrics(trades, equity, days),
            }
            row["negative_months"] = negative_months(trades)
            row["avg_r"] = float(trades["r_multiple"].mean()) if not trades.empty else 0.0
            summary_rows.append(row)
            if not trades.empty:
                trade_frames.append(trades)
                monthly_frames.append(monthly_performance(trades))

    summary = pd.DataFrame(summary_rows).sort_values(
        ["variant", "profit_factor"], ascending=[True, False]
    )
    trades_all = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    monthly_all = pd.concat(monthly_frames, ignore_index=True) if monthly_frames else pd.DataFrame()
    portfolio = portfolio_metrics(trades_all)

    write_csv(OUTPUT_DIR / "summary_per_symbol.csv", summary)
    write_csv(OUTPUT_DIR / "trades.csv", trades_all)
    write_csv(OUTPUT_DIR / "monthly_performance.csv", monthly_all)
    write_csv(OUTPUT_DIR / "portfolio_summary.csv", portfolio)

    print("=== Donchian Multi-Symbol (Jan–Dec 2025) ===")
    print(f"Period: {PERIOD_START} to {PERIOD_END}\n")

    if missing:
        print(f"Skipped: {missing}\n")

    # Per-symbol table
    cols = ["symbol", "variant", "total_trades", "winrate", "profit_factor",
            "profit_pct", "max_drawdown", "expectancy", "avg_r", "negative_months"]
    print("--- Per-Symbol Results ---")
    print(summary[cols].to_string(index=False))

    # Portfolio table
    print("\n--- Portfolio (equal allocation per symbol, $1000 each) ---")
    pcols = ["variant", "symbols", "total_trades", "trades_per_symbol",
             "winrate", "profit_factor", "portfolio_profit_pct", "negative_months_total"]
    print(portfolio[pcols].to_string(index=False))
    print(f"\nOutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
