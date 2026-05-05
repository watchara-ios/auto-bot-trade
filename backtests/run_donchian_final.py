"""
Donchian Final: Regime Filter + Walk-Forward Validation
=======================================================
Symbols  : BTC + SOL only (ETH removed — consistently PF < 0.85)
Period   : 2024-01-01 → 2025-12-31 (warmup from 2023-07-01)
Regime   : D1 ADX > 20 AND D1 EMA50/200 trend must confirm trade direction

Walk-forward folds
  Fold 1 : Train 2024-H1 (Jan–Jun 2024)  → Test 2024-H2 (Jul–Dec 2024)
  Fold 2 : Train 2024     (Jan–Dec 2024) → Test 2025     (Jan–Dec 2025)

For each fold × symbol, the best-PF candidate from train is applied to test.
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

OUTPUT_DIR = REPORT_DIR / "final"

SYMBOLS = {"BTCUSDT": "bitcoin", "SOLUSDT": "solana"}
WARMUP_START = "2023-07-01"
LONDON_NY = tuple(range(7, 21))

FOLDS = [
    ("fold1", "2024-01-01", "2024-06-30 23:59:59", "2024-07-01", "2024-12-31 23:59:59"),
    ("fold2", "2024-01-01", "2024-12-31 23:59:59", "2025-01-01", "2025-12-31 23:59:59"),
]

# Full-period snapshot for reference (no walk-forward selection)
FULL_PERIOD = ("2024-01-01", "2025-12-31 23:59:59")


def base_cfg() -> Config:
    return Config(
        adx_min=20.0,
        adx_max=28.0,
        use_volume_filter=True,
        volume_mult=1.2,
        use_atr_expansion=True,
    )


def candidates() -> list[tuple[str, Config]]:
    base = base_cfg()
    return [
        # No regime filter (reference)
        ("base_no_regime",    base),
        ("tierb_rr15",        replace(base, require_tier_b=True, rr=1.5)),
        ("sell_no_regime",    replace(base, allowed_side="SELL")),
        # With D1 regime filter
        ("base_regime",       replace(base, use_regime_filter=True)),
        ("tierb_regime",      replace(base, require_tier_b=True, use_regime_filter=True)),
        ("tierb_rr15_regime", replace(base, require_tier_b=True, rr=1.5, use_regime_filter=True)),
        ("tierb_rr18_regime", replace(base, require_tier_b=True, rr=1.8, use_regime_filter=True)),
        ("sell_regime",       replace(base, allowed_side="SELL", use_regime_filter=True)),
        ("sell_tierb_regime", replace(base, require_tier_b=True, allowed_side="SELL", use_regime_filter=True)),
        ("lnny_regime",       replace(base, include_utc_hours=LONDON_NY, use_regime_filter=True)),
        ("lnny_rr15_regime",  replace(base, include_utc_hours=LONDON_NY, rr=1.5, use_regime_filter=True)),
    ]


def load_symbol(prefix: str) -> dict | None:
    paths = {
        "m1":  DATA_DIR / f"{prefix}_2022_2025_1m.csv",
        "m5":  DATA_DIR / f"{prefix}_2022_2025_5m.csv",
        "m15": DATA_DIR / f"{prefix}_2022_2025_15m.csv",
    }
    return paths if all(p.exists() for p in paths.values()) else None


def slice_df(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return df.loc[start:end]


def pf(pnl: pd.Series) -> float:
    gp = pnl[pnl > 0].sum()
    gl = -pnl[pnl <= 0].sum()
    return float(gp / gl) if gl else 0.0


def neg_months(trades: pd.DataFrame) -> int:
    if trades.empty:
        return 0
    m = (
        pd.to_datetime(trades["exit_time"]).dt.to_period("M").astype(str)
    )
    return int(trades.groupby(m)["pnl"].sum().lt(0).sum())


def run_period(symbol, m1_full, m5_full, m15_full, cfg, variant, split_label, start, end):
    m5_warm = m5_full.loc[WARMUP_START:end]
    m15_warm = m15_full.loc[WARMUP_START:end]
    m5_prep = prepare(m5_warm, m15_warm, cfg)
    m5_prep = m5_prep.loc[start:end]
    m1 = slice_df(m1_full, start, end)
    days = max((m5_prep.index[-1].date() - m5_prep.index[0].date()).days + 1, 1)
    trades, equity = run_backtest(symbol, m1, m5_prep, cfg, variant, split_label)
    # Enrich r_multiple
    if not trades.empty:
        risk = (trades["entry"] - trades["sl"]).abs() * trades["qty"]
        trades["r_multiple"] = trades["pnl"] / risk.replace(0, np.nan)
    row = {"symbol": symbol, "variant": variant, "split": split_label, "days": days,
           **metrics(trades, equity, days)}
    row["neg_months"] = neg_months(trades)
    row["avg_r"] = float(trades["r_multiple"].mean()) if not trades.empty else 0.0
    return row, trades


def main():
    cands = candidates()
    all_full_rows = []
    all_full_trades = []
    wf_rows = []       # walk-forward OOS rows
    wf_trades = []

    for symbol, prefix in SYMBOLS.items():
        files = load_symbol(prefix)
        if not files:
            print(f"Missing data for {symbol}, skipping.")
            continue

        m1_full  = load_csv(files["m1"])
        m5_full  = load_csv(files["m5"])
        m15_full = load_csv(files["m15"])

        # ── Full-period grid (2024–2025) ─────────────────────────────────────
        print(f"\n[{symbol}] Full-period 2024–2025 …")
        for name, cfg in cands:
            row, trades = run_period(
                symbol, m1_full, m5_full, m15_full,
                cfg, name, "2024_2025", *FULL_PERIOD,
            )
            all_full_rows.append(row)
            if not trades.empty:
                all_full_trades.append(trades)

        # ── Walk-forward ──────────────────────────────────────────────────────
        for fold_name, tr_start, tr_end, te_start, te_end in FOLDS:
            print(f"[{symbol}] Walk-forward {fold_name}: train {tr_start[:7]}→{tr_end[:7]}, test {te_start[:7]}→{te_end[:7]} …")

            # 1. Run all candidates on train
            train_pf = {}
            for name, cfg in cands:
                row, _ = run_period(
                    symbol, m1_full, m5_full, m15_full,
                    cfg, name, f"{fold_name}_train", tr_start, tr_end,
                )
                train_pf[name] = row["profit_factor"] if row["total_trades"] >= 20 else 0.0

            best_name = max(train_pf, key=train_pf.get)
            best_cfg  = dict(cands)[best_name]
            print(f"  → best on train: {best_name} (PF={train_pf[best_name]:.3f})")

            # 2. Apply best config to test (OOS)
            row, trades = run_period(
                symbol, m1_full, m5_full, m15_full,
                best_cfg, best_name, f"{fold_name}_OOS", te_start, te_end,
            )
            row["fold"] = fold_name
            row["selected_config"] = best_name
            row["train_pf"] = train_pf[best_name]
            wf_rows.append(row)
            if not trades.empty:
                wf_trades.append(trades)

    # ── Save & print ─────────────────────────────────────────────────────────
    full_df = pd.DataFrame(all_full_rows).sort_values(
        ["symbol", "profit_factor"], ascending=[True, False]
    )
    wf_df = pd.DataFrame(wf_rows)
    trades_full = pd.concat(all_full_trades, ignore_index=True) if all_full_trades else pd.DataFrame()
    trades_wf   = pd.concat(wf_trades,       ignore_index=True) if wf_trades   else pd.DataFrame()

    write_csv(OUTPUT_DIR / "full_period_grid.csv",      full_df)
    write_csv(OUTPUT_DIR / "walkforward_oos.csv",        wf_df)
    write_csv(OUTPUT_DIR / "trades_full.csv",            trades_full)
    write_csv(OUTPUT_DIR / "trades_walkforward_oos.csv", trades_wf)

    cols = ["symbol", "variant", "total_trades", "winrate", "profit_factor",
            "profit_pct", "max_drawdown", "expectancy", "avg_r", "neg_months"]

    print("\n" + "=" * 70)
    print("FULL PERIOD GRID (2024–2025)  — sorted by symbol + profit_factor")
    print("=" * 70)
    print(full_df[cols].to_string(index=False))

    wf_cols = ["symbol", "fold", "selected_config", "train_pf",
               "total_trades", "winrate", "profit_factor", "profit_pct",
               "max_drawdown", "neg_months"]
    print("\n" + "=" * 70)
    print("WALK-FORWARD OOS RESULTS")
    print("=" * 70)
    print(wf_df[wf_cols].to_string(index=False))

    print(f"\nOutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
