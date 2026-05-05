"""
Donchian Weekly Regime Gate — BTC + SOL
=========================================
ทดสอบ 4 ระดับ regime filter เปรียบเทียบกัน:
  1. none       — ไม่มี regime filter
  2. d1_only    — D1 ADX>20 + D1 EMA trend
  3. w1_only    — W1 ADX>25 + W1 HH+HL / LH+LL (1 week)
  4. w1_strict  — W1 ADX>25 + W1 HH+HL / LH+LL (2 consecutive weeks)
  5. d1_w1      — D1 + W1 combined (1 week)
  6. d1_w1_strict — D1 + W1 combined (2 weeks)

ทดสอบบน config ที่ดีที่สุดจากการวิเคราะห์ก่อนหน้า (adx_20_28 + vol + atr)
พร้อม walk-forward 2 fold:
  Fold 1 : Train 2024-H1 → Test 2024-H2
  Fold 2 : Train 2024    → Test 2025
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

OUTPUT_DIR = REPORT_DIR / "weekly_regime"

SYMBOLS = {"BTCUSDT": "bitcoin", "SOLUSDT": "solana"}
WARMUP_START = "2023-07-01"

FOLDS = [
    ("fold1", "2024-01-01", "2024-06-30 23:59:59", "2024-07-01", "2024-12-31 23:59:59"),
    ("fold2", "2024-01-01", "2024-12-31 23:59:59", "2025-01-01", "2025-12-31 23:59:59"),
]
FULL = ("2024-01-01", "2025-12-31 23:59:59")


def base_cfg() -> Config:
    return Config(
        adx_min=20.0, adx_max=28.0,
        use_volume_filter=True, volume_mult=1.2,
        use_atr_expansion=True,
    )


def make_variants() -> list[tuple[str, Config]]:
    base = base_cfg()
    return [
        ("none",         base),
        ("d1_only",      replace(base, use_regime_filter=True)),
        ("w1_only",      replace(base, use_weekly_regime=True)),
        ("w1_strict",    replace(base, use_weekly_regime=True, weekly_swing_lookback=2)),
        ("d1_w1",        replace(base, use_regime_filter=True, use_weekly_regime=True)),
        ("d1_w1_strict", replace(base, use_regime_filter=True, use_weekly_regime=True,
                                 weekly_swing_lookback=2)),
        # Best per-symbol configs + weekly regime
        ("rr15_w1",      replace(base, rr=1.5, use_weekly_regime=True)),
        ("sell_w1",      replace(base, allowed_side="SELL", use_weekly_regime=True)),
        ("sell_d1_w1",   replace(base, allowed_side="SELL",
                                 use_regime_filter=True, use_weekly_regime=True)),
        ("rr15_d1_w1",   replace(base, rr=1.5,
                                 use_regime_filter=True, use_weekly_regime=True)),
        ("sell_rr15_w1", replace(base, allowed_side="SELL", rr=1.5,
                                 use_weekly_regime=True)),
        ("sell_rr15_d1_w1", replace(base, allowed_side="SELL", rr=1.5,
                                    use_regime_filter=True, use_weekly_regime=True)),
    ]


def load_symbol(prefix: str) -> dict | None:
    paths = {
        "m1":  DATA_DIR / f"{prefix}_2022_2025_1m.csv",
        "m5":  DATA_DIR / f"{prefix}_2022_2025_5m.csv",
        "m15": DATA_DIR / f"{prefix}_2022_2025_15m.csv",
    }
    return paths if all(p.exists() for p in paths.values()) else None


def pf_of(pnl: pd.Series) -> float:
    gp = pnl[pnl > 0].sum()
    gl = -pnl[pnl <= 0].sum()
    return float(gp / gl) if gl else 0.0


def neg_months(trades: pd.DataFrame) -> int:
    if trades.empty:
        return 0
    key = pd.to_datetime(trades["exit_time"]).dt.to_period("M").astype(str)
    return int(trades.groupby(key)["pnl"].sum().lt(0).sum())


def run_one(symbol, m1_full, m5_full, m15_full, cfg, variant, label, start, end):
    m5_w  = m5_full.loc[WARMUP_START:end]
    m15_w = m15_full.loc[WARMUP_START:end]
    m5p   = prepare(m5_w, m15_w, cfg).loc[start:end]
    m1    = m1_full.loc[start:end]
    days  = max((m5p.index[-1].date() - m5p.index[0].date()).days + 1, 1)
    trades, equity = run_backtest(symbol, m1, m5p, cfg, variant, label)
    if not trades.empty:
        risk = (trades["entry"] - trades["sl"]).abs() * trades["qty"]
        trades["r_multiple"] = trades["pnl"] / risk.replace(0, np.nan)
    row = {"symbol": symbol, "variant": variant, "split": label, "days": days,
           **metrics(trades, equity, days)}
    row["neg_months"] = neg_months(trades)
    row["avg_r"] = float(trades["r_multiple"].mean()) if not trades.empty else 0.0
    return row, trades


def main():
    variants = make_variants()
    variant_map = dict(variants)

    full_rows, full_trades_list = [], []
    wf_rows, wf_trades_list = [], []

    for symbol, prefix in SYMBOLS.items():
        files = load_symbol(prefix)
        if not files:
            print(f"  [skip] {symbol}: ไม่พบไฟล์ข้อมูล")
            continue

        m1_full  = load_csv(files["m1"])
        m5_full  = load_csv(files["m5"])
        m15_full = load_csv(files["m15"])

        # ── Full-period grid ──────────────────────────────────────────────
        print(f"[{symbol}] Full-period {FULL[0][:7]}–{FULL[1][:7]} …")
        for name, cfg in variants:
            row, trades = run_one(symbol, m1_full, m5_full, m15_full,
                                  cfg, name, "2024_2025", *FULL)
            full_rows.append(row)
            if not trades.empty:
                full_trades_list.append(trades)

        # ── Walk-forward ──────────────────────────────────────────────────
        for fold, tr_s, tr_e, te_s, te_e in FOLDS:
            print(f"[{symbol}] {fold}: train {tr_s[:7]}→{tr_e[:7]}, "
                  f"test {te_s[:7]}→{te_e[:7]} …")

            # Train — pick best PF among candidates with ≥20 trades
            train_pf = {}
            for name, cfg in variants:
                row, _ = run_one(symbol, m1_full, m5_full, m15_full,
                                 cfg, name, f"{fold}_train", tr_s, tr_e)
                train_pf[name] = row["profit_factor"] if row["total_trades"] >= 20 else 0.0

            best_name = max(train_pf, key=train_pf.get)
            best_cfg  = variant_map[best_name]
            print(f"  → เลือก: {best_name}  (train PF={train_pf[best_name]:.3f})")

            # Test — OOS
            row, trades = run_one(symbol, m1_full, m5_full, m15_full,
                                  best_cfg, best_name, f"{fold}_OOS", te_s, te_e)
            row["fold"]            = fold
            row["selected_config"] = best_name
            row["train_pf"]        = round(train_pf[best_name], 4)
            wf_rows.append(row)
            if not trades.empty:
                wf_trades_list.append(trades)

    # ── Save ─────────────────────────────────────────────────────────────
    full_df = pd.DataFrame(full_rows).sort_values(
        ["symbol", "profit_factor"], ascending=[True, False]
    )
    wf_df = pd.DataFrame(wf_rows)
    trades_full = (pd.concat(full_trades_list, ignore_index=True)
                   if full_trades_list else pd.DataFrame())
    trades_wf   = (pd.concat(wf_trades_list, ignore_index=True)
                   if wf_trades_list else pd.DataFrame())
    monthly_full = monthly_performance(trades_full)

    write_csv(OUTPUT_DIR / "full_grid.csv",       full_df)
    write_csv(OUTPUT_DIR / "walkforward_oos.csv",  wf_df)
    write_csv(OUTPUT_DIR / "trades_full.csv",      trades_full)
    write_csv(OUTPUT_DIR / "trades_wf_oos.csv",    trades_wf)
    write_csv(OUTPUT_DIR / "monthly_full.csv",     monthly_full)

    # ── Print ─────────────────────────────────────────────────────────────
    COLS = ["symbol", "variant", "total_trades", "winrate", "profit_factor",
            "profit_pct", "max_drawdown", "expectancy", "avg_r", "neg_months"]
    WF_COLS = ["symbol", "fold", "selected_config", "train_pf",
               "total_trades", "winrate", "profit_factor",
               "profit_pct", "max_drawdown", "neg_months"]

    print("\n" + "=" * 72)
    print("FULL GRID 2024–2025  (BTC + SOL × regime variants)")
    print("=" * 72)
    print(full_df[COLS].to_string(index=False))

    print("\n" + "=" * 72)
    print("WALK-FORWARD OOS")
    print("=" * 72)
    print(wf_df[WF_COLS].to_string(index=False))

    # Portfolio summary (equal $1 000 per symbol)
    if not trades_full.empty:
        print("\n" + "=" * 72)
        print("PORTFOLIO (BTC + SOL, $1 000 each, full 2024–2025)")
        print("=" * 72)
        port_rows = []
        for variant, grp in trades_full.groupby("variant"):
            syms   = grp["symbol"].nunique()
            wins   = (grp["pnl"] > 0).sum()
            total  = len(grp)
            key    = pd.to_datetime(grp["exit_time"]).dt.to_period("M").astype(str)
            neg_m  = int(grp.groupby(["symbol", key])["pnl"].sum().lt(0).sum())
            port_rows.append({
                "variant":      variant,
                "total_trades": total,
                "tpd":          round(total / (365 * 2), 2),
                "winrate":      round(wins / total * 100, 1) if total else 0,
                "pf":           round(pf_of(grp["pnl"]), 3),
                "portfolio_%":  round(grp["pnl"].sum() / (1000 * syms) * 100, 2),
                "neg_sym_months": neg_m,
            })
        port_df = pd.DataFrame(port_rows).sort_values("pf", ascending=False)
        write_csv(OUTPUT_DIR / "portfolio_summary.csv", port_df)
        print(port_df.to_string(index=False))

    print(f"\nOutputs → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
