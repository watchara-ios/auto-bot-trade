"""
Donchian H1 Backtest — BTC + SOL
==================================
แทน M5 signal ด้วย H1 signal เพื่อลด noise 12x โดยใช้ engine เดิมทั้งหมด:

  signal TF  : H1  (resample จาก M15)  — ทำหน้าที่แทน m5 ใน engine
  trend TF   : H4  (resample จาก M15)  — ทำหน้าที่แทน m15 ใน engine
  exit sim   : M1  (ข้อมูลเดิม)        — precise SL/TP fills

Period   : 2024-01-01 → 2025-12-31
Warmup   : 2023-07-01 (6 เดือนก่อน signal)

Walk-forward:
  Fold 1 : Train 2024-H1 → Test 2024-H2
  Fold 2 : Train 2024    → Test 2025

เปรียบเทียบ:
  - baseline ไม่มี regime
  - d1_regime (D1 ADX + EMA trend — พิสูจน์แล้วว่าช่วยบน M5)
  - w1_regime (W1 ADX + Swing structure)
  - d1_w1 combined
  - RR 1.5 และ 2.0
  - Tier B only
  - SELL side only
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

OUTPUT_DIR = REPORT_DIR / "h1"
WARMUP_START = "2023-07-01"
FULL = ("2024-01-01", "2025-12-31 23:59:59")
FOLDS = [
    ("fold1", "2024-01-01", "2024-06-30 23:59:59", "2024-07-01", "2024-12-31 23:59:59"),
    ("fold2", "2024-01-01", "2024-12-31 23:59:59", "2025-01-01", "2025-12-31 23:59:59"),
]
SYMBOLS = {"BTCUSDT": "bitcoin", "SOLUSDT": "solana"}
LONDON_NY = tuple(range(7, 21))


def resample_h1(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.resample("1h")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna(subset=["open", "close"])
    )


def resample_h4(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.resample("4h")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna(subset=["open", "close"])
    )


def load_symbol(prefix: str) -> dict | None:
    paths = {
        "m1":  DATA_DIR / f"{prefix}_2022_2025_1m.csv",
        "m15": DATA_DIR / f"{prefix}_2022_2025_15m.csv",
    }
    return paths if all(p.exists() for p in paths.values()) else None


def base_cfg() -> Config:
    """Best proven settings ported to H1 timeframe."""
    return Config(
        adx_min=20.0,
        adx_max=28.0,
        donchian_n=20,        # 20 H1 bars ≈ 1 trading day lookback
        swing_lookback=8,     # 8 H1 bars = 8 hours structure
        atr_period=14,
        adx_period=14,
        use_volume_filter=True,
        volume_mult=1.2,
        use_atr_expansion=True,
        rr=2.0,
        max_trades_per_day=2,
    )


def variants() -> list[tuple[str, Config]]:
    base = base_cfg()
    return [
        # ── Regime filter combinations ──────────────────────────────────
        ("none",              base),
        ("d1_regime",         replace(base, use_regime_filter=True)),
        ("w1_regime",         replace(base, use_weekly_regime=True)),
        ("d1_w1",             replace(base, use_regime_filter=True, use_weekly_regime=True)),
        # ── RR variations ───────────────────────────────────────────────
        ("rr15",              replace(base, rr=1.5)),
        ("rr15_d1",           replace(base, rr=1.5, use_regime_filter=True)),
        ("rr15_w1",           replace(base, rr=1.5, use_weekly_regime=True)),
        ("rr15_d1_w1",        replace(base, rr=1.5, use_regime_filter=True, use_weekly_regime=True)),
        # ── Tier B only ─────────────────────────────────────────────────
        ("tierb",             replace(base, require_tier_b=True)),
        ("tierb_d1",          replace(base, require_tier_b=True, use_regime_filter=True)),
        ("tierb_rr15_d1",     replace(base, require_tier_b=True, rr=1.5, use_regime_filter=True)),
        ("tierb_rr15_d1_w1",  replace(base, require_tier_b=True, rr=1.5,
                                       use_regime_filter=True, use_weekly_regime=True)),
        # ── SELL side (consistently stronger in M5 tests) ───────────────
        ("sell_d1",           replace(base, allowed_side="SELL", use_regime_filter=True)),
        ("sell_w1",           replace(base, allowed_side="SELL", use_weekly_regime=True)),
        ("sell_rr15_d1",      replace(base, allowed_side="SELL", rr=1.5, use_regime_filter=True)),
        ("sell_rr15_w1",      replace(base, allowed_side="SELL", rr=1.5, use_weekly_regime=True)),
        # ── Donchian window variations ───────────────────────────────────
        ("n12_d1",            replace(base, donchian_n=12, use_regime_filter=True)),
        ("n30_d1",            replace(base, donchian_n=30, use_regime_filter=True)),
        # ── Session filter ───────────────────────────────────────────────
        ("lnny_d1",           replace(base, include_utc_hours=LONDON_NY, use_regime_filter=True)),
        ("lnny_rr15_d1",      replace(base, include_utc_hours=LONDON_NY, rr=1.5,
                                       use_regime_filter=True)),
    ]


def pf_of(pnl: pd.Series) -> float:
    gp = pnl[pnl > 0].sum()
    gl = -pnl[pnl <= 0].sum()
    return float(gp / gl) if gl else 0.0


def neg_months(trades: pd.DataFrame) -> int:
    if trades.empty:
        return 0
    key = pd.to_datetime(trades["exit_time"]).dt.to_period("M").astype(str)
    return int(trades.groupby(key)["pnl"].sum().lt(0).sum())


def run_one(symbol, m1_full, m15_full, cfg, variant, label, start, end):
    # H1 and H4 built from M15 with warmup window for stable indicators
    m15_w = m15_full.loc[WARMUP_START:end]
    h1_w  = resample_h1(m15_w)
    h4_w  = resample_h4(m15_w)

    # prepare() receives h1 as "m5" and h4 as "m15" — engine is TF-agnostic
    h1_prep = prepare(h1_w, h4_w, cfg).loc[start:end]
    m1 = m1_full.loc[start:end]
    days = max((h1_prep.index[-1].date() - h1_prep.index[0].date()).days + 1, 1)

    trades, equity = run_backtest(symbol, m1, h1_prep, cfg, variant, label)
    if not trades.empty:
        risk = (trades["entry"] - trades["sl"]).abs() * trades["qty"]
        trades["r_multiple"] = trades["pnl"] / risk.replace(0, np.nan)

    row = {
        "symbol": symbol, "variant": variant, "split": label, "days": days,
        **metrics(trades, equity, days),
    }
    row["neg_months"] = neg_months(trades)
    row["avg_r"] = float(trades["r_multiple"].mean()) if not trades.empty else 0.0
    return row, trades


def main():
    var_list = variants()
    var_map = dict(var_list)

    full_rows, full_trades_list = [], []
    wf_rows,   wf_trades_list  = [], []

    for symbol, prefix in SYMBOLS.items():
        files = load_symbol(prefix)
        if not files:
            print(f"  [skip] {symbol}: ไม่พบไฟล์")
            continue

        m1_full  = load_csv(files["m1"])
        m15_full = load_csv(files["m15"])

        # ── Full-period grid ──────────────────────────────────────────
        print(f"\n[{symbol}] Full-period {FULL[0][:7]}–{FULL[1][:7]} …", flush=True)
        for name, cfg in var_list:
            print(f"  {name}", end=" ", flush=True)
            row, trades = run_one(symbol, m1_full, m15_full,
                                  cfg, name, "2024_2025", *FULL)
            full_rows.append(row)
            if not trades.empty:
                full_trades_list.append(trades)
        print()

        # ── Walk-forward ──────────────────────────────────────────────
        for fold, tr_s, tr_e, te_s, te_e in FOLDS:
            print(f"[{symbol}] {fold}: train {tr_s[:7]}→{tr_e[:7]}, "
                  f"test {te_s[:7]}→{te_e[:7]} …")
            train_pf = {}
            for name, cfg in var_list:
                row, _ = run_one(symbol, m1_full, m15_full,
                                 cfg, name, f"{fold}_train", tr_s, tr_e)
                train_pf[name] = row["profit_factor"] if row["total_trades"] >= 15 else 0.0

            best_name = max(train_pf, key=train_pf.get)
            best_cfg  = var_map[best_name]
            print(f"  → เลือก: {best_name}  (train PF={train_pf[best_name]:.3f})")

            row, trades = run_one(symbol, m1_full, m15_full,
                                  best_cfg, best_name, f"{fold}_OOS", te_s, te_e)
            row["fold"]            = fold
            row["selected_config"] = best_name
            row["train_pf"]        = round(train_pf[best_name], 4)
            wf_rows.append(row)
            if not trades.empty:
                wf_trades_list.append(trades)

    # ── Collate ───────────────────────────────────────────────────────
    full_df = pd.DataFrame(full_rows).sort_values(
        ["symbol", "profit_factor"], ascending=[True, False]
    )
    wf_df = pd.DataFrame(wf_rows)
    trades_full = (pd.concat(full_trades_list, ignore_index=True)
                   if full_trades_list else pd.DataFrame())
    trades_wf   = (pd.concat(wf_trades_list,   ignore_index=True)
                   if wf_trades_list   else pd.DataFrame())
    monthly = monthly_performance(trades_full)

    write_csv(OUTPUT_DIR / "full_grid.csv",      full_df)
    write_csv(OUTPUT_DIR / "walkforward_oos.csv", wf_df)
    write_csv(OUTPUT_DIR / "trades_full.csv",     trades_full)
    write_csv(OUTPUT_DIR / "trades_wf_oos.csv",   trades_wf)
    write_csv(OUTPUT_DIR / "monthly.csv",         monthly)

    # ── Print ─────────────────────────────────────────────────────────
    COLS = ["symbol", "variant", "total_trades", "winrate", "profit_factor",
            "profit_pct", "max_drawdown", "expectancy", "avg_r", "neg_months"]
    WF_COLS = ["symbol", "fold", "selected_config", "train_pf",
               "total_trades", "winrate", "profit_factor",
               "profit_pct", "max_drawdown", "neg_months"]

    print("\n" + "=" * 70)
    print("H1 DONCHIAN — FULL GRID 2024–2025")
    print("=" * 70)
    print(full_df[COLS].to_string(index=False))

    print("\n" + "=" * 70)
    print("H1 DONCHIAN — WALK-FORWARD OOS")
    print("=" * 70)
    print(wf_df[WF_COLS].to_string(index=False))

    # Portfolio
    if not trades_full.empty:
        port = []
        for v, grp in trades_full.groupby("variant"):
            syms = grp["symbol"].nunique()
            port.append({
                "variant":     v,
                "trades":      len(grp),
                "tpd":         round(len(grp) / (365 * 2), 2),
                "winrate":     round((grp["pnl"] > 0).mean() * 100, 1),
                "pf":          round(pf_of(grp["pnl"]), 3),
                "portfolio_%": round(grp["pnl"].sum() / (1000 * syms) * 100, 2),
            })
        port_df = pd.DataFrame(port).sort_values("pf", ascending=False)
        write_csv(OUTPUT_DIR / "portfolio.csv", port_df)
        print("\n" + "=" * 70)
        print("PORTFOLIO (BTC + SOL, $1 000 each, 2024–2025)")
        print("=" * 70)
        print(port_df.to_string(index=False))

    print(f"\nOutputs → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
