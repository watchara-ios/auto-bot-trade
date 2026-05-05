"""
Mean Reversion Sensitivity Test — BTC + SOL + ETH
===================================================
ทดสอบ variants หลัก 4 กลุ่ม:
  1. Regime gate       — ADX < 20 / 25 / 20+falling
  2. Signal quality    — RSI strict / loose / rejection candle
  3. TP type           — BB midline vs Fixed RR
  4. Side + Session    — SELL-only, London+NY session

Period   : 2024-01-01 → 2025-12-31
Warmup   : 2023-07-01 (6 เดือน สำหรับ indicator warmup)
Walk-forward:
  Fold 1 : Train 2024-H1 → Test 2024-H2
  Fold 2 : Train 2024    → Test 2025
"""

from dataclasses import replace

import numpy as np
import pandas as pd

from backtest_mean_reversion import (
    MRConfig,
    DATA_DIR,
    REPORT_DIR,
    find_symbol_files,
    load_csv,
    metrics,
    monthly_performance,
    prepare,
    run_backtest,
    write_csv,
)

OUTPUT_DIR = REPORT_DIR / "sensitivity"

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
WARMUP_START = "2023-07-01"
FULL = ("2024-01-01", "2025-12-31 23:59:59")
FOLDS = [
    ("fold1", "2024-01-01", "2024-06-30 23:59:59", "2024-07-01", "2024-12-31 23:59:59"),
    ("fold2", "2024-01-01", "2024-12-31 23:59:59", "2025-01-01", "2025-12-31 23:59:59"),
]
LONDON_NY = tuple(range(7, 21))


def base() -> MRConfig:
    return MRConfig(
        adx_max=20.0,
        require_rsi=True,
        rsi_ob=70.0,
        rsi_os=30.0,
        sl_atr_mult=1.5,
        tp_type="bb_mid",
        rr=1.5,
        risk_pct=0.01,
        max_trades_per_day=3,
    )


def variants() -> list[tuple[str, MRConfig]]:
    b = base()
    return [
        # ── Regime gate ─────────────────────────────────────────────────────
        ("adx20_bbmid",     b),
        ("adx25_bbmid",     replace(b, adx_max=25.0)),
        ("adx15_bbmid",     replace(b, adx_max=15.0)),
        ("adx20_falling",   replace(b, require_adx_falling=True)),
        # ── RSI thresholds ──────────────────────────────────────────────────
        ("adx20_rsi75_25",  replace(b, rsi_ob=75.0, rsi_os=25.0)),
        ("adx20_rsi80_20",  replace(b, rsi_ob=80.0, rsi_os=20.0)),
        ("adx20_no_rsi",    replace(b, require_rsi=False)),
        # ── Rejection candle ────────────────────────────────────────────────
        ("adx20_rejection", replace(b, require_rejection=True, min_wick_pct=0.35)),
        ("adx25_rejection", replace(b, adx_max=25.0, require_rejection=True)),
        # ── TP type ─────────────────────────────────────────────────────────
        ("adx20_rr15",      replace(b, tp_type="fixed_rr", rr=1.5)),
        ("adx20_rr20",      replace(b, tp_type="fixed_rr", rr=2.0)),
        ("adx25_rr15",      replace(b, adx_max=25.0, tp_type="fixed_rr", rr=1.5)),
        # ── SL width ────────────────────────────────────────────────────────
        ("adx20_sl1",       replace(b, sl_atr_mult=1.0)),
        ("adx20_sl2",       replace(b, sl_atr_mult=2.0)),
        # ── Side isolation ──────────────────────────────────────────────────
        ("adx20_sell_only", replace(b, allowed_side="SELL")),
        ("adx20_buy_only",  replace(b, allowed_side="BUY")),
        # ── Session filter ──────────────────────────────────────────────────
        ("adx20_lnny",      replace(b, include_utc_hours=LONDON_NY)),
        ("adx25_lnny",      replace(b, adx_max=25.0, include_utc_hours=LONDON_NY)),
        # ── Volume spike ────────────────────────────────────────────────────
        ("adx20_volume",    replace(b, volume_mult=1.2)),
        # ── BB std wider ────────────────────────────────────────────────────
        ("adx20_bb25",      replace(b, bb_std=2.5)),
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
    m15_w  = m15_full.loc[WARMUP_START:end]
    m15p   = prepare(m15_w, cfg).loc[start:end]
    m1     = m1_full.loc[start:end]
    days   = max((m15p.index[-1].date() - m15p.index[0].date()).days + 1, 1)
    trades, equity = run_backtest(symbol, m1, m15p, cfg, variant, label)
    if not trades.empty:
        risk = abs(trades["entry"] - trades["sl"]) * trades["qty"]
        trades["r_multiple"] = trades["pnl"] / risk.replace(0, np.nan)
    row = {"symbol": symbol, "variant": variant, "split": label, "days": days,
           **metrics(trades, equity, days)}
    row["neg_months"] = neg_months(trades)
    row["avg_r"] = float(trades["r_multiple"].mean()) if not trades.empty else 0.0
    return row, trades


def main():
    var_list = variants()
    var_map  = dict(var_list)

    full_rows, full_trades_list = [], []
    wf_rows,   wf_trades_list  = [], []

    for symbol in SYMBOLS:
        files = find_symbol_files(symbol)
        if not files or "m15" not in files:
            print(f"  [skip] {symbol}: ไม่พบไฟล์")
            continue

        m1_full  = load_csv(files["m1"])
        m15_full = load_csv(files["m15"])

        # ── Full-period grid ──────────────────────────────────────────────
        print(f"\n[{symbol}] Full-period {FULL[0][:7]}–{FULL[1][:7]} …", flush=True)
        for name, cfg in var_list:
            print(f"  {name}", end=" ", flush=True)
            row, trades = run_one(symbol, m1_full, m15_full,
                                  cfg, name, "2024_2025", *FULL)
            full_rows.append(row)
            if not trades.empty:
                full_trades_list.append(trades)
        print()

        # ── Walk-forward ──────────────────────────────────────────────────
        for fold, tr_s, tr_e, te_s, te_e in FOLDS:
            print(f"[{symbol}] {fold}: train {tr_s[:7]}→{tr_e[:7]}, "
                  f"test {te_s[:7]}→{te_e[:7]} …")
            train_pf = {}
            for name, cfg in var_list:
                row, _ = run_one(symbol, m1_full, m15_full,
                                 cfg, name, f"{fold}_train", tr_s, tr_e)
                train_pf[name] = row["profit_factor"] if row["total_trades"] >= 20 else 0.0

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

    # ── Collate & save ────────────────────────────────────────────────────
    full_df = pd.DataFrame(full_rows).sort_values(
        ["symbol", "profit_factor"], ascending=[True, False]
    )
    wf_df   = pd.DataFrame(wf_rows)
    trades_full = (pd.concat(full_trades_list, ignore_index=True)
                   if full_trades_list else pd.DataFrame())
    trades_wf   = (pd.concat(wf_trades_list,   ignore_index=True)
                   if wf_trades_list   else pd.DataFrame())
    monthly = monthly_performance(trades_full)

    write_csv(OUTPUT_DIR / "full_grid.csv",       full_df)
    write_csv(OUTPUT_DIR / "walkforward_oos.csv",  wf_df)
    write_csv(OUTPUT_DIR / "trades_full.csv",      trades_full)
    write_csv(OUTPUT_DIR / "trades_wf_oos.csv",    trades_wf)
    write_csv(OUTPUT_DIR / "monthly_full.csv",     monthly)

    # ── Print ─────────────────────────────────────────────────────────────
    COLS = ["symbol", "variant", "total_trades", "winrate", "profit_factor",
            "profit_pct", "max_drawdown", "expectancy", "trades_per_day",
            "avg_r", "neg_months"]
    WF_COLS = ["symbol", "fold", "selected_config", "train_pf",
               "total_trades", "winrate", "profit_factor",
               "profit_pct", "max_drawdown", "neg_months"]

    print("\n" + "=" * 72)
    print("MEAN REVERSION — FULL GRID 2024–2025")
    print("=" * 72)
    print(full_df[COLS].to_string(index=False))

    print("\n" + "=" * 72)
    print("MEAN REVERSION — WALK-FORWARD OOS")
    print("=" * 72)
    print(wf_df[WF_COLS].to_string(index=False))

    # ── Portfolio summary ─────────────────────────────────────────────────
    if not trades_full.empty:
        port = []
        for v, grp in trades_full.groupby("variant"):
            syms = grp["symbol"].nunique()
            key  = pd.to_datetime(grp["exit_time"]).dt.to_period("M").astype(str)
            neg_m = int(grp.groupby(["symbol", key])["pnl"].sum().lt(0).sum())
            port.append({
                "variant":     v,
                "symbols":     syms,
                "trades":      len(grp),
                "tpd":         round(len(grp) / (syms * 365 * 2), 2),
                "winrate":     round((grp["pnl"] > 0).mean() * 100, 1),
                "pf":          round(pf_of(grp["pnl"]), 3),
                "portfolio_%": round(grp["pnl"].sum() / (1000 * syms) * 100, 2),
                "neg_sym_mo":  neg_m,
            })
        port_df = pd.DataFrame(port).sort_values("pf", ascending=False)
        write_csv(OUTPUT_DIR / "portfolio.csv", port_df)

        print("\n" + "=" * 72)
        print("PORTFOLIO (BTC + ETH + SOL, $1 000 each, 2024–2025)")
        print("=" * 72)
        print(port_df.to_string(index=False))

    print(f"\nOutputs → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
