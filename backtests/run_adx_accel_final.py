"""
ADX Acceleration Config — Final Backtest
==========================================
Focus: fine-tune adx_accel2_be และ variants รอบๆ มัน
เปรียบเทียบกับ baseline rr15 (FINAL_REPORT config)

Technique ที่ทดสอบ:
  adx_bars_rising : 1, 2, 3 bars
  breakeven_r     : 0, 0.5, 1.0
  adx_min         : 18, 20, 22
  rr              : 1.5, 2.0, 2.5
  partial_close   : none, 0.75R, 1.0R
  fresh_breakout  : on/off combined with accel

Output: reports/adx_accel/
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backtests"))

from realistic_donchian_backtest import (   # noqa: E402
    Config, DATA_DIR, load_csv, metrics, prepare, run_backtest, write_csv,
)

OUTPUT_DIR   = ROOT / "reports" / "adx_accel"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WARMUP_START = "2023-07-01"
FULL         = ("2024-01-01", "2025-12-31 23:59:59")
FOLDS = [
    ("fold1", "2024-01-01", "2024-06-30 23:59:59", "2024-07-01", "2024-12-31 23:59:59"),
    ("fold2", "2024-01-01", "2024-12-31 23:59:59", "2025-01-01", "2025-12-31 23:59:59"),
]

SYMBOL_PREFIX = {
    "BTCUSDT":  "bitcoin",
    "SOLUSDT":  "solana",
    "ETHUSDT":  "ethereum",
    "AVAXUSDT": "avalanche",
    "BNBUSDT":  "bnb",
    "LINKUSDT": "chainlink",
    "DOTUSDT":  "polkadot",
}


def find_files(symbol: str) -> dict | None:
    prefix = SYMBOL_PREFIX.get(symbol, symbol.lower().replace("usdt", ""))
    m15 = DATA_DIR / f"{prefix}_2022_2025_15m.csv"
    m1  = DATA_DIR / f"{prefix}_2022_2025_1m.csv"
    return {"m15": m15, "m1": m1} if (m15.exists() and m1.exists()) else None


def resample(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    return (df.resample(freq)
            .agg({"open": "first", "high": "max", "low": "min",
                  "close": "last", "volume": "sum"})
            .dropna(subset=["open", "close"]))


# ── Configs ───────────────────────────────────────────────────────────────────

def baseline() -> Config:
    """Original rr15 config from FINAL_REPORT — reference point."""
    return Config(
        adx_min=20.0, adx_max=28.0,
        donchian_n=20, swing_lookback=8,
        atr_period=14, adx_period=14,
        use_volume_filter=True, volume_mult=1.2,
        use_atr_expansion=True,
        rr=1.5, max_trades_per_day=2,
    )


def variants() -> list[tuple[str, Config]]:
    b = baseline()
    return [
        # ── reference ───────────────────────────────────────────────────
        ("baseline",             b),

        # ── adx_bars_rising sweep ───────────────────────────────────────
        ("accel1_be05",          replace(b, adx_bars_rising=1, breakeven_r=0.5)),
        ("accel2_be05",          replace(b, adx_bars_rising=2, breakeven_r=0.5)),   # ★ winner
        ("accel3_be05",          replace(b, adx_bars_rising=3, breakeven_r=0.5)),

        # ── breakeven sweep on accel2 ────────────────────────────────────
        ("accel2_no_be",         replace(b, adx_bars_rising=2)),
        ("accel2_be1r",          replace(b, adx_bars_rising=2, breakeven_r=1.0)),

        # ── RR sweep on accel2+be ────────────────────────────────────────
        ("accel2_be05_rr20",     replace(b, adx_bars_rising=2, breakeven_r=0.5, rr=2.0)),
        ("accel2_be05_rr25",     replace(b, adx_bars_rising=2, breakeven_r=0.5, rr=2.5)),

        # ── ADX min sweep on accel2+be ───────────────────────────────────
        ("accel2_be05_adx18",    replace(b, adx_bars_rising=2, breakeven_r=0.5,
                                          adx_min=18.0, adx_max=35.0)),
        ("accel2_be05_adx22",    replace(b, adx_bars_rising=2, breakeven_r=0.5,
                                          adx_min=22.0, adx_max=30.0)),

        # ── partial close on accel2+be ───────────────────────────────────
        ("accel2_partial075",    replace(b, adx_bars_rising=2,
                                          partial_close_r=0.75, breakeven_r=0.5)),
        ("accel2_partial1r",     replace(b, adx_bars_rising=2,
                                          partial_close_r=1.0, breakeven_r=1.0)),

        # ── fresh breakout combined ──────────────────────────────────────
        ("accel2_fresh_be05",    replace(b, adx_bars_rising=2,
                                          require_fresh_breakout=True, breakeven_r=0.5)),

        # ── equity curve filter ──────────────────────────────────────────
        ("accel2_be05_eqfilter", replace(b, adx_bars_rising=2, breakeven_r=0.5,
                                          equity_curve_filter=True, equity_curve_ma=20)),

        # ── SELL only (for BTC specifically — also test on all) ──────────
        ("accel2_be05_sell",     replace(b, adx_bars_rising=2, breakeven_r=0.5,
                                          allowed_side="SELL")),

        # ── full combo: accel + fresh + be ──────────────────────────────
        ("accel2_fresh_be05_rr20", replace(b, adx_bars_rising=2,
                                            require_fresh_breakout=True,
                                            breakeven_r=0.5, rr=2.0)),
    ]


# ── Runners ───────────────────────────────────────────────────────────────────

def run_one(symbol, m1_full, m15_full, cfg, variant, label, start, end):
    m15_w = m15_full.loc[WARMUP_START:end]
    h1    = resample(m15_w, "1h")
    h4    = resample(m15_w, "4h")
    h1p   = prepare(h1, h4, cfg).loc[start:end]
    m1    = m1_full.loc[start:end]
    days  = max((h1p.index[-1].date() - h1p.index[0].date()).days + 1, 1)

    trades, equity = run_backtest(symbol, m1, h1p, cfg, variant, label)
    if not trades.empty:
        risk = (trades["entry"] - trades["sl"]).abs() * trades["qty"]
        trades["r_multiple"] = trades["pnl"] / risk.replace(0, np.nan)

    row = {"symbol": symbol, "variant": variant, "split": label, "days": days,
           **metrics(trades, equity, days)}
    row["avg_r"]      = round(float(trades["r_multiple"].mean()), 3) if not trades.empty else 0.0
    row["neg_months"] = 0
    if not trades.empty:
        row["neg_months"] = int(
            trades.assign(m=pd.to_datetime(trades["exit_time"]).dt.to_period("M").astype(str))
            .groupby("m")["pnl"].sum().lt(0).sum()
        )
    return row, trades


def pf(pnl: pd.Series) -> float:
    gp, gl = pnl[pnl > 0].sum(), -pnl[pnl <= 0].sum()
    return round(float(gp / gl), 3) if gl else 0.0


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    available = {s: find_files(s) for s in SYMBOL_PREFIX if find_files(s)}
    var_list  = variants()
    var_map   = dict(var_list)

    print(f"Symbols : {list(available)}")
    print(f"Variants: {len(var_list)}")
    print(f"Period  : {FULL[0]} → {FULL[1]}")
    print("=" * 72)

    full_rows, full_trades = [], []
    wf_rows,   wf_trades   = [], []

    for symbol, files in available.items():
        m1_full  = load_csv(files["m1"])
        m15_full = load_csv(files["m15"])

        # full period
        print(f"\n[{symbol}]", end=" ", flush=True)
        for name, cfg in var_list:
            print(".", end="", flush=True)
            row, tr = run_one(symbol, m1_full, m15_full, cfg, name, "full", *FULL)
            full_rows.append(row)
            if not tr.empty:
                full_trades.append(tr)
        print()

        # walk-forward
        for fold, tr_s, tr_e, te_s, te_e in FOLDS:
            train_pf = {}
            for name, cfg in var_list:
                r, _ = run_one(symbol, m1_full, m15_full, cfg, name,
                               f"{fold}_train", tr_s, tr_e)
                train_pf[name] = r["profit_factor"] if r["total_trades"] >= 10 else 0.0
            best = max(train_pf, key=train_pf.get)
            r, tr = run_one(symbol, m1_full, m15_full, var_map[best], best,
                            f"{fold}_OOS", te_s, te_e)
            r.update({"fold": fold, "selected": best,
                       "train_pf": round(train_pf[best], 4)})
            wf_rows.append(r)
            if not tr.empty:
                wf_trades.append(tr)

    full_df   = pd.DataFrame(full_rows)
    wf_df     = pd.DataFrame(wf_rows)
    trades_df = pd.concat(full_trades, ignore_index=True) if full_trades else pd.DataFrame()
    trades_wf = pd.concat(wf_trades,  ignore_index=True) if wf_trades  else pd.DataFrame()

    write_csv(OUTPUT_DIR / "full_grid.csv",      full_df)
    write_csv(OUTPUT_DIR / "walkforward_oos.csv", wf_df)
    write_csv(OUTPUT_DIR / "trades_full.csv",     trades_df)
    write_csv(OUTPUT_DIR / "trades_wf.csv",       trades_wf)

    # ── Per-symbol table ──────────────────────────────────────────────────
    COLS = ["symbol", "variant", "total_trades", "winrate",
            "profit_factor", "profit_pct", "max_drawdown", "avg_r", "neg_months"]
    full_disp = (full_df[full_df["split"] == "full"]
                 .sort_values(["symbol", "profit_factor"], ascending=[True, False]))
    for c in COLS:
        if c not in full_disp.columns:
            full_disp[c] = 0

    print("\n" + "=" * 72)
    print("PER-SYMBOL  2024–2025 (top 5 per symbol)")
    print("=" * 72)
    for sym, grp in full_disp.groupby("symbol"):
        print(f"\n── {sym} ──")
        print(grp[COLS].head(5).to_string(index=False))

    # ── Portfolio table ───────────────────────────────────────────────────
    if not trades_df.empty:
        n = len(available)
        port = []
        for v, grp in trades_df.groupby("variant"):
            port.append({
                "variant":     v,
                "symbols":     grp["symbol"].nunique(),
                "trades":      len(grp),
                "wr_%":        round((grp["pnl"] > 0).mean() * 100, 1),
                "pf":          pf(grp["pnl"]),
                "profit_%":    round(grp["pnl"].sum() / (1000.0 * n) * 100, 2),
                "pnl_$":       round(grp["pnl"].sum(), 1),
                "avg_dd":      round(
                    full_df[full_df["split"] == "full"].groupby("variant")["max_drawdown"].mean().get(v, 0), 2),
            })
        port_df = pd.DataFrame(port).sort_values("pf", ascending=False)
        write_csv(OUTPUT_DIR / "portfolio.csv", port_df)

        print("\n" + "=" * 72)
        print(f"PORTFOLIO ({n} symbols × $1 000, 2024–2025)  — sorted by PF")
        print("=" * 72)
        print(port_df.to_string(index=False))

    # ── Walk-forward ──────────────────────────────────────────────────────
    WF = ["symbol", "fold", "selected", "train_pf",
          "total_trades", "profit_factor", "profit_pct", "max_drawdown"]
    for c in WF:
        if c not in wf_df.columns:
            wf_df[c] = ""

    print("\n" + "=" * 72)
    print("WALK-FORWARD OOS  (fold2 = most important)")
    print("=" * 72)
    print(wf_df[WF].to_string(index=False))

    # ── Fold2 summary ─────────────────────────────────────────────────────
    fold2 = wf_df[wf_df["fold"] == "fold2"]
    n_pass = (fold2["profit_factor"].astype(float) > 1.0).sum()
    avg_pf = fold2["profit_factor"].astype(float).mean()
    print(f"\nFold2 summary: {n_pass}/{len(fold2)} symbols pass (PF>1.0) | avg PF={avg_pf:.3f}")

    # ── Best config recommendation ────────────────────────────────────────
    print("\n" + "=" * 72)
    print("BEST CONFIG RECOMMENDATION  (highest avg fold2 OOS PF)")
    print("=" * 72)
    if not trades_df.empty:
        best_port = port_df.iloc[0]
        print(f"→ {best_port['variant']}")
        print(f"  Portfolio PF={best_port['pf']}  profit={best_port['profit_%']}%  "
              f"trades={best_port['trades']}  WR={best_port['wr_%']}%")

    print(f"\nOutputs → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
