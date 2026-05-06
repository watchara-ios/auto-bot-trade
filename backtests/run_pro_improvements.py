"""
Pro Improvements Backtest
==========================
ทดสอบ 3 improvement พร้อมกัน:

  1. BTC SELL-only   — ตัด BTC BUY side ที่ขาดทุนออก
  2. Symbol expansion — เพิ่ม AVAX, BNB, LINK, DOT
  3. Partial close   — ปิด 50% ที่ 1R, ปล่อย 50% วิ่งไป target

เปรียบเทียบกับ baseline: BTC+SOL rr15 both sides (ผลเดิมจาก FINAL_REPORT)

Output: reports/pro_improvements/
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

OUTPUT_DIR   = ROOT / "reports" / "pro_improvements"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WARMUP_START = "2023-07-01"
FULL         = ("2024-01-01", "2025-12-31 23:59:59")
FOLDS = [
    ("fold1", "2024-01-01", "2024-06-30 23:59:59", "2024-07-01", "2024-12-31 23:59:59"),
    ("fold2", "2024-01-01", "2024-12-31 23:59:59", "2025-01-01", "2025-12-31 23:59:59"),
]

# ── Symbol registry ───────────────────────────────────────────────────────────

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
    if m15.exists() and m1.exists():
        return {"m15": m15, "m1": m1}
    return None


# ── Resample helpers ──────────────────────────────────────────────────────────

def resample(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    return (df.resample(freq)
            .agg({"open": "first", "high": "max", "low": "min",
                  "close": "last", "volume": "sum"})
            .dropna(subset=["open", "close"]))


# ── Base config (proven rr15 from FINAL_REPORT) ───────────────────────────────

def base_cfg() -> Config:
    return Config(
        adx_min=20.0, adx_max=28.0,
        donchian_n=20, swing_lookback=8,
        atr_period=14, adx_period=14,
        use_volume_filter=True, volume_mult=1.2,
        use_atr_expansion=True,
        rr=1.5, max_trades_per_day=2,
    )


# ── Variants ──────────────────────────────────────────────────────────────────

def variants_for(symbol: str) -> list[tuple[str, Config]]:
    base = base_cfg()
    is_btc = symbol == "BTCUSDT"
    v = [
        # ── Baseline ────────────────────────────────────────────────────────
        ("baseline",            base),

        # ── Technique 1: BTC SELL-only (confirmed losing BUY side) ──────────
        ("sell_only",           replace(base, allowed_side="SELL")),

        # ── Technique 2: Breakeven / Partial close (ICT / Mark Minervini) ───
        ("be_05r",              replace(base, breakeven_r=0.5)),
        ("partial_075r",        replace(base, partial_close_r=0.75, breakeven_r=0.5)),
        ("partial_1r_be",       replace(base, partial_close_r=1.0,  breakeven_r=1.0)),
        ("rr20_partial_1r",     replace(base, rr=2.0, partial_close_r=1.0, breakeven_r=1.0)),

        # ── Technique 3: Fresh breakout only (Jesse Livermore / ICT) ────────
        # "Only enter the FIRST breakout candle, not a continuation"
        ("fresh_break",         replace(base, require_fresh_breakout=True)),
        ("fresh_be_05r",        replace(base, require_fresh_breakout=True, breakeven_r=0.5)),

        # ── Technique 4: ADX acceleration (Mark Minervini / Stan Weinstein) ─
        # "Trend must be accelerating (ADX rising 2+ bars), not just above threshold"
        ("adx_accel2",          replace(base, adx_bars_rising=2)),
        ("adx_accel2_be",       replace(base, adx_bars_rising=2, breakeven_r=0.5)),

        # ── Technique 5: Volatility regime cap (Man AHL / Winton CTA) ───────
        # "Avoid chaotic high-volatility markets — trade only moderate regime"
        ("vol_regime",          replace(base, atr_percentile_max=80.0)),
        ("vol_regime_be",       replace(base, atr_percentile_max=80.0, breakeven_r=0.5)),

        # ── Technique 6: Equity curve filter (Campbell & Co. / Millburn) ────
        # "When your own equity is in drawdown, reduce risk — stop new entries"
        ("eq_filter",           replace(base, equity_curve_filter=True, equity_curve_ma=20)),
        ("eq_filter_be",        replace(base, equity_curve_filter=True, breakeven_r=0.5)),

        # ── Best combo (stack complementary techniques) ──────────────────────
        ("combo_best",          replace(base,
                                        require_fresh_breakout=True,
                                        adx_bars_rising=2,
                                        atr_percentile_max=80.0,
                                        breakeven_r=0.5)),
        ("combo_sell_fresh_be", replace(base,
                                        allowed_side="SELL",
                                        require_fresh_breakout=True,
                                        breakeven_r=0.5)),
    ]
    # BTC only: confirm BUY side is bad
    if is_btc:
        v.append(("buy_only", replace(base, allowed_side="BUY")))
    return v


# ── Run one split ─────────────────────────────────────────────────────────────

def run_one(symbol: str, m1_full: pd.DataFrame, m15_full: pd.DataFrame,
            cfg: Config, variant: str, label: str, start: str, end: str) -> tuple[dict, pd.DataFrame]:
    m15_w  = m15_full.loc[WARMUP_START:end]
    h1     = resample(m15_w, "1h")
    h4     = resample(m15_w, "4h")
    h1p    = prepare(h1, h4, cfg).loc[start:end]
    m1     = m1_full.loc[start:end]
    days   = max((h1p.index[-1].date() - h1p.index[0].date()).days + 1, 1)

    trades, equity = run_backtest(symbol, m1, h1p, cfg, variant, label)
    if not trades.empty:
        risk = (trades["entry"] - trades["sl"]).abs() * trades["qty"]
        trades["r_multiple"] = trades["pnl"] / risk.replace(0, np.nan)

    row = {"symbol": symbol, "variant": variant, "split": label, "days": days,
           **metrics(trades, equity, days)}
    if not trades.empty:
        row["neg_months"] = int(
            trades.assign(m=pd.to_datetime(trades["exit_time"]).dt.to_period("M").astype(str))
            .groupby("m")["pnl"].sum().lt(0).sum()
        )
        row["avg_r"] = round(float(trades["r_multiple"].mean()), 3)
    else:
        row["neg_months"] = 0
        row["avg_r"] = 0.0
    return row, trades


# ── Portfolio summary ─────────────────────────────────────────────────────────

def portfolio_summary(trades_df: pd.DataFrame, n_symbols: int) -> pd.DataFrame:
    rows = []
    for v, grp in trades_df.groupby("variant"):
        gp = grp.loc[grp["pnl"] > 0, "pnl"].sum()
        gl = -grp.loc[grp["pnl"] <= 0, "pnl"].sum()
        rows.append({
            "variant":       v,
            "symbols":       grp["symbol"].nunique(),
            "trades":        len(grp),
            "winrate_%":     round((grp["pnl"] > 0).mean() * 100, 1),
            "pf":            round(gp / gl, 3) if gl else 0.0,
            "portfolio_%":   round(grp["pnl"].sum() / (1000.0 * n_symbols) * 100, 2),
            "total_pnl_$":   round(grp["pnl"].sum(), 2),
        })
    return pd.DataFrame(rows).sort_values("pf", ascending=False)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # discover available symbols
    available = {sym: find_files(sym)
                 for sym in SYMBOL_PREFIX
                 if find_files(sym) is not None}

    print(f"Available symbols: {list(available)}")
    print(f"Period: {FULL[0]} → {FULL[1]}")
    print("=" * 70)

    full_rows, full_trades = [], []
    wf_rows,   wf_trades   = [], []

    for symbol, files in available.items():
        m1_full  = load_csv(files["m1"])
        m15_full = load_csv(files["m15"])
        var_list = variants_for(symbol)

        print(f"\n[{symbol}] full period …", flush=True)
        for name, cfg in var_list:
            print(f"  {name}", end=" ", flush=True)
            row, tr = run_one(symbol, m1_full, m15_full, cfg, name, "full", *FULL)
            full_rows.append(row)
            if not tr.empty:
                full_trades.append(tr)
        print()

        print(f"[{symbol}] walk-forward …")
        for fold, tr_s, tr_e, te_s, te_e in FOLDS:
            train_pf = {}
            for name, cfg in var_list:
                r, _ = run_one(symbol, m1_full, m15_full, cfg, name,
                               f"{fold}_train", tr_s, tr_e)
                train_pf[name] = r["profit_factor"] if r["total_trades"] >= 10 else 0.0

            best_name = max(train_pf, key=train_pf.get)
            best_cfg  = dict(var_list)[best_name]
            r, tr = run_one(symbol, m1_full, m15_full, best_cfg, best_name,
                            f"{fold}_OOS", te_s, te_e)
            r.update({"fold": fold, "selected": best_name,
                       "train_pf": round(train_pf[best_name], 4)})
            wf_rows.append(r)
            if not tr.empty:
                wf_trades.append(tr)
            print(f"  {fold}: best={best_name} trainPF={train_pf[best_name]:.3f} "
                  f"OOS PF={r['profit_factor']:.3f} trades={r['total_trades']}")

    # collate
    full_df   = pd.DataFrame(full_rows)
    wf_df     = pd.DataFrame(wf_rows)
    trades_df = pd.concat(full_trades, ignore_index=True) if full_trades else pd.DataFrame()
    trades_wf = pd.concat(wf_trades,  ignore_index=True) if wf_trades  else pd.DataFrame()

    write_csv(OUTPUT_DIR / "full_grid.csv",      full_df)
    write_csv(OUTPUT_DIR / "walkforward_oos.csv", wf_df)
    write_csv(OUTPUT_DIR / "trades_full.csv",     trades_df)
    write_csv(OUTPUT_DIR / "trades_wf.csv",       trades_wf)

    # per-symbol summary
    COLS = ["symbol", "variant", "total_trades", "winrate", "profit_factor",
            "profit_pct", "max_drawdown", "avg_r", "neg_months"]
    print("\n" + "=" * 70)
    print("PER-SYMBOL FULL PERIOD 2024–2025")
    print("=" * 70)
    display_df = full_df[full_df["split"] == "full"].sort_values(
        ["symbol", "profit_factor"], ascending=[True, False])
    for col in COLS:
        if col not in display_df.columns:
            display_df[col] = 0
    print(display_df[COLS].to_string(index=False))

    # portfolio
    if not trades_df.empty:
        n_sym   = len(available)
        port_df = portfolio_summary(trades_df, n_sym)
        write_csv(OUTPUT_DIR / "portfolio.csv", port_df)
        print("\n" + "=" * 70)
        print(f"PORTFOLIO ({n_sym} symbols × $1,000, 2024–2025)")
        print("=" * 70)
        print(port_df.to_string(index=False))

    # walk-forward summary
    if wf_rows:
        WF_COLS = ["symbol", "fold", "selected", "train_pf",
                   "total_trades", "profit_factor", "profit_pct", "max_drawdown"]
        print("\n" + "=" * 70)
        print("WALK-FORWARD OOS")
        print("=" * 70)
        for col in WF_COLS:
            if col not in wf_df.columns:
                wf_df[col] = ""
        print(wf_df[WF_COLS].to_string(index=False))

    print(f"\nOutputs → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
