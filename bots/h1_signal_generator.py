"""
H1 Donchian Signal Generator
==============================
Monitor BTCUSDT + SOLUSDT on H1 timeframe using the proven `rr15` config.
Sends Telegram alert when a Donchian breakout signal fires.

Config: H1 Donchian `rr15`
  - Signal TF : H1  (resampled from M15)
  - Trend TF  : H4  (resampled from M15)
  - adx_min=20, adx_max=28, rr=1.5
  - volume_filter=True (1.2x), atr_expansion=True
  - No regime filter (d1/w1 hurt H1 — see FINAL_REPORT.md)

Usage:
  python3 h1_signal_generator.py              # run once (useful for cron/testing)
  python3 h1_signal_generator.py --loop       # run continuously, polls every 5 min
                                              # (signals only on new H1 bar close)

Env vars (.env):
  TELEGRAM_BOT_TOKEN   — Telegram bot token
  TELEGRAM_CHAT_ID     — chat / channel ID
  BINANCE_BASE_URL     — optional, default https://fapi.binance.com
  SIGNAL_BALANCE       — account size for position sizing, default 1000
"""

import argparse
import csv
import os
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backtests"))

from realistic_donchian_backtest import Config, prepare, signal, trade_levels  # noqa: E402

load_dotenv(ROOT / ".env")

# ── Config ────────────────────────────────────────────────────────────────────

SYMBOLS = os.getenv("SIGNAL_SYMBOLS", "SOLUSDT").split(",")
BINANCE_BASE = os.getenv("BINANCE_BASE_URL", "https://fapi.binance.com")
BALANCE = float(os.getenv("SIGNAL_BALANCE", "300"))
RISK_PCT = 0.01        # 1% per trade = $3 per trade on $300
M15_LIMIT = 1200       # 1 200 × 15 min = 300 h of M15 (ample for H1+H4 warmup)
LOG_PATH = ROOT / "logs" / "h1_signals.csv"

STRATEGY_CFG = Config(
    adx_min=20.0,
    adx_max=28.0,
    donchian_n=20,
    swing_lookback=8,
    atr_period=14,
    adx_period=14,
    use_volume_filter=True,
    volume_mult=1.2,
    use_atr_expansion=True,
    rr=1.5,
    max_trades_per_day=2,
)

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
LOG_FIELDS = [
    "detected_at", "symbol", "side", "tier",
    "signal_bar_close", "entry_est", "sl", "tp", "rr",
    "risk_usd", "qty_est",
    "adx", "atr", "volume_ratio",
]

# ── Binance fetch ─────────────────────────────────────────────────────────────

def fetch_m15(symbol: str, limit: int = M15_LIMIT) -> pd.DataFrame:
    now_ms = int(time.time() * 1000)
    resp = requests.get(
        f"{BINANCE_BASE}/fapi/v1/klines",
        params={"symbol": symbol, "interval": "15m", "limit": limit},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "qav", "ntrades", "tbbase", "tbquote", "ignore",
    ])
    df = df[df["close_time"].astype(np.int64) < now_ms]   # drop current open bar
    df["time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_localize(None)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["time", "open", "high", "low", "close", "volume"]].dropna().set_index("time")


# ── Resample ──────────────────────────────────────────────────────────────────

def _resample(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    return (
        df.resample(freq)
        .agg({"open": "first", "high": "max", "low": "min",
              "close": "last", "volume": "sum"})
        .dropna(subset=["open", "close"])
    )


# ── Signal detection ──────────────────────────────────────────────────────────

def detect(symbol: str, m15_raw: pd.DataFrame, cfg: Config) -> dict | None:
    """Return signal dict for the last completed H1 bar, or None."""
    h1 = _resample(m15_raw, "1h")
    h4 = _resample(m15_raw, "4h")

    if len(h1) < cfg.donchian_n + cfg.swing_lookback + 10:
        return None

    h1p = prepare(h1, h4, cfg)
    last = h1p.iloc[-1]            # most recent completed H1 bar
    sig = signal(last, cfg)
    if sig is None:
        return None

    entry_est = float(last["close"])
    levels = trade_levels(last, entry_est, sig["side"], cfg)
    if levels is None:
        return None
    sl, tp, risk_dist = levels

    risk_usd = BALANCE * RISK_PCT
    qty_est = round(risk_usd / risk_dist, 4) if risk_dist > 0 else 0.0

    return {
        "detected_at":    datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "symbol":         symbol,
        "side":           sig["side"],
        "tier":           sig["tier"],
        "signal_bar_close": str(h1p.index[-1]),
        "entry_est":      round(entry_est, 4),
        "sl":             round(sl, 4),
        "tp":             round(tp, 4),
        "rr":             cfg.rr,
        "risk_usd":       round(risk_usd, 2),
        "qty_est":        qty_est,
        "adx":            round(float(last.get("m15_adx", 0)), 2),
        "atr":            round(float(last.get("atr", 0)), 4),
        "volume_ratio":   round(float(last.get("volume_ratio", 0)), 3),
    }


# ── Telegram ──────────────────────────────────────────────────────────────────

def send_telegram(msg: str) -> bool:
    token   = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        print("[telegram] TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set — skipping")
        return False
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat_id, "text": msg,
                  "parse_mode": "HTML", "disable_web_page_preview": "true"},
            timeout=10,
        )
        ok = resp.status_code == 200 and resp.json().get("ok", False)
        if not ok:
            print(f"[telegram] error: {resp.text[:200]}")
        return ok
    except Exception as e:
        print(f"[telegram] exception: {e}")
        return False


def format_alert(s: dict) -> str:
    side_emoji = "🟢 BUY" if s["side"] == "BUY" else "🔴 SELL"
    tier_label = f"Tier {s['tier']}"
    return (
        f"<b>H1 Donchian Signal — {s['symbol']}</b>\n"
        f"{side_emoji}  |  {tier_label}\n\n"
        f"Entry (est)  : <code>{s['entry_est']}</code>\n"
        f"Stop Loss    : <code>{s['sl']}</code>\n"
        f"Take Profit  : <code>{s['tp']}</code>  (RR {s['rr']})\n\n"
        f"Risk         : ${s['risk_usd']}  ({RISK_PCT*100:.0f}% of ${BALANCE:,.0f})\n"
        f"Qty (est)    : {s['qty_est']}\n\n"
        f"ADX          : {s['adx']}\n"
        f"ATR          : {s['atr']}\n"
        f"Vol ratio    : {s['volume_ratio']}x\n"
        f"Bar close    : {s['signal_bar_close']} UTC"
    )


# ── Logging ───────────────────────────────────────────────────────────────────

def log_signal(s: dict):
    write_header = not LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow({k: s.get(k, "") for k in LOG_FIELDS})


# ── State — avoid duplicate alerts on the same H1 bar ────────────────────────

_last_alerted: dict[str, str] = {}   # symbol → signal_bar_close string


def already_alerted(s: dict) -> bool:
    return _last_alerted.get(s["symbol"]) == s["signal_bar_close"]


def mark_alerted(s: dict):
    _last_alerted[s["symbol"]] = s["signal_bar_close"]


# ── One scan ──────────────────────────────────────────────────────────────────

def scan_all(cfg: Config = STRATEGY_CFG, verbose: bool = True) -> list[dict]:
    found = []
    for symbol in SYMBOLS:
        try:
            m15 = fetch_m15(symbol)
            sig = detect(symbol, m15, cfg)
            if sig is None:
                if verbose:
                    print(f"[{datetime.utcnow():%H:%M}] {symbol}: no signal")
                continue
            if already_alerted(sig):
                if verbose:
                    print(f"[{datetime.utcnow():%H:%M}] {symbol}: "
                          f"signal on {sig['signal_bar_close']} already sent")
                continue
            print(f"[{datetime.utcnow():%H:%M}] {symbol}: SIGNAL {sig['side']} Tier-{sig['tier']} "
                  f"entry={sig['entry_est']} sl={sig['sl']} tp={sig['tp']}")
            log_signal(sig)
            send_telegram(format_alert(sig))
            mark_alerted(sig)
            found.append(sig)
        except Exception as e:
            print(f"[{datetime.utcnow():%H:%M}] {symbol}: ERROR — {e}")
    return found


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="H1 Donchian signal generator")
    parser.add_argument("--loop", action="store_true",
                        help="Run continuously, polling every 5 minutes")
    parser.add_argument("--interval", type=int, default=300,
                        help="Poll interval in seconds (default 300 = 5 min)")
    parser.add_argument("--test-telegram", action="store_true",
                        help="Send a test Telegram message and exit")
    args = parser.parse_args()

    print(f"H1 Donchian Signal Generator  |  symbols: {SYMBOLS}")
    print(f"Config: adx {STRATEGY_CFG.adx_min}-{STRATEGY_CFG.adx_max}, "
          f"rr={STRATEGY_CFG.rr}, n={STRATEGY_CFG.donchian_n}, "
          f"vol={STRATEGY_CFG.volume_mult}x, atr_exp={STRATEGY_CFG.use_atr_expansion}")
    print(f"Balance: ${BALANCE:,.0f}  risk: {RISK_PCT*100:.0f}%  logs: {LOG_PATH}")
    print("-" * 60)

    if args.test_telegram:
        msg = (
            "<b>H1 Donchian — Test Message ✅</b>\n\n"
            f"Bot เชื่อมต่อสำเร็จ\n"
            f"Symbols: {SYMBOLS}\n"
            f"Balance: ${BALANCE:,.0f}  Risk/trade: ${BALANCE*RISK_PCT:.0f}\n"
            f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC"
        )
        ok = send_telegram(msg)
        print("Telegram OK" if ok else "Telegram FAILED — check token/chat_id")
        return

    if not args.loop:
        scan_all()
        return

    print(f"Loop mode: polling every {args.interval}s  (Ctrl+C to stop)\n")
    while True:
        scan_all()
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
