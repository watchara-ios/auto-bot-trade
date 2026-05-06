"""
FXCM Forex Bot
================
แทน MetaTrader5 (Windows-only) ด้วย fxcmpy REST/WebSocket API
Railway-deployable (Linux compatible)

Strategy : Donchian Breakout (เหมือน forex_bot.py เดิม)
  Signal TF : M5  |  Trend TF : M15  |  Exec confirm : M1

Setup:
  pip install fxcmpy python-dotenv requests

  FXCM Access Token:
    Demo : https://tradingstation.fxcm.com → Tools → API Token
    Live : https://tradingstation.fxcm.com (account required)

Env vars:
  FXCM_ACCESS_TOKEN     — required
  FXCM_SERVER           — demo | real  (default: demo)
  FXCM_SYMBOLS          — comma-separated (default: EUR/USD,GBP/USD,USD/JPY,XAU/USD)
  FXCM_DRY_RUN          — true | false (default: true)
  FXCM_RISK_PCT         — e.g. 1.0  = 1% per trade (default: 1.0)
  FXCM_RR               — Risk:Reward (default: 2.5)
  FXCM_ADX_MIN          — (default: 18.0)
  FXCM_ADX_MAX          — (default: 50.0)
  FXCM_DONCHIAN_N       — (default: 20)
  FXCM_TRADE_START_HOUR — Thai time (default: 14)
  FXCM_TRADE_END_HOUR   — Thai time (default: 23)
  FXCM_MAX_TRADES_DAY   — (default: 3)
  FXCM_MAX_DAILY_LOSS   — % of balance (default: 3.0)
  TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import fxcmpy
except ImportError:
    print("ERROR: fxcmpy not installed — run: pip install fxcmpy")
    sys.exit(1)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bots"))
sys.path.insert(0, str(ROOT / "backtests"))

from notifier import (  # noqa: E402
    notify_bot_started, notify_error,
    notify_order_opened, notify_order_result,
    notify_reconnected,
)
from donchian_core import DonchianCoreConfig, latest_signal  # noqa: E402

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

class Config:
    ACCESS_TOKEN  = os.getenv("FXCM_ACCESS_TOKEN", "")
    SERVER        = os.getenv("FXCM_SERVER", "demo")       # "demo" | "real"
    DRY_RUN       = os.getenv("FXCM_DRY_RUN", "true").lower() == "true"

    SYMBOLS = [s.strip() for s in
               os.getenv("FXCM_SYMBOLS", "EUR/USD,GBP/USD,USD/JPY,XAU/USD").split(",")
               if s.strip()]

    # Strategy
    RISK_PCT      = float(os.getenv("FXCM_RISK_PCT", "1.0")) / 100
    RR            = float(os.getenv("FXCM_RR", "2.5"))
    ADX_MIN       = float(os.getenv("FXCM_ADX_MIN", "18.0"))
    ADX_MAX       = float(os.getenv("FXCM_ADX_MAX", "50.0"))
    DONCHIAN_N    = int(os.getenv("FXCM_DONCHIAN_N", "20"))
    ATR_PCT_MIN   = float(os.getenv("FXCM_ATR_PCT_MIN", "50.0"))
    MIN_ATR_PCT   = float(os.getenv("FXCM_MIN_ATR_PCT", "0.0002"))
    ALLOWED_SIDE  = os.getenv("FXCM_ALLOWED_SIDE", "BOTH")

    # Session (Thai UTC+7)
    TRADE_START_HOUR = int(os.getenv("FXCM_TRADE_START_HOUR", "14"))
    TRADE_END_HOUR   = int(os.getenv("FXCM_TRADE_END_HOUR", "23"))

    # Risk
    MAX_TRADES_DAY   = int(os.getenv("FXCM_MAX_TRADES_DAY", "3"))
    MAX_DAILY_LOSS   = float(os.getenv("FXCM_MAX_DAILY_LOSS", "3.0")) / 100

    # Timing
    POLL_SECONDS     = int(os.getenv("FXCM_POLL_SECONDS", "60"))
    RETRY_SLEEP      = int(os.getenv("FXCM_RETRY_SLEEP", "15"))
    DISCONNECT_NOTIFY_COOLDOWN = 300

    # Paths
    LOG_DIR     = ROOT / "logs"
    LOG_FILE    = LOG_DIR / "fxcm_bot.log"
    TRADE_LOG   = LOG_DIR / "fxcm_trades.csv"
    STATE_FILE  = LOG_DIR / "fxcm_state.json"


Config.LOG_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────

def _build_logger() -> logging.Logger:
    lg = logging.getLogger("fxcm_bot")
    lg.setLevel(logging.INFO)
    lg.propagate = False
    h = TimedRotatingFileHandler(
        Config.LOG_FILE, when="midnight", interval=1, backupCount=30, encoding="utf-8"
    )
    h.suffix = "%Y-%m-%d"
    h.setFormatter(logging.Formatter("%(asctime)s  %(message)s"))
    lg.addHandler(h)
    return lg


_logger = _build_logger()


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")
    _logger.info(msg)


def warn(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] WARNING {msg}")
    _logger.warning(msg)


# ─────────────────────────────────────────────
# FXCM connection
# ─────────────────────────────────────────────

_con: fxcmpy.fxcmpy | None = None


def connect() -> fxcmpy.fxcmpy:
    global _con
    if not Config.ACCESS_TOKEN:
        raise RuntimeError("FXCM_ACCESS_TOKEN not set")
    log(f"Connecting FXCM ({Config.SERVER})...")
    _con = fxcmpy.fxcmpy(
        access_token=Config.ACCESS_TOKEN,
        log_level="error",
        server=Config.SERVER,
    )
    log("FXCM connected")
    return _con


def con() -> fxcmpy.fxcmpy:
    global _con
    if _con is None or not _con.is_connected():
        _con = connect()
    return _con


def ensure_connected() -> bool:
    try:
        c = con()
        return c.is_connected()
    except Exception as exc:
        warn(f"Connection check failed: {exc}")
        return False


# ─────────────────────────────────────────────
# Market data
# ─────────────────────────────────────────────

# FXCM period map (signal engine uses m1/m5/m15 labels)
_PERIOD = {"m1": "m1", "m5": "m5", "m15": "m15"}


def get_candles(symbol: str, period: str, bars: int = 500) -> pd.DataFrame:
    """Fetch OHLCV from FXCM, returns DataFrame indexed by datetime."""
    raw = con().get_candles(symbol, period=period, number=bars)
    # fxcmpy returns bid/ask columns — use bid for strategy
    df = pd.DataFrame({
        "open":   raw["bidopen"],
        "high":   raw["bidhigh"],
        "low":    raw["bidlow"],
        "close":  raw["bidclose"],
        "volume": raw["tickqty"],
    }, index=raw.index)
    df.index.name = "time"
    return df.dropna()


def get_spread_pct(symbol: str) -> float:
    """Return current spread as % of mid price (0.0 if unavailable)."""
    try:
        prices = con().get_prices(symbol)
        bid = float(prices["Rates"].iloc[0])
        ask = float(prices["Rates"].iloc[1])
        mid = (bid + ask) / 2
        return (ask - bid) / mid if mid else 0.0
    except Exception:
        return 0.0


# ─────────────────────────────────────────────
# Account helpers
# ─────────────────────────────────────────────

def get_balance() -> float:
    try:
        acc = con().get_accounts().iloc[0]
        return float(acc["balance"])
    except Exception as exc:
        warn(f"get_balance failed: {exc}")
        return 0.0


def get_equity() -> float:
    try:
        acc = con().get_accounts().iloc[0]
        return float(acc["equity"])
    except Exception as exc:
        warn(f"get_equity failed: {exc}")
        return 0.0


def get_open_positions() -> pd.DataFrame:
    try:
        return con().get_open_positions()
    except Exception as exc:
        warn(f"get_open_positions failed: {exc}")
        return pd.DataFrame()


def has_open_position(symbol: str | None = None) -> bool:
    pos = get_open_positions()
    if pos.empty:
        return False
    if symbol:
        return any(pos.get("currency", pd.Series()).str.upper() == symbol.upper())
    return len(pos) > 0


# ─────────────────────────────────────────────
# Position sizing
# ─────────────────────────────────────────────

def calc_amount_k(symbol: str, entry: float, sl: float, balance: float) -> int:
    """
    Return trade amount in K units (1K = 1,000 base currency units).
    Uses risk_pct % of balance.

    For USD-quote pairs (EUR/USD, GBP/USD, XAU/USD):
      P&L per unit = price_move → units = risk_usd / sl_dist

    For non-USD-quote (USD/JPY, USD/CHF, USD/CAD):
      P&L per unit in USD = price_move / entry → units = risk_usd * entry / sl_dist
    """
    sl_dist = abs(entry - sl)
    if sl_dist <= 0:
        return 0

    risk_usd = balance * Config.RISK_PCT
    quote = symbol.split("/")[1] if "/" in symbol else symbol[-3:]

    if quote.upper() == "USD":
        units = risk_usd / sl_dist
    else:
        units = (risk_usd * entry) / sl_dist

    amount_k = max(1, int(units / 1000))
    return amount_k


# ─────────────────────────────────────────────
# Signal generation (reuse donchian_core)
# ─────────────────────────────────────────────

def _core_config() -> DonchianCoreConfig:
    session_utc = tuple(range(
        Config.TRADE_START_HOUR - 7,
        Config.TRADE_END_HOUR   - 7 + 1,
    ))
    return DonchianCoreConfig(
        allowed_side        = Config.ALLOWED_SIDE,
        rr                  = Config.RR,
        donchian_n          = Config.DONCHIAN_N,
        adx_min             = Config.ADX_MIN,
        adx_max             = Config.ADX_MAX,
        session_hours_utc   = session_utc,
        atr_percentile_min  = Config.ATR_PCT_MIN,
        min_atr_pct         = Config.MIN_ATR_PCT,
        tier_a_risk         = Config.RISK_PCT * 0.25,
        tier_b_risk         = Config.RISK_PCT,
    )


def generate_signal(symbol: str) -> dict | None:
    """Return signal dict if entry conditions met, else None."""
    m1  = get_candles(symbol, "m1",  500)
    m5  = get_candles(symbol, "m5",  500)
    m15 = get_candles(symbol, "m15", 500)

    sig, reason, candle_time = latest_signal(symbol, m1, m5, m15, _core_config())

    if sig is None:
        log(f"{symbol} no signal: {reason}")
        return None

    log(f"{symbol} SIGNAL {sig['side']} tier={sig.get('tier')} {reason}")
    return {**sig, "symbol": symbol, "candle_time": candle_time}


# ─────────────────────────────────────────────
# Order execution
# ─────────────────────────────────────────────

def place_order(sig: dict, balance: float) -> bool:
    symbol   = sig["symbol"]
    is_buy   = sig["side"] == "BUY"
    entry    = float(sig["entry"])
    sl       = float(sig["sl"])
    tp       = float(sig["tp"])
    amount_k = calc_amount_k(symbol, entry, sl, balance)

    if amount_k <= 0:
        warn(f"{symbol} amount_k=0 — skip")
        return False

    log(
        f"ORDER {symbol} {'BUY' if is_buy else 'SELL'} "
        f"amount={amount_k}K entry~{entry:.5f} sl={sl:.5f} tp={tp:.5f} "
        f"risk={Config.RISK_PCT*100:.1f}%"
    )
    notify_order_opened(
        "FXCM", symbol, sig["side"], f"{amount_k}K",
        entry, sl, tp,
        tier=sig.get("tier", ""),
        risk_pct=Config.RISK_PCT,
        dry_run=Config.DRY_RUN,
    )

    if Config.DRY_RUN:
        log("DRY_RUN: order not sent")
        _append_trade(sig, amount_k, balance, dry_run=True)
        return True

    try:
        order = con().open_trade(
            symbol       = symbol,
            is_buy       = is_buy,
            rate         = 0,               # AtMarket — fill at current price
            is_in_pips   = False,
            amount       = amount_k,
            time_in_force= "GTC",
            order_type   = "AtMarket",
            limit        = tp,
            stop         = sl,
        )
        notify_order_result("FXCM", symbol, sig["side"], order, dry_run=False)
        _append_trade(sig, amount_k, balance, dry_run=False)
        log(f"{symbol} order placed: {order}")
        return True
    except Exception as exc:
        warn(f"{symbol} order failed: {exc}")
        notify_error("FXCM", f"{symbol} order error: {exc}")
        return False


def _append_trade(sig: dict, amount_k: int, balance: float, dry_run: bool) -> None:
    import csv
    row = {
        "time":       datetime.now().isoformat(timespec="seconds"),
        "symbol":     sig["symbol"],
        "side":       sig["side"],
        "tier":       sig.get("tier", ""),
        "amount_k":   amount_k,
        "entry_ref":  sig["entry"],
        "sl":         sig["sl"],
        "tp":         sig["tp"],
        "rr":         Config.RR,
        "risk_pct":   Config.RISK_PCT,
        "balance":    round(balance, 2),
        "dry_run":    dry_run,
    }
    path = Config.TRADE_LOG
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


# ─────────────────────────────────────────────
# State management
# ─────────────────────────────────────────────

def load_state() -> dict:
    p = Config.STATE_FILE
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return _fresh_state()


def _fresh_state() -> dict:
    s = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "day_start_balance": None,
        "trades_today": 0,
        "last_candle": {},          # symbol → candle_time str (dedup)
        "connection_down": False,
        "last_disconnect_notify": 0,
    }
    save_state(s)
    return s


def save_state(s: dict) -> None:
    Config.STATE_FILE.write_text(json.dumps(s, indent=2, default=str), encoding="utf-8")


def reset_day(s: dict, balance: float) -> None:
    today = datetime.now().strftime("%Y-%m-%d")
    if s.get("date") != today:
        s.update({"date": today, "day_start_balance": balance, "trades_today": 0, "last_candle": {}})
        save_state(s)
    elif s.get("day_start_balance") is None:
        s["day_start_balance"] = balance
        save_state(s)


# ─────────────────────────────────────────────
# Session / risk gates
# ─────────────────────────────────────────────

def in_session() -> bool:
    hour = datetime.now().hour  # local Thai time
    return Config.TRADE_START_HOUR <= hour <= Config.TRADE_END_HOUR


def can_open(s: dict, balance: float) -> tuple[bool, str]:
    day_start = float(s.get("day_start_balance") or balance)
    daily_loss = (day_start - balance) / max(day_start, 1)
    if daily_loss >= Config.MAX_DAILY_LOSS:
        return False, f"daily loss {daily_loss:.2%} >= {Config.MAX_DAILY_LOSS:.2%}"
    if s.get("trades_today", 0) >= Config.MAX_TRADES_DAY:
        return False, f"max {Config.MAX_TRADES_DAY} trades/day reached"
    return True, "ok"


# ─────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────

def main() -> None:
    log("=" * 60)
    mode = "LIVE" if not Config.DRY_RUN else "DRY RUN"
    log(f"FXCM Donchian Bot — {mode}")
    log(f"Symbols : {Config.SYMBOLS}")
    log(f"Server  : {Config.SERVER}")
    log(f"Session : Thai {Config.TRADE_START_HOUR}:00–{Config.TRADE_END_HOUR}:59")
    log(f"Risk    : {Config.RISK_PCT*100:.1f}%  RR {Config.RR}  ADX {Config.ADX_MIN}-{Config.ADX_MAX}")
    log("=" * 60)

    if not Config.ACCESS_TOKEN:
        raise RuntimeError("Set FXCM_ACCESS_TOKEN in .env")

    connect()
    state = load_state()

    notify_bot_started(
        "FXCM Donchian Bot", mode,
        f"Symbols: <code>{', '.join(Config.SYMBOLS)}</code>\n"
        f"Session Thai {Config.TRADE_START_HOUR}–{Config.TRADE_END_HOUR}  "
        f"RR {Config.RR}  risk {Config.RISK_PCT*100:.1f}%",
    )

    last_disconnect_notify = 0.0

    while True:
        try:
            # ── Reconnect if needed ───────────────────────────────────────
            if not ensure_connected():
                now = time.time()
                if now - last_disconnect_notify >= Config.DISCONNECT_NOTIFY_COOLDOWN:
                    notify_error("FXCM", "Connection lost — retrying")
                    last_disconnect_notify = now
                state["connection_down"] = True
                time.sleep(Config.RETRY_SLEEP)
                continue

            if state.get("connection_down"):
                notify_reconnected("FXCM", "Connection restored")
                state["connection_down"] = False

            balance = get_balance()
            reset_day(state, balance)

            day_start = float(state.get("day_start_balance") or balance)
            daily_pnl = balance - day_start
            log(
                f"balance={balance:.2f}  daily={daily_pnl:+.2f}  "
                f"trades={state.get('trades_today',0)}/{Config.MAX_TRADES_DAY}  "
                f"session={'✅' if in_session() else '🌙'}"
            )

            # ── Session gate ──────────────────────────────────────────────
            if not in_session():
                log("Outside session — sleeping")
                time.sleep(Config.POLL_SECONDS)
                continue

            # ── Risk gate ─────────────────────────────────────────────────
            ok, reason = can_open(state, balance)
            if not ok:
                log(f"Risk gate: {reason}")
                time.sleep(Config.POLL_SECONDS)
                continue

            # ── Skip if any position open ─────────────────────────────────
            if has_open_position():
                log("Position open — skip new entries")
                time.sleep(Config.POLL_SECONDS)
                continue

            # ── Scan symbols ──────────────────────────────────────────────
            for symbol in Config.SYMBOLS:
                log(f"--- {symbol} ---")
                try:
                    sig = generate_signal(symbol)
                    if sig is None:
                        continue

                    # Dedup — don't re-enter on same candle
                    ct = str(sig.get("candle_time", ""))
                    if state.get("last_candle", {}).get(symbol) == ct:
                        log(f"{symbol} already acted on candle {ct}")
                        continue

                    placed = place_order(sig, balance)
                    state.setdefault("last_candle", {})[symbol] = ct
                    if placed and not Config.DRY_RUN:
                        state["trades_today"] = state.get("trades_today", 0) + 1
                    save_state(state)

                    if placed:
                        break   # one trade per round

                except Exception as exc:
                    warn(f"{symbol} error: {exc}")
                    notify_error("FXCM", f"{symbol} error: {exc}")

        except KeyboardInterrupt:
            log("Stopped by user")
            break
        except Exception as exc:
            warn(f"Loop error: {exc}")
            time.sleep(Config.RETRY_SLEEP)
            continue

        time.sleep(Config.POLL_SECONDS)

    try:
        if _con and _con.is_connected():
            _con.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
