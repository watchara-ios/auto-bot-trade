"""
H1 Donchian Live Bot
======================
Strategy : H1 Donchian breakout `rr15` (proven config — see FINAL_REPORT.md)
  Signal TF  : H1  (resampled from M15 klines)
  Trend TF   : H4  (resampled from M15 klines)
  adx 20-28, volume 1.2x, atr_expansion, rr=1.5, risk=1%

Primary symbol : SOLUSDT  (PF=1.661, WR=52.8%, DD=-2.4%)
Optional       : BTCUSDT  (set HYBRID_SYMBOLS=SOLUSDT,BTCUSDT)

Key differences from bitcoin_bot_hybrid.py:
  - Only M15 klines needed (resample to H1 + H4 internally)
  - Signal engine: realistic_donchian_backtest.prepare/signal/trade_levels
  - No session filter (H1 signals valid all hours)
  - No D1/W1 regime filter (hurts H1 — see report)
  - Poll 300 s (5 min) — H1 bars close hourly, no need for faster poll

Env vars:
  BINANCE_API_KEY / BINANCE_SECRET   — Binance Futures credentials
  BINANCE_BASE_URL                   — default https://fapi.binance.com
  HYBRID_SYMBOLS                     — default SOLUSDT
  HYBRID_DRY_RUN                     — true / false (default false)
  HYBRID_POLL_SECONDS                — default 300
  SIGNAL_BALANCE                     — override balance for sizing (optional)
  TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID
"""

import csv
import hashlib
import hmac
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backtests"))

from realistic_donchian_backtest import (  # noqa: E402
    Config as StrategyConfig,
    prepare as strategy_prepare,
    signal as strategy_signal,
    trade_levels,
)
from notifier import (  # noqa: E402
    notify_bot_started,
    notify_error,
    notify_order_opened,
    notify_order_result,
    notify_reconnected,
)

load_dotenv(ROOT / ".env")


# ─────────────────────────────────────────────
# Bot Config
# ─────────────────────────────────────────────

class Config:
    # Credentials
    API_KEY  = os.getenv("BINANCE_API_KEY") or os.getenv("BINANCE2_API_KEY")
    SECRET   = os.getenv("BINANCE_SECRET")  or os.getenv("BINANCE2_SECRET")
    BASE_URL = os.getenv("BINANCE_BASE_URL", "https://fapi.binance.com").rstrip("/")

    # Symbols
    SYMBOLS = [s.strip().upper() for s in os.getenv("HYBRID_SYMBOLS", "SOLUSDT").split(",") if s.strip()]

    # Mode
    DRY_RUN     = os.getenv("HYBRID_DRY_RUN", "false").lower() == "true"
    LEVERAGE    = 1
    MARGIN_TYPE = "ISOLATED"

    # Timing
    POLL_SECONDS                      = int(os.getenv("HYBRID_POLL_SECONDS", "300"))
    CONNECTION_RETRY_SLEEP_SECONDS    = int(os.getenv("HYBRID_CONNECTION_RETRY_SLEEP_SECONDS", "20"))
    DISCONNECT_NOTIFY_COOLDOWN_SECONDS = int(os.getenv("HYBRID_DISCONNECT_NOTIFY_COOLDOWN_SECONDS", "300"))
    REQUEST_RETRIES                   = int(os.getenv("HYBRID_REQUEST_RETRIES", "3"))

    # Risk controls
    RISK_PCT             = 0.01    # 1% per trade
    MAX_TRADES_PER_DAY   = 2
    MAX_DAILY_LOSS_PCT   = 0.04    # stop new entries if down 4% on the day
    DAILY_PROFIT_STOP_PCT = 0.10   # take the day off if up 10%
    ENTRY_COOLDOWN        = 3600   # min seconds between entries per symbol (1 H1 bar)
    MAX_OPEN_SYMBOLS      = 1
    TRIGGER_GUARD_PCT     = 0.0005
    # Pro trade management (Minervini + ICT)
    BREAKEVEN_R          = 0.5    # move SL to entry when price reaches entry + 0.5×risk

    # M15 klines to fetch (resampled → H1+H4 inside)
    KLINE_LIMIT = 1200

    # Paths
    LOG_DIR    = ROOT / "logs"
    LOG_FILE   = LOG_DIR / "h1_donchian_bot.log"
    TRADE_LOG  = LOG_DIR / "h1_donchian_trades.csv"
    STATE_FILE = LOG_DIR / "h1_donchian_state.json"


Config.LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── Strategy config: accel2_be05_rr20 (best fold2 OOS avg PF=2.358) ─────────
STRATEGY_CFG = StrategyConfig(
    adx_min=20.0,
    adx_max=28.0,
    donchian_n=20,
    swing_lookback=8,
    atr_period=14,
    adx_period=14,
    use_volume_filter=True,
    volume_mult=1.2,
    use_atr_expansion=True,
    rr=2.0,               # upgraded from 1.5 — backtest avg fold2 PF=2.378
    adx_bars_rising=2,    # Minervini: ADX must accelerate ≥2 consecutive H4 bars
    max_trades_per_day=Config.MAX_TRADES_PER_DAY,
)


# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────

def _build_logger() -> logging.Logger:
    lg = logging.getLogger("h1_donchian_bot")
    lg.setLevel(logging.INFO)
    lg.propagate = False
    h = TimedRotatingFileHandler(
        Config.LOG_FILE, when="midnight", interval=1, backupCount=30, encoding="utf-8"
    )
    h.suffix = "%Y-%m-%d"
    h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
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
# Binance API
# ─────────────────────────────────────────────

class Binance:
    time_offset: int = 0
    _exchange_info_cache: dict = {}

    @classmethod
    def sync_time(cls) -> None:
        try:
            data = requests.get(f"{Config.BASE_URL}/fapi/v1/time", timeout=10).json()
            cls.time_offset = int(data["serverTime"]) - int(time.time() * 1000)
            log(f"Time synced offset={cls.time_offset}ms")
        except Exception as exc:
            warn(f"Time sync failed: {exc}")

    @staticmethod
    def _headers() -> dict:
        return {"X-MBX-APIKEY": Config.API_KEY}

    @staticmethod
    def _sign(params: dict) -> str:
        query = urlencode(params)
        sig = hmac.new(Config.SECRET.encode(), query.encode(), hashlib.sha256).hexdigest()
        return f"{query}&signature={sig}"

    @staticmethod
    def public_get(path: str, params: dict | None = None) -> dict:
        last_exc = None
        for attempt in range(Config.REQUEST_RETRIES):
            try:
                r = requests.get(f"{Config.BASE_URL}{path}", params=params or {}, timeout=15)
                r.raise_for_status()
                return r.json()
            except Exception as exc:
                last_exc = exc
                if attempt < Config.REQUEST_RETRIES - 1:
                    time.sleep(min(2 ** attempt, 5))
        raise RuntimeError(f"Public GET {path} failed: {last_exc}")

    @staticmethod
    def signed(method: str, path: str, params: dict | None = None) -> dict:
        base = dict(params or {})
        last_exc = None
        for attempt in range(Config.REQUEST_RETRIES):
            try:
                p = {**base,
                     "timestamp": int(time.time() * 1000) + Binance.time_offset,
                     "recvWindow": 10000}
                url = f"{Config.BASE_URL}{path}?{Binance._sign(p)}"
                fn = {"GET": requests.get, "POST": requests.post, "DELETE": requests.delete}[method]
                r = fn(url, headers=Binance._headers(), timeout=15)
                data = r.json()
                if r.status_code >= 400:
                    raise RuntimeError(f"Binance {r.status_code}: {data}")
                return data
            except Exception as exc:
                last_exc = exc
                if "-1003" in str(exc) or "429" in str(exc):
                    # rate limited — stop retrying immediately, let main loop back off
                    break
                if "-1021" in str(exc):
                    Binance.sync_time()
                if attempt < Config.REQUEST_RETRIES - 1:
                    time.sleep(min(2 ** attempt, 5))
        raise RuntimeError(f"Signed {method} {path} failed: {last_exc}")

    @staticmethod
    def get_klines(symbol: str, interval: str, limit: int) -> pd.DataFrame:
        now_ms = int(time.time() * 1000)
        data = Binance.public_get("/fapi/v1/klines",
                                  {"symbol": symbol, "interval": interval, "limit": limit})
        df = pd.DataFrame(data, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "qav", "ntrades", "tbbase", "tbquote", "ignore",
        ])
        df = df[df["close_time"].astype(np.int64) < now_ms]
        df["time"] = pd.to_datetime(df["open_time"], unit="ms")
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df[["time", "open", "high", "low", "close", "volume"]].set_index("time").dropna()

    @staticmethod
    def exchange_info(symbol: str) -> dict:
        if symbol not in Binance._exchange_info_cache:
            data = Binance.public_get("/fapi/v1/exchangeInfo")
            for item in data["symbols"]:
                if item["symbol"] == symbol:
                    Binance._exchange_info_cache[symbol] = item
                    break
            else:
                raise RuntimeError(f"Symbol not found: {symbol}")
        return Binance._exchange_info_cache[symbol]

    @staticmethod
    def balance() -> float:
        override = os.getenv("SIGNAL_BALANCE")
        if override:
            return float(override)
        for item in Binance.signed("GET", "/fapi/v2/balance"):
            if item.get("asset") == "USDT":
                return float(item.get("balance", 0))
        return 0.0

    @staticmethod
    def positions() -> list:
        return Binance.signed("GET", "/fapi/v2/positionRisk")

    @staticmethod
    def position(symbol: str) -> dict:
        for p in Binance.positions():
            if p.get("symbol") == symbol:
                return {
                    "amount": float(p.get("positionAmt", 0)),
                    "entry":  float(p.get("entryPrice", 0)),
                    "mark":   float(p.get("markPrice", 0)),
                    "raw": p,
                }
        return {"amount": 0.0, "entry": 0.0, "mark": 0.0, "raw": None}

    @staticmethod
    def open_orders(symbol: str) -> list:
        return Binance.signed("GET", "/fapi/v1/openOrders", {"symbol": symbol})

    @staticmethod
    def open_algo_orders(symbol: str) -> list:
        try:
            return Binance.signed("GET", "/fapi/v1/openAlgoOrders", {"symbol": symbol})
        except Exception as exc:
            warn(f"{symbol} openAlgoOrders fallback: {exc}")
            return Binance.signed("GET", "/fapi/v1/openOrders",
                                  {"symbol": symbol, "conditional": "true"})

    @staticmethod
    def cancel_all_orders(symbol: str, conditional: bool = False) -> dict:
        params = {"symbol": symbol}
        if conditional:
            params["conditional"] = "true"
        return Binance.signed("DELETE", "/fapi/v1/allOpenOrders", params)

    @staticmethod
    def market_order(symbol: str, side: str, qty: float, reduce_only: bool = False) -> dict:
        params = {"symbol": symbol, "side": side, "type": "MARKET", "quantity": qty}
        if reduce_only:
            params["reduceOnly"] = "true"
        if Config.DRY_RUN:
            log(f"DRY_RUN market_order {params}")
            return {"dry_run": True, **params}
        return Binance.signed("POST", "/fapi/v1/order", params)

    @staticmethod
    def algo_order(symbol: str, side: str, order_type: str,
                   trigger_price: float, qty: float, position_side: str,
                   auto_close: bool = False, current_price: float = 0.0) -> dict:
        current = current_price or trigger_price
        tp = round(trigger_price, 4)

        if position_side == "BUY":
            if order_type == "STOP_MARKET" and tp >= current:
                if auto_close:
                    return Binance.market_order(symbol, side, qty, reduce_only=True)
                tp = round(current * (1 - Config.TRIGGER_GUARD_PCT), 4)
            elif order_type == "TAKE_PROFIT_MARKET" and tp <= current:
                if auto_close:
                    return Binance.market_order(symbol, side, qty, reduce_only=True)
                tp = round(current * (1 + Config.TRIGGER_GUARD_PCT), 4)
        else:
            if order_type == "STOP_MARKET" and tp <= current:
                if auto_close:
                    return Binance.market_order(symbol, side, qty, reduce_only=True)
                tp = round(current * (1 + Config.TRIGGER_GUARD_PCT), 4)
            elif order_type == "TAKE_PROFIT_MARKET" and tp >= current:
                if auto_close:
                    return Binance.market_order(symbol, side, qty, reduce_only=True)
                tp = round(current * (1 - Config.TRIGGER_GUARD_PCT), 4)

        params = {
            "algoType": "CONDITIONAL",
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "triggerPrice": tp,
            "quantity": qty,
            "reduceOnly": "true",
            "workingType": "CONTRACT_PRICE",
        }
        if Config.DRY_RUN:
            log(f"DRY_RUN algo {order_type} {params}")
            return {"dry_run": True, **params}
        return Binance.signed("POST", "/fapi/v1/algoOrder", params)

    @staticmethod
    def set_margin(symbol: str) -> None:
        try:
            Binance.signed("POST", "/fapi/v1/marginType",
                           {"symbol": symbol, "marginType": Config.MARGIN_TYPE})
            log(f"{symbol} margin={Config.MARGIN_TYPE}")
        except Exception as exc:
            if "-4046" in str(exc):
                log(f"{symbol} already {Config.MARGIN_TYPE}")
            else:
                warn(f"{symbol} set margin: {exc}")

    @staticmethod
    def set_leverage(symbol: str) -> None:
        try:
            Binance.signed("POST", "/fapi/v1/leverage",
                           {"symbol": symbol, "leverage": Config.LEVERAGE})
            log(f"{symbol} leverage={Config.LEVERAGE}x")
        except Exception as exc:
            warn(f"{symbol} set leverage: {exc}")


# ─────────────────────────────────────────────
# Market data
# ─────────────────────────────────────────────

def fetch_m15(symbol: str) -> pd.DataFrame:
    return Binance.get_klines(symbol, "15m", Config.KLINE_LIMIT)


def resample_h1(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.resample("1h")
        .agg({"open": "first", "high": "max", "low": "min",
              "close": "last", "volume": "sum"})
        .dropna(subset=["open", "close"])
    )


def resample_h4(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.resample("4h")
        .agg({"open": "first", "high": "max", "low": "min",
              "close": "last", "volume": "sum"})
        .dropna(subset=["open", "close"])
    )


# ─────────────────────────────────────────────
# Signal generation
# ─────────────────────────────────────────────

def generate_signal(symbol: str, m15_raw: pd.DataFrame) -> dict | None:
    """Detect signal on last closed H1 bar. Returns signal dict or None."""
    h1 = resample_h1(m15_raw)
    h4 = resample_h4(m15_raw)

    min_bars = STRATEGY_CFG.donchian_n + STRATEGY_CFG.swing_lookback + 20
    if len(h1) < min_bars:
        log(f"{symbol} not enough H1 bars ({len(h1)} < {min_bars})")
        return None

    h1p = strategy_prepare(h1, h4, STRATEGY_CFG)
    last = h1p.iloc[-1]
    sig  = strategy_signal(last, STRATEGY_CFG)
    if sig is None:
        return None

    entry = float(last["close"])
    levels = trade_levels(last, entry, sig["side"], STRATEGY_CFG)
    if levels is None:
        return None
    sl, tp, _ = levels

    exit_side = "SELL" if sig["side"] == "BUY" else "BUY"
    return {
        "symbol":           symbol,
        "side":             sig["side"],
        "exit_side":        exit_side,
        "tier":             sig["tier"],
        "entry":            entry,
        "sl":               round(sl, 4),
        "tp":               round(tp, 4),
        "signal_bar_close": str(h1p.index[-1]),
        "adx":              round(float(last.get("m15_adx", 0)), 2),
        "atr":              round(float(last.get("atr", 0)), 4),
        "volume_ratio":     round(float(last.get("volume_ratio", 0)), 3),
    }


# ─────────────────────────────────────────────
# Position sizing
# ─────────────────────────────────────────────

def round_qty(symbol: str, qty: float) -> float:
    step, min_qty = 0.001, 0.0
    for f in Binance.exchange_info(symbol)["filters"]:
        if f["filterType"] == "LOT_SIZE":
            step    = float(f["stepSize"])
            min_qty = float(f["minQty"])
            break
    precision = max(0, int(round(-np.log10(step))))
    rounded   = round(float(np.floor(qty / step) * step), precision)
    return rounded if rounded >= min_qty else 0.0


def quantity_for_signal(symbol: str, balance: float, sig: dict) -> float:
    risk_usd = balance * Config.RISK_PCT
    sl_dist  = abs(sig["entry"] - sig["sl"])
    if sl_dist <= 0:
        return 0.0
    return round_qty(symbol, risk_usd / sl_dist)


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
    state = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "day_start_balance": None,
        "trades_today": 0,
        "last_trade_time": {},
        "last_signal_bar": {},      # symbol → signal_bar_close str (dedup)
        "be_trigger": {},           # symbol → price level that triggers breakeven move
        "be_applied": {},           # symbol → bool, True once SL moved to entry
        "dry_run": Config.DRY_RUN,
        "connection_down": False,
        "last_disconnect_notify": 0,
    }
    save_state(state)
    return state


def save_state(state: dict) -> None:
    Config.STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def reset_day(state: dict, balance: float) -> None:
    today = datetime.now().strftime("%Y-%m-%d")
    mode_changed = state.get("dry_run") != Config.DRY_RUN
    day_changed  = state.get("date") != today
    if mode_changed or day_changed:
        state.update({
            "date": today,
            "day_start_balance": balance,
            "trades_today": 0,
            "last_trade_time": {},
            "dry_run": Config.DRY_RUN,
        })
        save_state(state)
        return
    if state.get("day_start_balance") is None:
        state["day_start_balance"] = balance
        save_state(state)


# ─────────────────────────────────────────────
# Risk checks
# ─────────────────────────────────────────────

def can_open_new(state: dict, balance: float, positions_by_symbol: dict) -> tuple[bool, str]:
    day_start = float(state.get("day_start_balance") or balance)
    daily_ret = (balance - day_start) / max(day_start, 1)

    if daily_ret <= -Config.MAX_DAILY_LOSS_PCT:
        return False, f"daily loss stop {daily_ret:.2%}"
    if daily_ret >= Config.DAILY_PROFIT_STOP_PCT:
        return False, f"daily profit stop {daily_ret:.2%}"
    if state.get("trades_today", 0) >= Config.MAX_TRADES_PER_DAY:
        return False, "max trades per day"

    open_syms = [s for s, p in positions_by_symbol.items()
                 if s in Config.SYMBOLS and abs(p.get("amount", 0)) > 0]
    if len(open_syms) >= Config.MAX_OPEN_SYMBOLS:
        return False, f"max open symbols {len(open_syms)}"

    return True, "ok"


# ─────────────────────────────────────────────
# Position helpers
# ─────────────────────────────────────────────

def get_positions_by_symbol() -> dict:
    return {
        p["symbol"]: {
            "amount": float(p.get("positionAmt", 0)),
            "entry":  float(p.get("entryPrice", 0)),
            "mark":   float(p.get("markPrice", 0)),
        }
        for p in Binance.positions()
        if p.get("symbol")
    }


def ensure_protection(symbol: str, pos: dict) -> None:
    if abs(pos["amount"]) <= 0 or Config.DRY_RUN:
        return
    try:
        orders = Binance.open_orders(symbol) + Binance.open_algo_orders(symbol)
        protected = any(
            str(o.get("reduceOnly")).lower() == "true"
            or str(o.get("closePosition")).lower() == "true"
            for o in orders
        )
        if protected:
            return
    except Exception as exc:
        warn(f"{symbol} cannot inspect orders: {exc}")
        return

    # No protection found — place emergency SL/TP based on 1×ATR
    entry    = pos["entry"] or pos["mark"]
    atr_est  = entry * 0.015   # rough 1.5% ATR estimate
    side     = "BUY" if pos["amount"] > 0 else "SELL"
    exit_side = "SELL" if side == "BUY" else "BUY"
    sl = entry - atr_est if side == "BUY" else entry + atr_est
    tp = entry + atr_est * STRATEGY_CFG.rr if side == "BUY" else entry - atr_est * STRATEGY_CFG.rr
    qty = round_qty(symbol, abs(pos["amount"]))
    warn(f"{symbol} unprotected — emergency SL={sl:.4f} TP={tp:.4f}")
    Binance.algo_order(symbol, exit_side, "STOP_MARKET",        sl, qty, side, auto_close=True, current_price=entry)
    Binance.algo_order(symbol, exit_side, "TAKE_PROFIT_MARKET", tp, qty, side, auto_close=True, current_price=entry)


def cleanup_orphan_orders(symbol: str) -> None:
    if Config.DRY_RUN:
        return
    try:
        regular = Binance.open_orders(symbol)
        algo    = Binance.open_algo_orders(symbol)
        if not regular and not algo:
            return
        warn(f"{symbol} no position but {len(regular)+len(algo)} orphan orders — canceling")
        if regular:
            Binance.cancel_all_orders(symbol)
        if algo:
            Binance.cancel_all_orders(symbol, conditional=True)
    except Exception as exc:
        warn(f"{symbol} orphan cleanup: {exc}")


# ─────────────────────────────────────────────
# Breakeven management  (ICT: "free ride" — move SL to entry at 0.5R)
# ─────────────────────────────────────────────

def move_sl_to_breakeven(symbol: str, pos: dict) -> None:
    """Cancel existing SL algo order and replace it at entry (breakeven)."""
    if Config.DRY_RUN:
        log(f"{symbol} DRY_RUN: would move SL to breakeven entry={pos['entry']:.4f}")
        return
    entry    = pos["entry"]
    amt      = abs(pos["amount"])
    side     = "BUY" if pos["amount"] > 0 else "SELL"
    exit_side = "SELL" if side == "BUY" else "BUY"
    qty      = round_qty(symbol, amt)
    try:
        Binance.cancel_all_orders(symbol, conditional=True)
        Binance.algo_order(symbol, exit_side, "STOP_MARKET",
                           entry, qty, side, current_price=pos["mark"])
        log(f"{symbol} ✅ breakeven: SL moved to entry={entry:.4f}")
    except Exception as exc:
        warn(f"{symbol} breakeven move failed: {exc}")


def check_breakeven(symbol: str, pos: dict, state: dict) -> None:
    """Called each cycle when a position is open — applies breakeven once."""
    if abs(pos["amount"]) <= 0:
        # position closed — clear breakeven state
        state.setdefault("be_trigger", {}).pop(symbol, None)
        state.setdefault("be_applied", {}).pop(symbol, None)
        return

    if state.get("be_applied", {}).get(symbol, False):
        return  # already applied this trade

    trigger = state.get("be_trigger", {}).get(symbol)
    if trigger is None:
        return  # no trigger recorded for this trade

    mark = pos["mark"]
    side = "BUY" if pos["amount"] > 0 else "SELL"
    hit  = (mark >= trigger) if side == "BUY" else (mark <= trigger)

    if hit:
        log(f"{symbol} price {mark:.4f} reached BE trigger {trigger:.4f} — moving SL")
        move_sl_to_breakeven(symbol, pos)
        state.setdefault("be_applied", {})[symbol] = True
        save_state(state)


# ─────────────────────────────────────────────
# Connection state
# ─────────────────────────────────────────────

def mark_connected(state: dict) -> None:
    if state.get("connection_down"):
        notify_reconnected("BINANCE", "H1 Donchian bot recovered")
        log("Binance connection recovered")
    state["connection_down"] = False
    state["last_disconnect_notify"] = 0


def mark_disconnected(state: dict, error: Exception) -> None:
    now = time.time()
    cooldown_ok = now - float(state.get("last_disconnect_notify") or 0) >= Config.DISCONNECT_NOTIFY_COOLDOWN_SECONDS
    if not state.get("connection_down") or cooldown_ok:
        notify_error("BINANCE", f"Disconnected: {error}")
        state["last_disconnect_notify"] = now
    state["connection_down"] = True
    save_state(state)


# ─────────────────────────────────────────────
# Trade execution
# ─────────────────────────────────────────────

def execute_signal(sig: dict, balance: float, state: dict, pos_amount: float = 0.0) -> None:
    symbol = sig["symbol"]

    # Dedup: don't re-enter on the same H1 bar
    if state.get("last_signal_bar", {}).get(symbol) == sig["signal_bar_close"]:
        log(f"{symbol} signal on bar {sig['signal_bar_close']} already acted on")
        return

    # Cooldown
    last_trade = float(state.get("last_trade_time", {}).get(symbol, 0))
    if time.time() - last_trade < Config.ENTRY_COOLDOWN:
        remaining = int(Config.ENTRY_COOLDOWN - (time.time() - last_trade))
        log(f"{symbol} cooldown {remaining}s remaining")
        return

    if abs(pos_amount) > 0:
        log(f"{symbol} position already open — skip")
        return

    qty = quantity_for_signal(symbol, balance, sig)
    if qty <= 0:
        log(f"{symbol} qty too small at balance={balance:.2f}")
        return

    risk_usd = balance * Config.RISK_PCT
    log(
        f"OPEN {symbol} {sig['side']} tier={sig['tier']} qty={qty} "
        f"entry~{sig['entry']:.4f} sl={sig['sl']} tp={sig['tp']} "
        f"risk=${risk_usd:.2f} ({Config.RISK_PCT*100:.0f}%)"
    )
    notify_order_opened(
        "BINANCE", symbol, sig["side"], qty,
        sig["entry"], sig["sl"], sig["tp"],
        tier=sig["tier"],
        risk_pct=Config.RISK_PCT,
        dry_run=Config.DRY_RUN,
    )

    order_result = Binance.market_order(symbol, sig["side"], qty)
    notify_order_result("BINANCE", symbol, sig["side"], order_result, dry_run=Config.DRY_RUN)

    Binance.algo_order(symbol, sig["exit_side"], "STOP_MARKET",        sig["sl"], qty, sig["side"], current_price=sig["entry"])
    Binance.algo_order(symbol, sig["exit_side"], "TAKE_PROFIT_MARKET", sig["tp"], qty, sig["side"], current_price=sig["entry"])

    # Update state
    state.setdefault("last_signal_bar", {})[symbol] = sig["signal_bar_close"]
    state.setdefault("last_trade_time", {})[symbol] = time.time()
    if not Config.DRY_RUN:
        state["trades_today"] = state.get("trades_today", 0) + 1
    # Record breakeven trigger: entry ± BREAKEVEN_R × risk_dist
    risk_dist = abs(sig["entry"] - sig["sl"])
    sign = 1 if sig["side"] == "BUY" else -1
    state.setdefault("be_trigger", {})[symbol] = round(
        sig["entry"] + sign * risk_dist * Config.BREAKEVEN_R, 6)
    state.setdefault("be_applied", {})[symbol] = False
    save_state(state)

    _append_trade({
        "time":             datetime.now().isoformat(timespec="seconds"),
        "symbol":           symbol,
        "side":             sig["side"],
        "tier":             sig["tier"],
        "qty":              qty,
        "entry_ref":        sig["entry"],
        "sl":               sig["sl"],
        "tp":               sig["tp"],
        "rr":               STRATEGY_CFG.rr,
        "risk_pct":         Config.RISK_PCT,
        "signal_bar_close": sig["signal_bar_close"],
        "adx":              sig["adx"],
        "atr":              sig["atr"],
        "volume_ratio":     sig["volume_ratio"],
        "dry_run":          Config.DRY_RUN,
    })


def _append_trade(row: dict) -> None:
    path = Config.TRADE_LOG
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


# ─────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────

def _setup() -> None:
    if not Config.API_KEY or not Config.SECRET:
        raise RuntimeError("Missing BINANCE_API_KEY + BINANCE_SECRET")
    Binance.sync_time()
    for symbol in Config.SYMBOLS:
        Binance.set_margin(symbol)
        Binance.set_leverage(symbol)


def main() -> None:
    log("=" * 64)
    mode_label = "LIVE" if not Config.DRY_RUN else "DRY RUN"
    log(f"H1 Donchian Bot — {mode_label}")
    log(
        f"Symbols={Config.SYMBOLS} | adx {STRATEGY_CFG.adx_min}-{STRATEGY_CFG.adx_max} | "
        f"RR {STRATEGY_CFG.rr} | vol {STRATEGY_CFG.volume_mult}x | "
        f"risk {Config.RISK_PCT*100:.0f}% | poll {Config.POLL_SECONDS}s"
    )
    log("=" * 64)

    notify_bot_started(
        "H1 Donchian Bot",
        mode_label,
        f"Symbols: <code>{','.join(Config.SYMBOLS)}</code>\n"
        f"Config: adx {STRATEGY_CFG.adx_min}-{STRATEGY_CFG.adx_max}, "
        f"rr={STRATEGY_CFG.rr}, risk={Config.RISK_PCT*100:.0f}%",
    )
    _setup()
    state = load_state()

    while True:
        try:
            balance             = Binance.balance()
            positions_by_symbol = get_positions_by_symbol()
            reset_day(state, balance)
            mark_connected(state)

            open_positions = {
                s: p for s, p in positions_by_symbol.items()
                if s in Config.SYMBOLS and abs(p.get("amount", 0)) > 0
            }
            day_start  = float(state.get("day_start_balance") or balance)
            daily_ret  = (balance - day_start) / max(day_start, 1) * 100

            log(
                f"balance={balance:.2f} USDT | daily={daily_ret:+.2f}% | "
                f"trades={state.get('trades_today',0)}/{Config.MAX_TRADES_PER_DAY} | "
                f"open={len(open_positions)}"
            )

            open_allowed, open_reason = can_open_new(state, balance, positions_by_symbol)
            if not open_allowed:
                log(f"New entries paused: {open_reason}")

            for symbol in Config.SYMBOLS:
                log(f"--- {symbol} ---")
                pos = positions_by_symbol.get(symbol, {"amount": 0.0, "entry": 0.0, "mark": 0.0})

                if abs(pos["amount"]) > 0:
                    be_applied = state.get("be_applied", {}).get(symbol, False)
                    be_trigger = state.get("be_trigger", {}).get(symbol, 0.0)
                    log(f"{symbol} open pos={pos['amount']:.4f} entry={pos['entry']:.4f} "
                        f"mark={pos['mark']:.4f} | BE={'✅' if be_applied else f'@{be_trigger:.4f}'}")
                    ensure_protection(symbol, pos)
                    check_breakeven(symbol, pos, state)
                    continue

                cleanup_orphan_orders(symbol)

                if not open_allowed:
                    log(f"{symbol} skip: {open_reason}")
                    continue

                m15 = fetch_m15(symbol)
                sig  = generate_signal(symbol, m15)

                if sig is None:
                    log(f"{symbol} no signal")
                    continue

                log(
                    f"{symbol} SIGNAL {sig['side']} tier={sig['tier']} "
                    f"bar={sig['signal_bar_close']} adx={sig['adx']} vol={sig['volume_ratio']}x"
                )
                execute_signal(sig, balance, state, pos_amount=pos.get("amount", 0.0))

        except KeyboardInterrupt:
            warn("Stopped by user")
            break
        except Exception as exc:
            warn(f"Loop error: {exc}")
            mark_disconnected(state, exc)
            if "-1003" in str(exc) or "429" in str(exc):
                # rate limited — back off 120 s before retrying
                warn("Rate limited by Binance — sleeping 120 s")
                time.sleep(120)
            else:
                if "-1021" in str(exc):
                    Binance.sync_time()
                time.sleep(Config.CONNECTION_RETRY_SLEEP_SECONDS)
            continue

        time.sleep(Config.POLL_SECONDS)


if __name__ == "__main__":
    main()
