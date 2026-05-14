"""
hybrid_bot.py — Donchian Breakout Futures Bot (Binance)
Refactored: cleaner structure, better RR, more signals, live-safe.
"""

import os
import sys
import time
import hmac
import json
import hashlib
import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from donchian_core import DonchianCoreConfig, latest_signal, _REJECT_STATS
from demo_testcase_logger import log_demo_testcase
from notifier import (
    notify_bot_started,
    notify_error,
    notify_order_opened,
    notify_order_result,
    notify_reconnected,
)

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

load_dotenv()


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

class Config:
    # Credentials
    API_KEY = os.getenv("BINANCE_API_KEY")
    SECRET  = os.getenv("BINANCE_SECRET")
    BASE_URL = os.getenv("BINANCE_BASE_URL", "https://fapi.binance.com").rstrip("/")
    BASE_URLS = [
        u.strip().rstrip("/")
        for u in os.getenv(
            "BINANCE_BASE_URLS",
            ",".join([
                BASE_URL,
                "https://fapi1.binance.com",
                "https://fapi2.binance.com",
                "https://fapi3.binance.com",
            ]),
        ).split(",")
        if u.strip()
    ]

    # Symbols & timeframes
    SYMBOLS    = [s.strip().upper() for s in os.getenv("HYBRID_SYMBOLS", "BTCUSDT").split(",") if s.strip()]
    TIMEFRAMES = ("1m", "5m", "15m")
    KLINE_LIMIT = int(os.getenv("HYBRID_KLINE_LIMIT", "600"))

    # Mode
    DRY_RUN     = os.getenv("HYBRID_DRY_RUN", "false").lower() == "true"
    LEVERAGE    = int(os.getenv("HYBRID_LEVERAGE", "3"))
    MARGIN_TYPE = "ISOLATED"

    # Timing
    POLL_SECONDS                     = int(os.getenv("HYBRID_POLL_SECONDS", "30"))
    OUTSIDE_SESSION_SLEEP_SECONDS    = int(os.getenv("HYBRID_OUTSIDE_SESSION_SLEEP_SECONDS", "300"))
    CONNECTION_RETRY_SLEEP_SECONDS   = int(os.getenv("HYBRID_CONNECTION_RETRY_SLEEP_SECONDS", "20"))
    RATE_LIMIT_SLEEP_SECONDS         = int(os.getenv("HYBRID_RATE_LIMIT_SLEEP_SECONDS", "180"))
    DISCONNECT_NOTIFY_COOLDOWN_SECONDS = int(os.getenv("HYBRID_DISCONNECT_NOTIFY_COOLDOWN_SECONDS", "300"))
    REQUEST_RETRIES = int(os.getenv("HYBRID_REQUEST_RETRIES", "3"))

    # Debug / logging
    LOG_TIMEFRAME_SUMMARY    = os.getenv("HYBRID_LOG_TIMEFRAME_SUMMARY", "false").lower() == "true"
    ENABLE_DEMO_TESTCASE_LOG = os.getenv("HYBRID_ENABLE_DEMO_TESTCASE_LOG", "false").lower() == "true"

    # Risk controls
    ENTRY_COOLDOWN          = 300        # seconds between entries per symbol
    MAX_OPEN_SYMBOLS        = 1
    MAX_TRADES_PER_DAY      = int(os.getenv("HYBRID_MAX_TRADES_PER_DAY", "5"))
    MAX_DAILY_LOSS_PCT      = float(os.getenv("HYBRID_MAX_DAILY_LOSS_PCT", "0.04"))
    DAILY_PROFIT_TARGET_PCT = float(os.getenv("HYBRID_DAILY_PROFIT_TARGET_PCT", "0.06"))  # 1.5× max loss
    MAX_TOTAL_EXPOSURE_PCT  = 60.0
    MARGIN_BUFFER_PCT       = float(os.getenv("HYBRID_MARGIN_BUFFER_PCT", "0.95"))

    # Position sizing
    TIER_A_RISK = 0.0025   # 0.25% per trade (conservative, compounding faster)
    TIER_B_RISK = 0.01     # 1.0% for high-conviction

    # ── RR improvements ──────────────────────────────────────────────────────
    # Problem: TP was 2× SL but price rarely reached it → now using ATR-based
    # asymmetric targets. SL tighter (0.8×ATR), TP wider (2.4×ATR → RR = 3).
    RR          = 3.0      # raised from 2.0
    SL_ATR      = 0.8      # tighter SL → less loss per stop-out
    TP_ATR      = SL_ATR * RR   # = 2.4
    TRAILING_ATR = 0.7     # trail closer so we lock in more profit

    TRIGGER_GUARD_PCT = 0.0005

    # ── Trend EMAs (passed to donchian_core) ─────────────────────────────────
    EMA_FAST_TREND = int(os.getenv("HYBRID_EMA_FAST", "20"))
    EMA_SLOW_TREND = int(os.getenv("HYBRID_EMA_SLOW", "50"))

    # ── Signal filters ────────────────────────────────────────────────────────
    DONCHIAN_N         = int(os.getenv("HYBRID_DONCHIAN_N", "20"))
    ADX_MIN            = float(os.getenv("HYBRID_ADX_MIN", "15.0"))    # loosened from 18
    ADX_MAX            = float(os.getenv("HYBRID_ADX_MAX", "55.0"))    # raised from 40
    ATR_PERCENTILE_MIN = float(os.getenv("HYBRID_ATR_PCT_MIN", "35.0"))  # lowered from 55
    VOLUME_MULT        = float(os.getenv("HYBRID_VOLUME_MULT", "0.9")) # lowered from 1.1
    MIN_ATR_PCT        = float(os.getenv("HYBRID_MIN_ATR_PCT", "0.0008"))  # crypto vol > forex
    REQUIRE_ATR_EXPANSION = os.getenv("HYBRID_REQUIRE_ATR_EXPANSION", "false").lower() == "true"
    ATR_EXPANSION_PERIOD  = int(os.getenv("HYBRID_ATR_EXPANSION_PERIOD", "50"))

    ALLOWED_SIDE = os.getenv("HYBRID_ALLOWED_SIDE", "BOTH")  # was "BUY" — crypto needs both sides

    # Crypto is 24/7; no fixed session — default to all hours
    SESSION_HOURS_UTC = tuple(
        int(h.strip())
        for h in os.getenv(
            "HYBRID_SESSION_HOURS_UTC",
            ",".join(str(h) for h in range(24)),
        ).split(",")
        if h.strip()
    )

    # AI validator
    USE_AI_VALIDATOR = os.getenv("HYBRID_USE_AI", "false").lower() == "true"
    DEEPSEEK_KEY     = os.getenv("DEEPSEEK_API_KEY")
    AI_MODEL         = "deepseek-chat"

    # Paths
    LOG_DIR    = Path("logs")
    LOG_FILE   = LOG_DIR / "hybrid_bot.log"
    TRADE_LOG  = LOG_DIR / "hybrid_trades.csv"
    STATE_FILE = LOG_DIR / "hybrid_state.json"
    KILL_FILE  = LOG_DIR / "hybrid_STOP"
    MAX_CONSECUTIVE_LOSSES = int(os.getenv("HYBRID_MAX_CONSECUTIVE_LOSSES", "3"))


Config.LOG_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────

def _build_logger() -> logging.Logger:
    lg = logging.getLogger("hybrid_bot")
    lg.setLevel(logging.INFO)
    lg.propagate = False
    handler = TimedRotatingFileHandler(
        Config.LOG_FILE, when="midnight", interval=1, backupCount=30, encoding="utf-8"
    )
    handler.suffix = "%Y-%m-%d"
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    lg.addHandler(handler)
    return lg


_logger = _build_logger()
_ts = lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{_ts()}] {msg}")
    _logger.info(msg)


def warn(msg: str) -> None:
    print(f"[{_ts()}] WARNING {msg}")
    _logger.warning(msg)


# ─────────────────────────────────────────────
# Binance API
# ─────────────────────────────────────────────

class BinanceRateLimitError(RuntimeError):
    """Raised when Binance blocks/rate-limits this IP or endpoint."""


class Binance:
    time_offset: int = 0
    _exchange_info_cache: dict = {}
    _base_url_index: int = 0

    # ── helpers ──────────────────────────────

    @classmethod
    def sync_time(cls) -> None:
        try:
            data = requests.get(f"{cls.base_url()}/fapi/v1/time", timeout=10).json()
            cls.time_offset = int(data["serverTime"]) - int(time.time() * 1000)
            log(f"⏱️ Time synced offset={cls.time_offset}ms")
        except Exception as exc:
            warn(f"⏱️ Time sync failed: {exc}")

    @staticmethod
    def _headers() -> dict:
        return {"X-MBX-APIKEY": Config.API_KEY}

    @classmethod
    def base_url(cls) -> str:
        return Config.BASE_URLS[cls._base_url_index % len(Config.BASE_URLS)]

    @classmethod
    def rotate_base_url(cls) -> str:
        cls._base_url_index = (cls._base_url_index + 1) % len(Config.BASE_URLS)
        url = cls.base_url()
        warn(f"🌐 Binance endpoint rotated → {url}")
        return url

    @staticmethod
    def _retry_after_seconds(response) -> int | None:
        value = response.headers.get("Retry-After")
        if not value:
            return None
        try:
            return max(1, int(float(value)))
        except ValueError:
            return None

    @staticmethod
    def _sign(params: dict) -> str:
        query = urlencode(params)
        sig = hmac.new(Config.SECRET.encode(), query.encode(), hashlib.sha256).hexdigest()
        return f"{query}&signature={sig}"

    # ── request wrappers ─────────────────────

    @staticmethod
    def public_get(path: str, params: dict | None = None) -> dict:
        last_exc: Exception | None = None
        for attempt in range(Config.REQUEST_RETRIES):
            try:
                url = f"{Binance.base_url()}{path}"
                r = requests.get(url, params=params or {}, timeout=15)
                if r.status_code in (418, 429):
                    retry_after = Binance._retry_after_seconds(r)
                    msg = (
                        f"Binance public rate-limit/block {r.status_code} on {path} "
                        f"base={Binance.base_url()} retry_after={retry_after} body={r.text[:160]}"
                    )
                    Binance.rotate_base_url()
                    raise BinanceRateLimitError(msg)
                r.raise_for_status()
                return r.json()
            except BinanceRateLimitError as exc:
                last_exc = exc
                sleep_for = Config.RATE_LIMIT_SLEEP_SECONDS
                if attempt < Config.REQUEST_RETRIES - 1:
                    warn(f"⏳ {exc}; sleeping {sleep_for}s before retry")
                    time.sleep(sleep_for)
                else:
                    break
            except Exception as exc:
                last_exc = exc
                if attempt < Config.REQUEST_RETRIES - 1:
                    time.sleep(min(2 ** attempt, 5))
        if isinstance(last_exc, BinanceRateLimitError):
            raise last_exc
        raise RuntimeError(f"Public GET {path} failed: {last_exc}")

    @staticmethod
    def signed(method: str, path: str, params: dict | None = None) -> dict:
        base = dict(params or {})
        last_exc: Exception | None = None
        for attempt in range(Config.REQUEST_RETRIES):
            try:
                p = {**base, "timestamp": int(time.time() * 1000) + Binance.time_offset, "recvWindow": 10000}
                url = f"{Binance.base_url()}{path}?{Binance._sign(p)}"
                fn = {"GET": requests.get, "POST": requests.post, "DELETE": requests.delete}[method]
                r = fn(url, headers=Binance._headers(), timeout=15)
                if r.status_code in (418, 429):
                    retry_after = Binance._retry_after_seconds(r)
                    msg = (
                        f"Binance signed rate-limit/block {r.status_code} on {path} "
                        f"base={Binance.base_url()} retry_after={retry_after} body={r.text[:160]}"
                    )
                    Binance.rotate_base_url()
                    raise BinanceRateLimitError(msg)
                data = r.json()
                if r.status_code >= 400:
                    raise RuntimeError(f"Binance {r.status_code}: {data}")
                return data
            except BinanceRateLimitError as exc:
                last_exc = exc
                sleep_for = Config.RATE_LIMIT_SLEEP_SECONDS
                if attempt < Config.REQUEST_RETRIES - 1:
                    warn(f"⏳ {exc}; sleeping {sleep_for}s before retry")
                    time.sleep(sleep_for)
                else:
                    break
            except Exception as exc:
                last_exc = exc
                if "-1021" in str(exc):
                    Binance.sync_time()
                if attempt < Config.REQUEST_RETRIES - 1:
                    time.sleep(min(2 ** attempt, 5))
        if isinstance(last_exc, BinanceRateLimitError):
            raise last_exc
        raise RuntimeError(f"Signed {method} {path} failed: {last_exc}")

    # ── market data ──────────────────────────

    @staticmethod
    def get_klines(symbol: str, interval: str, limit: int = 250) -> pd.DataFrame:
        data = Binance.public_get("/fapi/v1/klines", {"symbol": symbol, "interval": interval, "limit": limit})
        df = pd.DataFrame(data, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "num_trades",
            "taker_buy_base", "taker_buy_quote", "ignore",
        ])
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

    # ── account ──────────────────────────────

    @staticmethod
    def balance() -> float:
        for item in Binance.signed("GET", "/fapi/v2/balance"):
            if item.get("asset") == "USDT":
                return float(item.get("balance", 0))
        return 0.0

    @staticmethod
    def available_balance() -> float:
        for item in Binance.signed("GET", "/fapi/v2/balance"):
            if item.get("asset") == "USDT":
                return float(item.get("availableBalance", item.get("balance", 0)))
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
            warn(f"🧹 {symbol} openAlgoOrders fallback: {exc}")
            return Binance.signed("GET", "/fapi/v1/openOrders", {"symbol": symbol, "conditional": "true"})

    @staticmethod
    def cancel_all_orders(symbol: str, conditional: bool = False) -> dict:
        params = {"symbol": symbol}
        if conditional:
            params["conditional"] = "true"
        return Binance.signed("DELETE", "/fapi/v1/allOpenOrders", params)

    # ── trading ──────────────────────────────

    @staticmethod
    def market_order(symbol: str, side: str, qty: float, reduce_only: bool = False) -> dict:
        params = {"symbol": symbol, "side": side, "type": "MARKET", "quantity": qty}
        if reduce_only:
            params["reduceOnly"] = "true"
        if Config.DRY_RUN:
            log(f"🧪 DRY_RUN market_order {params}")
            return {"dry_run": True, **params}
        return Binance.signed("POST", "/fapi/v1/order", params)

    @staticmethod
    def algo_order(
        symbol: str,
        side: str,
        order_type: str,
        trigger_price: float,
        qty: float,
        position_side: str,
        auto_close: bool = False,
    ) -> dict:
        """Place STOP_MARKET or TAKE_PROFIT_MARKET algo order with guard logic."""
        pos = Binance.position(symbol)
        current = pos["mark"] or pos["entry"] or trigger_price

        tp = round(trigger_price, 2)

        # Adjust trigger if already breached
        if position_side == "BUY":
            if order_type == "STOP_MARKET" and tp >= current:
                if auto_close:
                    warn(f"🚪 {symbol} long SL crossed ({tp} >= {current:.2f}); closing market")
                    return Binance.market_order(symbol, side, qty, reduce_only=True)
                tp = round(current * (1 - Config.TRIGGER_GUARD_PCT), 2)
            elif order_type == "TAKE_PROFIT_MARKET" and tp <= current:
                if auto_close:
                    warn(f"💰 {symbol} long TP reached ({tp} <= {current:.2f}); closing market")
                    return Binance.market_order(symbol, side, qty, reduce_only=True)
                tp = round(current * (1 + Config.TRIGGER_GUARD_PCT), 2)
        else:  # SELL
            if order_type == "STOP_MARKET" and tp <= current:
                if auto_close:
                    warn(f"🚪 {symbol} short SL crossed ({tp} <= {current:.2f}); closing market")
                    return Binance.market_order(symbol, side, qty, reduce_only=True)
                tp = round(current * (1 + Config.TRIGGER_GUARD_PCT), 2)
            elif order_type == "TAKE_PROFIT_MARKET" and tp >= current:
                if auto_close:
                    warn(f"💰 {symbol} short TP reached ({tp} >= {current:.2f}); closing market")
                    return Binance.market_order(symbol, side, qty, reduce_only=True)
                tp = round(current * (1 - Config.TRIGGER_GUARD_PCT), 2)

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
            log(f"🧪 DRY_RUN algo {order_type} {params}")
            return {"dry_run": True, **params}
        return Binance.signed("POST", "/fapi/v1/algoOrder", params)

    # ── setup ─────────────────────────────────

    @staticmethod
    def set_margin(symbol: str) -> None:
        try:
            Binance.signed("POST", "/fapi/v1/marginType", {"symbol": symbol, "marginType": Config.MARGIN_TYPE})
            log(f"⚙️ {symbol} margin={Config.MARGIN_TYPE}")
        except Exception as exc:
            if "-4046" in str(exc):
                log(f"✅ {symbol} already {Config.MARGIN_TYPE}")
            else:
                warn(f"⚙️ {symbol} set margin: {exc}")

    @staticmethod
    def set_leverage(symbol: str) -> None:
        try:
            Binance.signed("POST", "/fapi/v1/leverage", {"symbol": symbol, "leverage": Config.LEVERAGE})
            log(f"⚙️ {symbol} leverage={Config.LEVERAGE}x")
        except Exception as exc:
            warn(f"⚙️ {symbol} set leverage: {exc}")


# ─────────────────────────────────────────────
# Indicators
# ─────────────────────────────────────────────

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    delta = s.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    return 100 - (100 / (1 + gain / loss.replace(0, np.nan)))


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    tr = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            (df["high"] - df["close"].shift()).abs(),
            (df["low"]  - df["close"].shift()).abs(),
        ),
    )
    return tr.rolling(n).mean()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema20"]       = ema(df["close"], 20)
    df["ema50"]       = ema(df["close"], 50)
    df["ema200"]      = ema(df["close"], 200)
    df["rsi"]         = rsi(df["close"])
    df["atr"]         = atr(df)
    df["atr_pct"]     = df["atr"] / df["close"]
    df["vol_ma20"]    = df["volume"].rolling(20).mean()
    df["vol_ratio"]   = df["volume"] / df["vol_ma20"]
    df["body_ratio"]  = (df["close"] - df["open"]).abs() / (df["high"] - df["low"]).replace(0, np.nan)
    df["ema20_slope"] = df["ema20"] - df["ema20"].shift(5)
    return df


# ─────────────────────────────────────────────
# Market data helpers
# ─────────────────────────────────────────────

def fetch_market(symbol: str) -> dict[str, pd.DataFrame]:
    return {tf: Binance.get_klines(symbol, tf, Config.KLINE_LIMIT) for tf in Config.TIMEFRAMES}


def last_closed(df: pd.DataFrame) -> pd.Series:
    return df.iloc[-2]


def summarize_timeframes(symbol: str, market: dict) -> None:
    if not Config.LOG_TIMEFRAME_SUMMARY:
        return
    parts = []
    for tf in Config.TIMEFRAMES:
        row = last_closed(prepare(market[tf]))
        trend = "UP" if row.close > row.ema20 else ("DOWN" if row.close < row.ema20 else "FLAT")
        parts.append(
            f"{tf} close={row.close:.2f} ema20={row.ema20:.2f} rsi={row.rsi:.1f} "
            f"atr%={row.atr_pct*100:.2f} vol={row.vol_ratio:.2f} {trend}"
        )
    log(f"📈 {symbol} | " + " | ".join(parts))


# ─────────────────────────────────────────────
# Signal generation
# ─────────────────────────────────────────────

def _core_config() -> DonchianCoreConfig:
    return DonchianCoreConfig(
        allowed_side          = Config.ALLOWED_SIDE,
        ema_fast              = Config.EMA_FAST_TREND,
        ema_slow              = Config.EMA_SLOW_TREND,
        min_atr_pct           = Config.MIN_ATR_PCT,
        tier_a_risk           = Config.TIER_A_RISK,
        tier_b_risk           = Config.TIER_B_RISK,
        rr                    = Config.RR,
        donchian_n            = Config.DONCHIAN_N,
        adx_min               = Config.ADX_MIN,
        adx_max               = Config.ADX_MAX,
        session_hours_utc     = Config.SESSION_HOURS_UTC,
        atr_percentile_min    = Config.ATR_PERCENTILE_MIN,
        volume_mult           = Config.VOLUME_MULT,
        require_atr_expansion = Config.REQUIRE_ATR_EXPANSION,
        atr_expansion_period  = Config.ATR_EXPANSION_PERIOD,
        max_trades_per_day    = Config.MAX_TRADES_PER_DAY,
        require_trend_alignment = False,  # EMA20/50 crossover lags too far; let Donchian breakout decide
    )


def generate_signal(symbol: str, market: dict) -> tuple[dict | None, str]:
    signal, reason, _ = latest_signal(
        symbol, market["1m"], market["5m"], market["15m"], _core_config()
    )
    if not signal:
        return None, reason
    summary = (
        f"SIGNAL {signal['side']} tier={signal['tier']} score={signal['score']} "
        f"entry={signal['entry']:.2f} sl={signal['sl']:.2f} tp={signal['tp']:.2f} | {signal['reason']}"
    )
    return signal, summary


# ─────────────────────────────────────────────
# AI validator (optional)
# ─────────────────────────────────────────────

def ai_validate(signal: dict, market: dict) -> tuple[bool, str]:
    if not Config.USE_AI_VALIDATOR or not Config.DEEPSEEK_KEY or OpenAI is None:
        return True, "AI disabled"
    if signal.get("tier") == "B":
        return True, "strong technical score"

    try:
        client = OpenAI(api_key=Config.DEEPSEEK_KEY, base_url="https://api.deepseek.com/v1")
        r5  = last_closed(prepare(market["5m"]))
        r15 = last_closed(prepare(market["15m"]))
        trend_15m = "UP" if r15.close > r15.ema50 else "DOWN"
        rsi_extreme = r5.rsi > 75 if signal["side"] == "BUY" else r5.rsi < 25
        prompt = (
            f"Validate this crypto futures setup. Return strict JSON only.\n"
            f"Symbol: {signal['symbol']}  Side: {signal['side']}  Tier: {signal.get('tier')}\n"
            f"Score: {signal.get('breakout_strength_score')}\n"
            f"5m  close={r5.close:.2f} ema20={r5.ema20:.2f} rsi={r5.rsi:.1f} "
            f"atr%={r5.atr_pct:.4f} vol={r5.vol_ratio:.2f}\n"
            f"15m close={r15.close:.2f} ema20={r15.ema20:.2f} ema50={r15.ema50:.2f} "
            f"rsi={r15.rsi:.1f} trend={trend_15m}\n"
            f"RSI extreme vs signal direction: {rsi_extreme}\n"
            f"Block if: SELL in strong 15m uptrend (price >> EMA50), "
            f"BUY in strong 15m downtrend, or RSI extreme against signal direction. "
            f'Return {{"action":"ALLOW"|"BLOCK","reason":"short (max 60 chars)"}}'
        )
        resp = client.chat.completions.create(
            model=Config.AI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=120,
        )
        content = resp.choices[0].message.content.strip().replace("```json", "").replace("```", "").strip()
        result = json.loads(content)
        return result.get("action", "ALLOW") == "ALLOW", result.get("reason", "")
    except Exception as exc:
        warn(f"🧠 AI validator failed (allowing): {exc}")
        return True, "AI failed open"


# ─────────────────────────────────────────────
# State management
# ─────────────────────────────────────────────

def load_state() -> dict:
    if Path(Config.STATE_FILE).exists():
        return json.loads(Path(Config.STATE_FILE).read_text(encoding="utf-8"))
    return _fresh_state()


def _fresh_state() -> dict:
    state = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "day_start_balance": None,
        "trades_today": 0,
        "last_trade_time": {},
        "dry_run": Config.DRY_RUN,
        "connection_down": False,
        "last_disconnect_notify": 0,
        "consecutive_losses": 0,
        "prev_open_symbols": [],
    }
    save_state(state)
    return state


def save_state(state: dict) -> None:
    Path(Config.STATE_FILE).write_text(json.dumps(state, indent=2), encoding="utf-8")


def reset_day(state: dict, balance: float) -> None:
    today = datetime.now().strftime("%Y-%m-%d")

    mode_changed = state.get("dry_run") != Config.DRY_RUN
    day_changed  = state.get("date") != today

    if mode_changed:
        log(f"🔄 Mode changed dry_run={state.get('dry_run')} → {Config.DRY_RUN}; resetting counters")

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
        state["dry_run"] = Config.DRY_RUN
        save_state(state)


# ─────────────────────────────────────────────
# Risk / position checks
# ─────────────────────────────────────────────

def daily_return_pct(state: dict, balance: float) -> float:
    day_start = float(state.get("day_start_balance") or balance)
    return (balance - day_start) / max(day_start, 1) * 100


def _account_snapshot(balance: float, positions_by_symbol: dict) -> tuple[list, float]:
    open_symbols, exposure = [], 0.0
    for symbol in Config.SYMBOLS:
        pos = positions_by_symbol.get(symbol, {"amount": 0.0, "entry": 0.0, "mark": 0.0})
        amt = abs(pos["amount"])
        if amt > 0:
            open_symbols.append(symbol)
            exposure += amt * (pos["mark"] or pos["entry"]) / max(balance, 1) * 100
    return open_symbols, exposure


def can_open_new(state: dict, balance: float, positions_by_symbol: dict) -> tuple[bool, str]:
    day_start  = float(state.get("day_start_balance") or balance)
    daily_ret  = (balance - day_start) / max(day_start, 1)

    if daily_ret <= -Config.MAX_DAILY_LOSS_PCT:
        return False, f"daily loss stop {daily_ret:.2%}"
    if daily_ret >= Config.DAILY_PROFIT_TARGET_PCT:
        return False, f"daily profit target reached {daily_ret:.2%}"
    if state.get("trades_today", 0) >= Config.MAX_TRADES_PER_DAY:
        return False, "max trades per day reached"

    open_symbols, exposure = _account_snapshot(balance, positions_by_symbol)
    if len(open_symbols) >= Config.MAX_OPEN_SYMBOLS:
        return False, f"max open symbols {len(open_symbols)}/{Config.MAX_OPEN_SYMBOLS}"
    if exposure >= Config.MAX_TOTAL_EXPOSURE_PCT:
        return False, f"total exposure {exposure:.2f}% >= {Config.MAX_TOTAL_EXPOSURE_PCT}%"

    return True, "ok"


# ─────────────────────────────────────────────
# Session timing
# ─────────────────────────────────────────────

def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def is_near_funding(now: datetime | None = None) -> bool:
    """True in the 5 min before each 8-hour funding window (00, 08, 16 UTC)."""
    t = (now or _utc_now())
    h, m = t.hour, t.minute
    # 5 min before: 23:55-23:59, 07:55-07:59, 15:55-15:59
    if h in (23, 7, 15) and m >= 55:
        return True
    # First 5 min after (price often spikes): 00:00-00:04, 08:00-08:04, 16:00-16:04
    if h in (0, 8, 16) and m < 5:
        return True
    return False


def entry_session_active(now: datetime | None = None) -> bool:
    return (now or _utc_now()).hour in Config.SESSION_HOURS_UTC


def seconds_until_next_session(now: datetime | None = None) -> int:
    now = now or _utc_now()
    if entry_session_active(now):
        return 0
    allowed = sorted(Config.SESSION_HOURS_UTC)
    if not allowed:
        return Config.OUTSIDE_SESSION_SLEEP_SECONDS
    for hour in allowed:
        candidate = now.replace(hour=hour, minute=0, second=0, microsecond=0)
        if candidate > now:
            return max(1, int((candidate - now).total_seconds()))
    tomorrow = (now.date() + timedelta(days=1))
    candidate = datetime.combine(tomorrow, datetime.min.time()).replace(tzinfo=timezone.utc, hour=allowed[0])
    return max(1, int((candidate - now).total_seconds()))


# ─────────────────────────────────────────────
# Position helpers
# ─────────────────────────────────────────────

def get_positions_by_symbol() -> dict:
    return {
        p["symbol"]: {
            "amount": float(p.get("positionAmt", 0)),
            "entry":  float(p.get("entryPrice", 0)),
            "mark":   float(p.get("markPrice", 0)),
            "raw": p,
        }
        for p in Binance.positions()
        if p.get("symbol")
    }


def active_positions(positions_by_symbol: dict) -> dict:
    return {
        sym: pos for sym, pos in positions_by_symbol.items()
        if sym in Config.SYMBOLS and abs(pos.get("amount", 0)) > 0
    }


# ─────────────────────────────────────────────
# Order sizing
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


def quantity_for_signal(symbol: str, balance: float, signal: dict) -> float:
    risk_amount    = balance * float(signal.get("risk_pct", Config.TIER_A_RISK))
    stop_distance  = abs(signal["entry"] - signal["sl"])
    if stop_distance <= 0:
        return 0.0
    return round_qty(symbol, risk_amount / stop_distance)


def cap_qty_to_available_margin(symbol: str, qty: float, entry: float, available_margin: float) -> tuple[float, str]:
    if Config.DRY_RUN:
        return qty, "dry_run"
    leverage = max(float(Config.LEVERAGE), 1.0)
    usable_margin = max(available_margin, 0.0) * Config.MARGIN_BUFFER_PCT
    required_margin = qty * entry / leverage
    if required_margin <= usable_margin:
        return qty, f"required_margin={required_margin:.2f} <= usable={usable_margin:.2f}"
    capped_qty = round_qty(symbol, usable_margin * leverage / entry)
    if capped_qty <= 0:
        return 0.0, f"insufficient margin: required={required_margin:.2f} usable={usable_margin:.2f}"
    return capped_qty, f"qty capped by margin {qty}->{capped_qty} required={required_margin:.2f} usable={usable_margin:.2f}"


# ─────────────────────────────────────────────
# Connection state
# ─────────────────────────────────────────────

def mark_connected(state: dict) -> None:
    if state.get("connection_down"):
        notify_reconnected("BINANCE", "Main loop recovered")
        log("✅ Binance connection recovered")
    state["connection_down"] = False
    state["last_disconnect_notify"] = 0


def mark_disconnected(state: dict, error: Exception) -> None:
    now = time.time()
    cooldown_expired = now - float(state.get("last_disconnect_notify") or 0) >= Config.DISCONNECT_NOTIFY_COOLDOWN_SECONDS
    if not state.get("connection_down") or cooldown_expired:
        notify_error("BINANCE", f"Disconnected: {error}")
        state["last_disconnect_notify"] = now
    state["connection_down"] = True
    save_state(state)


# ─────────────────────────────────────────────
# Position protection
# ─────────────────────────────────────────────

def ensure_protection(symbol: str, pos: dict, market: dict) -> None:
    """Place emergency SL/TP if an open position has no reduce-only orders."""
    if abs(pos["amount"]) <= 0:
        return

    try:
        orders = ([] if Config.DRY_RUN else Binance.open_orders(symbol)) + \
                 ([] if Config.DRY_RUN else Binance.open_algo_orders(symbol))
        protected = any(
            str(o.get("reduceOnly")).lower() == "true" or str(o.get("closePosition")).lower() == "true"
            for o in orders
        )
        if protected:
            log(f"🛡️ {symbol} position is protected (orders={len(orders)})")
            return
    except Exception as exc:
        warn(f"🛡️ {symbol} cannot inspect orders: {exc}")
        return

    atr_series = atr(market["5m"])
    atr_val = float(atr_series.iloc[-2]) if len(atr_series) >= 2 and not pd.isna(atr_series.iloc[-2]) else pos["entry"] * 0.003
    side     = "BUY" if pos["amount"] > 0 else "SELL"
    exit_side = "SELL" if side == "BUY" else "BUY"
    entry    = pos["entry"] or pos["mark"] or float(market["5m"].iloc[-2]["close"])

    sl = entry - atr_val * Config.SL_ATR if side == "BUY" else entry + atr_val * Config.SL_ATR
    tp = entry + atr_val * Config.TP_ATR  if side == "BUY" else entry - atr_val * Config.TP_ATR
    qty = round_qty(symbol, abs(pos["amount"]))

    warn(f"🛡️ {symbol} unprotected — placing emergency SL={sl:.2f} TP={tp:.2f}")
    Binance.algo_order(symbol, exit_side, "STOP_MARKET",        sl, qty, side, auto_close=True)
    Binance.algo_order(symbol, exit_side, "TAKE_PROFIT_MARKET", tp, qty, side, auto_close=True)


def cleanup_orphan_orders(symbol: str) -> None:
    if Config.DRY_RUN:
        return
    try:
        regular = Binance.open_orders(symbol)
        algo    = Binance.open_algo_orders(symbol)
        total   = len(regular) + len(algo)
        if total <= 0:
            return
        warn(f"🧹 {symbol} no position but {total} orphan orders; canceling")
        if regular:
            Binance.cancel_all_orders(symbol)
        if algo:
            Binance.cancel_all_orders(symbol, conditional=True)
    except Exception as exc:
        warn(f"🧹 {symbol} orphan cleanup failed: {exc}")


# ─────────────────────────────────────────────
# Trade execution
# ─────────────────────────────────────────────

def execute_signal(signal: dict, balance: float, state: dict, market: dict) -> None:
    symbol = signal["symbol"]

    # Guards
    if abs(Binance.position(symbol)["amount"]) > 0:
        log(f"⏸️ {symbol} skip: position already open")
        return
    last_trade = state.get("last_trade_time", {}).get(symbol, 0)
    if time.time() - last_trade < Config.ENTRY_COOLDOWN:
        log(f"⏸️ {symbol} skip: cooldown")
        return
    qty = quantity_for_signal(symbol, balance, signal)
    if qty <= 0:
        log(f"⏸️ {symbol} skip: qty too small")
        return
    available_margin = balance if Config.DRY_RUN else Binance.available_balance()
    capped_qty, margin_msg = cap_qty_to_available_margin(symbol, qty, float(signal["entry"]), available_margin)
    if capped_qty <= 0:
        warn(f"⏸️ {symbol} skip: {margin_msg}")
        return
    if capped_qty < qty:
        warn(f"💰 {symbol} {margin_msg}")
        qty = capped_qty
    allowed, reason = ai_validate(signal, market)
    if not allowed:
        log(f"🧠 {symbol} AI blocked: {reason}")
        return

    log(
        f"🚀 OPEN REQUEST {symbol} {signal['side']} qty={qty} tier={signal.get('tier')} "
        f"risk={signal.get('risk_pct', Config.TIER_A_RISK)*100:.2f}% "
        f"entry~{signal['entry']:.2f} sl={signal['sl']:.2f} tp={signal['tp']:.2f}"
    )

    try:
        order_result = Binance.market_order(symbol, signal["side"], qty)
    except Exception as exc:
        msg = str(exc)
        if "-2019" in msg or "Margin is insufficient" in msg:
            warn(f"⏸️ {symbol} order rejected: insufficient margin | qty={qty} | {margin_msg}")
            notify_error("BINANCE", f"Order rejected: insufficient margin for {symbol} qty={qty}")
            return
        raise
    notify_order_result("BINANCE", symbol, signal["side"], order_result, dry_run=Config.DRY_RUN)
    notify_order_opened(
        "BINANCE", symbol, signal["side"], qty,
        signal["entry"], signal["sl"], signal["tp"],
        tier=signal.get("tier", ""),
        risk_pct=signal.get("risk_pct", Config.TIER_A_RISK),
        dry_run=Config.DRY_RUN,
    )

    Binance.algo_order(symbol, signal["exit_side"], "STOP_MARKET",        signal["sl"], qty, signal["side"])
    Binance.algo_order(symbol, signal["exit_side"], "TAKE_PROFIT_MARKET", signal["tp"], qty, signal["side"])

    # Update state
    state[f"balance_at_entry_{symbol}"] = balance
    is_live = not Config.DRY_RUN
    if is_live:
        state["trades_today"] = state.get("trades_today", 0) + 1
    state.setdefault("last_trade_time", {})[symbol] = time.time()
    state["dry_run"] = Config.DRY_RUN
    save_state(state)

    if Config.DRY_RUN:
        log("🧪 DRY_RUN: trade logged only, daily counter not consumed")

    _append_trade({
        "time":       datetime.now().isoformat(timespec="seconds"),
        "symbol":     symbol,
        "side":       signal["side"],
        "qty":        qty,
        "entry_ref":  signal["entry"],
        "sl":         signal["sl"],
        "tp":         signal["tp"],
        "score":      signal["score"],
        "tier":       signal.get("tier"),
        "risk_pct":   signal.get("risk_pct"),
        "type":       signal["type"],
        "dry_run":    Config.DRY_RUN,
        "reason":     signal["reason"],
    })


def _append_trade(row: dict) -> None:
    expected = list(row.keys())
    if Config.TRADE_LOG.exists():
        try:
            first_line = Config.TRADE_LOG.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
            if first_line.split(",") != expected:
                backup = Config.TRADE_LOG.with_name(
                    f"{Config.TRADE_LOG.stem}.schema_bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                )
                Config.TRADE_LOG.rename(backup)
                warn(f"🧾 Schema changed; old log rotated to {backup}")
        except Exception as exc:
            warn(f"🧾 Trade log schema check failed: {exc}")
    pd.DataFrame([row]).to_csv(
        Config.TRADE_LOG, mode="a",
        header=not Config.TRADE_LOG.exists(), index=False
    )


# ─────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────

def _setup() -> None:
    if not Config.API_KEY or not Config.SECRET:
        raise RuntimeError("Missing API key/secret — set BINANCE_API_KEY + BINANCE_SECRET")
    Binance.sync_time()
    try:
        balance = Binance.balance()
        log(f"✅ Binance auth OK | USDT balance={balance:.2f}")
    except Exception as exc:
        msg = str(exc)
        if "-2015" in msg or "Invalid API-key" in msg:
            raise RuntimeError(
                "Binance auth failed: invalid API key, IP whitelist, or Futures trading permission. "
                "Check BINANCE_API_KEY, BINANCE_SECRET, Binance Futures permissions, IP restrictions, "
                "and whether BINANCE_BASE_URL matches mainnet/testnet keys."
            ) from exc
        raise
    for symbol in Config.SYMBOLS:
        Binance.set_margin(symbol)
        Binance.set_leverage(symbol)


def _maybe_demo_log(symbol, market, core_cfg, **kwargs):
    if Config.ENABLE_DEMO_TESTCASE_LOG:
        log_demo_testcase("bitcoin_demo", symbol, market["1m"], market["5m"], market["15m"], core_cfg, **kwargs)


def main() -> None:
    log("═" * 64)
    log("🤖 Hybrid Donchian Bot — LIVE MODE" if not Config.DRY_RUN else "🧪 Hybrid Donchian Bot — DRY RUN")
    log(
        f"Strategy: Donchian{Config.DONCHIAN_N} {Config.ALLOWED_SIDE} | "
        f"ADX {Config.ADX_MIN:g}–{Config.ADX_MAX:g} | ATRpct>={Config.ATR_PERCENTILE_MIN:g} | "
        f"RR 1:{Config.RR:g} | SL={Config.SL_ATR}×ATR TP={Config.TP_ATR:.1f}×ATR | "
        f"Leverage={Config.LEVERAGE}x | Session UTC={Config.SESSION_HOURS_UTC} | Symbols={Config.SYMBOLS}"
    )
    log("═" * 64)

    notify_bot_started(
        "Binance Donchian Bot",
        "DRY_RUN" if Config.DRY_RUN else "LIVE",
        f"Symbols: <code>{','.join(Config.SYMBOLS)}</code>",
    )
    _setup()
    state = load_state()
    state.setdefault("last_stats_hour", -1)
    state.setdefault("last_stats_date", None)
    core_cfg = _core_config()

    while True:
        try:
            if Config.KILL_FILE.exists():
                log("[KILL] STOP file detected — exiting cleanly")
                save_state(state)
                break
            balance              = Binance.balance()
            available_balance    = balance if Config.DRY_RUN else Binance.available_balance()
            positions_by_symbol  = get_positions_by_symbol()
            reset_day(state, balance)
            mark_connected(state)

            # ── Daily stats reset + hourly log ───────────────────────────
            _now = datetime.now()
            today = str(_now.date())
            if state.get("last_stats_date") != today:
                if _REJECT_STATS:
                    log(f"[STATS] Daily summary: {dict(_REJECT_STATS.most_common())}")
                _REJECT_STATS.clear()
                state["last_stats_date"] = today
            elif _now.hour != state.get("last_stats_hour", -1) and _REJECT_STATS:
                total = _REJECT_STATS.get("_total_attempts", 1)
                top = {
                    k: f"{v}({v/total:.0%})"
                    for k, v in _REJECT_STATS.most_common()
                    if not k.startswith("_")
                }
                log(f"[STATS] Hourly rejections (of {total} attempts): {dict(list(top.items())[:8])}")
                state["last_stats_hour"] = _now.hour

            session_active  = entry_session_active()
            open_positions  = active_positions(positions_by_symbol)
            daily_ret       = daily_return_pct(state, balance)

            # Detect position closures → update consecutive_losses
            prev_open = set(state.get("prev_open_symbols", []))
            curr_open  = set(open_positions.keys())
            for closed_sym in prev_open - curr_open:
                bal_at_entry = state.get(f"balance_at_entry_{closed_sym}")
                if bal_at_entry is not None:
                    if balance < float(bal_at_entry):
                        state["consecutive_losses"] = state.get("consecutive_losses", 0) + 1
                        log(f"[CONSEC] Loss on {closed_sym} — consecutive={state['consecutive_losses']}")
                    else:
                        if state.get("consecutive_losses", 0) > 0:
                            log(f"[CONSEC] Win/BE on {closed_sym} — consecutive_losses reset")
                        state["consecutive_losses"] = 0
                    state.pop(f"balance_at_entry_{closed_sym}", None)
            state["prev_open_symbols"] = list(curr_open)

            log(
                f"📊 Balance={balance:.2f} USDT | available={available_balance:.2f} USDT | "
                f"lev={Config.LEVERAGE}x | daily={daily_ret:+.2f}% | "
                f"trades={state.get('trades_today',0)}/{Config.MAX_TRADES_PER_DAY} | "
                f"session={'✅' if session_active else '🌙'} | "
                f"open_positions={len(open_positions)}"
            )

            # Sleep outside session if nothing to manage
            if not session_active and not open_positions:
                sleep_for = min(Config.OUTSIDE_SESSION_SLEEP_SECONDS, seconds_until_next_session())
                log(f"🌙 Outside session — sleeping {sleep_for}s")
                save_state(state)
                time.sleep(sleep_for)
                continue

            open_allowed, open_reason = can_open_new(state, balance, positions_by_symbol)
            if not session_active:
                open_allowed, open_reason = False, "outside entry session"
            if state.get("consecutive_losses", 0) >= Config.MAX_CONSECUTIVE_LOSSES:
                open_allowed = False
                open_reason  = f"consecutive losses stop ({state['consecutive_losses']})"
            if not open_allowed:
                log(f"🚦 New entries paused: {open_reason}")

            for symbol in Config.SYMBOLS:
                log(f"🔎 {symbol}")
                market = fetch_market(symbol)
                summarize_timeframes(symbol, market)

                pos = positions_by_symbol.get(symbol, {"amount": 0.0, "entry": 0.0, "mark": 0.0, "raw": None})

                if abs(pos["amount"]) > 0:
                    log(f"📌 {symbol} open pos={pos['amount']} entry={pos['entry']} mark={pos['mark']}")
                    ensure_protection(symbol, pos, market)
                    _maybe_demo_log(symbol, market, core_cfg,
                                    external_blocked_by="OPEN_POSITION_EXISTS",
                                    external_block_reason=f"amount={pos['amount']}")
                    continue

                cleanup_orphan_orders(symbol)

                if not open_allowed:
                    block = "DAILY_TRADE_LIMIT" if "max trades" in open_reason else "RISK_LIMIT"
                    _maybe_demo_log(symbol, market, core_cfg,
                                    external_blocked_by=block, external_block_reason=open_reason)
                    continue

                if is_near_funding():
                    log(f"⏸️ {symbol} skip: near funding window")
                    continue

                _maybe_demo_log(symbol, market, core_cfg)
                signal, reason = generate_signal(symbol, market)
                if not signal:
                    log(f"⏸️ {symbol} no signal: {reason}")
                    continue

                log(f"✅ {symbol} {reason}")
                execute_signal(signal, balance, state, market)

        except KeyboardInterrupt:
            warn("🛑 Stopped by user")
            break
        except BinanceRateLimitError as exc:
            warn(f"🚧 Binance rate-limit/block: {exc}")
            mark_disconnected(state, exc)
            sleep_for = Config.RATE_LIMIT_SLEEP_SECONDS
            log(f"⏳ Sleeping {sleep_for}s after Binance 418/429")
            time.sleep(sleep_for)
            continue
        except Exception as exc:
            warn(f"💥 Loop error: {exc}")
            mark_disconnected(state, exc)
            if "-1021" in str(exc):
                Binance.sync_time()
            time.sleep(Config.CONNECTION_RETRY_SLEEP_SECONDS)
            continue

        time.sleep(Config.POLL_SECONDS)


if __name__ == "__main__":
    main()
