import os
import sys
import time
import hmac
import json
import hashlib
import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from datetime import datetime, timedelta
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from donchian_core import DonchianCoreConfig, latest_signal
from demo_testcase_logger import log_demo_testcase
from notifier import notify_bot_started, notify_error, notify_order_opened, notify_order_result, notify_reconnected

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


load_dotenv()


class Config:
    API_KEY = os.getenv("BINANCE_API_KEY") or os.getenv("BINANCE2_API_KEY")
    SECRET = os.getenv("BINANCE_SECRET") or os.getenv("BINANCE2_SECRET")
    BASE_URL = os.getenv("BINANCE_BASE_URL", "https://demo-fapi.binance.com").rstrip("/")

    # Micro-edge config is validated on BTCUSDT BUY-only. Override with HYBRID_SYMBOLS if needed.
    SYMBOLS = [s.strip().upper() for s in os.getenv("HYBRID_SYMBOLS", "BTCUSDT").split(",") if s.strip()]
    TIMEFRAMES = ("1m", "5m", "15m")
    KLINE_LIMIT = int(os.getenv("HYBRID_KLINE_LIMIT", "600"))

    DRY_RUN = os.getenv("HYBRID_DRY_RUN", "true").lower() == "true"
    LEVERAGE = 1
    MARGIN_TYPE = "ISOLATED"

    POLL_SECONDS = int(os.getenv("HYBRID_POLL_SECONDS", "30"))
    OUTSIDE_SESSION_SLEEP_SECONDS = int(os.getenv("HYBRID_OUTSIDE_SESSION_SLEEP_SECONDS", "300"))
    CONNECTION_RETRY_SLEEP_SECONDS = int(os.getenv("HYBRID_CONNECTION_RETRY_SLEEP_SECONDS", "20"))
    DISCONNECT_NOTIFY_COOLDOWN_SECONDS = int(os.getenv("HYBRID_DISCONNECT_NOTIFY_COOLDOWN_SECONDS", "300"))
    REQUEST_RETRIES = int(os.getenv("HYBRID_REQUEST_RETRIES", "3"))
    LOG_TIMEFRAME_SUMMARY = os.getenv("HYBRID_LOG_TIMEFRAME_SUMMARY", "false").lower() == "true"
    ENABLE_DEMO_TESTCASE_LOG = os.getenv("HYBRID_ENABLE_DEMO_TESTCASE_LOG", "false").lower() == "true"
    ENTRY_COOLDOWN = 300
    MAX_OPEN_SYMBOLS = 1
    MAX_TRADES_PER_DAY = 2
    MAX_DAILY_LOSS_PCT = 0.04
    DAILY_PROFIT_TARGET_PCT = 0.10
    MAX_TOTAL_EXPOSURE_PCT = 60.0
    TIER_A_RISK = 0.0025
    TIER_B_RISK = 0.01
    RR = 2.0
    DONCHIAN_N = 20
    ADX_MIN = 20.0
    ADX_MAX = 30.0
    ALLOWED_SIDE = "BUY"
    SESSION_HOURS_UTC = tuple(
        int(h.strip())
        for h in os.getenv("HYBRID_SESSION_HOURS_UTC", "8,9,10,11,12").split(",")
        if h.strip()
    )
    ATR_PERCENTILE_MIN = 65.0
    VOLUME_MULT = 1.2
    REQUIRE_ATR_EXPANSION = True
    ATR_EXPANSION_PERIOD = 50

    SL_ATR = 1.0
    TP_ATR_NORMAL = SL_ATR * RR
    TP_ATR_STRONG = SL_ATR * RR
    TRAILING_ATR = 0.9
    TRIGGER_GUARD_PCT = 0.0005

    USE_AI_VALIDATOR = os.getenv("HYBRID_USE_AI", "false").lower() == "true"
    DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY")
    AI_MODEL = "deepseek-chat"

    LOG_DIR = Path("logs")
    LOG_FILE = LOG_DIR / f"hybrid_bot_{datetime.now().strftime('%Y-%m-%d')}.log"
    TRADE_LOG = LOG_DIR / "hybrid_trades.csv"
    STATE_FILE = LOG_DIR / "hybrid_state.json"


Config.LOG_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("hybrid_bot")
logger.setLevel(logging.INFO)
logger.propagate = False

file_handler = TimedRotatingFileHandler(
    Config.LOG_FILE,
    when="midnight",
    interval=1,
    backupCount=30,
    encoding="utf-8",
)
file_handler.suffix = "%Y-%m-%d"
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(file_handler)


def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")
    logger.info(msg)


def warn(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] WARNING {msg}")
    logger.warning(msg)


class Binance:
    time_offset = 0
    _exchange_info_cache = {}

    @classmethod
    def sync_time(cls):
        try:
            data = requests.get(f"{Config.BASE_URL}/fapi/v1/time", timeout=10).json()
            cls.time_offset = int(data["serverTime"]) - int(time.time() * 1000)
            log(f"⏱️ Time synced offset={cls.time_offset}ms")
        except Exception as e:
            warn(f"⏱️ Time sync failed: {e}")

    @staticmethod
    def headers():
        return {"X-MBX-APIKEY": Config.API_KEY}

    @staticmethod
    def sign(params):
        query = urlencode(params)
        sig = hmac.new(Config.SECRET.encode(), query.encode(), hashlib.sha256).hexdigest()
        return f"{query}&signature={sig}"

    @staticmethod
    def public_get(path, params=None):
        last_error = None
        for attempt in range(Config.REQUEST_RETRIES):
            try:
                r = requests.get(f"{Config.BASE_URL}{path}", params=params or {}, timeout=15)
                r.raise_for_status()
                return r.json()
            except Exception as e:
                last_error = e
                if attempt < Config.REQUEST_RETRIES - 1:
                    time.sleep(min(2 ** attempt, 5))
        raise RuntimeError(f"Binance public request failed {path}: {last_error}")

    @staticmethod
    def signed(method, path, params=None):
        base_params = dict(params or {})
        last_error = None
        for attempt in range(Config.REQUEST_RETRIES):
            try:
                signed_params = dict(base_params)
                signed_params["timestamp"] = int(time.time() * 1000) + Binance.time_offset
                signed_params["recvWindow"] = 10000
                url = f"{Config.BASE_URL}{path}?{Binance.sign(signed_params)}"
                if method == "GET":
                    r = requests.get(url, headers=Binance.headers(), timeout=15)
                elif method == "POST":
                    r = requests.post(url, headers=Binance.headers(), timeout=15)
                elif method == "DELETE":
                    r = requests.delete(url, headers=Binance.headers(), timeout=15)
                else:
                    raise ValueError(method)
                data = r.json()
                if r.status_code >= 400:
                    raise RuntimeError(f"Binance error {r.status_code}: {data}")
                return data
            except Exception as e:
                last_error = e
                if "-1021" in str(e):
                    Binance.sync_time()
                if attempt < Config.REQUEST_RETRIES - 1:
                    time.sleep(min(2 ** attempt, 5))
        raise RuntimeError(f"Binance signed request failed {path}: {last_error}")

    @staticmethod
    def get_klines(symbol, interval, limit=250):
        data = Binance.public_get(
            "/fapi/v1/klines",
            {"symbol": symbol, "interval": interval, "limit": limit},
        )
        df = pd.DataFrame(data, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "num_trades",
            "taker_buy_base", "taker_buy_quote", "ignore",
        ])
        df["time"] = pd.to_datetime(df["open_time"], unit="ms")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df[["time", "open", "high", "low", "close", "volume"]].set_index("time").dropna()

    @staticmethod
    def balance():
        data = Binance.signed("GET", "/fapi/v2/balance", {})
        for item in data:
            if item.get("asset") == "USDT":
                return float(item.get("balance", 0))
        return 0.0

    @staticmethod
    def positions():
        return Binance.signed("GET", "/fapi/v2/positionRisk", {})

    @staticmethod
    def position(symbol):
        for p in Binance.positions():
            if p.get("symbol") == symbol:
                return {
                    "amount": float(p.get("positionAmt", 0)),
                    "entry": float(p.get("entryPrice", 0)),
                    "mark": float(p.get("markPrice", 0)),
                    "raw": p,
                }
        return {"amount": 0.0, "entry": 0.0, "mark": 0.0, "raw": None}

    @staticmethod
    def open_orders(symbol):
        return Binance.signed("GET", "/fapi/v1/openOrders", {"symbol": symbol})

    @staticmethod
    def open_algo_orders(symbol):
        try:
            return Binance.signed("GET", "/fapi/v1/openAlgoOrders", {"symbol": symbol})
        except Exception as e:
            warn(f"🧹 {symbol} openAlgoOrders failed, fallback to conditional openOrders: {e}")
            return Binance.signed("GET", "/fapi/v1/openOrders", {"symbol": symbol, "conditional": "true"})

    @staticmethod
    def cancel_all_open_orders(symbol, conditional=False):
        params = {"symbol": symbol}
        if conditional:
            params["conditional"] = "true"
        return Binance.signed("DELETE", "/fapi/v1/allOpenOrders", params)

    @staticmethod
    def exchange_info(symbol):
        if symbol in Binance._exchange_info_cache:
            return Binance._exchange_info_cache[symbol]
        data = Binance.public_get("/fapi/v1/exchangeInfo")
        for item in data["symbols"]:
            if item["symbol"] == symbol:
                Binance._exchange_info_cache[symbol] = item
                return item
        raise RuntimeError(f"Symbol not found: {symbol}")

    @staticmethod
    def set_margin(symbol):
        try:
            Binance.signed("POST", "/fapi/v1/marginType", {"symbol": symbol, "marginType": Config.MARGIN_TYPE})
            log(f"⚙️ Set margin {symbol}={Config.MARGIN_TYPE}")
        except Exception as e:
            if "-4046" in str(e):
                log(f"✅ Set margin {symbol}: already {Config.MARGIN_TYPE}")
            else:
                warn(f"⚙️ Set margin {symbol}: {e}")

    @staticmethod
    def set_leverage(symbol):
        try:
            Binance.signed("POST", "/fapi/v1/leverage", {"symbol": symbol, "leverage": Config.LEVERAGE})
            log(f"⚙️ Set leverage {symbol}={Config.LEVERAGE}x")
        except Exception as e:
            warn(f"⚙️ Set leverage {symbol}: {e}")

    @staticmethod
    def market_order(symbol, side, qty, reduce_only=False):
        params = {"symbol": symbol, "side": side, "type": "MARKET", "quantity": qty}
        if reduce_only:
            params["reduceOnly"] = "true"
        if Config.DRY_RUN:
            log(f"🧪 DRY_RUN market_order {params}")
            return {"dry_run": True, **params}
        return Binance.signed("POST", "/fapi/v1/order", params)

    @staticmethod
    def stop_order(symbol, side, stop_price, qty, order_type, position_side=None, auto_close=False):
        pos = Binance.position(symbol)
        current = pos["mark"] or pos["entry"]
        if current <= 0:
            current = stop_price

        trigger_price = round(stop_price, 2)
        if position_side == "BUY":
            if order_type == "STOP_MARKET" and trigger_price >= current:
                if auto_close:
                    warn(f"🚪 {symbol} long SL already crossed ({trigger_price} >= {current:.2f}); closing market")
                    return Binance.market_order(symbol, side, qty, reduce_only=True)
                trigger_price = round(current * (1 - Config.TRIGGER_GUARD_PCT), 2)
            elif order_type == "TAKE_PROFIT_MARKET" and trigger_price <= current:
                if auto_close:
                    warn(f"💰 {symbol} long TP already reached ({trigger_price} <= {current:.2f}); closing market")
                    return Binance.market_order(symbol, side, qty, reduce_only=True)
                trigger_price = round(current * (1 + Config.TRIGGER_GUARD_PCT), 2)
        elif position_side == "SELL":
            if order_type == "STOP_MARKET" and trigger_price <= current:
                if auto_close:
                    warn(f"🚪 {symbol} short SL already crossed ({trigger_price} <= {current:.2f}); closing market")
                    return Binance.market_order(symbol, side, qty, reduce_only=True)
                trigger_price = round(current * (1 + Config.TRIGGER_GUARD_PCT), 2)
            elif order_type == "TAKE_PROFIT_MARKET" and trigger_price >= current:
                if auto_close:
                    warn(f"💰 {symbol} short TP already reached ({trigger_price} >= {current:.2f}); closing market")
                    return Binance.market_order(symbol, side, qty, reduce_only=True)
                trigger_price = round(current * (1 - Config.TRIGGER_GUARD_PCT), 2)

        params = {
            "algoType": "CONDITIONAL",
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "triggerPrice": trigger_price,
            "quantity": qty,
            "reduceOnly": "true",
            "workingType": "CONTRACT_PRICE",
        }
        if Config.DRY_RUN:
            log(f"🧪 DRY_RUN algo {order_type} {params}")
            return {"dry_run": True, **params}
        return Binance.signed("POST", "/fapi/v1/algoOrder", params)


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(s, n=14):
    delta = s.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df, n=14):
    tr = np.maximum(
        df["high"] - df["low"],
        np.maximum((df["high"] - df["close"].shift()).abs(), (df["low"] - df["close"].shift()).abs()),
    )
    return tr.rolling(n).mean()


def prepare(df):
    df = df.copy()
    df["ema20"] = ema(df["close"], 20)
    df["ema50"] = ema(df["close"], 50)
    df["ema200"] = ema(df["close"], 200)
    df["rsi"] = rsi(df["close"])
    df["atr"] = atr(df)
    df["atr_pct"] = df["atr"] / df["close"]
    df["vol_ma20"] = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / df["vol_ma20"]
    df["body_ratio"] = (df["close"] - df["open"]).abs() / (df["high"] - df["low"]).replace(0, np.nan)
    df["ema20_slope"] = df["ema20"] - df["ema20"].shift(5)
    return df


def fetch_market(symbol):
    return {tf: Binance.get_klines(symbol, tf, Config.KLINE_LIMIT) for tf in Config.TIMEFRAMES}


def last_closed(df):
    return df.iloc[-2]


def previous(df):
    return df.iloc[-3]


def summarize_timeframes(symbol, market):
    if not Config.LOG_TIMEFRAME_SUMMARY:
        return
    parts = []
    for tf in Config.TIMEFRAMES:
        row = last_closed(prepare(market[tf]))
        trend = "UP" if row.close > row.ema20 else "DOWN" if row.close < row.ema20 else "FLAT"
        parts.append(
            f"{tf} close={row.close:.2f} ema20={row.ema20:.2f} ema50={row.ema50:.2f} "
            f"rsi={row.rsi:.1f} atr={row.atr:.2f} atr%={row.atr_pct*100:.2f} vol={row.vol_ratio:.2f} {trend}"
        )
    log(f"📈 {symbol} | " + " | ".join(parts))


def trend_side(row_15m, row_1h):
    buy = (
        row_1h.close > row_1h.ema50
        and row_1h.ema20_slope >= 0
        and row_15m.close > row_15m.ema20
        and row_15m.close > row_15m.ema50
    )
    sell = (
        row_1h.close < row_1h.ema50
        and row_1h.ema20_slope <= 0
        and row_15m.close < row_15m.ema20
        and row_15m.close < row_15m.ema50
    )
    if buy:
        return "BUY"
    if sell:
        return "SELL"
    return None


def breakout_side(df_15m):
    row = last_closed(df_15m)
    lookback = df_15m.iloc[-Config.BREAKOUT_LOOKBACK - 2:-2]
    high = lookback["high"].max()
    low = lookback["low"].min()
    if row.close > high:
        return "BUY", f"15m close broke prev{Config.BREAKOUT_LOOKBACK} high {high:.2f}"
    if row.close < low:
        return "SELL", f"15m close broke prev{Config.BREAKOUT_LOOKBACK} low {low:.2f}"
    return None, ""


def impulse_side(df_5m, df_15m, df_1h):
    row_5m = last_closed(df_5m)
    lookback = df_5m.iloc[-Config.IMPULSE_LOOKBACK - 2:-2]
    if len(lookback) < Config.IMPULSE_LOOKBACK:
        return None, ""

    move_from_high = (row_5m.close - lookback["high"].max()) / lookback["high"].max()
    move_from_low = (row_5m.close - lookback["low"].min()) / lookback["low"].min()
    row_15m = last_closed(df_15m)
    row_1h = last_closed(df_1h)

    sell = (
        move_from_high <= -Config.IMPULSE_MOVE_PCT
        and row_5m.close < row_5m.ema20
        and row_15m.close < row_15m.ema20
        and row_1h.close < row_1h.ema20
        and row_5m.rsi > 22
    )
    buy = (
        move_from_low >= Config.IMPULSE_MOVE_PCT
        and row_5m.close > row_5m.ema20
        and row_15m.close > row_15m.ema20
        and row_1h.close > row_1h.ema20
        and row_5m.rsi < 78
    )

    if sell:
        return "SELL", f"5m impulse breakdown {move_from_high*100:.2f}% from recent high"
    if buy:
        return "BUY", f"5m impulse breakout {move_from_low*100:.2f}% from recent low"
    return None, ""


def generate_signal(symbol, market):
    signal, reason, _ = latest_signal(
        symbol,
        market["1m"],
        market["5m"],
        market["15m"],
        core_config(),
    )
    if not signal:
        return None, reason
    return signal, (
        f"SIGNAL {signal['side']} tier={signal['tier']} score={signal['score']} "
        f"entry={signal['entry']:.2f} sl={signal['sl']:.2f} tp={signal['tp']:.2f} | {signal['reason']}"
    )


def maybe_log_demo_testcase(*args, **kwargs):
    if not Config.ENABLE_DEMO_TESTCASE_LOG:
        return
    log_demo_testcase(*args, **kwargs)


def core_config():
    return DonchianCoreConfig(
        allowed_side=Config.ALLOWED_SIDE,
        tier_a_risk=Config.TIER_A_RISK,
        tier_b_risk=Config.TIER_B_RISK,
        rr=Config.RR,
        donchian_n=Config.DONCHIAN_N,
        adx_min=Config.ADX_MIN,
        adx_max=Config.ADX_MAX,
        session_hours_utc=Config.SESSION_HOURS_UTC,
        atr_percentile_min=Config.ATR_PERCENTILE_MIN,
        volume_mult=Config.VOLUME_MULT,
        require_atr_expansion=Config.REQUIRE_ATR_EXPANSION,
        atr_expansion_period=Config.ATR_EXPANSION_PERIOD,
    )


def ai_validate(signal, market):
    if not Config.USE_AI_VALIDATOR or not Config.DEEPSEEK_KEY or OpenAI is None:
        return True, "AI disabled"
    if signal.get("tier") == "B":
        return True, "strong technical score"
    try:
        client = OpenAI(api_key=Config.DEEPSEEK_KEY, base_url="https://api.deepseek.com/v1")
        row_5m = last_closed(prepare(market["5m"]))
        row_15m = last_closed(prepare(market["15m"]))
        prompt = f"""Validate this crypto futures setup. Return strict JSON only.
Symbol: {signal['symbol']}
Side: {signal['side']}
Type: {signal['type']}
Tier: {signal.get('tier')}
Breakout strength score: {signal.get('breakout_strength_score')}
5m close={row_5m.close:.2f} ema20={row_5m.ema20:.2f} rsi={row_5m.rsi:.1f} atr_pct={row_5m.atr_pct:.4f} vol_ratio={row_5m.vol_ratio:.2f}
15m close={row_15m.close:.2f} ema20={row_15m.ema20:.2f} ema50={row_15m.ema50:.2f} rsi={row_15m.rsi:.1f}
Return {{"action":"ALLOW" or "BLOCK","reason":"short"}}. Block only on obvious contradiction or extreme chop."""
        resp = client.chat.completions.create(
            model=Config.AI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=120,
        )
        content = resp.choices[0].message.content.strip()
        if "```" in content:
            content = content.split("```")[1].replace("json", "", 1).strip()
        result = json.loads(content)
        action = result.get("action", "ALLOW")
        reason = result.get("reason", "")
        return action == "ALLOW", reason
    except Exception as e:
        warn(f"🧠 AI validator failed, allowing technical signal: {e}")
        return True, "AI failed open"


def load_state():
    if Path(Config.STATE_FILE).exists():
        return json.loads(Path(Config.STATE_FILE).read_text(encoding="utf-8"))
    state = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "day_start_balance": None,
        "trades_today": 0,
        "last_trade_time": {},
        "dry_run": Config.DRY_RUN,
    }
    save_state(state)
    return state


def save_state(state):
    Path(Config.STATE_FILE).write_text(json.dumps(state, indent=2), encoding="utf-8")


def reset_day(state, balance):
    today = datetime.now().strftime("%Y-%m-%d")
    if state.get("dry_run") != Config.DRY_RUN:
        log(
            f"🔄 Runtime mode changed dry_run={state.get('dry_run')} -> {Config.DRY_RUN}; "
            "resetting daily counters"
        )
        state["dry_run"] = Config.DRY_RUN
        state["day_start_balance"] = balance
        state["trades_today"] = 0
        state["last_trade_time"] = {}
        save_state(state)
    if state.get("date") != today:
        state["date"] = today
        state["day_start_balance"] = balance
        state["trades_today"] = 0
        state["last_trade_time"] = {}
        state["dry_run"] = Config.DRY_RUN
        save_state(state)
    if state.get("day_start_balance") is None:
        state["day_start_balance"] = balance
        state["dry_run"] = Config.DRY_RUN
        save_state(state)


def append_trade(row):
    pd.DataFrame([row]).to_csv(
        Config.TRADE_LOG,
        mode="a",
        header=not Path(Config.TRADE_LOG).exists(),
        index=False,
    )


def round_qty(symbol, qty):
    info = Binance.exchange_info(symbol)
    step = 0.001
    min_qty = 0.0
    for f in info["filters"]:
        if f["filterType"] == "LOT_SIZE":
            step = float(f["stepSize"])
            min_qty = float(f["minQty"])
            break
    precision = max(0, int(round(-np.log10(step))))
    rounded = np.floor(qty / step) * step
    rounded = round(float(rounded), precision)
    return rounded if rounded >= min_qty else 0.0


def quantity_for_signal(symbol, balance, signal):
    risk_amount = balance * float(signal.get("risk_pct", Config.TIER_A_RISK))
    stop_distance = abs(signal["entry"] - signal["sl"])
    if stop_distance <= 0:
        return 0.0
    qty = risk_amount / stop_distance
    return round_qty(symbol, qty)


def account_snapshot(balance, positions_by_symbol=None):
    positions_by_symbol = positions_by_symbol or get_positions_by_symbol()
    open_symbols = []
    exposure = 0.0
    for symbol in Config.SYMBOLS:
        pos = positions_by_symbol.get(symbol, {"amount": 0.0, "entry": 0.0, "mark": 0.0})
        amt = abs(pos["amount"])
        if amt > 0:
            open_symbols.append(symbol)
            exposure += amt * (pos["mark"] or pos["entry"]) / max(balance, 1) * 100
    return open_symbols, exposure


def can_open_new(state, balance, positions_by_symbol=None):
    day_start = float(state.get("day_start_balance") or balance)
    daily_ret = (balance - day_start) / day_start if day_start else 0.0
    if daily_ret <= -Config.MAX_DAILY_LOSS_PCT:
        return False, f"daily loss stop {daily_ret:.2%}"
    if daily_ret >= Config.DAILY_PROFIT_TARGET_PCT:
        return False, f"daily profit target reached {daily_ret:.2%}"
    if state.get("trades_today", 0) >= Config.MAX_TRADES_PER_DAY:
        return False, "max trades per day reached"
    open_symbols, exposure = account_snapshot(balance, positions_by_symbol)
    if len(open_symbols) >= Config.MAX_OPEN_SYMBOLS:
        return False, f"max open symbols {len(open_symbols)}/{Config.MAX_OPEN_SYMBOLS}"
    if exposure >= Config.MAX_TOTAL_EXPOSURE_PCT:
        return False, f"total exposure {exposure:.2f}% >= {Config.MAX_TOTAL_EXPOSURE_PCT}%"
    return True, "ok"


def daily_return_pct(state, balance):
    day_start = float(state.get("day_start_balance") or balance)
    if day_start <= 0:
        return 0.0
    return (balance - day_start) / day_start * 100


def utc_now():
    return datetime.utcnow()


def entry_session_active(now=None):
    now = now or utc_now()
    return now.hour in Config.SESSION_HOURS_UTC


def seconds_until_next_entry_session(now=None):
    now = now or utc_now()
    if entry_session_active(now):
        return 0
    allowed_hours = sorted(Config.SESSION_HOURS_UTC)
    if not allowed_hours:
        return Config.OUTSIDE_SESSION_SLEEP_SECONDS
    for hour in allowed_hours:
        candidate = now.replace(hour=hour, minute=0, second=0, microsecond=0)
        if candidate > now:
            return max(1, int((candidate - now).total_seconds()))
    tomorrow = now.date() + timedelta(days=1)
    candidate = datetime.combine(tomorrow, datetime.min.time()).replace(hour=allowed_hours[0])
    return max(1, int((candidate - now).total_seconds()))


def get_positions_by_symbol():
    positions = {}
    for p in Binance.positions():
        symbol = p.get("symbol")
        if symbol:
            positions[symbol] = {
                "amount": float(p.get("positionAmt", 0)),
                "entry": float(p.get("entryPrice", 0)),
                "mark": float(p.get("markPrice", 0)),
                "raw": p,
            }
    return positions


def active_positions(positions_by_symbol):
    return {
        symbol: pos
        for symbol, pos in positions_by_symbol.items()
        if symbol in Config.SYMBOLS and abs(pos.get("amount", 0)) > 0
    }


def mark_loop_connected(state):
    if state.get("connection_down"):
        notify_reconnected("BINANCE", "Main loop recovered and Binance API responded")
        log("✅ Binance connection recovered")
        state["connection_down"] = False
        state["last_disconnect_notify"] = 0
        save_state(state)
        return
    state["connection_down"] = False
    state["last_disconnect_notify"] = 0


def mark_loop_disconnected(state, error):
    now = time.time()
    should_notify = (
        not state.get("connection_down")
        or now - float(state.get("last_disconnect_notify") or 0) >= Config.DISCONNECT_NOTIFY_COOLDOWN_SECONDS
    )
    state["connection_down"] = True
    if should_notify:
        notify_error("BINANCE", f"Disconnected / loop error: {error}")
        state["last_disconnect_notify"] = now
    save_state(state)


def ensure_protection(symbol, pos, market):
    amt = abs(pos["amount"])
    if amt <= 0:
        return
    try:
        regular_orders = Binance.open_orders(symbol) if not Config.DRY_RUN else []
        algo_orders = Binance.open_algo_orders(symbol) if not Config.DRY_RUN else []
        orders = regular_orders + algo_orders
        has_reduce = any(
            str(o.get("reduceOnly")).lower() == "true"
            or str(o.get("closePosition")).lower() == "true"
            for o in orders
        )
        if has_reduce:
            log(f"🛡️ {symbol} position protected by reduce-only/algo order (orders={len(orders)})")
            return
    except Exception as e:
        warn(f"🛡️ {symbol} cannot inspect open orders: {e}")
        return

    row_5m = last_closed(market["5m"])
    entry = pos["entry"] or pos["mark"] or row_5m.close
    side = "BUY" if pos["amount"] > 0 else "SELL"
    exit_side = "SELL" if side == "BUY" else "BUY"
    atr_series = atr(market["5m"])
    atr_val = float(atr_series.iloc[-2]) if len(atr_series) >= 2 and not pd.isna(atr_series.iloc[-2]) else entry * 0.003
    if side == "BUY":
        sl = entry - atr_val * Config.SL_ATR
        tp = entry + atr_val * Config.TP_ATR_NORMAL
    else:
        sl = entry + atr_val * Config.SL_ATR
        tp = entry - atr_val * Config.TP_ATR_NORMAL
    qty = round_qty(symbol, amt)
    warn(f"🛡️ {symbol} has unprotected position; placing emergency SL/TP side={side} qty={qty} sl={sl:.2f} tp={tp:.2f}")
    Binance.stop_order(symbol, exit_side, sl, qty, "STOP_MARKET", position_side=side, auto_close=True)
    Binance.stop_order(symbol, exit_side, tp, qty, "TAKE_PROFIT_MARKET", position_side=side, auto_close=True)


def cleanup_orphan_orders(symbol):
    if Config.DRY_RUN:
        return
    try:
        regular_orders = Binance.open_orders(symbol)
        algo_orders = Binance.open_algo_orders(symbol)
        total = len(regular_orders) + len(algo_orders)
        if total <= 0:
            return
        warn(f"🧹 {symbol} has no position but {total} open orders remain; canceling")
        if regular_orders:
            Binance.cancel_all_open_orders(symbol)
        if algo_orders:
            Binance.cancel_all_open_orders(symbol, conditional=True)
    except Exception as e:
        warn(f"🧹 {symbol} orphan order cleanup failed: {e}")


def execute_signal(signal, balance, state, market):
    symbol = signal["symbol"]
    pos = Binance.position(symbol)
    if abs(pos["amount"]) > 0:
        log(f"⏸️ {symbol} skip entry: position already open amount={pos['amount']}")
        return
    last = state.get("last_trade_time", {}).get(symbol, 0)
    if time.time() - last < Config.ENTRY_COOLDOWN:
        log(f"⏸️ {symbol} skip entry: cooldown")
        return

    qty = quantity_for_signal(symbol, balance, signal)
    if qty <= 0:
        log(f"⏸️ {symbol} skip entry: qty too small")
        return

    allowed, reason = ai_validate(signal, market)
    if not allowed:
        log(f"🧠 {symbol} AI blocked signal: {reason}")
        return

    log(
        f"🚀 OPEN {symbol} {signal['side']} qty={qty} tier={signal.get('tier')} "
        f"risk={signal.get('risk_pct', Config.TIER_A_RISK)*100:.2f}% "
        f"type={signal['type']} entry~{signal['entry']:.2f} sl={signal['sl']:.2f} tp={signal['tp']:.2f}"
    )
    notify_order_opened(
        "BINANCE",
        symbol,
        signal["side"],
        qty,
        signal["entry"],
        signal["sl"],
        signal["tp"],
        tier=signal.get("tier", ""),
        risk_pct=signal.get("risk_pct", Config.TIER_A_RISK),
        dry_run=Config.DRY_RUN,
    )
    order_result = Binance.market_order(symbol, signal["side"], qty)
    notify_order_result("BINANCE", symbol, signal["side"], order_result, dry_run=Config.DRY_RUN)
    Binance.stop_order(symbol, signal["exit_side"], signal["sl"], qty, "STOP_MARKET", position_side=signal["side"])
    Binance.stop_order(symbol, signal["exit_side"], signal["tp"], qty, "TAKE_PROFIT_MARKET", position_side=signal["side"])

    if Config.DRY_RUN:
        log("🧪 DRY_RUN signal recorded only in trade log; not counting toward live daily trade limit")
    else:
        state["trades_today"] = state.get("trades_today", 0) + 1
        state.setdefault("last_trade_time", {})[symbol] = time.time()
        state["dry_run"] = Config.DRY_RUN
        save_state(state)
    append_trade({
        "time": datetime.now().isoformat(timespec="seconds"),
        "symbol": symbol,
        "side": signal["side"],
        "qty": qty,
        "entry_ref": signal["entry"],
        "sl": signal["sl"],
        "tp": signal["tp"],
        "score": signal["score"],
        "tier": signal.get("tier"),
        "risk_pct": signal.get("risk_pct"),
        "type": signal["type"],
        "dry_run": Config.DRY_RUN,
        "reason": signal["reason"],
    })


def setup():
    if not Config.API_KEY or not Config.SECRET:
        raise RuntimeError("Missing BINANCE_API_KEY/BINANCE_SECRET or BINANCE2_API_KEY/BINANCE2_SECRET")
    Binance.sync_time()
    for symbol in Config.SYMBOLS:
        Binance.set_margin(symbol)
        Binance.set_leverage(symbol)


def main():
    log("═" * 64)
    log("🤖 Hybrid Donchian BTC/ETH/SOL bot started")
    log(
        f"🧪 DRY_RUN={Config.DRY_RUN} | 🧠 AI_VALIDATOR={Config.USE_AI_VALIDATOR} | "
        f"strategy=MicroEdge Donchian{Config.DONCHIAN_N} {Config.ALLOWED_SIDE} "
        f"ADX {Config.ADX_MIN:g}-{Config.ADX_MAX:g} ATRpct>={Config.ATR_PERCENTILE_MIN:g} "
        f"UTC={','.join(map(str, Config.SESSION_HOURS_UTC))} RR=1:{Config.RR:g} | "
        f"symbols={','.join(Config.SYMBOLS)} | base_url={Config.BASE_URL}"
    )
    log(
        f"⚡ Fast mode: kline_limit={Config.KLINE_LIMIT}, poll={Config.POLL_SECONDS}s, "
        f"outside_session_sleep={Config.OUTSIDE_SESSION_SLEEP_SECONDS}s, "
        f"summary_log={Config.LOG_TIMEFRAME_SUMMARY}, testcase_log={Config.ENABLE_DEMO_TESTCASE_LOG}"
    )
    log("═" * 64)
    notify_bot_started(
        "Binance Donchian Bot",
        "DRY_RUN" if Config.DRY_RUN else "LIVE/DEMO",
        f"Symbols: <code>{','.join(Config.SYMBOLS)}</code>",
    )
    setup()
    state = load_state()

    while True:
        try:
            balance = Binance.balance()
            reset_day(state, balance)
            positions_by_symbol = get_positions_by_symbol()
            mark_loop_connected(state)
            daily_ret = daily_return_pct(state, balance)
            session_active = entry_session_active()
            open_positions = active_positions(positions_by_symbol)
            log(
                f"📊 Balance={balance:.2f} | daily={daily_ret:.2f}%/"
                f"{Config.DAILY_PROFIT_TARGET_PCT*100:.0f}% | trades_today={state.get('trades_today', 0)} | "
                f"session_active={session_active}"
            )

            if not session_active and not open_positions:
                sleep_for = min(Config.OUTSIDE_SESSION_SLEEP_SECONDS, seconds_until_next_entry_session())
                log(f"🌙 Outside entry session and no open position; sleeping {sleep_for}s")
                save_state(state)
                time.sleep(sleep_for)
                continue

            open_allowed, open_reason = can_open_new(state, balance, positions_by_symbol)
            if not session_active:
                open_allowed = False
                open_reason = "outside entry session"
            if not open_allowed:
                log(f"🚦 New entries paused: {open_reason}. Existing positions still managed.")

            for symbol in Config.SYMBOLS:
                log(f"🔎 --- {symbol} ---")
                market = fetch_market(symbol)
                summarize_timeframes(symbol, market)

                pos = positions_by_symbol.get(symbol, {"amount": 0.0, "entry": 0.0, "mark": 0.0, "raw": None})
                if abs(pos["amount"]) > 0:
                    maybe_log_demo_testcase(
                        "bitcoin_demo",
                        symbol,
                        market["1m"],
                        market["5m"],
                        market["15m"],
                        core_config(),
                        external_blocked_by="OPEN_POSITION_EXISTS",
                        external_block_reason=f"position amount={pos['amount']}",
                    )
                    log(f"📌 {symbol} existing position amount={pos['amount']} entry={pos['entry']} mark={pos['mark']}")
                    ensure_protection(symbol, pos, market)
                    continue
                cleanup_orphan_orders(symbol)

                if not open_allowed:
                    block = "DAILY_TRADE_LIMIT" if "max trades" in open_reason else "RISK_LIMIT"
                    maybe_log_demo_testcase(
                        "bitcoin_demo",
                        symbol,
                        market["1m"],
                        market["5m"],
                        market["15m"],
                        core_config(),
                        external_blocked_by=block,
                        external_block_reason=open_reason,
                    )
                    continue

                maybe_log_demo_testcase("bitcoin_demo", symbol, market["1m"], market["5m"], market["15m"], core_config())
                signal, reason = generate_signal(symbol, market)
                if not signal:
                    log(f"⏸️ {symbol} no entry: {reason}")
                    continue
                log(f"✅ {symbol} {reason}")
                execute_signal(signal, balance, state, market)

        except KeyboardInterrupt:
            warn("🛑 Bot stopped by user")
            break
        except Exception as e:
            warn(f"💥 Loop error: {e}")
            mark_loop_disconnected(state, e)
            if "-1021" in str(e):
                Binance.sync_time()
            time.sleep(Config.CONNECTION_RETRY_SLEEP_SECONDS)
            continue

        time.sleep(Config.POLL_SECONDS)


if __name__ == "__main__":
    main()
