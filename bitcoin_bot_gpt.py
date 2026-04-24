# ============================================================
# V28 BINANCE FUTURES DEMO BOT
# ------------------------------------------------------------
# DEMO / TESTNET ONLY
#
# Required:
#   pip install pandas numpy requests python-dotenv scikit-learn joblib
#
# .env example:
#   BINANCE_API_KEY=your_testnet_key
#   BINANCE_SECRET=your_testnet_secret
#
# Run:
#   python3 v28_binance_futures_demo_bot.py
#
# Important:
#   - Uses Binance Futures Testnet endpoint:
#       https://demo-fapi.binance.com
#   - Default DRY_RUN = True. Set DRY_RUN = False only after checking logs.
#   - This is NOT financial advice and not guaranteed profitable.
# ============================================================

import os
import time
import hmac
import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from urllib.parse import urlencode

import requests
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# =====================
# LOAD ENV
# =====================
load_dotenv()

# =====================
# CONFIG
# =====================
class Config:
    API_KEY = os.getenv("BINANCE_API_KEY")
    SECRET = os.getenv("BINANCE_SECRET")
    BASE_URL = "https://demo-fapi.binance.com"

    SYMBOLS = ["BTCUSDT","ETHUSDT","SOLUSDT"]  # Start with BTC only. Add ETHUSDT/SOLUSDT after stable.
    INTERVAL = "5m"
    KLINE_LIMIT = 500

    # Safety
    DRY_RUN = True          # True = log signal only. False = place demo orders.
    LEVERAGE = 1
    MARGIN_TYPE = "ISOLATED"

    START_BALANCE_FALLBACK = 5000.0
    RISK_PER_TRADE = 0.005
    MAX_OPEN_POSITIONS = 3
    MAX_TRADES_PER_DAY = 3
    MAX_DAILY_LOSS_PCT = -0.03

    # Strategy / AI-gate fallback
    AI_PROB_THRESHOLD = 0.50

    # Rule filters from V26/V27 family
    SESSION_START = 13
    SESSION_END = 21

    RR = 2.1
    SL_ATR = 1.2
    TP_ATR = SL_ATR * RR

    MIN_ATR_PCT = 0.0013
    MAX_ATR_PCT = 0.0080
    MIN_VOLUME_RATIO = 0.50
    MIN_MOMENTUM_PCT = 0.00030
    MIN_EMA20_DISTANCE = 0.00025
    MIN_BODY_RATIO = 0.25

    POLL_SECONDS = 30

    LOG_FILE = "v29_xray_demo_bot.log"
    TRADE_LOG = "v29_xray_demo_trades.csv"
    STATE_FILE = "v29_xray_demo_state.json"


# =====================
# LOGGING
# =====================
logging.basicConfig(
    filename=Config.LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log(msg):
    print(f"[{now()}] {msg}")
    logging.info(msg)


# =====================
# BASIC VALIDATION
# =====================
if not Config.API_KEY or not Config.SECRET:
    raise RuntimeError("Missing BINANCE_API_KEY or BINANCE_SECRET in .env")


# =====================
# BINANCE FUTURES API
# =====================
class BinanceFutures:
    @staticmethod
    def _headers():
        return {"X-MBX-APIKEY": Config.API_KEY}

    @staticmethod
    def _sign(params):
        query = urlencode(params)
        sig = hmac.new(
            Config.SECRET.encode("utf-8"),
            query.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
        return f"{query}&signature={sig}"

    @staticmethod
    def public_get(path, params=None):
        url = Config.BASE_URL + path
        r = requests.get(url, params=params or {}, timeout=15)
        r.raise_for_status()
        return r.json()

    @staticmethod
    def signed_request(method, path, params=None):
        params = params or {}
        params["timestamp"] = int(time.time() * 1000)
        params["recvWindow"] = 5000

        query = BinanceFutures._sign(params)
        url = Config.BASE_URL + path + "?" + query

        if method == "GET":
            r = requests.get(url, headers=BinanceFutures._headers(), timeout=15)
        elif method == "POST":
            r = requests.post(url, headers=BinanceFutures._headers(), timeout=15)
        elif method == "DELETE":
            r = requests.delete(url, headers=BinanceFutures._headers(), timeout=15)
        else:
            raise ValueError(f"Unsupported method: {method}")

        try:
            data = r.json()
        except Exception:
            data = {"raw": r.text}

        if r.status_code >= 400:
            raise RuntimeError(f"Binance error {r.status_code}: {data}")

        return data

    @staticmethod
    def get_klines(symbol, interval="5m", limit=500):
        data = BinanceFutures.public_get(
            "/fapi/v1/klines",
            {"symbol": symbol, "interval": interval, "limit": limit}
        )
        df = pd.DataFrame(data, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "num_trades",
            "taker_buy_base", "taker_buy_quote", "ignore"
        ])

        df["time"] = pd.to_datetime(df["open_time"], unit="ms")
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        return df[["time", "open", "high", "low", "close", "volume"]].set_index("time").dropna()

    @staticmethod
    def account_balance_usdt():
        data = BinanceFutures.signed_request("GET", "/fapi/v2/balance", {})
        for item in data:
            if item.get("asset") == "USDT":
                return float(item.get("balance", 0))
        return Config.START_BALANCE_FALLBACK

    @staticmethod
    def positions():
        return BinanceFutures.signed_request("GET", "/fapi/v2/positionRisk", {})

    @staticmethod
    def get_position(symbol):
        positions = BinanceFutures.positions()
        for p in positions:
            if p.get("symbol") == symbol:
                amt = float(p.get("positionAmt", 0))
                entry = float(p.get("entryPrice", 0))
                return {"amount": amt, "entry": entry, "raw": p}
        return {"amount": 0.0, "entry": 0.0, "raw": None}

    @staticmethod
    def set_leverage(symbol):
        try:
            BinanceFutures.signed_request(
                "POST",
                "/fapi/v1/leverage",
                {"symbol": symbol, "leverage": Config.LEVERAGE}
            )
            log(f"Set leverage {symbol} = {Config.LEVERAGE}x")
        except Exception as e:
            log(f"Set leverage warning {symbol}: {e}")

    @staticmethod
    def set_margin_type(symbol):
        try:
            BinanceFutures.signed_request(
                "POST",
                "/fapi/v1/marginType",
                {"symbol": symbol, "marginType": Config.MARGIN_TYPE}
            )
            log(f"Set margin {symbol} = {Config.MARGIN_TYPE}")
        except Exception as e:
            # Binance returns error if margin type is already set. Safe to ignore.
            log(f"Set margin warning {symbol}: {e}")

    @staticmethod
    def market_order(symbol, side, quantity):
        params = {
            "symbol": symbol,
            "side": side,
            "type": "MARKET",
            "quantity": quantity,
        }

        if Config.DRY_RUN:
            log(f"DRY_RUN market_order {params}")
            return {"dry_run": True, **params}

        return BinanceFutures.signed_request("POST", "/fapi/v1/order", params)

    @staticmethod
    def stop_market_order(symbol, side, stop_price, quantity):
        params = {
            "symbol": symbol,
            "side": side,
            "type": "STOP_MARKET",
            "stopPrice": round(stop_price, 2),
            "quantity": quantity,
            "reduceOnly": "true",
            "workingType": "MARK_PRICE",
        }

        if Config.DRY_RUN:
            log(f"DRY_RUN stop_market_order {params}")
            return {"dry_run": True, **params}

        return BinanceFutures.signed_request("POST", "/fapi/v1/order", params)

    @staticmethod
    def take_profit_market_order(symbol, side, stop_price, quantity):
        params = {
            "symbol": symbol,
            "side": side,
            "type": "TAKE_PROFIT_MARKET",
            "stopPrice": round(stop_price, 2),
            "quantity": quantity,
            "reduceOnly": "true",
            "workingType": "MARK_PRICE",
        }

        if Config.DRY_RUN:
            log(f"DRY_RUN take_profit_market_order {params}")
            return {"dry_run": True, **params}

        return BinanceFutures.signed_request("POST", "/fapi/v1/order", params)

    @staticmethod
    def exchange_info(symbol):
        data = BinanceFutures.public_get("/fapi/v1/exchangeInfo")
        for s in data["symbols"]:
            if s["symbol"] == symbol:
                return s
        raise RuntimeError(f"Symbol not found: {symbol}")


# =====================
# STATE
# =====================
def load_state():
    if Path(Config.STATE_FILE).exists():
        return json.loads(Path(Config.STATE_FILE).read_text(encoding="utf-8"))
    state = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "day_start_balance": None,
        "trades_today": 0,
    }
    save_state(state)
    return state

def save_state(state):
    Path(Config.STATE_FILE).write_text(json.dumps(state, indent=2), encoding="utf-8")

def reset_day_if_needed(state, balance):
    today = datetime.now().strftime("%Y-%m-%d")
    if state.get("date") != today:
        state["date"] = today
        state["day_start_balance"] = balance
        state["trades_today"] = 0
        save_state(state)

    if state.get("day_start_balance") is None:
        state["day_start_balance"] = balance
        save_state(state)

    return state

def append_trade_log(row):
    pd.DataFrame([row]).to_csv(
        Config.TRADE_LOG,
        mode="a",
        header=not Path(Config.TRADE_LOG).exists(),
        index=False
    )


# =====================
# INDICATORS
# =====================
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def atr(df, n=14):
    tr = np.maximum(
        df["high"] - df["low"],
        np.maximum(abs(df["high"] - df["close"].shift(1)), abs(df["low"] - df["close"].shift(1)))
    )
    return tr.rolling(n).mean()

def prepare_df(df):
    df = df.copy()
    df["ema20"] = ema(df["close"], 20)
    df["ema50"] = ema(df["close"], 50)
    df["atr"] = atr(df)
    df["vol_ma20"] = df["volume"].rolling(20).mean()
    df["atr_pct"] = df["atr"] / df["close"]
    df["volume_ratio"] = df["volume"] / df["vol_ma20"]
    df["body_ratio"] = abs(df["close"] - df["open"]) / (df["high"] - df["low"]).replace(0, np.nan)
    df["ema20_distance"] = abs(df["close"] - df["ema20"]) / df["close"]
    return df


# =====================
# SIGNAL ENGINE
# =====================
def ai_probability_placeholder(features):
    """
    Safe placeholder.

    V26 showed AI filter works in backtest, but for live demo we need exported model
    to avoid mismatched logic. This placeholder approximates quality score from filters.

    Later version:
      - export sklearn model with joblib
      - load model here
      - return model.predict_proba([features])[0][1]
    """

    score = 0.50

    if features["volume_ratio"] >= 1.0:
        score += 0.04
    if features["momentum_pct"] >= 0.0010:
        score += 0.04
    if features["body_ratio"] >= 0.55:
        score += 0.03
    if features["ema20_distance"] >= 0.0010:
        score += 0.03
    if Config.MIN_ATR_PCT <= features["atr_pct"] <= 0.005:
        score += 0.03

    return min(score, 0.75)

def generate_signal(symbol, df):
    """
    X-ray version:
    returns (signal, reason)
    - signal = dict when bot should enter
    - reason = text explaining pass/fail filter
    """
    df = prepare_df(df)

    if len(df) < 60:
        return None, f"{symbol}: not enough candles"

    row = df.iloc[-2]   # last closed candle
    prev = df.iloc[-3]
    t = row.name
    price = float(row["close"])

    if not (Config.SESSION_START <= t.hour <= Config.SESSION_END):
        return None, f"{symbol}: outside session hour={t.hour}"

    if pd.isna(row["atr"]) or row["atr"] <= 0:
        return None, f"{symbol}: ATR not ready"

    atr_val = float(row["atr"])
    atr_pct = float(row["atr_pct"]) if not pd.isna(row["atr_pct"]) else 0.0
    volume_ratio = float(row["volume_ratio"]) if not pd.isna(row["volume_ratio"]) else 0.0
    momentum_pct = abs(price - float(prev["close"])) / float(prev["close"])
    ema20_distance = float(row["ema20_distance"]) if not pd.isna(row["ema20_distance"]) else 0.0
    body_ratio = float(row["body_ratio"]) if not pd.isna(row["body_ratio"]) else 0.0

    metrics = (
        f"price={price:.2f} atr_pct={atr_pct:.5f} "
        f"vol_ratio={volume_ratio:.2f} momentum={momentum_pct:.5f} "
        f"ema20_dist={ema20_distance:.5f} body={body_ratio:.2f}"
    )

    if not (Config.MIN_ATR_PCT <= atr_pct <= Config.MAX_ATR_PCT):
        return None, (
            f"{symbol}: ATR filter fail {metrics} "
            f"need {Config.MIN_ATR_PCT:.5f}-{Config.MAX_ATR_PCT:.5f}"
        )

    if volume_ratio < Config.MIN_VOLUME_RATIO:
        return None, (
            f"{symbol}: volume filter fail {metrics} "
            f"need >= {Config.MIN_VOLUME_RATIO:.2f}"
        )

    if momentum_pct < Config.MIN_MOMENTUM_PCT:
        return None, (
            f"{symbol}: momentum filter fail {metrics} "
            f"need >= {Config.MIN_MOMENTUM_PCT:.5f}"
        )

    if ema20_distance < Config.MIN_EMA20_DISTANCE:
        return None, (
            f"{symbol}: EMA20 distance fail {metrics} "
            f"need >= {Config.MIN_EMA20_DISTANCE:.5f}"
        )

    if body_ratio < Config.MIN_BODY_RATIO:
        return None, (
            f"{symbol}: body filter fail {metrics} "
            f"need >= {Config.MIN_BODY_RATIO:.2f}"
        )

    side = None

    # Practical demo direction logic:
    # BUY  = price above EMA50 + close breaks previous high
    # SELL = price below EMA50 + close breaks previous low
    if row["close"] > row["ema50"] and price > float(prev["high"]):
        side = "BUY"
    elif row["close"] < row["ema50"] and price < float(prev["low"]):
        side = "SELL"

    if not side:
        return None, (
            f"{symbol}: direction fail {metrics} "
            f"close={row['close']:.2f} ema50={row['ema50']:.2f} "
            f"prev_high={prev['high']:.2f} prev_low={prev['low']:.2f}"
        )

    slip = atr_val * Config.SLIPPAGE_ATR

    if side == "BUY":
        entry = price + slip
        sl = entry - Config.SL_ATR * atr_val
        tp = entry + Config.TP_ATR * atr_val
    else:
        entry = price - slip
        sl = entry + Config.SL_ATR * atr_val
        tp = entry - Config.TP_ATR * atr_val

    features = {
        "atr_pct": atr_pct,
        "volume_ratio": volume_ratio,
        "momentum_pct": momentum_pct,
        "ema20_distance": ema20_distance,
        "body_ratio": body_ratio,
    }

    ai_prob = ai_probability_placeholder(features)

    if ai_prob < Config.AI_PROB_THRESHOLD:
        return None, (
            f"{symbol}: AI score fail ai={ai_prob:.3f} "
            f"need >= {Config.AI_PROB_THRESHOLD:.2f} {metrics}"
        )

    signal = {
        "symbol": symbol,
        "time": str(t),
        "side": side,
        "price": price,
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "atr": atr_val,
        "ai_prob": ai_prob,
        **features,
    }

    return signal, f"{symbol}: SIGNAL {side} ai={ai_prob:.3f} {metrics}"


# =====================
# POSITION SIZING
# =====================
def get_quantity(symbol, balance, entry, sl):
    risk_amount = balance * Config.RISK_PER_TRADE
    stop_distance = abs(entry - sl)

    if stop_distance <= 0:
        return 0.0

    qty = risk_amount / stop_distance

    # Round quantity according to symbol stepSize.
    info = BinanceFutures.exchange_info(symbol)
    step_size = None
    min_qty = None

    for f in info["filters"]:
        if f["filterType"] == "LOT_SIZE":
            step_size = float(f["stepSize"])
            min_qty = float(f["minQty"])
            break

    if not step_size:
        return round(qty, 3)

    precision = max(0, int(round(-np.log10(step_size))))
    qty = np.floor(qty / step_size) * step_size
    qty = round(qty, precision)

    if min_qty and qty < min_qty:
        return 0.0

    return qty


# =====================
# RISK CHECK
# =====================
def can_trade(state, balance):
    reset_day_if_needed(state, balance)

    day_start = float(state["day_start_balance"])
    daily_ret = (balance - day_start) / day_start if day_start else 0

    if daily_ret <= Config.MAX_DAILY_LOSS_PCT:
        log(f"Daily stop hit: {daily_ret:.2%}")
        return False

    if state["trades_today"] >= Config.MAX_TRADES_PER_DAY:
        log("Max trades per day reached")
        return False

    open_count = 0
    for symbol in Config.SYMBOLS:
        pos = BinanceFutures.get_position(symbol)
        log(f"Position {symbol}: amount={pos['amount']} entry={pos['entry']}")
        if abs(pos["amount"]) > 0:
            open_count += 1

    if open_count >= Config.MAX_OPEN_POSITIONS:
        log(f"Max open positions reached: {open_count}/{Config.MAX_OPEN_POSITIONS}")
        return False

    return True


# =====================
# EXECUTION
# =====================
def execute_signal(signal, balance, state):
    symbol = signal["symbol"]

    pos = BinanceFutures.get_position(symbol)
    if abs(pos["amount"]) > 0:
        log(f"Skip {symbol}: already has position {pos['amount']}")
        return

    qty = get_quantity(symbol, balance, signal["entry"], signal["sl"])
    if qty <= 0:
        log(f"Skip {symbol}: qty too small")
        return

    if signal["side"] == "BUY":
        entry_side = "BUY"
        exit_side = "SELL"
    else:
        entry_side = "SELL"
        exit_side = "BUY"

    log(
        f"SIGNAL {symbol} {signal['side']} qty={qty} "
        f"entry≈{signal['entry']:.2f} sl={signal['sl']:.2f} tp={signal['tp']:.2f} "
        f"ai={signal['ai_prob']:.3f}"
    )

    entry_order = BinanceFutures.market_order(symbol, entry_side, qty)
    sl_order = BinanceFutures.stop_market_order(symbol, exit_side, signal["sl"], qty)
    tp_order = BinanceFutures.take_profit_market_order(symbol, exit_side, signal["tp"], qty)

    state["trades_today"] += 1
    save_state(state)

    append_trade_log({
        "time": now(),
        "symbol": symbol,
        "side": signal["side"],
        "qty": qty,
        "entry_ref": signal["entry"],
        "sl": signal["sl"],
        "tp": signal["tp"],
        "ai_prob": signal["ai_prob"],
        "dry_run": Config.DRY_RUN,
        "entry_order": json.dumps(entry_order),
        "sl_order": json.dumps(sl_order),
        "tp_order": json.dumps(tp_order),
    })


# =====================
# MAIN LOOP
# =====================
def setup():
    for symbol in Config.SYMBOLS:
        BinanceFutures.set_margin_type(symbol)
        BinanceFutures.set_leverage(symbol)

def main():
    log("🚀 V29 X-Ray Binance Futures Demo Bot started")
    log(f"BASE_URL={Config.BASE_URL}")
    log(f"DRY_RUN={Config.DRY_RUN}")

    setup()
    state = load_state()

    while True:
        try:
            balance = BinanceFutures.account_balance_usdt()
            reset_day_if_needed(state, balance)

            log(f"Balance USDT={balance:.2f}")

            if not can_trade(state, balance):
                time.sleep(Config.POLL_SECONDS)
                continue

            for symbol in Config.SYMBOLS:
                pos = BinanceFutures.get_position(symbol)
                if abs(pos["amount"]) > 0:
                    log(f"Skip {symbol}: position already open amount={pos['amount']} entry={pos['entry']}")
                    continue

                df = BinanceFutures.get_klines(symbol, Config.INTERVAL, Config.KLINE_LIMIT)
                signal, reason = generate_signal(symbol, df)

                if signal:
                    log(reason)
                    execute_signal(signal, balance, state)
                else:
                    log(reason)

        except Exception as e:
            log(f"ERROR: {e}")

        time.sleep(Config.POLL_SECONDS)


if __name__ == "__main__":
    main()
