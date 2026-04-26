import os
import time
import hmac
import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


load_dotenv()


class Config:
    API_KEY = os.getenv("BINANCE_API_KEY")
    SECRET = os.getenv("BINANCE_SECRET")
    BASE_URL = "https://demo-fapi.binance.com"

    SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    TIMEFRAMES = ("1m", "5m", "15m", "1h")
    KLINE_LIMIT = 250

    DRY_RUN = os.getenv("HYBRID_DRY_RUN", "true").lower() == "true"
    LEVERAGE = 1
    MARGIN_TYPE = "ISOLATED"

    POLL_SECONDS = 30
    ENTRY_COOLDOWN = 300
    MAX_OPEN_SYMBOLS = 3
    MAX_TRADES_PER_DAY = 6
    MAX_DAILY_LOSS_PCT = 0.04
    MAX_TOTAL_EXPOSURE_PCT = 60.0
    RISK_PER_TRADE = 0.006

    MIN_SCORE = 58
    STRONG_SCORE = 72
    MIN_ATR_PCT = 0.0010
    MAX_ATR_PCT = 0.0090
    MIN_VOLUME_RATIO = 0.65
    BREAKOUT_LOOKBACK = 20

    SL_ATR = 1.25
    TP_ATR_NORMAL = 1.8
    TP_ATR_STRONG = 2.4
    TRAILING_ATR = 0.9

    USE_AI_VALIDATOR = os.getenv("HYBRID_USE_AI", "false").lower() == "true"
    DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY")
    AI_MODEL = "deepseek-chat"

    LOG_FILE = "hybrid_bot.log"
    TRADE_LOG = "hybrid_trades.csv"
    STATE_FILE = "hybrid_state.json"


logging.basicConfig(
    filename=Config.LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")
    logging.info(msg)


def warn(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] WARNING {msg}")
    logging.warning(msg)


class Binance:
    time_offset = 0

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
        r = requests.get(f"{Config.BASE_URL}{path}", params=params or {}, timeout=15)
        r.raise_for_status()
        return r.json()

    @staticmethod
    def signed(method, path, params=None):
        params = params or {}
        params["timestamp"] = int(time.time() * 1000) + Binance.time_offset
        params["recvWindow"] = 10000
        url = f"{Config.BASE_URL}{path}?{Binance.sign(params)}"
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
    def exchange_info(symbol):
        data = Binance.public_get("/fapi/v1/exchangeInfo")
        for item in data["symbols"]:
            if item["symbol"] == symbol:
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
    def stop_order(symbol, side, stop_price, qty, order_type):
        params = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "stopPrice": round(stop_price, 2),
            "quantity": qty,
            "reduceOnly": "true",
            "workingType": "MARK_PRICE",
        }
        if Config.DRY_RUN:
            log(f"🧪 DRY_RUN {order_type} {params}")
            return {"dry_run": True, **params}
        return Binance.signed("POST", "/fapi/v1/order", params)


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
    return {tf: prepare(Binance.get_klines(symbol, tf, Config.KLINE_LIMIT)) for tf in Config.TIMEFRAMES}


def last_closed(df):
    return df.iloc[-2]


def previous(df):
    return df.iloc[-3]


def summarize_timeframes(symbol, market):
    parts = []
    for tf in Config.TIMEFRAMES:
        row = last_closed(market[tf])
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


def generate_signal(symbol, market):
    row_5m = last_closed(market["5m"])
    prev_5m = previous(market["5m"])
    row_15m = last_closed(market["15m"])
    row_1h = last_closed(market["1h"])

    if any(pd.isna(row_5m[x]) for x in ["atr", "atr_pct", "vol_ratio", "rsi", "ema20"]):
        return None, "5m indicators not ready"
    if any(pd.isna(row_15m[x]) for x in ["ema20", "ema50", "rsi"]):
        return None, "15m indicators not ready"
    if any(pd.isna(row_1h[x]) for x in ["ema20", "ema50"]):
        return None, "1h indicators not ready"

    atr_pct = float(row_5m.atr_pct)
    if not (Config.MIN_ATR_PCT <= atr_pct <= Config.MAX_ATR_PCT):
        return None, f"ATR filter fail atr%={atr_pct*100:.2f}"

    side = trend_side(row_15m, row_1h)
    signal_type = "trend"
    reasons = []

    breakout, breakout_reason = breakout_side(market["15m"])
    if breakout:
        side = breakout
        signal_type = "breakout"
        reasons.append(breakout_reason)

    if side is None:
        return None, "15m/1h trend not aligned"

    momentum = (row_5m.close - prev_5m.close) / prev_5m.close
    if side == "BUY":
        entry_ok = row_5m.close > row_5m.ema20 and momentum > -0.0002 and row_5m.rsi < 76
        reclaim = prev_5m.close <= prev_5m.ema20 and row_5m.close > row_5m.ema20
    else:
        entry_ok = row_5m.close < row_5m.ema20 and momentum < 0.0002 and row_5m.rsi > 24
        reclaim = prev_5m.close >= prev_5m.ema20 and row_5m.close < row_5m.ema20

    if reclaim and signal_type == "trend":
        signal_type = "reclaim"
        reasons.append("5m reclaimed EMA20 in aligned 15m/1h trend")

    if not entry_ok:
        return None, f"5m entry not ready side={side} close={row_5m.close:.2f} ema20={row_5m.ema20:.2f} momentum={momentum:.5f}"

    score = 45
    score += 12 if signal_type == "breakout" else 8 if signal_type == "reclaim" else 5
    score += 8 if row_5m.vol_ratio >= 1.0 else 4 if row_5m.vol_ratio >= Config.MIN_VOLUME_RATIO else -8
    score += 7 if abs(momentum) >= 0.0008 else 3 if abs(momentum) >= 0.00025 else 0
    score += 8 if row_15m.close > row_15m.ema20 > row_15m.ema50 and side == "BUY" else 0
    score += 8 if row_15m.close < row_15m.ema20 < row_15m.ema50 and side == "SELL" else 0
    score += 7 if row_1h.close > row_1h.ema50 and side == "BUY" else 0
    score += 7 if row_1h.close < row_1h.ema50 and side == "SELL" else 0
    score -= 6 if row_5m.vol_ratio < Config.MIN_VOLUME_RATIO else 0

    score = max(0, min(100, score))
    if score < Config.MIN_SCORE:
        return None, f"score too low {score} side={side} type={signal_type}"

    rr = Config.TP_ATR_STRONG if score >= Config.STRONG_SCORE else Config.TP_ATR_NORMAL
    entry = float(row_5m.close)
    stop_distance = float(row_5m.atr) * Config.SL_ATR
    take_distance = float(row_5m.atr) * rr
    if side == "BUY":
        sl = entry - stop_distance
        tp = entry + take_distance
        exit_side = "SELL"
    else:
        sl = entry + stop_distance
        tp = entry - take_distance
        exit_side = "BUY"

    signal = {
        "symbol": symbol,
        "side": side,
        "exit_side": exit_side,
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "atr": float(row_5m.atr),
        "atr_pct": atr_pct,
        "score": score,
        "type": signal_type,
        "reason": "; ".join(reasons) or "aligned trend",
    }
    return signal, f"SIGNAL {side} score={score} type={signal_type} entry={entry:.2f} sl={sl:.2f} tp={tp:.2f}"


def ai_validate(signal, market):
    if not Config.USE_AI_VALIDATOR or not Config.DEEPSEEK_KEY or OpenAI is None:
        return True, "AI disabled"
    if signal["score"] >= Config.STRONG_SCORE:
        return True, "strong technical score"
    try:
        client = OpenAI(api_key=Config.DEEPSEEK_KEY, base_url="https://api.deepseek.com/v1")
        row_5m = last_closed(market["5m"])
        row_15m = last_closed(market["15m"])
        row_1h = last_closed(market["1h"])
        prompt = f"""Validate this crypto futures setup. Return strict JSON only.
Symbol: {signal['symbol']}
Side: {signal['side']}
Type: {signal['type']}
Score: {signal['score']}
5m close={row_5m.close:.2f} ema20={row_5m.ema20:.2f} rsi={row_5m.rsi:.1f} atr_pct={row_5m.atr_pct:.4f} vol_ratio={row_5m.vol_ratio:.2f}
15m close={row_15m.close:.2f} ema20={row_15m.ema20:.2f} ema50={row_15m.ema50:.2f} rsi={row_15m.rsi:.1f}
1h close={row_1h.close:.2f} ema20={row_1h.ema20:.2f} ema50={row_1h.ema50:.2f} ema200={row_1h.ema200:.2f}
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
    state = {"date": datetime.now().strftime("%Y-%m-%d"), "day_start_balance": None, "trades_today": 0, "last_trade_time": {}}
    save_state(state)
    return state


def save_state(state):
    Path(Config.STATE_FILE).write_text(json.dumps(state, indent=2), encoding="utf-8")


def reset_day(state, balance):
    today = datetime.now().strftime("%Y-%m-%d")
    if state.get("date") != today:
        state["date"] = today
        state["day_start_balance"] = balance
        state["trades_today"] = 0
        state["last_trade_time"] = {}
        save_state(state)
    if state.get("day_start_balance") is None:
        state["day_start_balance"] = balance
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
    risk_amount = balance * Config.RISK_PER_TRADE
    stop_distance = abs(signal["entry"] - signal["sl"])
    if stop_distance <= 0:
        return 0.0
    qty = risk_amount / stop_distance
    return round_qty(symbol, qty)


def account_snapshot(balance):
    open_symbols = []
    exposure = 0.0
    for symbol in Config.SYMBOLS:
        pos = Binance.position(symbol)
        amt = abs(pos["amount"])
        if amt > 0:
            open_symbols.append(symbol)
            exposure += amt * (pos["mark"] or pos["entry"]) / max(balance, 1) * 100
    return open_symbols, exposure


def can_open_new(state, balance):
    day_start = float(state.get("day_start_balance") or balance)
    daily_ret = (balance - day_start) / day_start if day_start else 0.0
    if daily_ret <= -Config.MAX_DAILY_LOSS_PCT:
        return False, f"daily loss stop {daily_ret:.2%}"
    if state.get("trades_today", 0) >= Config.MAX_TRADES_PER_DAY:
        return False, "max trades per day reached"
    open_symbols, exposure = account_snapshot(balance)
    if len(open_symbols) >= Config.MAX_OPEN_SYMBOLS:
        return False, f"max open symbols {len(open_symbols)}/{Config.MAX_OPEN_SYMBOLS}"
    if exposure >= Config.MAX_TOTAL_EXPOSURE_PCT:
        return False, f"total exposure {exposure:.2f}% >= {Config.MAX_TOTAL_EXPOSURE_PCT}%"
    return True, "ok"


def ensure_protection(symbol, pos, market):
    amt = abs(pos["amount"])
    if amt <= 0:
        return
    try:
        orders = Binance.open_orders(symbol) if not Config.DRY_RUN else []
        has_reduce = any(str(o.get("reduceOnly")).lower() == "true" for o in orders)
        if has_reduce:
            log(f"🛡️ {symbol} position protected by reduce-only open order")
            return
    except Exception as e:
        warn(f"🛡️ {symbol} cannot inspect open orders: {e}")
        return

    row_5m = last_closed(market["5m"])
    entry = pos["entry"] or pos["mark"] or row_5m.close
    side = "BUY" if pos["amount"] > 0 else "SELL"
    exit_side = "SELL" if side == "BUY" else "BUY"
    atr_val = float(row_5m.atr) if not pd.isna(row_5m.atr) else entry * 0.003
    if side == "BUY":
        sl = entry - atr_val * Config.SL_ATR
        tp = entry + atr_val * Config.TP_ATR_NORMAL
    else:
        sl = entry + atr_val * Config.SL_ATR
        tp = entry - atr_val * Config.TP_ATR_NORMAL
    qty = round_qty(symbol, amt)
    warn(f"🛡️ {symbol} has unprotected position; placing emergency SL/TP side={side} qty={qty} sl={sl:.2f} tp={tp:.2f}")
    Binance.stop_order(symbol, exit_side, sl, qty, "STOP_MARKET")
    Binance.stop_order(symbol, exit_side, tp, qty, "TAKE_PROFIT_MARKET")


def execute_signal(signal, balance, state):
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

    allowed, reason = ai_validate(signal, fetch_market(symbol))
    if not allowed:
        log(f"🧠 {symbol} AI blocked signal: {reason}")
        return

    log(
        f"🚀 OPEN {symbol} {signal['side']} qty={qty} score={signal['score']} "
        f"type={signal['type']} entry~{signal['entry']:.2f} sl={signal['sl']:.2f} tp={signal['tp']:.2f}"
    )
    Binance.market_order(symbol, signal["side"], qty)
    Binance.stop_order(symbol, signal["exit_side"], signal["sl"], qty, "STOP_MARKET")
    Binance.stop_order(symbol, signal["exit_side"], signal["tp"], qty, "TAKE_PROFIT_MARKET")

    state["trades_today"] = state.get("trades_today", 0) + 1
    state.setdefault("last_trade_time", {})[symbol] = time.time()
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
    log("🤖 Hybrid BTC/ETH/SOL bot started")
    log(f"🧪 DRY_RUN={Config.DRY_RUN} | 🧠 AI_VALIDATOR={Config.USE_AI_VALIDATOR} | symbols={','.join(Config.SYMBOLS)}")
    log("═" * 64)
    setup()
    state = load_state()

    while True:
        try:
            balance = Binance.balance()
            reset_day(state, balance)
            log(f"📊 Balance={balance:.2f} | trades_today={state.get('trades_today', 0)}")

            open_allowed, open_reason = can_open_new(state, balance)
            if not open_allowed:
                log(f"🚦 New entries paused: {open_reason}. Existing positions still managed.")

            for symbol in Config.SYMBOLS:
                log(f"🔎 --- {symbol} ---")
                market = fetch_market(symbol)
                summarize_timeframes(symbol, market)

                pos = Binance.position(symbol)
                if abs(pos["amount"]) > 0:
                    log(f"📌 {symbol} existing position amount={pos['amount']} entry={pos['entry']} mark={pos['mark']}")
                    ensure_protection(symbol, pos, market)
                    continue

                if not open_allowed:
                    continue

                signal, reason = generate_signal(symbol, market)
                if not signal:
                    log(f"⏸️ {symbol} no entry: {reason}")
                    continue
                log(f"✅ {symbol} {reason}")
                execute_signal(signal, balance, state)

        except KeyboardInterrupt:
            warn("🛑 Bot stopped by user")
            break
        except Exception as e:
            warn(f"💥 Loop error: {e}")
            if "-1021" in str(e):
                Binance.sync_time()

        time.sleep(Config.POLL_SECONDS)


if __name__ == "__main__":
    main()
