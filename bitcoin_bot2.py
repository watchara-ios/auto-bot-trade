#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BTC/USDT Binance Demo Bot - Enhanced V2 Strategy + AI Close Assistant
(cleaned 4h error + env var fix)
"""

import os
import sys
import time
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
import ccxt
from openai import OpenAI

# ==================== CONFIG ====================
load_dotenv()

BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_SECRET")          # <-- ชื่อตรงกับ .env
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

if not all([BINANCE_API_KEY, BINANCE_API_SECRET, DEEPSEEK_API_KEY]):
    raise ValueError("Missing .env keys: BINANCE_API_KEY, BINANCE_SECRET, DEEPSEEK_API_KEY")

# ==================== EXCHANGE SETUP (DEMO) ====================
exchange = ccxt.binance({
    'apiKey': BINANCE_API_KEY,
    'secret': BINANCE_API_SECRET,
    'enableRateLimit': True,
    'options': {'defaultType': 'spot'},
})

# แก้ไขปัญหา sapi endpoints โดยเปลี่ยนเฉพาะโดเมนเป็น demo.binance.com
for key in exchange.urls['api']:
    if isinstance(exchange.urls['api'][key], str):
        exchange.urls['api'][key] = exchange.urls['api'][key].replace('api.binance.com', 'demo.binance.com')
    elif isinstance(exchange.urls['api'][key], dict):
        for sub_key in exchange.urls['api'][key]:
            if isinstance(exchange.urls['api'][key][sub_key], str):
                exchange.urls['api'][key][sub_key] = exchange.urls['api'][key][sub_key].replace('api.binance.com', 'demo.binance.com')

# ทดสอบการเชื่อมต่อ
try:
    server_time = exchange.fetch_time()
    logger = logging.getLogger(__name__)
    print(f"✅ Connected to Binance Demo - Server time: {datetime.fromtimestamp(server_time/1000)}")
except Exception as e:
    print(f"❌ Demo connection failed: {e}")
    sys.exit(1)

# ==================== DEEPSEEK AI ====================
deepseek_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
DEEPSEEK_MODEL = "deepseek-chat"

# ==================== PARAMETERS ====================
SYMBOL = "BTC/USDT"

# Strategy (from best backtest)
FAST_EMA = 12
SLOW_EMA = 30
RSI_PERIOD = 14
RSI_LOW = 35
RSI_HIGH = 70
ADX_THRESH = 22
USE_VOLUME_FILTER = False
VOL_MA_PERIOD = 20
SL_ATR_MULT = 1.8
TP_ATR_MULT = 4.5

# Risk Management
MAX_DAILY_PROFIT_PCT = 20.0
MAX_DAILY_LOSS_PCT = 15.0
RISK_PER_TRADE = 0.015        # 1.5% of balance per trade
MAX_POSITIONS = 1

# Time intervals
AI_REVIEW_INTERVAL = 30       # seconds
FULL_REVIEW_INTERVAL = 3600   # hour

# ==================== LOGGING ====================
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, f"demo_bot_{datetime.now().strftime('%Y%m%d')}.log"), encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ==================== INDICATORS ====================
def compute_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df['high'], df['low'], df['close']
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df['high'], df['low'], df['close']
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    return dx.rolling(period).mean()

# ==================== DATA FETCHING ====================
def fetch_ohlcv(timeframe, limit=100):
    """Fetch OHLCV data. Returns None on failure."""
    try:
        data = exchange.fetch_ohlcv(SYMBOL, timeframe, limit=limit)
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        logger.error(f"Error fetching {timeframe}: {e}")
        return None

def fetch_ohlcv_silent(timeframe, limit=100):
    """Fetch OHLCV but log errors at DEBUG level only."""
    try:
        return fetch_ohlcv(timeframe, limit)
    except Exception as e:
        logger.debug(f"Silent fail for {timeframe}: {e}")
        return None

# ==================== STRATEGY SIGNAL ====================
def get_signal() -> str | None:
    # 1h trend
    df_1h = fetch_ohlcv('1h', 100)
    if df_1h is None: return None
    ema50_1h = compute_ema(df_1h['close'], 50)
    h1_uptrend = df_1h['close'].iloc[-1] > ema50_1h.iloc[-1]

    # 4h trend (optional – silent fail)
    h4_uptrend = True
    df_4h = fetch_ohlcv_silent('4h', 100)   # <-- ใช้ silent version
    if df_4h is not None and len(df_4h) >= 50:
        ema50_4h = compute_ema(df_4h['close'], 50)
        h4_uptrend = df_4h['close'].iloc[-1] > ema50_4h.iloc[-1]

    # 5m entry
    df_5m = fetch_ohlcv('5m', 100)
    if df_5m is None: return None
    fast = compute_ema(df_5m['close'], FAST_EMA)
    slow = compute_ema(df_5m['close'], SLOW_EMA)
    rsi = compute_rsi(df_5m['close'], RSI_PERIOD)
    adx = compute_adx(df_5m, 14)

    latest = len(df_5m)-1
    prev = latest-1

    # Volume filter (off by default)
    if USE_VOLUME_FILTER:
        vol_ma = df_5m['volume'].rolling(VOL_MA_PERIOD).mean()
        vol_ok = df_5m['volume'].iloc[latest] > vol_ma.iloc[latest]
    else:
        vol_ok = True

    buy_cond = (h1_uptrend and h4_uptrend and
                fast.iloc[latest] > slow.iloc[latest] and
                fast.iloc[prev] <= slow.iloc[prev] and
                rsi.iloc[latest] < RSI_LOW and
                adx.iloc[latest] > ADX_THRESH and
                vol_ok)

    if buy_cond:
        return "BUY"
    return None

# ==================== POSITION SIZING ====================
def calculate_qty(balance_usdt: float, atr: float, entry_price: float) -> float:
    risk_amount = balance_usdt * RISK_PER_TRADE
    stop_distance = atr * SL_ATR_MULT
    if stop_distance <= 0:
        return 0.0
    qty = risk_amount / stop_distance
    min_qty = 10.0 / entry_price if entry_price > 0 else 0.0
    max_qty = balance_usdt * 0.95 / entry_price
    qty = max(min_qty, min(qty, max_qty))
    qty = np.floor(qty * 1e6) / 1e6
    return qty

# ==================== ORDER EXECUTION ====================
def execute_buy():
    try:
        ticker = exchange.fetch_ticker(SYMBOL)
        current_price = ticker['last']
        df_5m = fetch_ohlcv('5m', 30)
        if df_5m is None: return False
        atr_val = compute_atr(df_5m, 14).iloc[-1]

        balance_info = exchange.fetch_balance()
        usdt_balance = balance_info['USDT']['free']
        qty = calculate_qty(usdt_balance, atr_val, current_price)
        if qty <= 0:
            logger.warning("Quantity is zero")
            return False

        buy_order = exchange.create_order(SYMBOL, 'market', 'buy', qty)
        logger.info(f"✅ BUY {buy_order['filled']} BTC @ ~{buy_order['average']}")

        entry_price = buy_order['average']
        sl_price = entry_price - atr_val * SL_ATR_MULT
        tp_price = entry_price + atr_val * TP_ATR_MULT

        exchange.create_order(SYMBOL, 'stop_loss_limit', 'sell', qty,
                              sl_price, {'stopPrice': sl_price,
                                         'price': round(sl_price * 0.999, 2)})
        exchange.create_order(SYMBOL, 'limit', 'sell', qty, round(tp_price, 2))
        logger.info(f"🔒 SL={sl_price:.2f} | TP={tp_price:.2f}")
        return True
    except Exception as e:
        logger.error(f"Buy execution failed: {e}")
        return False

# ==================== POSITION TRACKING ====================
last_entry_price = 0.0

def has_position() -> bool:
    try:
        balance = exchange.fetch_balance()
        btc = balance['BTC']['free']
        return btc > 0.00001
    except:
        return False

def get_position_details():
    global last_entry_price
    try:
        ticker = exchange.fetch_ticker(SYMBOL)
        balance = exchange.fetch_balance()
        btc = balance['BTC']['free']
        if btc > 0 and last_entry_price > 0:
            current_price = ticker['last']
            profit = (current_price - last_entry_price) * btc
            return {
                'symbol': SYMBOL,
                'qty': btc,
                'entry': last_entry_price,
                'current': current_price,
                'profit': profit
            }
        return None
    except Exception as e:
        logger.error(f"get_position_details error: {e}")
        return None

# ==================== AI REVIEW (DEEPSEEK) ====================
def ai_review_position():
    details = get_position_details()
    if not details:
        return

    market_info = ""
    for tf in ["5m", "15m", "1h"]:
        df = fetch_ohlcv(tf, 50)
        if df is not None and len(df) >= 20:
            ema20 = compute_ema(df['close'], 20)
            last_cls = df['close'].iloc[-1]
            last_ema = ema20.iloc[-1]
            direction = "Price>EMA" if last_cls > last_ema else "Price<EMA"
            market_info += f"{tf}: Close={last_cls:.2f}, EMA20={last_ema:.2f} ({direction}); "

    entry = details['entry']
    current = details['current']
    profit = details['profit']
    pips = (current - entry) * 100

    prompt = f"""You are a crypto position manager. Decide NOW: CLOSE (exit) or HOLD.

Symbol: {SYMBOL}
Side: BUY
Entry: {entry:.2f}
Current: {current:.2f}
Pips: {pips:.1f}
Profit: {profit:.2f} USD

Market condition:
{market_info}

Decision rules:
- CLOSE if trend reversed (price below EMA20 on both M5 and H1)
- CLOSE if profit > 30 pips and momentum seems weak
- CLOSE if loss < -15 pips with no recovery sign
- Otherwise HOLD

Reply ONLY with JSON: {{"action":"CLOSE" or "HOLD", "confidence":0-100, "reason":"short reason"}}
"""
    try:
        resp = deepseek_client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{"role":"system","content":"You are a trading assistant. Reply only with valid JSON."},
                      {"role":"user","content":prompt}],
            temperature=0.1,
            max_tokens=150,
            response_format={"type":"json_object"}
        )
        content = resp.choices[0].message.content.strip()
        content = content.replace("```json","").replace("```","").strip()
        decision = json.loads(content)
        action = decision.get("action","HOLD").upper()
        reason = decision.get("reason","")
        logger.info(f"🤖 AI Decision: {action} | {reason}")
        if action == "CLOSE":
            close_all_positions()
    except Exception as e:
        logger.error(f"AI Review error: {e}")

def close_all_positions():
    global last_entry_price
    try:
        balance = exchange.fetch_balance()
        btc_qty = balance['BTC']['free']
        if btc_qty > 0.00001:
            open_orders = exchange.fetch_open_orders(SYMBOL)
            for order in open_orders:
                exchange.cancel_order(order['id'], SYMBOL)
            exchange.create_order(SYMBOL, 'market', 'sell', btc_qty)
            logger.info(f"🔴 Closed position: sold {btc_qty} BTC")
            last_entry_price = 0.0
    except Exception as e:
        logger.error(f"Close failed: {e}")

# ==================== MAIN LOOP ====================
def main():
    global last_entry_price
    logger.info("🚀 Binance Demo BTC Bot started (V2 + AI)")
    last_ai_time = time.time()
    last_full_time = time.time()

    while True:
        try:
            now = time.time()

            # AI Quick Review every 30s if holding
            if now - last_ai_time >= AI_REVIEW_INTERVAL:
                if has_position():
                    ai_review_position()
                last_ai_time = now

            # Full Review every hour
            if now - last_full_time >= FULL_REVIEW_INTERVAL:
                logger.info("=== Hourly Full Review ===")
                if has_position():
                    ai_review_position()
                else:
                    signal = get_signal()
                    if signal == "BUY":
                        logger.info("BUY signal detected, opening...")
                        if execute_buy():
                            trades = exchange.fetch_my_trades(SYMBOL, limit=1)
                            if trades:
                                last_entry_price = trades[-1]['price']
                last_full_time = now

            # Idle check every loop (if no position)
            if not has_position():
                signal = get_signal()
                if signal == "BUY":
                    logger.info("Idle BUY signal, opening...")
                    if execute_buy():
                        trades = exchange.fetch_my_trades(SYMBOL, limit=1)
                        if trades:
                            last_entry_price = trades[-1]['price']

            time.sleep(1)

        except KeyboardInterrupt:
            logger.info("⏹️ Bot stopped by user")
            break
        except Exception as e:
            logger.exception(f"Unexpected error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()