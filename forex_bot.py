#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MT5 Forex Multi-Pair Bot - Enhanced V2 Strategy + AI Close Assistant
- ดึงข้อมูลจาก MT5 (Forex)
- วิเคราะห์สัญญาณซื้อจาก EMA/RSI/ADX/Trend (5m/1h/4h)
- เลือกคู่เงินที่สัญญาณดีที่สุดตาม Score
- AI Review ทุก 30 วินาที ช่วยตัดสินใจปิดก่อน
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
import MetaTrader5 as mt5
from openai import OpenAI

# ==================== CONFIG ====================
load_dotenv()

MT5_LOGIN = int(os.getenv("MT5_LOGIN"))
MT5_PASSWORD = os.getenv("MT5_PASSWORD")
MT5_SERVER = os.getenv("MT5_SERVER")
MT5_PATH = os.getenv("MT5_PATH", "")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    raise ValueError("Missing DEEPSEEK_API_KEY in .env")
deepseek_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
DEEPSEEK_MODEL = "deepseek-chat"

# รายชื่อคู่เงิน (Base names; โค้ดจะหาชื่อจริงที่มี suffix เช่น EURUSD.sml โดยอัตโนมัติ)
FOREX_WATCHLIST = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD",
    "USDCAD", "NZDUSD", "EURGBP", "EURJPY", "GBPJPY"
]

# Strategy parameters (จาก Backtest ที่ดีที่สุด)
FAST_EMA = 12
SLOW_EMA = 30
RSI_PERIOD = 14
RSI_LOW = 35
RSI_HIGH = 70
ADX_THRESH = 22
USE_VOLUME_FILTER = False       # Forex volume ไม่ใช้ดีกว่า
VOL_MA_PERIOD = 20
SL_ATR_MULT = 1.8
TP_ATR_MULT = 4.5

# Risk & Trade Management
MAX_DAILY_PROFIT_PERCENT = 20.0
MAX_DAILY_LOSS_PERCENT = 15.0
RISK_PER_TRADE = 0.015          # 1.5% ของ Balance
MAX_CONCURRENT_TRADES = 2       # เปิดได้สูงสุด 2 คู่พร้อมกัน
DEVIATION = 20
MAGIC_NUMBER = 654321

# AI Review intervals
AI_REVIEW_INTERVAL = 30         # วินาที
FULL_REVIEW_INTERVAL = 3600     # ชั่วโมง

# Logging
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, f"mt5_bot_{datetime.now().strftime('%Y%m%d')}.log"), encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ==================== MT5 HELPERS ====================
def connect_mt5():
    if MT5_PATH and os.path.exists(MT5_PATH):
        ok = mt5.initialize(path=MT5_PATH, login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER)
    else:
        ok = mt5.initialize(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER)
    if not ok:
        logger.error(f"MT5 init failed: {mt5.last_error()}")
        return False
    logger.info("Connected to MT5")
    # enable all watchlist symbols
    all_symbols = mt5.symbols_get()
    if not all_symbols:
        logger.error("No symbols found")
        return False
    for base in FOREX_WATCHLIST:
        found = None
        for s in all_symbols:
            if s.name == base or s.name.startswith(base + ".") or s.name == base + ".sml":
                found = s.name
                break
        if found and mt5.symbol_select(found, True):
            logger.info(f"Enabled {found}")
        elif found:
            logger.warning(f"Could not select {found}")
        else:
            logger.warning(f"{base} not found")
    return True

def get_account_info():
    acc = mt5.account_info()
    if acc is None:
        return None
    today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    history = mt5.history_deals_get(today_start, datetime.now())
    daily_profit = sum(deal.profit for deal in history) if history else 0.0
    pct = (daily_profit / acc.balance) * 100 if acc.balance > 0 else 0.0
    return {"balance": acc.balance, "equity": acc.equity, "daily_profit": daily_profit, "daily_pnl_pct": pct}

def get_open_positions():
    return mt5.positions_get()

def close_position(position, comment="AI_Close"):
    symbol = position.symbol
    tick = mt5.symbol_info_tick(symbol)
    if not tick: return False
    if position.type == mt5.POSITION_TYPE_BUY:
        order_type = mt5.ORDER_TYPE_SELL
        price = tick.bid
    else:
        order_type = mt5.ORDER_TYPE_BUY
        price = tick.ask
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": position.volume,
        "type": order_type,
        "position": position.ticket,
        "price": price,
        "deviation": DEVIATION,
        "magic": MAGIC_NUMBER,
        "comment": comment[:30],
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.error(f"Close failed: {result.comment}")
        return False
    logger.info(f"Closed {symbol} PnL: {position.profit:.2f}")
    return True

# ==================== INDICATORS (on DataFrame) ====================
def compute_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_atr(df, period=14):
    high, low, close = df['high'], df['low'], df['close']
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def compute_adx(df, period=14):
    high, low, close = df['high'], df['low'], df['close']
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    return dx.rolling(period).mean()

# ==================== STRATEGY SIGNAL + SCORE ====================
def analyze_symbol(symbol):
    """
    Returns dict { 'signal': 'BUY' or None, 'score': float, 'atr': float } or None if data insufficient.
    score: higher = stronger setup (e.g., ADX + RSI oversold strength)
    """
    # Fetch 5m data (100 bars)
    rates_5m = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 100)
    if rates_5m is None or len(rates_5m) < 50:
        return None
    df_5m = pd.DataFrame(rates_5m)
    # Compute indicators
    fast = compute_ema(df_5m['close'], FAST_EMA)
    slow = compute_ema(df_5m['close'], SLOW_EMA)
    rsi = compute_rsi(df_5m['close'], RSI_PERIOD)
    adx = compute_adx(df_5m, 14)
    atr = compute_atr(df_5m, 14)

    latest = len(df_5m)-1
    prev = latest-1

    # Entry condition on 5m
    if not (fast.iloc[latest] > slow.iloc[latest] and fast.iloc[prev] <= slow.iloc[prev]):
        return None
    if not (rsi.iloc[latest] < RSI_LOW and adx.iloc[latest] > ADX_THRESH):
        return None

    # 1h trend
    rates_1h = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 100)
    if rates_1h is None or len(rates_1h) < 50:
        return None
    df_1h = pd.DataFrame(rates_1h)
    ema50_1h = compute_ema(df_1h['close'], 50)
    if df_1h['close'].iloc[-1] <= ema50_1h.iloc[-1]:
        return None  # must be above 1h EMA50

    # 4h trend (optional)
    try:
        rates_4h = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H4, 0, 100)
        if rates_4h is not None and len(rates_4h) >= 50:
            df_4h = pd.DataFrame(rates_4h)
            ema50_4h = compute_ema(df_4h['close'], 50)
            if df_4h['close'].iloc[-1] <= ema50_4h.iloc[-1]:
                return None
    except:
        pass  # if H4 not available, skip this filter

    # Calculate score: combination of ADX strength and RSI oversold depth
    score = (adx.iloc[latest] - 20) * 0.5 + (35 - rsi.iloc[latest]) * 2   # tune weights
    return {
        'signal': 'BUY',
        'score': score,
        'atr': atr.iloc[latest]
    }

def select_best_symbols(max_picks):
    """วิเคราะห์ทุกคู่ใน watchlist แล้วเลือก MAX_CONCURRENT_TRADES คู่ที่คะแนนสูงสุด"""
    candidates = []
    for sym in FOREX_WATCHLIST:
        # หาชื่อจริงของ symbol
        found = None
        for s in mt5.symbols_get():
            if s.name == sym or s.name.startswith(sym + ".") or s.name == sym + ".sml":
                found = s.name
                break
        if not found:
            continue
        # ตรวจสอบว่ามี position เปิดอยู่แล้วหรือไม่
        if mt5.positions_get(symbol=found):
            continue
        analysis = analyze_symbol(found)
        if analysis:
            candidates.append((found, analysis))
    # เรียงตามคะแนนจากมากไปน้อย
    candidates.sort(key=lambda x: x[1]['score'], reverse=True)
    best = candidates[:max_picks]
    logger.info(f"🔍 Scanned {len(FOREX_WATCHLIST)} pairs, found {len(candidates)} signals. Best: {[c[0] for c in best]}")
    return best

# ==================== ORDER EXECUTION ====================
def calculate_lot(symbol, atr, account_balance):
    """คำนวณ lot size ตามความเสี่ยง 1.5% ของพอร์ต"""
    risk_amount = account_balance * RISK_PER_TRADE
    tick_value = mt5.symbol_info(symbol).trade_tick_value
    if tick_value is None or tick_value == 0:
        lot = 0.01
    else:
        stop_distance_price = atr * SL_ATR_MULT
        # มูลค่าเป็นเงินต่อการเคลื่อนไหว 1 pip (สำหรับ 1 lot)
        # แต่เราใช้ atr เป็น price distance, คูณกับ trade_tick_value / tick_size
        # สมมติว่า 1 pip = 10 points, trade_tick_value อาจจะเป็นต่อ point
        # เพื่อความง่ายใช้ stop_distance * 10 เป็น pips แล้วคำนวณ lot
        lot = risk_amount / (stop_distance_price * 10)  # empirical
    lot = max(0.01, min(lot, 0.1))
    step = mt5.symbol_info(symbol).volume_step
    lot = round(lot / step) * step
    return round(lot, 2)

def execute_buy(symbol, analysis):
    """เปิด BUY market order พร้อม SL/TP"""
    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        return False
    entry = tick.ask
    atr = analysis['atr']
    sl = entry - atr * SL_ATR_MULT
    tp = entry + atr * TP_ATR_MULT

    acc = get_account_info()
    if not acc:
        return False
    lot = calculate_lot(symbol, atr, acc['balance'])

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": mt5.ORDER_TYPE_BUY,
        "price": entry,
        "sl": sl,
        "tp": tp,
        "deviation": DEVIATION,
        "magic": MAGIC_NUMBER,
        "comment": f"V2_{datetime.now().strftime('%H%M')}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.error(f"BUY {symbol} failed: {result.comment}")
        return False
    logger.info(f"✅ BUY {symbol} {lot} lot @ {entry}, SL={sl:.5f}, TP={tp:.5f}")
    return True

# ==================== AI REVIEW FOR ALL OPEN POSITIONS ====================
def ai_review_all_positions():
    positions = get_open_positions()
    if not positions:
        return
    for pos in positions:
        sym = pos.symbol
        tick = mt5.symbol_info_tick(sym)
        if not tick:
            continue
        # รวบรวมข้อมูล EMA20 สำหรับ 5m, 15m, 1h
        market_info = ""
        for tf, mt5_tf in [("M5", mt5.TIMEFRAME_M5), ("M15", mt5.TIMEFRAME_M15), ("H1", mt5.TIMEFRAME_H1)]:
            rates = mt5.copy_rates_from_pos(sym, mt5_tf, 0, 30)
            if rates is not None and len(rates) >= 20:
                df = pd.DataFrame(rates)
                ema20 = compute_ema(df['close'], 20)
                last_close = df['close'].iloc[-1]
                last_ema = ema20.iloc[-1]
                dir_flag = "Price>EMA" if last_close > last_ema else "Price<EMA"
                market_info += f"{tf}: C={last_close:.5f}, EMA20={last_ema:.5f} ({dir_flag}); "

        profit = pos.profit
        entry = pos.price_open
        current = tick.bid if pos.type == mt5.POSITION_TYPE_BUY else tick.ask
        # คำนวณ pips คร่าว ๆ (ใช้ symbol info)
        pip_size = 10 * mt5.symbol_info(sym).point
        pips = (current - entry) / pip_size if pos.type == mt5.POSITION_TYPE_BUY else (entry - current) / pip_size

        prompt = f"""You are a forex position manager. Decide: CLOSE or HOLD.

Symbol: {sym}
Side: {"BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL"}
Entry: {entry:.5f}
Current: {current:.5f}
Pips: {pips:.1f}
Profit: {profit:.2f} USD

Market conditions:
{market_info}

Rules:
- CLOSE if trend reversed (price below EMA20 on M15 and H1 for BUY)
- CLOSE if profit > 30 pips and momentum weakening
- CLOSE if loss < -15 pips and no sign of recovery
- Otherwise HOLD

Reply ONLY JSON: {{"action": "CLOSE" or "HOLD", "confidence": 0-100, "reason": "short reason"}}
"""
        try:
            response = deepseek_client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[{"role":"system","content":"You are a trading assistant. Reply only with valid JSON."},
                          {"role":"user","content":prompt}],
                temperature=0.1,
                max_tokens=150,
                response_format={"type":"json_object"}
            )
            content = response.choices[0].message.content.strip()
            content = content.replace("```json","").replace("```","").strip()
            decision = json.loads(content)
            action = decision.get("action","HOLD").upper()
            reason = decision.get("reason","")
            logger.info(f"🤖 AI {sym}: {action} ({reason})")
            if action == "CLOSE":
                close_position(pos, comment=f"AI_{reason[:20]}")
        except Exception as e:
            logger.error(f"AI review failed for {sym}: {e}")

# ==================== MAIN LOOP ====================
def main():
    if not connect_mt5():
        return
    logger.info("🚀 MT5 Forex Multi-Pair Bot started (V2 + AI)")

    last_ai_time = time.time()
    last_full_time = time.time()

    while True:
        try:
            # Check daily limits
            acc = get_account_info()
            if not acc:
                time.sleep(30)
                continue
            if acc['daily_pnl_pct'] >= MAX_DAILY_PROFIT_PERCENT:
                logger.info("Daily profit target reached. Pausing...")
                time.sleep(300)
                continue
            if acc['daily_pnl_pct'] <= -MAX_DAILY_LOSS_PERCENT:
                logger.info("Daily loss limit hit. Stopping bot.")
                break

            now = time.time()

            # AI quick review every 30s if any open positions
            if now - last_ai_time >= AI_REVIEW_INTERVAL:
                if get_open_positions():
                    logger.debug("AI quick review...")
                    ai_review_all_positions()
                last_ai_time = now

            # Full review & find new opportunities every 1 hour
            if now - last_full_time >= FULL_REVIEW_INTERVAL:
                logger.info("=== Hourly Full Review ===")
                # Review positions
                if get_open_positions():
                    ai_review_all_positions()
                # เปิดเทรดใหม่ถ้าจำนวนน้อยกว่า MAX_CONCURRENT_TRADES
                current_trades = len(get_open_positions())
                if current_trades < MAX_CONCURRENT_TRADES:
                    slots = MAX_CONCURRENT_TRADES - current_trades
                    best_pairs = select_best_symbols(slots)
                    for sym, analysis in best_pairs:
                        execute_buy(sym, analysis)
                        time.sleep(1)
                last_full_time = now

            # Check for opportunities every 5 minutes if we have free slots
            if len(get_open_positions()) < MAX_CONCURRENT_TRADES:
                slots = MAX_CONCURRENT_TRADES - len(get_open_positions())
                best_pairs = select_best_symbols(slots)
                for sym, analysis in best_pairs:
                    execute_buy(sym, analysis)
                    time.sleep(1)

            time.sleep(1)

        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
            break
        except Exception as e:
            logger.exception(f"Unhandled error: {e}")
            time.sleep(5)

    mt5.shutdown()
    logger.info("MT5 disconnected")

if __name__ == "__main__":
    main()