"""
MT5 AI Trading Bot - Full Version with Position Review & Auto-Fill up to 5 Trades
- Collects all pairs data (Bid/Ask, Daily Change, EMA across M5/M15/H1)
- Sends one request to DeepSeek API for all analysis (new opportunities)
- Reviews open positions every hour using AI
- Automatically opens new trades until reaching MAX_CONCURRENT_TRADES (default 5)
- Fixed numpy array truth value ambiguity errors
- No emoji, fully Unicode-safe for Windows console
"""

import os
import sys
import time
import json
import logging
import codecs
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import MetaTrader5 as mt5
from openai import OpenAI
from typing import Dict, Optional, List, Tuple

# ========== FIX UNICODE FOR WINDOWS CONSOLE ==========
if sys.platform == "win32":
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "strict")

# ========== LOAD CONFIGURATION ==========
load_dotenv()

# MT5 settings
MT5_LOGIN = int(os.getenv("MT5_LOGIN"))
MT5_PASSWORD = os.getenv("MT5_PASSWORD")
MT5_SERVER = os.getenv("MT5_SERVER")
MT5_PATH = os.getenv("MT5_PATH", "")

# DeepSeek settings
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    raise ValueError("ERROR: DEEPSEEK_API_KEY not found in .env")
deepseek_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
DEEPSEEK_MODEL = "deepseek-chat"

# Trading parameters
MAX_DAILY_PROFIT_PERCENT = 8.0
MAX_DAILY_LOSS_PERCENT = 5.0
MAX_CONCURRENT_TRADES = 5          # เป้าหมายสูงสุด 5 คู่
RISK_PER_TRADE_PERCENT = 1.0       # ความเสี่ยงต่อออเดอร์ (% ของ balance)
DEVIATION = 20
MAGIC_NUMBER = 123456
REVIEW_INTERVAL = 3600             # รีวิวทุก 1 ชั่วโมง (วินาที)

# Watchlist (base names, will auto-detect suffixes like .sml)
WATCHLIST = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD",
    "USDCAD", "NZDUSD", "EURGBP", "EURJPY", "GBPJPY"
]
SYMBOL_BLACKLIST = []   # คู่ที่ไม่อยากเทรด (ใส่ชื่อจริง เช่น "EURUSD.sml")

TIMEFRAMES = {
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "H1": mt5.TIMEFRAME_H1
}

# ========== LOGGING SETUP ==========
LOG_DIR = "trading_logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, f"bot_{datetime.now().strftime('%Y%m%d')}.log"), encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# ========== MT5 HELPERS ==========
def connect_mt5() -> bool:
    """Connect to MT5 and enable symbols."""
    if MT5_PATH and os.path.exists(MT5_PATH):
        initialized = mt5.initialize(path=MT5_PATH, login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER)
    else:
        initialized = mt5.initialize(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER)

    if not initialized:
        logger.error(f"MT5 init failed: {mt5.last_error()}")
        return False

    logger.info("Connected to MT5")
    all_symbols = mt5.symbols_get()
    if not all_symbols:
        logger.error("No symbols found")
        return False

    enabled = []
    for base in WATCHLIST:
        found = None
        for s in all_symbols:
            if s.name == base or s.name.startswith(base + ".") or s.name == base + ".sml":
                found = s.name
                break
        if found and found not in SYMBOL_BLACKLIST:
            if mt5.symbol_select(found, True):
                enabled.append(found)
                logger.info(f"Enabled {found}")
        elif found:
            logger.warning(f"Symbol {found} is blacklisted, skipping")
        else:
            logger.warning(f"Symbol {base} not found")

    if not enabled:
        logger.error("No usable symbols. Please add symbols to Market Watch.")
        return False

    logger.info(f"Available symbols: {enabled}")
    return True

def get_available_symbols() -> List[str]:
    """Return list of enabled symbols (actual MT5 names)."""
    all_symbols = mt5.symbols_get()
    if not all_symbols:
        return []
    available = []
    for base in WATCHLIST:
        for s in all_symbols:
            if s.name == base or s.name.startswith(base + ".") or s.name == base + ".sml":
                if s.name not in SYMBOL_BLACKLIST:
                    available.append(s.name)
                break
    return available

def get_symbol_mapping() -> Dict[str, str]:
    """Create mapping from base name (without suffix) to actual MT5 symbol name."""
    available = get_available_symbols()
    mapping = {}
    for sym in available:
        base = sym.replace(".sml", "").replace(".", "")
        mapping[base] = sym
        mapping[base.upper()] = sym
        mapping[base.lower()] = sym
        mapping[sym] = sym
        if not sym.endswith(".sml"):
            mapping[sym + ".sml"] = sym
    return mapping

def get_account_info() -> Optional[Dict]:
    """Fetch account info and daily P&L."""
    acc = mt5.account_info()
    if acc is None:
        return None
    today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    history = mt5.history_deals_get(today_start, datetime.now())
    daily_profit = 0.0
    if history:
        daily_profit = sum(deal.profit for deal in history)
    daily_profit_percent = (daily_profit / acc.balance) * 100 if acc.balance > 0 else 0
    return {
        "balance": acc.balance,
        "equity": acc.equity,
        "daily_profit": daily_profit,
        "daily_profit_percent": daily_profit_percent
    }

def get_open_positions() -> List:
    """Return list of open positions (as mt5 objects)."""
    positions = mt5.positions_get()
    return positions if positions else []

def get_open_positions_count() -> int:
    return len(get_open_positions())

def get_position_symbols() -> List[str]:
    """Return list of symbols that currently have open positions."""
    positions = get_open_positions()
    return [pos.symbol for pos in positions]

def close_position(position) -> bool:
    """Close a specific position (market order)."""
    symbol = position.symbol
    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        logger.error(f"Cannot get tick for {symbol}")
        return False
    
    order_type = mt5.ORDER_TYPE_BUY if position.type == mt5.POSITION_TYPE_SELL else mt5.ORDER_TYPE_SELL
    price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": position.volume,
        "type": order_type,
        "position": position.ticket,
        "price": price,
        "deviation": DEVIATION,
        "magic": MAGIC_NUMBER,
        "comment": f"AI_CLOSE_{datetime.now().strftime('%H%M')}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.error(f"Close position failed: {result.retcode} - {result.comment}")
        return False
    
    pnl = position.profit
    logger.info(f"CLOSED {symbol}: PnL = {pnl:.2f}")
    return True


# ========== DATA COLLECTION FOR ALL PAIRS ==========
def get_all_pairs_data() -> List[Dict]:
    """Collect Bid/Ask, Daily Change, and EMA data for all symbols."""
    symbols = get_available_symbols()
    all_data = []

    for symbol in symbols:
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            continue

        # Daily change
        rates_daily = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, 0, 2)
        if rates_daily is None or len(rates_daily) < 2:
            daily_change_pct = 0.0
        else:
            yesterday_close = rates_daily[0]['close']
            current_price = tick.bid
            daily_change_pct = ((current_price - yesterday_close) / yesterday_close) * 100

        # Spread
        symbol_info = mt5.symbol_info(symbol)
        spread = symbol_info.spread if symbol_info else 0

        # EMA data for multiple timeframes
        ema_summary = {}
        for tf_name, tf_value in TIMEFRAMES.items():
            rates = mt5.copy_rates_from_pos(symbol, tf_value, 0, 50)
            if rates is None or len(rates) < 30:
                continue
            df = pd.DataFrame(rates)
            df['close'] = df['close']
            df['ema5'] = df['close'].ewm(span=5, adjust=False).mean()
            df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
            df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else latest

            # Trend
            if latest['ema5'] > latest['ema20'] > latest['ema50']:
                trend = "STRONG_BULLISH"
            elif latest['ema5'] > latest['ema20']:
                trend = "BULLISH"
            elif latest['ema5'] < latest['ema20'] < latest['ema50']:
                trend = "STRONG_BEARISH"
            elif latest['ema5'] < latest['ema20']:
                trend = "BEARISH"
            else:
                trend = "CONSOLIDATION"

            # Crossover
            crossover = None
            if prev['ema5'] <= prev['ema20'] and latest['ema5'] > latest['ema20']:
                crossover = "GOLDEN_CROSS"
            elif prev['ema5'] >= prev['ema20'] and latest['ema5'] < latest['ema20']:
                crossover = "DEATH_CROSS"

            ema_summary[tf_name] = {
                "ema5": round(latest['ema5'], 5),
                "ema20": round(latest['ema20'], 5),
                "trend": trend,
                "crossover": crossover,
                "price_vs_ema20": "ABOVE" if latest['close'] > latest['ema20'] else "BELOW"
            }

        all_data.append({
            "symbol": symbol,
            "bid": tick.bid,
            "ask": tick.ask,
            "spread": spread,
            "daily_change_percent": round(daily_change_pct, 2),
            "volume": tick.volume if hasattr(tick, 'volume') else 0,
            "ema": ema_summary
        })

    return all_data


# ========== AI POSITION REVIEW (ทุกชั่วโมง) ==========
def ai_review_positions() -> None:
    """Review each open position and close if AI recommends."""
    positions = get_open_positions()
    if not positions:
        logger.info("No open positions to review.")
        return

    logger.info(f"Reviewing {len(positions)} open positions...")
    for pos in positions:
        symbol = pos.symbol
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            continue
        
        ema_data = {}
        for tf_name, tf_value in TIMEFRAMES.items():
            rates = mt5.copy_rates_from_pos(symbol, tf_value, 0, 30)
            if rates is not None and len(rates) >= 20:
                df = pd.DataFrame(rates)
                df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
                ema_data[tf_name] = {
                    "ema20": round(df['ema20'].iloc[-1], 5),
                    "price": tick.bid if pos.type == mt5.POSITION_TYPE_BUY else tick.ask
                }
        
        prompt = f"""You are a forex position manager. Analyze this open position and decide whether to HOLD or CLOSE.

Symbol: {symbol}
Position side: {"BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL"}
Entry price: {pos.price_open:.5f}
Current price: {tick.bid if pos.type == mt5.POSITION_TYPE_BUY else tick.ask:.5f}
Current profit/loss: {pos.profit:.2f} USD
Unrealized pips: {pos.profit / (pos.volume * 0.1) if pos.volume > 0 else 0:.1f} pips (approx)
EMA20 (M5): {ema_data.get('M5', {}).get('ema20', 'N/A')}
EMA20 (M15): {ema_data.get('M15', {}).get('ema20', 'N/A')}
EMA20 (H1): {ema_data.get('H1', {}).get('ema20', 'N/A')}

Decision rules:
- CLOSE if trend reversed against position (e.g., BUY but price < EMA20 on M15 and H1)
- CLOSE if profit has reached >= 30 pips and momentum weakening
- CLOSE if loss exceeds 20 pips and no reversal sign
- Otherwise HOLD

Respond with JSON: {{"action": "HOLD" or "CLOSE", "confidence": 0-100, "reason": "short reason"}}
"""
        try:
            response = deepseek_client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": "You are a forex position manager. Reply only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=200,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content.strip()
            content = content.replace("```json", "").replace("```", "").strip()
            decision = json.loads(content)
            action = decision.get("action", "HOLD")
            reason = decision.get("reason", "")
            logger.info(f"Review {symbol}: {action} - {reason}")
            if action == "CLOSE":
                if close_position(pos):
                    logger.info(f"Closed {symbol} based on AI review: {reason}")
                else:
                    logger.error(f"Failed to close {symbol}")
            time.sleep(1)
        except Exception as e:
            logger.error(f"Review error for {symbol}: {e}")


# ========== AI FIND NEW OPPORTUNITIES (จนกว่าจะครบ MAX_CONCURRENT_TRADES) ==========
def ai_find_new_opportunities() -> None:
    """Ask AI to recommend new trades for symbols that don't have open positions."""
    current_count = get_open_positions_count()
    if current_count >= MAX_CONCURRENT_TRADES:
        logger.info(f"Already have {current_count}/{MAX_CONCURRENT_TRADES} trades. No need new opportunities.")
        return
    
    needed = MAX_CONCURRENT_TRADES - current_count
    logger.info(f"Looking for up to {needed} new trade opportunities...")
    
    all_symbols = get_available_symbols()
    pos_symbols = get_position_symbols()
    candidates = [s for s in all_symbols if s not in pos_symbols]
    if not candidates:
        logger.info("No candidate symbols available for new trades.")
        return
    
    data_for_ai = []
    for symbol in candidates:
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            continue
        
        # FIXED: use 'is not None' instead of truth value of numpy array
        rates_daily = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, 0, 2)
        daily_change = 0.0
        if rates_daily is not None and len(rates_daily) >= 2:
            daily_change = ((tick.bid - rates_daily[0]['close']) / rates_daily[0]['close']) * 100
        
        ema_summary = {}
        for tf_name, tf_value in TIMEFRAMES.items():
            rates = mt5.copy_rates_from_pos(symbol, tf_value, 0, 30)
            if rates is not None and len(rates) >= 20:
                df = pd.DataFrame(rates)
                df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
                df['ema5'] = df['close'].ewm(span=5, adjust=False).mean()
                latest = df.iloc[-1]
                trend = "BULLISH" if latest['ema5'] > latest['ema20'] else "BEARISH"
                ema_summary[tf_name] = {
                    "ema5": round(latest['ema5'], 5),
                    "ema20": round(latest['ema20'], 5),
                    "trend": trend,
                    "price_vs_ema20": "ABOVE" if latest['close'] > latest['ema20'] else "BELOW"
                }
        data_for_ai.append({
            "symbol": symbol,
            "bid": tick.bid,
            "ask": tick.ask,
            "daily_change_percent": round(daily_change, 2),
            "ema": ema_summary
        })
    
    if not data_for_ai:
        logger.info("No valid data for candidates.")
        return
    
    # Build prompt
    pairs_text = []
    for p in data_for_ai:
        ema_lines = []
        for tf, e in p['ema'].items():
            ema_lines.append(f"{tf}: {e['trend']}, EMA5/20={e['ema5']:.4f}/{e['ema20']:.4f}, price {e['price_vs_ema20']} EMA20")
        ema_str = "; ".join(ema_lines)
        pairs_text.append(
            f"- {p['symbol']}: Bid={p['bid']:.5f}, Ask={p['ask']:.5f}, DailyChange={p['daily_change_percent']}%, EMA: {ema_str}"
        )
    prompt_data = "\n".join(pairs_text)
    
    prompt = f"""You are a forex scalping AI. Analyze the following pairs and return trading signals for the BEST up to {needed} pairs only.

Data:
{prompt_data}

Rules:
- Entry: BUY if M5/M15/H1 all show BULLISH and price above EMA20. SELL if all show BEARISH and price below EMA20.
- Stop Loss: 15-20 pips (0.0015-0.0020) from entry.
- Take Profit: 30-40 pips (Risk/Reward >= 1:2).
- Confidence: HIGH (all timeframes aligned), MEDIUM (mostly aligned), LOW (weak alignment).

Return ONLY valid JSON array (max {needed} objects):
[
  {{
    "symbol": "EURUSD",
    "decision": "BUY",
    "confidence": "HIGH",
    "entry_price": number (use Ask for BUY, Bid for SELL),
    "stop_loss": number,
    "take_profit": number,
    "reasoning": "short reason"
  }}
]
If no good setup, return empty array [].
Do not include any text outside JSON.
"""
    try:
        response = deepseek_client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": "You are a forex trading AI. Reply only with valid JSON array."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=800,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content.strip()
        content = content.replace("```json", "").replace("```", "").strip()
        signals = json.loads(content)
        if isinstance(signals, dict):
            signals = signals.get("signals", signals.get("selected", []))
        if not isinstance(signals, list):
            signals = []
        
        logger.info(f"AI proposed {len(signals)} new signals")
        
        symbol_map = get_symbol_mapping()
        account = get_account_info()
        if not account:
            return
        executed = 0
        for sig in signals:
            if executed >= needed:
                break
            ai_symbol = sig.get('symbol')
            real_symbol = symbol_map.get(ai_symbol) or symbol_map.get(ai_symbol + ".sml")
            if not real_symbol:
                logger.warning(f"Cannot map symbol {ai_symbol}, skipping")
                continue
            sig['symbol'] = real_symbol
            if real_symbol in get_position_symbols():
                logger.info(f"Symbol {real_symbol} already has a position, skipping")
                continue
            if send_order(sig, account['balance']):
                executed += 1
                time.sleep(2)
        logger.info(f"Opened {executed} new trades.")
    except Exception as e:
        logger.error(f"AI new opportunities error: {e}")


# ========== ORDER EXECUTION WITH LOT NORMALIZATION ==========
def normalize_lot(symbol: str, lot: float) -> float:
    """Adjust lot to match broker's volume requirements."""
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        return round(max(0.01, min(lot, 0.1)), 2)
    
    min_lot = symbol_info.volume_min
    max_lot = symbol_info.volume_max
    step_lot = symbol_info.volume_step
    
    lot = max(min_lot, min(lot, max_lot))
    if step_lot > 0:
        lot = round(lot / step_lot) * step_lot
        lot = round(lot, 6)
    return lot

def send_order(signal: Dict, account_balance: float) -> bool:
    """Place market order with SL/TP using normalized lot."""
    symbol = signal['symbol']
    decision = signal['decision']
    entry = signal['entry_price']
    sl = signal['stop_loss']
    tp = signal['take_profit']
    confidence = signal.get('confidence', 'MEDIUM')

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        logger.error(f"No tick for {symbol}")
        return False

    if decision == 'BUY':
        order_type = mt5.ORDER_TYPE_BUY
        price = tick.ask
        sl_points = abs(entry - sl) * 10000 if sl else 20
        if sl and sl >= entry:
            logger.error(f"Invalid SL for BUY: {sl} >= {entry}")
            return False
        if tp and tp <= entry:
            logger.error(f"Invalid TP for BUY: {tp} <= {entry}")
            return False
    elif decision == 'SELL':
        order_type = mt5.ORDER_TYPE_SELL
        price = tick.bid
        sl_points = abs(sl - entry) * 10000 if sl else 20
        if sl and sl <= entry:
            logger.error(f"Invalid SL for SELL: {sl} <= {entry}")
            return False
        if tp and tp >= entry:
            logger.error(f"Invalid TP for SELL: {tp} >= {entry}")
            return False
    else:
        return False

    # Calculate base lot from risk
    risk_amount = account_balance * (RISK_PER_TRADE_PERCENT / 100.0)
    pip_value_per_0_01lot = 0.1  # approximate for 4-digit pairs
    lot = (risk_amount / (sl_points * pip_value_per_0_01lot)) * 0.01
    lot = max(0.01, min(lot, 0.1))

    # Reduce lot for lower confidence
    if confidence == 'MEDIUM':
        lot = max(0.01, lot * 0.5)
    elif confidence == 'LOW':
        lot = max(0.01, lot * 0.25)
    lot = round(lot, 2)
    
    lot = normalize_lot(symbol, lot)

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": DEVIATION,
        "magic": MAGIC_NUMBER,
        "comment": f"AI_{decision}_{datetime.now().strftime('%H%M')}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }

    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.error(f"Order failed: {result.retcode} - {result.comment}")
        return False

    logger.info(f"ORDER SUCCESS: {decision} {lot} lots {symbol} @ {price}, SL={sl}, TP={tp}")
    return True


# ========== MAIN LOOP ==========
def main():
    logger.info("Starting Forex Bot with Position Review and Auto-Fill (Max 5 trades)")
    if not connect_mt5():
        return

    logger.info(f"Daily target: {MAX_DAILY_PROFIT_PERCENT}% profit, max loss {MAX_DAILY_LOSS_PERCENT}%")
    logger.info(f"Max concurrent trades: {MAX_CONCURRENT_TRADES}")
    logger.info(f"Review interval: {REVIEW_INTERVAL} seconds")

    last_review_time = time.time()
    
    try:
        while True:
            # Check account and daily limits
            account = get_account_info()
            if not account:
                time.sleep(60)
                continue

            logger.info(f"Balance: {account['balance']:.2f}, Daily P&L: {account['daily_profit_percent']:.2f}%")

            if account['daily_profit_percent'] >= MAX_DAILY_PROFIT_PERCENT:
                logger.info("Daily profit target reached. Stopping.")
                break
            if account['daily_profit_percent'] <= -MAX_DAILY_LOSS_PERCENT:
                logger.info("Daily loss limit hit. Stopping.")
                break

            # === รีวิวทุกชั่วโมง ===
            now = time.time()
            if now - last_review_time >= REVIEW_INTERVAL:
                logger.info("=== Starting scheduled review ===")
                ai_review_positions()
                # หลังจากรีวิวแล้ว ให้หาโอกาสเปิดใหม่จนกว่าจะครบ MAX_CONCURRENT_TRADES
                ai_find_new_opportunities()
                last_review_time = now
                logger.info("=== Review cycle completed ===")
                time.sleep(60)
                continue

            # === ปกติ: ถ้ายังมีตำแหน่งว่าง ให้ลองหาสัญญาณทันที (ไม่ต้องรอชั่วโมง) ===
            open_positions = get_open_positions_count()
            if open_positions < MAX_CONCURRENT_TRADES:
                logger.info(f"Open positions: {open_positions}/{MAX_CONCURRENT_TRADES}. Looking for immediate opportunities...")
                ai_find_new_opportunities()
            
            logger.info("Sleeping 5 minutes...")
            time.sleep(300)

    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    finally:
        mt5.shutdown()
        logger.info("MT5 disconnected")

if __name__ == "__main__":
    main()