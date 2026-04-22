"""
MT5 AI Trading Bot with DeepSeek Reasoner (English Prompt)
Full version with robust error handling, fallback logic, and risk management.
Target: 8% daily profit with max 5% daily loss.
"""

import os
import time
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
import MetaTrader5 as mt5
from openai import OpenAI
from typing import Dict, Optional, List, Tuple

# ========================= CONFIGURATION =========================
load_dotenv()

# MT5 settings
MT5_LOGIN = int(os.getenv("MT5_LOGIN"))
MT5_PASSWORD = os.getenv("MT5_PASSWORD")
MT5_SERVER = os.getenv("MT5_SERVER")
MT5_PATH = os.getenv("MT5_PATH", "")  # optional path to terminal64.exe

# DeepSeek settings
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    raise ValueError("❌ DEEPSEEK_API_KEY not found in .env")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-reasoner"  # or "deepseek-chat" if needed

# Trading parameters
WATCHLIST = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD"]  # base names
RISK_PER_TRADE_PERCENT = 1.0          # risk 1% of equity per trade
MAX_DAILY_PROFIT_PERCENT = 8.0
MAX_DAILY_LOSS_PERCENT = 5.0
MAX_CONCURRENT_TRADES = 2
DEVIATION = 20
MAGIC_NUMBER = 123456
TIMEFRAME = mt5.TIMEFRAME_M5
BARS_COUNT = 100
LOT_FIXED = 0.01                      # fallback lot size

# Logging setup
LOG_DIR = "trading_logs"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, f"bot_{datetime.now().strftime('%Y%m%d')}.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize DeepSeek client
deepseek_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)


# ========================= MT5 HELPERS =========================
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
    
    # Get all symbols in terminal
    all_symbols = mt5.symbols_get()
    if all_symbols:
        logger.info(f"Total symbols in MT5: {len(all_symbols)}")
        sample = [s.name for s in all_symbols[:10]]
        logger.info(f"Sample: {sample}")
    
    # Enable our symbols (handle suffixes like .sml)
    enabled = []
    for base in WATCHLIST:
        found = None
        for s in all_symbols or []:
            if s.name == base or s.name.startswith(base + ".") or s.name == base + ".sml":
                found = s.name
                break
        if found:
            if mt5.symbol_select(found, True):
                enabled.append(found)
                logger.info(f"Enabled {found}")
            else:
                logger.warning(f"Could not select {found}")
        else:
            logger.warning(f"Symbol {base} not found in terminal")
    
    if not enabled:
        logger.error("No usable symbols found. Please add symbols to Market Watch.")
        return False
    
    logger.info(f"Usable symbols: {enabled}")
    return True

def get_available_symbols() -> List[str]:
    """Return list of enabled symbols that exist in MT5."""
    all_symbols = mt5.symbols_get()
    if not all_symbols:
        return []
    available = []
    for base in WATCHLIST:
        for s in all_symbols:
            if s.name == base or s.name.startswith(base + ".") or s.name == base + ".sml":
                available.append(s.name)
                break
    return available

def get_account_info() -> Optional[Dict]:
    """Fetch account info and calculate daily P&L."""
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
        "margin": acc.margin,
        "free_margin": acc.margin_free,
        "daily_profit": daily_profit,
        "daily_profit_percent": daily_profit_percent
    }

def get_open_positions_count() -> int:
    positions = mt5.positions_get()
    return len(positions) if positions else 0


# ========================= TECHNICAL INDICATORS =========================
def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

def get_market_data(symbol: str) -> Optional[Dict]:
    """Fetch OHLCV data and compute indicators."""
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        logger.warning(f"No tick data for {symbol}")
        return None
    
    rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME, 0, BARS_COUNT)
    if rates is None or len(rates) < 50:
        logger.warning(f"Insufficient data for {symbol} (got {len(rates) if rates else 0} bars)")
        return None
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df['ema5'] = calculate_ema(df['close'], 5)
    df['ema20'] = calculate_ema(df['close'], 20)
    df['rsi'] = calculate_rsi(df['close'], 14)
    df['atr'] = calculate_atr(df['high'], df['low'], df['close'], 14)
    
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    trend = "bullish" if latest['ema5'] > latest['ema20'] else "bearish"
    
    # ATR in pips (for 4-digit forex)
    atr_pips = latest['atr'] * 10000
    
    return {
        "symbol": symbol,
        "timeframe": "M5",
        "timestamp": latest['time'].isoformat(),
        "open": latest['open'],
        "high": latest['high'],
        "low": latest['low'],
        "close": latest['close'],
        "bid": tick.bid,
        "ask": tick.ask,
        "ema5": latest['ema5'],
        "ema20": latest['ema20'],
        "rsi": latest['rsi'],
        "atr_pips": atr_pips,
        "trend": trend,
        "prev_close": prev['close'],
        "volume": latest['tick_volume']
    }


# ========================= DEEPSEEK REASONER (ENGLISH PROMPT) =========================
def ask_deepseek_for_signal(market_data: Dict) -> Optional[Dict]:
    """
    Send market data to DeepSeek Reasoner with an English prompt.
    Returns a JSON with decision, confidence, entry, SL, TP, reasoning.
    """
    symbol = market_data['symbol']
    # Clean prompt – concise and strict
    prompt = f"""You are a professional forex scalper. Analyze {symbol} on M5 timeframe.

Latest data:
- Price: {market_data['close']:.5f} (Bid {market_data['bid']:.5f}, Ask {market_data['ask']:.5f})
- Trend: {market_data['trend']} (EMA5 {market_data['ema5']:.5f}, EMA20 {market_data['ema20']:.5f})
- RSI(14): {market_data['rsi']:.2f}
- ATR(14) in pips: {market_data['atr_pips']:.1f}
- High/Low: {market_data['high']:.5f} / {market_data['low']:.5f}
- Tick volume: {market_data['volume']}

Provide trading signal. Use strict JSON format:
{{
  "decision": "BUY" or "SELL" or "WAIT",
  "confidence": "HIGH" or "MEDIUM" or "LOW",
  "entry_price": number (or null if WAIT),
  "stop_loss": number (or null if WAIT),
  "take_profit": number (or null if WAIT),
  "reasoning": "short reason"
}}

Rules:
- Stop loss ≤ 20 pips (0.0020 for 4-digit pairs)
- Risk/Reward at least 1:1.5
- Avoid overbought/oversold extremes for entry
- Do not include any extra text outside JSON."""
    
    for attempt in range(2):
        try:
            response = deepseek_client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": "You are a trading assistant. Reply only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=400,
                response_format={"type": "json_object"}  # DeepSeek supports this
            )
            content = response.choices[0].message.content.strip()
            # Remove markdown fences if any
            content = content.replace("```json", "").replace("```", "").strip()
            if not content:
                logger.warning(f"Empty response from DeepSeek for {symbol}, retrying...")
                continue
            
            result = json.loads(content)
            # Validate
            if result.get('decision') == 'WAIT':
                return result
            required = ['entry_price', 'stop_loss', 'take_profit']
            if all(k in result for k in required):
                # Ensure numbers are float
                for k in required:
                    if result[k] is not None:
                        result[k] = float(result[k])
                return result
            else:
                logger.warning(f"Missing fields in DeepSeek response: {result}")
                continue
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error for {symbol}: {e}")
            if 'content' in locals():
                logger.debug(f"Raw response: {content[:200]}")
            continue
        except Exception as e:
            logger.error(f"DeepSeek API error for {symbol}: {e}")
            continue
    return None

def fallback_signal(market_data: Dict) -> Dict:
    """Simple rule-based signal when AI fails."""
    symbol = market_data['symbol']
    rsi = market_data['rsi']
    trend = market_data['trend']
    atr = market_data['atr_pips'] / 10000.0  # convert pips to price
    
    if trend == 'bullish' and 40 < rsi < 70:
        decision = 'BUY'
        confidence = 'LOW'
        entry = market_data['ask']
        sl = entry - atr * 1.5
        tp = entry + atr * 2.5
    elif trend == 'bearish' and 30 < rsi < 60:
        decision = 'SELL'
        confidence = 'LOW'
        entry = market_data['bid']
        sl = entry + atr * 1.5
        tp = entry - atr * 2.5
    else:
        decision = 'WAIT'
        confidence = 'LOW'
        entry = sl = tp = None
    
    logger.info(f"Using fallback signal for {symbol}: {decision}")
    return {
        "decision": decision,
        "confidence": confidence,
        "entry_price": entry,
        "stop_loss": sl,
        "take_profit": tp,
        "reasoning": "Fallback rule-based signal (AI error)"
    }


# ========================= RISK & ORDER MANAGEMENT =========================
def check_daily_limits(account: Dict) -> Tuple[bool, str]:
    profit_pct = account['daily_profit_percent']
    if profit_pct >= MAX_DAILY_PROFIT_PERCENT:
        return False, f"Daily profit target {MAX_DAILY_PROFIT_PERCENT}% reached. Stopping."
    if profit_pct <= -MAX_DAILY_LOSS_PERCENT:
        return False, f"Daily loss limit {MAX_DAILY_LOSS_PERCENT}% hit. Stopping."
    return True, ""

def calculate_lot_size(risk_percent: float, stop_loss_points: float, account_balance: float) -> float:
    """Calculate lot size based on risk percentage and stop loss in points."""
    if stop_loss_points <= 0:
        return LOT_FIXED
    risk_amount = account_balance * (risk_percent / 100.0)
    # For 4-digit forex, 1 pip = 0.0001, but we use points (0.0001)
    # value per 0.01 lot per point = $0.1 approx (for USD pairs)
    pip_value_per_0_01lot = 0.1
    lot = (risk_amount / (stop_loss_points * pip_value_per_0_01lot)) * 0.01
    lot = max(0.01, min(lot, 0.1))  # cap at 0.1 lot
    return round(lot, 2)

def send_order(symbol: str, signal: Dict, account_balance: float) -> bool:
    """Place a market order with SL/TP."""
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
        sl_points = abs(entry - sl) * 10000 if sl else 0
        order_type = mt5.ORDER_TYPE_BUY
        price = tick.ask
        if sl and sl >= entry:
            logger.error(f"Invalid SL for BUY: {sl} >= {entry}")
            return False
        if tp and tp <= entry:
            logger.error(f"Invalid TP for BUY: {tp} <= {entry}")
            return False
    elif decision == 'SELL':
        sl_points = abs(sl - entry) * 10000 if sl else 0
        order_type = mt5.ORDER_TYPE_SELL
        price = tick.bid
        if sl and sl <= entry:
            logger.error(f"Invalid SL for SELL: {sl} <= {entry}")
            return False
        if tp and tp >= entry:
            logger.error(f"Invalid TP for SELL: {tp} >= {entry}")
            return False
    else:
        return False
    
    # Calculate lot size based on risk
    lot = calculate_lot_size(RISK_PER_TRADE_PERCENT, sl_points, account_balance)
    # Reduce lot for MEDIUM/LOW confidence
    if confidence == 'MEDIUM':
        lot = max(0.01, lot * 0.5)
        logger.info(f"Confidence MEDIUM -> reduced lot to {lot}")
    elif confidence == 'LOW':
        lot = max(0.01, lot * 0.25)
        logger.info(f"Confidence LOW -> reduced lot to {lot}")
    
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
        "comment": f"DeepSeek_{decision}_{datetime.now().strftime('%H%M')}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.error(f"Order failed for {symbol}: {result.retcode} - {result.comment}")
        return False
    
    logger.info(f"✅ {decision} {lot} lots of {symbol} @ {price}, SL={sl}, TP={tp}")
    return True


# ========================= MAIN LOOP =========================
def analyze_and_trade():
    """Main trading routine: scan symbols, get AI signals, execute trades."""
    logger.info(f"🔍 Starting analysis at {datetime.now()}")
    
    account = get_account_info()
    if account is None:
        logger.error("Cannot fetch account info")
        return
    
    can_trade, msg = check_daily_limits(account)
    logger.info(f"Balance: {account['balance']:.2f}, Daily P&L: {account['daily_profit_percent']:.2f}%")
    if not can_trade:
        logger.info(msg)
        return
    
    open_positions = get_open_positions_count()
    if open_positions >= MAX_CONCURRENT_TRADES:
        logger.info(f"Max concurrent trades reached ({MAX_CONCURRENT_TRADES})")
        return
    
    available = get_available_symbols()
    if not available:
        logger.warning("No available symbols to trade")
        return
    
    logger.info(f"Scanning symbols: {available}")
    signals_to_trade = []
    
    for symbol in available:
        market_data = get_market_data(symbol)
        if market_data is None:
            continue
        
        logger.info(f"Asking DeepSeek for {symbol}...")
        signal = ask_deepseek_for_signal(market_data)
        if signal is None:
            logger.warning(f"DeepSeek failed, using fallback for {symbol}")
            signal = fallback_signal(market_data)
        
        if signal['decision'] == 'WAIT':
            logger.info(f"{symbol}: WAIT")
            continue
        
        logger.info(f"{symbol}: {signal['decision']} (confidence={signal['confidence']})")
        logger.info(f"  Reasoning: {signal['reasoning']}")
        
        # Accept HIGH or MEDIUM confidence; LOW only if we have no other signals
        if signal['confidence'] in ['HIGH', 'MEDIUM']:
            signals_to_trade.append((symbol, signal))
        else:
            logger.info(f"Skipping {symbol} due to LOW confidence")
    
    # Limit number of new trades
    remaining = MAX_CONCURRENT_TRADES - open_positions
    for symbol, signal in signals_to_trade[:remaining]:
        success = send_order(symbol, signal, account['balance'])
        if success:
            time.sleep(2)  # avoid rate limits

def main():
    if not connect_mt5():
        return
    
    logger.info(f"🚀 DeepSeek Trading Bot started")
    logger.info(f"🎯 Daily target: {MAX_DAILY_PROFIT_PERCENT}% profit, max loss {MAX_DAILY_LOSS_PERCENT}%")
    logger.info(f"📊 Watchlist: {WATCHLIST}")
    logger.info(f"⏱️  Analysis every 5 minutes\n")
    
    try:
        while True:
            analyze_and_trade()
            logger.info("😴 Sleeping for 5 minutes... (Ctrl+C to stop)")
            time.sleep(300)
    except KeyboardInterrupt:
        logger.info("🛑 Bot stopped by user")
    finally:
        mt5.shutdown()
        logger.info("MT5 disconnected")

if __name__ == "__main__":
    main()