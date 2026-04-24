#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MT5 Forex Multi-Pair Bot - V4.1 Dynamic Entry

Goal ของ V4.1:
- ลด overtrade / ลดโดน SL รัว
- ปิด AI close ออกก่อน เพื่อให้ระบบ exit ด้วย SL/TP ที่วัดผลได้
- เพิ่ม cooldown ต่อ symbol
- เพิ่ม spread filter
- เพิ่ม session filter
- block คู่ที่ประวัติเสียหนัก เช่น GBPJPY
- รองรับ filling mode auto retry แก้ retcode=10030
- ทำโครง news filter ไว้เสียบ API ภายหลัง

คำเตือน: ใช้ DEMO/DRY_RUN ก่อนเท่านั้น จนกว่าจะ forward test ผ่าน
"""

import os
import sys
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import MetaTrader5 as mt5
import pandas as pd
from dotenv import load_dotenv

# ==================== CONFIG ====================
load_dotenv()

MT5_LOGIN = int(os.getenv("MT5_LOGIN", "0"))
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "")
MT5_SERVER = os.getenv("MT5_SERVER", "")
MT5_PATH = os.getenv("MT5_PATH", "")

# DEMO safety
DRY_RUN = os.getenv("DRY_RUN", "false").lower() == "true"

FOREX_WATCHLIST = []  # V4.1 dynamic auto scan

# จาก trade history ล่าสุด GBPJPY เสียหนัก จึง block ก่อน
BLOCK_PAIRS = {
    "GBPJPY", "GBPJPY.sml"
}

# Strategy parameters
FAST_EMA = 12
SLOW_EMA = 30
EMA_TREND_PERIOD_H1 = 50
EMA_TREND_PERIOD_H4 = 50
RSI_PERIOD = 14
ADX_PERIOD = 14
ATR_PERIOD = 14

# V4 เข้มกว่า V3: ลด signal ที่กลาง ๆ
ADX_MIN = 18
RSI_BUY_MIN = 45
RSI_BUY_MAX = 70
RSI_SELL_MIN = 30
RSI_SELL_MAX = 55
MIN_SCORE_TO_TRADE = 4.0

# Exit / risk
SL_ATR_MULT = 1.6
TP_ATR_MULT = 2.4       # RR ประมาณ 1.5:1, ไม่ไกลเกินจนไม่ค่อยโดน TP
RISK_PER_TRADE = 0.005  # 0.5% ต่อไม้ ลดจาก 1.5%
MAX_LOT = 0.05
MIN_LOT = 0.01
MAX_CONCURRENT_TRADES = 5
MAX_DAILY_PROFIT_PERCENT = 8.0
MAX_DAILY_LOSS_PERCENT = 3.0

# Execution filters
DEVIATION = 20
MAGIC_NUMBER = 654321
SCAN_INTERVAL = 300           # 5 นาที
COOLDOWN_MINUTES = 20         # V4.1 ลดลง เพื่อให้มีโอกาสเข้าได้มากขึ้น
AFTER_CLOSE_COOLDOWN_MINUTES = 30

# Spread filter: max spread เป็น pips ต่อ symbol base
DEFAULT_MAX_SPREAD_PIPS = 3.0
MAX_SPREAD_PIPS_BY_SYMBOL = {
    "EURUSD": 1.8,
    "GBPUSD": 2.2,
    "USDJPY": 2.0,
    "USDCHF": 2.4,
    "AUDUSD": 2.2,
    "USDCAD": 2.5,
    "NZDUSD": 2.5,
    "EURGBP": 2.2,
    "EURJPY": 3.0,
    "GBPJPY": 3.5,
}

# Forex session filter, local machine time. Bangkok = UTC+7 typically.
# ให้เทรดช่วง London/NY overlap และหลังตลาดเอเชียเริ่มนิ่งขึ้น
TRADE_HOURS_LOCAL = set(range(7, 24))  # 07:00-23:59 local
AVOID_FRIDAY_AFTER_HOUR = 21

# News filter placeholder
USE_NEWS_FILTER = False
NEWS_BLOCK_BEFORE_MIN = 30
NEWS_BLOCK_AFTER_MIN = 30
NEWS_CACHE: List[Dict] = []

# Logging
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, f"mt5_bot_v4_{datetime.now().strftime('%Y%m%d')}.log"), encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

LAST_TRADE_TIME: Dict[str, datetime] = {}
LAST_CLOSED_TIME: Dict[str, datetime] = {}
SYMBOL_CACHE: Dict[str, str] = {}

AUTO_SYMBOL_SCAN = os.getenv("AUTO_SYMBOL_SCAN", "true").lower() == "true"
MAX_SYMBOLS_TO_SCAN = int(os.getenv("MAX_SYMBOLS_TO_SCAN", "40"))
MAJOR_CURRENCIES = {"USD", "EUR", "GBP", "JPY", "CHF", "AUD", "NZD", "CAD"}
BLOCK_KEYWORDS = {"XAU", "XAG", "BTC", "ETH", "US30", "NAS", "SPX", "OIL", "BRENT", "WTI", "CRYPTO", "INDEX"}


def normalize_forex_name(name: str) -> str:
    return name.replace(".sml", "").replace(".", "").replace("_", "").upper()


def is_forex_symbol_name(name: str) -> bool:
    clean = normalize_forex_name(name)
    if any(k in clean for k in BLOCK_KEYWORDS):
        return False
    if len(clean) < 6:
        return False
    base, quote = clean[:3], clean[3:6]
    return base in MAJOR_CURRENCIES and quote in MAJOR_CURRENCIES


def get_all_tradable_forex_symbols() -> List[str]:
    symbols = mt5.symbols_get()
    if not symbols:
        return []
    candidates = []
    for s in symbols:
        name = s.name
        if not is_forex_symbol_name(name):
            continue
        base = normalize_forex_name(name)[:6]
        if base in BLOCK_PAIRS or name in BLOCK_PAIRS:
            continue
        if not mt5.symbol_select(name, True):
            continue
        info = mt5.symbol_info(name)
        tick = mt5.symbol_info_tick(name)
        if info is None or tick is None:
            continue
        if info.trade_mode == mt5.SYMBOL_TRADE_MODE_DISABLED:
            continue
        spread = current_spread_pips(name)
        if spread is None or spread <= 0:
            continue
        max_spread = MAX_SPREAD_PIPS_BY_SYMBOL.get(base, DEFAULT_MAX_SPREAD_PIPS)
        if spread > max_spread * 1.5:
            continue
        rates = mt5.copy_rates_from_pos(name, mt5.TIMEFRAME_M5, 0, 100)
        if rates is None or len(rates) < 80:
            continue
        candidates.append((name, spread))
    candidates.sort(key=lambda x: x[1])
    return [x[0] for x in candidates[:MAX_SYMBOLS_TO_SCAN]]


# ==================== MT5 HELPERS ====================
def connect_mt5() -> bool:
    global FOREX_WATCHLIST
    if MT5_PATH and os.path.exists(MT5_PATH):
        ok = mt5.initialize(path=MT5_PATH, login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER)
    else:
        ok = mt5.initialize(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER)
    if not ok:
        logger.error(f"MT5 init failed: {mt5.last_error()}")
        return False
    logger.info("Connected to MT5")
    all_symbols = mt5.symbols_get()
    if not all_symbols:
        logger.error("No symbols found")
        return False
    if AUTO_SYMBOL_SCAN:
        FOREX_WATCHLIST = get_all_tradable_forex_symbols()
        logger.info(f"✅ AUTO_SYMBOL_SCAN=true, tradable forex symbols={FOREX_WATCHLIST}")
        logger.info(f"✅ Total tradable forex pairs: {len(FOREX_WATCHLIST)}")
    else:
        FOREX_WATCHLIST = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD", "EURGBP", "EURJPY", "GBPJPY"]
    enabled = []
    for base in FOREX_WATCHLIST:
        found = base if mt5.symbol_info(base) else find_symbol_name(base, all_symbols)
        if found and mt5.symbol_select(found, True):
            SYMBOL_CACHE[base] = found
            enabled.append(found)
        elif found:
            logger.warning(f"Could not select {found}")
        else:
            logger.warning(f"{base} not found")
    FOREX_WATCHLIST = sorted(set(enabled))
    logger.info(f"Enabled symbols for scanning: {FOREX_WATCHLIST}")
    return len(FOREX_WATCHLIST) > 0

def find_symbol_name(base: str, all_symbols=None) -> Optional[str]:
    if base in SYMBOL_CACHE:
        return SYMBOL_CACHE[base]
    if all_symbols is None:
        all_symbols = mt5.symbols_get()
    if not all_symbols:
        return None
    for s in all_symbols:
        if s.name == base or s.name.startswith(base + ".") or s.name == base + ".sml":
            SYMBOL_CACHE[base] = s.name
            return s.name
    return None


def clean_base_symbol(symbol: str) -> str:
    clean = normalize_forex_name(symbol)
    return clean[:6] if len(clean) >= 6 else symbol.split(".")[0]


def get_account_info() -> Optional[dict]:
    acc = mt5.account_info()
    if acc is None:
        return None
    today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    history = mt5.history_deals_get(today_start, datetime.now())
    daily_profit = sum(deal.profit for deal in history) if history else 0.0
    pct = (daily_profit / acc.balance) * 100 if acc.balance > 0 else 0.0
    return {"balance": acc.balance, "equity": acc.equity, "daily_profit": daily_profit, "daily_pnl_pct": pct}


def get_open_positions():
    positions = mt5.positions_get()
    return positions if positions else []


def pip_size(symbol: str) -> float:
    info = mt5.symbol_info(symbol)
    if info is None:
        return 0.0001
    # JPY pairs usually 0.01 pip, 5-digit non-JPY usually point*10
    if "JPY" in clean_base_symbol(symbol):
        return 0.01
    return info.point * 10 if info.digits in (3, 5) else info.point


def current_spread_pips(symbol: str) -> Optional[float]:
    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        return None
    ps = pip_size(symbol)
    return (tick.ask - tick.bid) / ps if ps else None


def spread_ok(symbol: str) -> Tuple[bool, str]:
    spread = current_spread_pips(symbol)
    if spread is None:
        return False, "no tick/spread"
    base = clean_base_symbol(symbol)
    max_spread = MAX_SPREAD_PIPS_BY_SYMBOL.get(base, DEFAULT_MAX_SPREAD_PIPS)
    if spread > max_spread:
        return False, f"spread too high {spread:.1f} > {max_spread:.1f} pips"
    return True, f"spread {spread:.1f} pips"


def session_ok() -> Tuple[bool, str]:
    now = datetime.now()
    if now.weekday() >= 5:
        return False, "weekend"
    if now.weekday() == 4 and now.hour >= AVOID_FRIDAY_AFTER_HOUR:
        return False, "avoid late Friday"
    if now.hour not in TRADE_HOURS_LOCAL:
        return False, f"outside trade hours local={now.hour}"
    return True, "session ok"


def get_supported_filling_modes(symbol: str) -> List[int]:
    info = mt5.symbol_info(symbol)
    possible_modes = [mt5.ORDER_FILLING_RETURN, mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_FOK]
    if info is None:
        return possible_modes

    available = []
    filling_mode = getattr(info, "filling_mode", None)
    for mode in possible_modes:
        try:
            if filling_mode == mode or (isinstance(filling_mode, int) and mode != 0 and (filling_mode & mode) == mode):
                available.append(mode)
        except Exception:
            pass
    for mode in possible_modes:
        if mode not in available:
            available.append(mode)
    return available


def order_send_with_filling_retry(request: dict, symbol: str, action_label: str):
    last_result = None
    for filling_mode in get_supported_filling_modes(symbol):
        req = dict(request)
        req["type_filling"] = filling_mode
        result = mt5.order_send(req)
        last_result = result

        if result is None:
            logger.error(f"{action_label} {symbol} failed: order_send None, mt5_error={mt5.last_error()}")
            continue

        if result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"{action_label} {symbol} sent with filling_mode={filling_mode}")
            return result

        logger.warning(f"{action_label} {symbol} filling={filling_mode} rejected: retcode={result.retcode}, comment={result.comment}")
        if result.retcode != 10030:  # Unsupported filling mode
            return result
    return last_result


# ==================== INDICATORS ====================
def compute_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, pd.NA)
    return 100 - (100 / (1 + rs))


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    plus_di = 100 * plus_dm.rolling(period).mean() / atr
    minus_di = 100 * minus_dm.rolling(period).mean() / atr
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, pd.NA)) * 100
    return dx.rolling(period).mean()


def load_rates(symbol: str, timeframe, bars: int) -> Optional[pd.DataFrame]:
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    if rates is None or len(rates) < max(60, bars // 2):
        return None
    return pd.DataFrame(rates)


# ==================== NEWS FILTER PLACEHOLDER ====================
def fetch_news_calendar() -> List[dict]:
    """
    เสียบ Trading Economics / calendar API ตรงนี้ภายหลัง
    return format ตัวอย่าง:
    [{"currency":"USD", "time": datetime(...), "impact":"high"}]
    """
    return []


def symbol_currencies(symbol: str) -> Tuple[str, str]:
    base = clean_base_symbol(symbol)
    return base[:3], base[3:6]


def is_news_block(symbol: str) -> Tuple[bool, str]:
    if not USE_NEWS_FILTER:
        return False, "news filter off"
    now = datetime.utcnow()
    c1, c2 = symbol_currencies(symbol)
    for news in NEWS_CACHE:
        if news.get("impact", "").lower() != "high":
            continue
        if news.get("currency") not in {c1, c2}:
            continue
        news_time = news.get("time")
        if not isinstance(news_time, datetime):
            continue
        start = news_time - timedelta(minutes=NEWS_BLOCK_BEFORE_MIN)
        end = news_time + timedelta(minutes=NEWS_BLOCK_AFTER_MIN)
        if start <= now <= end:
            return True, f"high impact news {news.get('currency')} at {news_time} UTC"
    return False, "no news block"


# ==================== STRATEGY ====================
def analyze_symbol(symbol: str, debug: bool = False) -> Optional[dict]:
    def reject(reason: str):
        if debug:
            logger.info(f"❌ {symbol}: {reason}")
        return None

    base = clean_base_symbol(symbol)
    if base in BLOCK_PAIRS or symbol in BLOCK_PAIRS:
        return reject("blocked pair from history")

    ok, reason = session_ok()
    if not ok:
        return reject(reason)

    ok, reason = spread_ok(symbol)
    if not ok:
        return reject(reason)

    blocked, news_reason = is_news_block(symbol)
    if blocked:
        return reject(news_reason)

    now = datetime.now()
    if symbol in LAST_TRADE_TIME and now - LAST_TRADE_TIME[symbol] < timedelta(minutes=COOLDOWN_MINUTES):
        return reject("symbol cooldown after trade")
    if symbol in LAST_CLOSED_TIME and now - LAST_CLOSED_TIME[symbol] < timedelta(minutes=AFTER_CLOSE_COOLDOWN_MINUTES):
        return reject("symbol cooldown after close")

    df_5m = load_rates(symbol, mt5.TIMEFRAME_M5, 150)
    if df_5m is None:
        return reject("M5 data insufficient")

    fast = compute_ema(df_5m["close"], FAST_EMA)
    slow = compute_ema(df_5m["close"], SLOW_EMA)
    rsi = compute_rsi(df_5m["close"], RSI_PERIOD)
    adx = compute_adx(df_5m, ADX_PERIOD)
    atr = compute_atr(df_5m, ATR_PERIOD)

    i = len(df_5m) - 1
    last_close = df_5m["close"].iloc[i]
    last_fast = fast.iloc[i]
    last_slow = slow.iloc[i]
    last_rsi = rsi.iloc[i]
    last_adx = adx.iloc[i]
    last_atr = atr.iloc[i]

    if pd.isna(last_rsi) or pd.isna(last_adx) or pd.isna(last_atr):
        return reject("indicator not ready")
    if last_adx < ADX_MIN:
        return reject(f"ADX weak {last_adx:.1f} < {ADX_MIN}")

    # Candle confirmation: ไม่เข้าไล่แท่งยาวมากเกินไป
    recent_atr = last_atr
    candle_body = abs(df_5m["close"].iloc[i] - df_5m["open"].iloc[i])
    if candle_body > recent_atr * 1.2:
        return reject("last candle too extended")

    signal = None
    if last_fast > last_slow and RSI_BUY_MIN <= last_rsi <= RSI_BUY_MAX and last_close > last_fast:
        signal = "BUY"
    elif last_fast < last_slow and RSI_SELL_MIN <= last_rsi <= RSI_SELL_MAX and last_close < last_fast:
        signal = "SELL"
    else:
        return reject(f"no setup fast={last_fast:.5f}, slow={last_slow:.5f}, close={last_close:.5f}, RSI={last_rsi:.1f}")

    # H1 hard trend filter
    df_1h = load_rates(symbol, mt5.TIMEFRAME_H1, 120)
    if df_1h is None:
        return reject("H1 data insufficient")
    ema_h1 = compute_ema(df_1h["close"], EMA_TREND_PERIOD_H1)
    h1_close = df_1h["close"].iloc[-1]
    h1_ema = ema_h1.iloc[-1]
    if signal == "BUY" and h1_close <= h1_ema:
        return reject(f"BUY blocked by H1 trend {h1_close:.5f} <= {h1_ema:.5f}")
    if signal == "SELL" and h1_close >= h1_ema:
        return reject(f"SELL blocked by H1 trend {h1_close:.5f} >= {h1_ema:.5f}")

    # H4 soft score, ไม่ตัดทิ้งเพราะบาง broker data ไม่ครบ
    h4_bonus = 0.0
    df_4h = load_rates(symbol, mt5.TIMEFRAME_H4, 120)
    if df_4h is not None:
        ema_h4 = compute_ema(df_4h["close"], EMA_TREND_PERIOD_H4)
        h4_close = df_4h["close"].iloc[-1]
        h4_ema = ema_h4.iloc[-1]
        if signal == "BUY" and h4_close > h4_ema:
            h4_bonus = 4.0
        elif signal == "SELL" and h4_close < h4_ema:
            h4_bonus = 4.0
        else:
            h4_bonus = -5.0

    trend_gap = abs(last_fast - last_slow) / last_atr if last_atr else 0
    rsi_mid = (RSI_BUY_MIN + RSI_BUY_MAX) / 2 if signal == "BUY" else (RSI_SELL_MIN + RSI_SELL_MAX) / 2
    rsi_quality = max(0.0, 20.0 - abs(last_rsi - rsi_mid))
    score = (last_adx - ADX_MIN) * 0.8 + trend_gap * 3.0 + rsi_quality * 0.35 + h4_bonus

    if score < MIN_SCORE_TO_TRADE:
        return reject(f"score too low {score:.2f} < {MIN_SCORE_TO_TRADE}")

    logger.info(
        f"✅ V4.1 Signal {signal} {symbol}: score={score:.2f}, RSI={last_rsi:.1f}, ADX={last_adx:.1f}, ATR={last_atr:.5f}, {reason}"
    )
    return {"signal": signal, "score": score, "atr": float(last_atr)}


def select_best_symbols(max_picks: int) -> List[Tuple[str, dict]]:
    candidates = []
    all_symbols = mt5.symbols_get()
    for base in FOREX_WATCHLIST:
        symbol = find_symbol_name(base, all_symbols)
        if not symbol:
            continue
        if mt5.positions_get(symbol=symbol):
            continue
        analysis = analyze_symbol(symbol, debug=True)
        if analysis:
            candidates.append((symbol, analysis))

    candidates.sort(key=lambda x: x[1]["score"], reverse=True)
    best = candidates[:max_picks]
    logger.info(f"🔍 V4.1 scanned {len(FOREX_WATCHLIST)} pairs, found {len(candidates)} signals. Best: {[b[0] for b in best]}")
    return best


# ==================== ORDER EXECUTION ====================
def calculate_lot(symbol: str, atr: float, balance: float) -> float:
    info = mt5.symbol_info(symbol)
    if info is None or atr <= 0:
        return MIN_LOT

    risk_amount = balance * RISK_PER_TRADE
    stop_distance = atr * SL_ATR_MULT
    tick_size = info.trade_tick_size if info.trade_tick_size else info.point
    tick_value = info.trade_tick_value if info.trade_tick_value else 0

    if tick_size <= 0 or tick_value <= 0:
        lot = MIN_LOT
    else:
        value_per_lot_at_sl = (stop_distance / tick_size) * tick_value
        lot = risk_amount / value_per_lot_at_sl if value_per_lot_at_sl > 0 else MIN_LOT

    lot = max(MIN_LOT, min(float(lot), MAX_LOT))
    step = info.volume_step if info.volume_step else 0.01
    lot = round(lot / step) * step
    lot = max(info.volume_min, min(lot, info.volume_max, MAX_LOT))
    return round(lot, 2)


def execute_trade(symbol: str, analysis: dict) -> bool:
    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        logger.error(f"No tick for {symbol}")
        return False

    ok, reason = spread_ok(symbol)
    if not ok:
        logger.info(f"Skip execution {symbol}: {reason}")
        return False

    signal = analysis["signal"]
    atr = analysis["atr"]

    if signal == "BUY":
        order_type = mt5.ORDER_TYPE_BUY
        entry = tick.ask
        sl = entry - atr * SL_ATR_MULT
        tp = entry + atr * TP_ATR_MULT
    elif signal == "SELL":
        order_type = mt5.ORDER_TYPE_SELL
        entry = tick.bid
        sl = entry + atr * SL_ATR_MULT
        tp = entry - atr * TP_ATR_MULT
    else:
        logger.error(f"Unknown signal {signal}")
        return False

    acc = get_account_info()
    if not acc:
        logger.error("No account info")
        return False
    lot = calculate_lot(symbol, atr, acc["balance"])

    logger.info(f"ORDER PLAN: {signal} {symbol} lot={lot}, entry={entry:.5f}, SL={sl:.5f}, TP={tp:.5f}, {reason}")

    if DRY_RUN:
        logger.info(f"🧪 DRY_RUN: would {signal} {symbol}")
        LAST_TRADE_TIME[symbol] = datetime.now()
        return True

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": order_type,
        "price": entry,
        "sl": sl,
        "tp": tp,
        "deviation": DEVIATION,
        "magic": MAGIC_NUMBER,
        "comment": f"V4_{signal}_{datetime.now().strftime('%H%M')}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = order_send_with_filling_retry(request, symbol, signal)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        retcode = result.retcode if result else "None"
        comment = result.comment if result else mt5.last_error()
        logger.error(f"{signal} {symbol} failed: retcode={retcode}, comment={comment}")
        return False

    LAST_TRADE_TIME[symbol] = datetime.now()
    logger.info(f"✅ {signal} {symbol} {lot} lot @ {entry:.5f}, SL={sl:.5f}, TP={tp:.5f}")
    return True


def refresh_closed_position_cooldown():
    """อัปเดต LAST_CLOSED_TIME จาก deal วันนี้ เพื่อกันเข้าไม้ซ้ำหลังเพิ่งโดน SL/TP"""
    start = datetime.now() - timedelta(hours=12)
    deals = mt5.history_deals_get(start, datetime.now())
    if not deals:
        return
    for d in deals:
        try:
            if d.magic != MAGIC_NUMBER or not d.symbol:
                continue
            # entry 1/3 = OUT/OUT_BY
            if d.entry in (1, 3):
                LAST_CLOSED_TIME[d.symbol] = datetime.fromtimestamp(d.time)
        except Exception:
            continue


# ==================== MAIN LOOP ====================
def main():
    if not connect_mt5():
        return

    logger.info("🚀 MT5 Forex Bot V4 Risk Control started")
    logger.info(f"DRY_RUN={DRY_RUN}, BLOCK_PAIRS={BLOCK_PAIRS}, MAX_CONCURRENT_TRADES={MAX_CONCURRENT_TRADES}")

    last_scan_time = 0.0
    last_news_refresh = 0.0

    while True:
        try:
            acc = get_account_info()
            if not acc:
                time.sleep(30)
                continue

            if acc["daily_pnl_pct"] >= MAX_DAILY_PROFIT_PERCENT:
                logger.info(f"Daily profit target reached {acc['daily_pnl_pct']:.2f}%. Pausing...")
                time.sleep(300)
                continue

            if acc["daily_pnl_pct"] <= -MAX_DAILY_LOSS_PERCENT:
                logger.warning(f"Daily loss limit hit {acc['daily_pnl_pct']:.2f}%. Stopping bot.")
                break

            now = time.time()

            if USE_NEWS_FILTER and now - last_news_refresh >= 3600:
                try:
                    NEWS_CACHE[:] = fetch_news_calendar()
                    logger.info(f"News cache refreshed: {len(NEWS_CACHE)} items")
                except Exception as e:
                    logger.error(f"News refresh failed: {e}")
                last_news_refresh = now

            if now - last_scan_time >= SCAN_INTERVAL:
                refresh_closed_position_cooldown()
                open_positions = get_open_positions()
                current_trades = len([p for p in open_positions if p.magic == MAGIC_NUMBER])

                if current_trades < MAX_CONCURRENT_TRADES:
                    slots = MAX_CONCURRENT_TRADES - current_trades
                    best_pairs = select_best_symbols(slots)
                    for symbol, analysis in best_pairs:
                        execute_trade(symbol, analysis)
                        time.sleep(1)
                else:
                    logger.info(f"Max concurrent trades reached: {current_trades}")

                last_scan_time = now

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
