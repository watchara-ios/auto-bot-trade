import json
import os
import time
from datetime import datetime, time as dtime
from pathlib import Path
import MetaTrader5 as mt5
import pandas as pd
import numpy as np

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

if load_dotenv:
    load_dotenv()


# =========================================================
# CONFIG
# =========================================================
SYMBOL = "XAUUSD"          # แก้ตามชื่อ symbol ใน MT5 เช่น XAUUSD, XAUUSDm, GOLD, EURUSD
SYMBOL_ALIASES = ["XAUUSD", "GOLD"]
LOT = 0.01

# Multi Timeframe
TF_TREND = mt5.TIMEFRAME_H1
TF_SETUP = mt5.TIMEFRAME_M15
TF_ENTRY = mt5.TIMEFRAME_M5

BARS = 500

EMA_FAST = 20
EMA_SLOW = 50
EMA_BIG = 200
RSI_PERIOD = 14

RR = 3.0
SL_BUFFER_POINTS = 100

MAX_TRADES_PER_DAY = 2
MAX_DAILY_LOSS_PERCENT = 3.0

CHECK_INTERVAL_SECONDS = int(os.getenv("FOREX_CHECK_INTERVAL_SECONDS", "300"))

# เทรดเฉพาะช่วงเวลาไทยโดยประมาณ
TRADE_START_HOUR = int(os.getenv("FOREX_TRADE_START_HOUR", "14"))
TRADE_END_HOUR = int(os.getenv("FOREX_TRADE_END_HOUR", "23"))
BLOCK_ENTRY_HOURS = {
    int(hour.strip())
    for hour in os.getenv("FOREX_BLOCK_ENTRY_HOURS", "").split(",")
    if hour.strip()
}
EXIT_AFTER_SESSION_END = True

DRY_RUN = True   # True = ไม่ยิง order จริง / False = ยิงจริง

# DeepSeek daily market scan
AI_DAILY_SCAN_ENABLED = os.getenv("FOREX_AI_DAILY_SCAN_ENABLED", "true").lower() == "true"
REQUIRE_AI_WATCHLIST = os.getenv("FOREX_REQUIRE_AI_WATCHLIST", "true").lower() == "true"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_MODEL = os.getenv("FOREX_DEEPSEEK_MODEL", "deepseek-reasoner")
AI_SCAN_HOUR = int(os.getenv("FOREX_AI_SCAN_HOUR", str(TRADE_START_HOUR)))
AI_STATE_FILE = Path("logs") / "forex_ai_state.json"
AI_RECOMMENDATION_LOG = Path("logs") / "forex_ai_recommendations.jsonl"
AI_MAJOR_PAIRS = [
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "USDCHF",
    "USDCAD",
    "AUDUSD",
    "NZDUSD",
    "EURJPY",
    "GBPJPY",
    "XAUUSD",
]
AI_SYMBOL_UNIVERSE_LIMIT = int(os.getenv("FOREX_AI_SYMBOL_UNIVERSE_LIMIT", "30"))
AI_ALLOWED_BASE_SYMBOLS = [
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "USDCHF",
    "USDCAD",
    "AUDUSD",
    "NZDUSD",
    "EURJPY",
    "GBPJPY",
    "EURGBP",
    "EURCHF",
    "EURCAD",
    "EURAUD",
    "EURNZD",
    "GBPCHF",
    "GBPCAD",
    "GBPAUD",
    "GBPNZD",
    "AUDJPY",
    "CADJPY",
    "CHFJPY",
    "NZDJPY",
    "AUDCAD",
    "AUDCHF",
    "AUDNZD",
    "CADCHF",
    "NZDCAD",
    "NZDCHF",
    "XAUUSD",
]

# Quality filters
MAX_SPREAD_POINTS = 80
MIN_TREND_GAP_PCT = 0.00035
MIN_M5_ATR_POINTS = 120
MAX_M5_ATR_POINTS = 1200
MIN_BODY_RATIO = 0.35
MIN_VOLUME_MULT = 1.0


# =========================================================
# MT5 CONNECT
# =========================================================
def resolve_symbol(preferred_symbol):
    candidates = [preferred_symbol]
    preferred_upper = preferred_symbol.upper()
    if preferred_upper.startswith("XAU") or "GOLD" in preferred_upper:
        for alias in SYMBOL_ALIASES:
            if alias not in candidates:
                candidates.append(alias)

    for name in candidates:
        info = mt5.symbol_info(name)
        if info is not None:
            return name, info

    all_symbols = mt5.symbols_get()
    if all_symbols is None:
        raise RuntimeError(f"Cannot load MT5 symbols: {mt5.last_error()}")

    names = [s.name for s in all_symbols]
    upper_names = [(name, name.upper()) for name in names]

    for alias in candidates:
        alias_upper = alias.upper()
        for name, upper_name in upper_names:
            if upper_name.startswith(alias_upper):
                return name, mt5.symbol_info(name)

    for alias in candidates:
        alias_upper = alias.upper()
        for name, upper_name in upper_names:
            if alias_upper in upper_name:
                return name, mt5.symbol_info(name)

    gold_like = [name for name, upper_name in upper_names if "XAU" in upper_name or "GOLD" in upper_name]
    sample = ", ".join(gold_like[:20]) if gold_like else ", ".join(names[:20])
    raise RuntimeError(f"Symbol not found: {preferred_symbol}. Available similar symbols: {sample}")


def connect_mt5():
    global SYMBOL

    print(f"[{datetime.now()}] 🔌 Connecting MT5...", flush=True)
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")

    resolved_symbol, symbol_info = resolve_symbol(SYMBOL)
    if resolved_symbol != SYMBOL:
        print(f"[{datetime.now()}] 🔎 Symbol resolved: {SYMBOL} -> {resolved_symbol}", flush=True)
        SYMBOL = resolved_symbol

    if not symbol_info.visible:
        if not mt5.symbol_select(SYMBOL, True):
            raise RuntimeError(f"Cannot select symbol in Market Watch: {SYMBOL} | {mt5.last_error()}")
        symbol_info = mt5.symbol_info(SYMBOL)

    print(f"[{datetime.now()}] ✅ Connected MT5 | Symbol={SYMBOL}", flush=True)


def select_trading_symbol(preferred_symbol, reason=""):
    global SYMBOL

    resolved_symbol, symbol_info = resolve_symbol(preferred_symbol)
    if not symbol_info.visible:
        if not mt5.symbol_select(resolved_symbol, True):
            raise RuntimeError(f"Cannot select symbol in Market Watch: {resolved_symbol} | {mt5.last_error()}")

    old_symbol = SYMBOL
    SYMBOL = resolved_symbol
    if old_symbol != SYMBOL:
        print(
            f"[{datetime.now()}] 🎯 Trading symbol switched: {old_symbol} -> {SYMBOL}"
            + (f" | {reason}" if reason else ""),
            flush=True,
        )
    else:
        print(
            f"[{datetime.now()}] 🎯 Trading symbol selected: {SYMBOL}"
            + (f" | {reason}" if reason else ""),
            flush=True,
        )
    return SYMBOL


def normalize_symbol_name(symbol_name):
    upper_name = symbol_name.upper()
    if "XAU" in upper_name or "GOLD" in upper_name:
        return "XAUUSD"

    letters = "".join(ch for ch in upper_name if ch.isalpha())
    for base_symbol in AI_ALLOWED_BASE_SYMBOLS:
        if letters.startswith(base_symbol):
            return base_symbol
    return None


def get_ai_candidate_symbols():
    all_symbols = mt5.symbols_get()
    if not all_symbols:
        return AI_MAJOR_PAIRS

    available = {}
    for item in all_symbols:
        base_symbol = normalize_symbol_name(item.name)
        if not base_symbol:
            continue
        if base_symbol not in available:
            available[base_symbol] = item.name

    ordered = []
    for symbol in AI_MAJOR_PAIRS + AI_ALLOWED_BASE_SYMBOLS:
        if symbol in available and symbol not in ordered:
            ordered.append(symbol)

    return ordered[:AI_SYMBOL_UNIVERSE_LIMIT] or AI_MAJOR_PAIRS


# =========================================================
# DATA
# =========================================================
def get_ohlcv(symbol, timeframe, bars=500):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)

    if rates is None or len(rates) == 0:
        raise RuntimeError(f"No data from MT5: {symbol}")

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.rename(columns={"tick_volume": "volume"}, inplace=True)

    return df


# =========================================================
# INDICATORS
# =========================================================
def add_indicators(df):
    df = df.copy()

    df["ema20"] = df["close"].ewm(span=EMA_FAST, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=EMA_SLOW, adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=EMA_BIG, adjust=False).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(RSI_PERIOD).mean()
    avg_loss = loss.rolling(RSI_PERIOD).mean()

    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))

    df["vol_avg"] = df["volume"].rolling(20).mean()
    df["ema20_50_gap_pct"] = abs(df["ema20"] - df["ema50"]) / df["close"]
    df["body_ratio"] = abs(df["close"] - df["open"]) / (df["high"] - df["low"]).replace(0, np.nan)

    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()

    return df


# =========================================================
# SWING / SUPPORT / RESISTANCE
# =========================================================
def add_swing_levels(df, lookback=5):
    df = df.copy()
    df["swing_high"] = False
    df["swing_low"] = False

    for i in range(lookback, len(df) - lookback):
        high_range = df["high"].iloc[i - lookback:i + lookback + 1]
        low_range = df["low"].iloc[i - lookback:i + lookback + 1]

        if df["high"].iloc[i] == high_range.max():
            df.loc[df.index[i], "swing_high"] = True

        if df["low"].iloc[i] == low_range.min():
            df.loc[df.index[i], "swing_low"] = True

    return df


def get_last_sr(df):
    swing_highs = df[df["swing_high"]]
    swing_lows = df[df["swing_low"]]

    resistance = swing_highs["high"].iloc[-1] if len(swing_highs) else None
    support = swing_lows["low"].iloc[-1] if len(swing_lows) else None

    return support, resistance


# =========================================================
# TREND / BOS
# =========================================================
def detect_trend(df):
    last = df.iloc[-2]  # ใช้แท่งปิดแล้ว ไม่ใช้แท่งกำลังวิ่ง

    if last["ema20_50_gap_pct"] < MIN_TREND_GAP_PCT:
        return "SIDEWAY"

    if last["close"] > last["ema20"] > last["ema50"] and last["close"] > last["ema200"]:
        return "UP"

    if last["close"] < last["ema20"] < last["ema50"] and last["close"] < last["ema200"]:
        return "DOWN"

    return "SIDEWAY"


def detect_bos(df):
    last = df.iloc[-2]
    old_df = df.iloc[:-2]

    support, resistance = get_last_sr(old_df)

    if resistance and last["close"] > resistance:
        return "BULLISH_BOS"

    if support and last["close"] < support:
        return "BEARISH_BOS"

    return "NO_BOS"


# =========================================================
# DAILY LIMIT
# =========================================================
def get_today_orders_count():
    today = datetime.now().date()
    start = datetime.combine(today, dtime.min)
    end = datetime.combine(today, dtime.max)

    deals = mt5.history_deals_get(start, end)
    if deals is None:
        return 0

    count = 0
    for d in deals:
        if d.symbol == SYMBOL and d.entry == mt5.DEAL_ENTRY_IN:
            count += 1

    return count


def get_today_profit():
    today = datetime.now().date()
    start = datetime.combine(today, dtime.min)
    end = datetime.combine(today, dtime.max)

    deals = mt5.history_deals_get(start, end)
    if deals is None:
        return 0.0

    profit = 0.0
    for d in deals:
        if d.symbol == SYMBOL:
            profit += d.profit

    return profit


def pass_daily_risk_filter():
    account = mt5.account_info()
    if account is None:
        return False, "No account info"

    today_orders = get_today_orders_count()
    today_profit = get_today_profit()

    max_loss_money = account.balance * (MAX_DAILY_LOSS_PERCENT / 100)

    if today_orders >= MAX_TRADES_PER_DAY:
        return False, f"Max trades reached: {today_orders}/{MAX_TRADES_PER_DAY}"

    if today_profit <= -max_loss_money:
        return False, f"Daily loss limit reached: {today_profit:.2f}"

    return True, "Daily risk OK"


# =========================================================
# SESSION FILTER
# =========================================================
def pass_session_filter():
    now = datetime.now()
    hour = now.hour

    if hour in BLOCK_ENTRY_HOURS:
        print(f"[{datetime.now()}] ⛔ Entry blocked by FOREX_BLOCK_ENTRY_HOURS={sorted(BLOCK_ENTRY_HOURS)}")
        return False

    if TRADE_START_HOUR <= hour <= TRADE_END_HOUR:
        return True

    return False


def should_exit_after_session():
    if not EXIT_AFTER_SESSION_END:
        return False
    return datetime.now().hour > TRADE_END_HOUR


def get_spread_points():
    tick = mt5.symbol_info_tick(SYMBOL)
    info = mt5.symbol_info(SYMBOL)
    if tick is None or info is None or info.point <= 0:
        return None
    return (tick.ask - tick.bid) / info.point


def pass_spread_filter():
    spread_points = get_spread_points()
    if spread_points is None:
        return False, "No tick/spread info"
    if spread_points > MAX_SPREAD_POINTS:
        return False, f"Spread too wide: {spread_points:.1f} points > {MAX_SPREAD_POINTS}"
    return True, f"Spread OK: {spread_points:.1f} points"


# =========================================================
# DEEPSEEK DAILY MARKET SCAN
# =========================================================
def read_ai_state():
    if AI_STATE_FILE.exists():
        try:
            return json.loads(AI_STATE_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def write_ai_state(state):
    AI_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    AI_STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def clean_ai_json(content):
    content = content.strip()
    if "```json" in content:
        content = content.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in content:
        content = content.split("```", 1)[1].split("```", 1)[0]
    return json.loads(content.strip())


def build_daily_ai_prompt(candidate_symbols):
    now = datetime.now()
    return f"""You are a professional Forex analyst. Every day at 14:00 (Thailand time, UTC+7), you analyze the current market situation and rate major currency pairs based on the criteria below.

**Time context:** 2:00 PM Thailand time (UTC+7) – this is the London-New York overlap session, offering high liquidity and moderate to high volatility.

**Factors to consider (latest available data):**
- Recent major economic news and upcoming events (within 2–3 hours) – e.g., NFP, CPI, GDP, PMI, central bank meetings
- Latest geopolitical events affecting USD, EUR, JPY, GBP
- Technical trends (key support/resistance, turning points, timeframe 1H/4H)
- Currency correlations (e.g., EUR/USD vs USD/CHF)
- Session-specific suitability – at 14:00 UTC+7, pairs with EUR, GBP, USD are most active

**Candidate symbols available in this MT5 account:** {", ".join(candidate_symbols)}

**Output format:**
Return ONLY a valid JSON object. No explanations, no markdown, no extra text before or after.
DeepSeek-R1: You must NOT output your reasoning chain – only the final JSON.

**JSON schema:**
{{
  "timestamp": "YYYY-MM-DD HH:MM:SS",
  "recommendations": [
    {{
      "symbol": "EURUSD",
      "rating": 1,
      "reason": "Short reason (one sentence, max 80 chars)"
    }},
    {{
      "symbol": "GBPUSD",
      "rating": 2,
      "reason": "..."
    }}
  ]
}}

**Rating scale:**
- rating 1 = Most tradable (high profit probability, low-medium risk)
- rating 2 = Very tradable
- rating 3 = Moderately tradable
- rating 4 = Caution (high risk or unclear signals)
- rating 5 = Avoid trading

**Rules:**
- Include only pairs with rating 1, 2, or 3 (minimum 3 pairs, maximum 7 pairs)
- Each reason must reference the 2:00 PM session (e.g., London-NY overlap, upcoming news, 1H chart pattern)
- Use current date for "timestamp": {now.strftime("%Y-%m-%d")}

Now, analyze the Forex market and return only the JSON."""


def build_ai_watchlist(recommendations, candidate_symbols=None):
    candidate_symbols = candidate_symbols or get_ai_candidate_symbols()
    ranked = sorted(
        [r for r in recommendations if int(r.get("rating", 99)) in {1, 2, 3}],
        key=lambda r: (
            int(r.get("rating", 99)),
            candidate_symbols.index(str(r.get("symbol", "")).upper())
            if str(r.get("symbol", "")).upper() in candidate_symbols
            else 999,
        ),
    )

    watchlist = []
    errors = []
    for rec in ranked:
        symbol = str(rec.get("symbol", "")).strip().upper()
        if not symbol:
            continue
        try:
            resolved_symbol, symbol_info = resolve_symbol(symbol)
            if not symbol_info.visible:
                if not mt5.symbol_select(resolved_symbol, True):
                    raise RuntimeError(f"Cannot select symbol in Market Watch: {resolved_symbol} | {mt5.last_error()}")
            watchlist.append({
                "symbol": resolved_symbol,
                "rating": int(rec.get("rating", 99)),
                "reason": rec.get("reason", ""),
                "source_symbol": symbol,
            })
        except Exception as e:
            errors.append(f"{symbol}: {e}")

    if watchlist:
        symbols_text = ", ".join([f"{item['symbol']}(r{item['rating']})" for item in watchlist])
        print(f"[{datetime.now()}] 🧠 AI MT5 watchlist: {symbols_text}", flush=True)
    else:
        print(f"[{datetime.now()}] 🧠 No AI recommended symbol is available in MT5: {' | '.join(errors)}", flush=True)
    return watchlist


def choose_ai_trading_symbol(recommendations):
    watchlist = build_ai_watchlist(recommendations)
    if not watchlist:
        return None, None
    first = watchlist[0]
    selected = select_trading_symbol(
        first["symbol"],
        f"AI rating={first['rating']} reason={first['reason']}",
    )
    return selected, first


def get_trading_watchlist():
    today = datetime.now().strftime("%Y-%m-%d")
    state = read_ai_state()

    if AI_DAILY_SCAN_ENABLED and REQUIRE_AI_WATCHLIST:
        if state.get("last_scan_date") != today:
            print(
                f"[{datetime.now()}] 🧠 Waiting for today's AI news scan before trading",
                flush=True,
            )
            return []

    symbols = state.get("candidate_symbols") or []
    if not symbols and state.get("selected_symbol"):
        symbols = [state["selected_symbol"]]
    if not symbols and AI_DAILY_SCAN_ENABLED and REQUIRE_AI_WATCHLIST:
        print(
            f"[{datetime.now()}] 🧠 Today's AI scan has no tradable MT5 watchlist; skip entries",
            flush=True,
        )
        return []
    if not symbols:
        symbols = [SYMBOL]

    unique_symbols = []
    for symbol in symbols:
        if symbol and symbol not in unique_symbols:
            unique_symbols.append(symbol)
    return unique_symbols


def apply_ai_selected_symbol_from_state():
    state = read_ai_state()
    selected_symbol = state.get("selected_symbol")
    if not selected_symbol:
        return
    try:
        select_trading_symbol(selected_symbol, "loaded from today's AI state")
    except Exception as e:
        print(f"[{datetime.now()}] 🧠 Cannot apply AI selected symbol {selected_symbol}: {e}", flush=True)


def run_daily_ai_market_scan_once():
    if not AI_DAILY_SCAN_ENABLED:
        return

    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    if now.hour < AI_SCAN_HOUR:
        return

    state = read_ai_state()
    if state.get("last_scan_date") == today:
        if not state.get("candidate_symbols") and state.get("last_recommendations"):
            watchlist = build_ai_watchlist(state["last_recommendations"])
            if watchlist:
                state["candidate_symbols"] = [item["symbol"] for item in watchlist]
                state["candidate_recommendations"] = watchlist
                state["selected_symbol"] = watchlist[0]["symbol"]
                state["selected_recommendation"] = watchlist[0]
                write_ai_state(state)
        return

    # New trading day: do not let stale recommendations become today's watchlist.
    state.pop("candidate_symbols", None)
    state.pop("candidate_recommendations", None)
    state.pop("selected_symbol", None)
    state.pop("selected_recommendation", None)

    if not DEEPSEEK_API_KEY:
        print(f"[{datetime.now()}] 🧠 Skip DeepSeek daily scan: missing DEEPSEEK_API_KEY", flush=True)
        state["last_scan_date"] = today
        state["last_scan_at"] = now.strftime("%Y-%m-%d %H:%M:%S")
        state["last_scan_error"] = "missing DEEPSEEK_API_KEY"
        write_ai_state(state)
        return
    if OpenAI is None:
        print(f"[{datetime.now()}] 🧠 Skip DeepSeek daily scan: openai package not installed", flush=True)
        state["last_scan_date"] = today
        state["last_scan_at"] = now.strftime("%Y-%m-%d %H:%M:%S")
        state["last_scan_error"] = "openai package not installed"
        write_ai_state(state)
        return

    try:
        print(f"[{datetime.now()}] 🧠 DeepSeek daily forex scan starting...", flush=True)
        candidate_symbols = get_ai_candidate_symbols()
        print(f"[{datetime.now()}] 🧠 MT5 candidate symbols for AI: {', '.join(candidate_symbols)}", flush=True)
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1")
        resp = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": build_daily_ai_prompt(candidate_symbols)}],
            temperature=0.2,
            max_tokens=700,
        )
        result = clean_ai_json(resp.choices[0].message.content)
        result.setdefault("timestamp", now.strftime("%Y-%m-%d %H:%M:%S"))

        AI_RECOMMENDATION_LOG.parent.mkdir(parents=True, exist_ok=True)
        with AI_RECOMMENDATION_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        state["last_scan_date"] = today
        state["last_scan_at"] = now.strftime("%Y-%m-%d %H:%M:%S")
        state["candidate_universe"] = candidate_symbols
        state["last_recommendations"] = result.get("recommendations", [])
        watchlist = build_ai_watchlist(state["last_recommendations"], candidate_symbols)
        if watchlist:
            state["candidate_symbols"] = [item["symbol"] for item in watchlist]
            state["candidate_recommendations"] = watchlist
            state["selected_symbol"] = watchlist[0]["symbol"]
            state["selected_recommendation"] = watchlist[0]
        write_ai_state(state)

        print(f"[{datetime.now()}] 🧠 DeepSeek recommendations: {json.dumps(result, ensure_ascii=False)}", flush=True)
    except Exception as e:
        print(f"[{datetime.now()}] 🧠 DeepSeek daily scan failed: {e}", flush=True)


# =========================================================
# POSITION FILTER
# =========================================================
def has_open_position():
    positions = mt5.positions_get(symbol=SYMBOL)
    return positions is not None and len(positions) > 0


def has_any_open_position(symbols):
    positions = mt5.positions_get()
    if positions is None:
        return False, ""

    symbol_set = set(symbols)
    for pos in positions:
        if pos.symbol in symbol_set:
            return True, pos.symbol
    return False, ""


# =========================================================
# SIGNAL LOGIC
# =========================================================
def generate_signal():
    df_h1 = add_swing_levels(add_indicators(get_ohlcv(SYMBOL, TF_TREND, BARS)))
    df_m15 = add_swing_levels(add_indicators(get_ohlcv(SYMBOL, TF_SETUP, BARS)))
    df_m5 = add_swing_levels(add_indicators(get_ohlcv(SYMBOL, TF_ENTRY, BARS)))

    h1_trend = detect_trend(df_h1)
    m15_trend = detect_trend(df_m15)
    m15_bos = detect_bos(df_m15)

    entry_candle = df_m5.iloc[-2]
    prev_candle = df_m5.iloc[-3]

    support, resistance = get_last_sr(df_m5.iloc[:-2])

    signal = {
        "time": entry_candle["time"],
        "symbol": SYMBOL,
        "side": "NO_TRADE",
        "price": float(entry_candle["close"]),
        "h1_trend": h1_trend,
        "m15_trend": m15_trend,
        "m15_bos": m15_bos,
        "rsi_m5": float(entry_candle["rsi"]),
        "volume": float(entry_candle["volume"]),
        "vol_avg": float(entry_candle["vol_avg"]) if not np.isnan(entry_candle["vol_avg"]) else 0,
        "atr_points": None,
        "body_ratio": float(entry_candle["body_ratio"]) if not np.isnan(entry_candle["body_ratio"]) else 0,
        "h1_gap_pct": float(df_h1.iloc[-2]["ema20_50_gap_pct"]),
        "m15_gap_pct": float(df_m15.iloc[-2]["ema20_50_gap_pct"]),
        "support": support,
        "resistance": resistance,
        "reason": ""
    }

    info = mt5.symbol_info(SYMBOL)
    point = info.point if info and info.point > 0 else 0
    atr_points = float(entry_candle["atr"] / point) if point and not np.isnan(entry_candle["atr"]) else 0
    signal["atr_points"] = round(atr_points, 1)

    quality_failures = []
    if atr_points < MIN_M5_ATR_POINTS or atr_points > MAX_M5_ATR_POINTS:
        quality_failures.append(f"ATR points {atr_points:.1f} outside {MIN_M5_ATR_POINTS}-{MAX_M5_ATR_POINTS}")
    if signal["body_ratio"] < MIN_BODY_RATIO:
        quality_failures.append(f"body ratio {signal['body_ratio']:.2f} < {MIN_BODY_RATIO}")
    if signal["volume"] < signal["vol_avg"] * MIN_VOLUME_MULT:
        quality_failures.append(f"volume {signal['volume']:.0f} < avg*{MIN_VOLUME_MULT}")
    if signal["h1_gap_pct"] < MIN_TREND_GAP_PCT or signal["m15_gap_pct"] < MIN_TREND_GAP_PCT:
        quality_failures.append(
            f"trend gap weak H1={signal['h1_gap_pct']:.5f} M15={signal['m15_gap_pct']:.5f}"
        )
    if quality_failures:
        signal["reason"] = "Quality filter fail: " + "; ".join(quality_failures)
        return signal, df_m5

    # BUY setup
    buy_condition = (
        h1_trend == "UP"
        and m15_trend == "UP"
        and m15_bos in ["BULLISH_BOS", "NO_BOS"]
        and entry_candle["close"] > entry_candle["ema20"]
        and entry_candle["ema20"] > entry_candle["ema50"]
        and 45 <= entry_candle["rsi"] <= 65
        and entry_candle["close"] > prev_candle["close"]
        and entry_candle["close"] > prev_candle["high"]
    )

    # SELL setup
    sell_condition = (
        h1_trend == "DOWN"
        and m15_trend == "DOWN"
        and m15_bos in ["BEARISH_BOS", "NO_BOS"]
        and entry_candle["close"] < entry_candle["ema20"]
        and entry_candle["ema20"] < entry_candle["ema50"]
        and 35 <= entry_candle["rsi"] <= 55
        and entry_candle["close"] < prev_candle["close"]
        and entry_candle["close"] < prev_candle["low"]
    )

    if buy_condition:
        signal["side"] = "BUY"
        signal["reason"] = "H1 UP + M15 UP + M5 pullback continuation"

    elif sell_condition:
        signal["side"] = "SELL"
        signal["reason"] = "H1 DOWN + M15 DOWN + M5 pullback continuation"

    else:
        signal["reason"] = "No clean multi-timeframe setup"

    return signal, df_m5


# =========================================================
# TP / SL
# =========================================================
def calculate_tp_sl(signal):
    info = mt5.symbol_info(SYMBOL)
    if info is None:
        raise RuntimeError("No symbol info")

    point = info.point
    entry = signal["price"]
    side = signal["side"]

    support = signal["support"]
    resistance = signal["resistance"]
    atr_distance = max(signal.get("atr_points") or MIN_M5_ATR_POINTS, MIN_M5_ATR_POINTS) * point
    min_stop = MIN_M5_ATR_POINTS * point * 0.8
    max_stop = MAX_M5_ATR_POINTS * point

    if side == "BUY":
        if support:
            sl = support - SL_BUFFER_POINTS * point
        else:
            sl = entry - atr_distance

        if abs(entry - sl) < min_stop or abs(entry - sl) > max_stop:
            sl = entry - atr_distance

        risk = abs(entry - sl)
        tp = entry + risk * RR

    elif side == "SELL":
        if resistance:
            sl = resistance + SL_BUFFER_POINTS * point
        else:
            sl = entry + atr_distance

        if abs(entry - sl) < min_stop or abs(entry - sl) > max_stop:
            sl = entry + atr_distance

        risk = abs(entry - sl)
        tp = entry - risk * RR

    else:
        return None

    digits = info.digits

    return {
        "entry": round(entry, digits),
        "sl": round(sl, digits),
        "tp": round(tp, digits),
        "risk_points": round(abs(entry - sl) / point, 1),
        "reward_points": round(abs(tp - entry) / point, 1),
        "rr": RR
    }


# =========================================================
# ORDER
# =========================================================
def send_order(signal, tp_sl):
    tick = mt5.symbol_info_tick(SYMBOL)
    info = mt5.symbol_info(SYMBOL)

    if tick is None or info is None:
        print("❌ No tick/info")
        return None

    side = signal["side"]

    if side == "BUY":
        order_type = mt5.ORDER_TYPE_BUY
        price = tick.ask

    elif side == "SELL":
        order_type = mt5.ORDER_TYPE_SELL
        price = tick.bid

    else:
        return None

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": LOT,
        "type": order_type,
        "price": price,
        "sl": tp_sl["sl"],
        "tp": tp_sl["tp"],
        "deviation": 30,
        "magic": 20260424,
        "comment": "MTF_PRO_BOT",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    if DRY_RUN:
        print("🧪 DRY_RUN ORDER")
        print(request)
        return request

    result = mt5.order_send(request)
    print("📌 ORDER RESULT:", result)
    return result


# =========================================================
# MAIN LOOP
# =========================================================
def run_bot():
    print(
        f"[{datetime.now()}] 🚀 Forex bot booting | pid={os.getpid()} | cwd={os.getcwd()} | "
        f"session={TRADE_START_HOUR}:00-{TRADE_END_HOUR}:59 | blocked_hours={sorted(BLOCK_ENTRY_HOURS)} | "
        f"interval={CHECK_INTERVAL_SECONDS}s | DRY_RUN={DRY_RUN}",
        flush=True,
    )
    connect_mt5()

    last_entry_candle_time = {}

    print(f"[{datetime.now()}] 🚀 Bot started", flush=True)

    while True:
        try:
            run_daily_ai_market_scan_once()

            if not pass_session_filter():
                if should_exit_after_session():
                    print(f"[{datetime.now()}] 🌙 Session ended, bot exiting")
                    break
                print(f"[{datetime.now()}] ⏳ Outside trading session")
                time.sleep(CHECK_INTERVAL_SECONDS)
                continue

            watchlist = get_trading_watchlist()
            if not watchlist:
                print(f"[{datetime.now()}] ⏳ No AI-selected symbol for today, waiting", flush=True)
                time.sleep(CHECK_INTERVAL_SECONDS)
                continue

            print(f"[{datetime.now()}] 👀 AI watchlist scan: {', '.join(watchlist)}", flush=True)

            has_position, position_symbol = has_any_open_position(watchlist)
            if has_position:
                print(f"[{datetime.now()}] 📌 Existing position detected on {position_symbol}, skip new entries")
                time.sleep(CHECK_INTERVAL_SECONDS)
                continue

            order_sent = False
            for preferred_symbol in watchlist:
                try:
                    select_trading_symbol(preferred_symbol, "AI watchlist candidate")
                    print(f"[{datetime.now()}] 🔎 Analyzing {SYMBOL}", flush=True)

                    spread_ok, spread_msg = pass_spread_filter()
                    if not spread_ok:
                        print(f"[{datetime.now()}] ⏸ {SYMBOL} {spread_msg}")
                        continue

                    risk_ok, risk_msg = pass_daily_risk_filter()
                    if not risk_ok:
                        print(f"[{datetime.now()}] 🛑 {SYMBOL} {risk_msg}")
                        continue

                    if has_open_position():
                        print(f"[{datetime.now()}] 📌 {SYMBOL} existing position detected, skip")
                        continue

                    signal, df_m5 = generate_signal()

                    current_candle_time = signal["time"]

                    # กันยิงซ้ำในแท่งเดียวกัน แยกตาม symbol
                    if last_entry_candle_time.get(SYMBOL) == current_candle_time:
                        print(f"[{datetime.now()}] ⏸ {SYMBOL} already checked candle {current_candle_time}")
                        continue

                    print("\n==============================")
                    print(f"Time: {datetime.now()}")
                    print("SIGNAL:", signal)

                    if signal["side"] == "NO_TRADE":
                        print(f"⏸ {SYMBOL} No trade")
                        last_entry_candle_time[SYMBOL] = current_candle_time
                        continue

                    tp_sl = calculate_tp_sl(signal)
                    print("TP/SL:", tp_sl)

                    send_order(signal, tp_sl)

                    last_entry_candle_time[SYMBOL] = current_candle_time
                    order_sent = True
                    break

                except Exception as e:
                    print(f"[{datetime.now()}] ❌ {preferred_symbol} analysis error: {e}", flush=True)

            if not order_sent:
                print(f"[{datetime.now()}] ⏳ No order from AI watchlist this round", flush=True)

            time.sleep(CHECK_INTERVAL_SECONDS)

        except KeyboardInterrupt:
            print("🛑 Bot stopped by user")
            break

        except Exception as e:
            print("❌ ERROR:", e)
            time.sleep(CHECK_INTERVAL_SECONDS)

    mt5.shutdown()


if __name__ == "__main__":
    run_bot()
