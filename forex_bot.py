import os
import time
from datetime import datetime, time as dtime
import MetaTrader5 as mt5
import pandas as pd
import numpy as np


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

RR = 1.5
SL_BUFFER_POINTS = 100

MAX_TRADES_PER_DAY = 2
MAX_DAILY_LOSS_PERCENT = 3.0

CHECK_INTERVAL_SECONDS = 30

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
# POSITION FILTER
# =========================================================
def has_open_position():
    positions = mt5.positions_get(symbol=SYMBOL)
    return positions is not None and len(positions) > 0


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
        f"DRY_RUN={DRY_RUN}",
        flush=True,
    )
    connect_mt5()

    last_entry_candle_time = None

    print(f"[{datetime.now()}] 🚀 Bot started", flush=True)

    while True:
        try:
            if not pass_session_filter():
                if should_exit_after_session():
                    print(f"[{datetime.now()}] 🌙 Session ended, bot exiting")
                    break
                print(f"[{datetime.now()}] ⏳ Outside trading session")
                time.sleep(CHECK_INTERVAL_SECONDS)
                continue

            spread_ok, spread_msg = pass_spread_filter()
            if not spread_ok:
                print(f"[{datetime.now()}] ⏸ {spread_msg}")
                time.sleep(CHECK_INTERVAL_SECONDS)
                continue

            risk_ok, risk_msg = pass_daily_risk_filter()
            if not risk_ok:
                print(f"[{datetime.now()}] 🛑 {risk_msg}")
                time.sleep(CHECK_INTERVAL_SECONDS)
                continue

            if has_open_position():
                print(f"[{datetime.now()}] 📌 Existing position detected, skip")
                time.sleep(CHECK_INTERVAL_SECONDS)
                continue

            signal, df_m5 = generate_signal()

            current_candle_time = signal["time"]

            # กันยิงซ้ำในแท่งเดียวกัน
            if last_entry_candle_time == current_candle_time:
                time.sleep(CHECK_INTERVAL_SECONDS)
                continue

            print("\n==============================")
            print(f"Time: {datetime.now()}")
            print("SIGNAL:", signal)

            if signal["side"] == "NO_TRADE":
                print("⏸ No trade")
                last_entry_candle_time = current_candle_time
                time.sleep(CHECK_INTERVAL_SECONDS)
                continue

            tp_sl = calculate_tp_sl(signal)
            print("TP/SL:", tp_sl)

            send_order(signal, tp_sl)

            last_entry_candle_time = current_candle_time

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
