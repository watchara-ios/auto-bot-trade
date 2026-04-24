import time
from datetime import datetime, time as dtime
import MetaTrader5 as mt5
import pandas as pd
import numpy as np


# =========================================================
# CONFIG
# =========================================================
SYMBOL = "XAUUSD"          # แก้ตามชื่อ symbol ใน MT5 เช่น XAUUSD, BTCUSD, EURUSD
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

RR = 1.0
SL_BUFFER_POINTS = 100

MAX_TRADES_PER_DAY = 5
MAX_DAILY_LOSS_PERCENT = 3.0

CHECK_INTERVAL_SECONDS = 30

# เทรดเฉพาะช่วงเวลาไทยโดยประมาณ
TRADE_START_HOUR = 14
TRADE_END_HOUR = 23

DRY_RUN = True   # True = ไม่ยิง order จริง / False = ยิงจริง


# =========================================================
# MT5 CONNECT
# =========================================================
def connect_mt5():
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")

    symbol_info = mt5.symbol_info(SYMBOL)
    if symbol_info is None:
        raise RuntimeError(f"Symbol not found: {SYMBOL}")

    if not symbol_info.visible:
        mt5.symbol_select(SYMBOL, True)

    print(f"✅ Connected MT5 | Symbol={SYMBOL}")


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

    if TRADE_START_HOUR <= hour <= TRADE_END_HOUR:
        return True

    return False


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
        "support": support,
        "resistance": resistance,
        "reason": ""
    }

    # BUY setup
    buy_condition = (
        h1_trend == "UP"
        and m15_trend == "UP"
        and entry_candle["close"] > entry_candle["ema20"]
        and entry_candle["ema20"] > entry_candle["ema50"]
        and 40 <= entry_candle["rsi"] <= 68
        and entry_candle["volume"] >= entry_candle["vol_avg"] * 0.8
        and entry_candle["close"] > prev_candle["close"]
    )

    # SELL setup
    sell_condition = (
        h1_trend == "DOWN"
        and m15_trend == "DOWN"
        and entry_candle["close"] < entry_candle["ema20"]
        and entry_candle["ema20"] < entry_candle["ema50"]
        and 32 <= entry_candle["rsi"] <= 60
        and entry_candle["volume"] >= entry_candle["vol_avg"] * 0.8
        and entry_candle["close"] < prev_candle["close"]
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

    if side == "BUY":
        if support:
            sl = support - SL_BUFFER_POINTS * point
        else:
            sl = entry - 900 * point

        risk = abs(entry - sl)
        tp = entry + risk * RR

    elif side == "SELL":
        if resistance:
            sl = resistance + SL_BUFFER_POINTS * point
        else:
            sl = entry + 900 * point

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
    connect_mt5()

    last_entry_candle_time = None

    print("🚀 Bot started")
    print(f"DRY_RUN={DRY_RUN}")

    while True:
        try:
            if not pass_session_filter():
                print(f"[{datetime.now()}] ⏳ Outside trading session")
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