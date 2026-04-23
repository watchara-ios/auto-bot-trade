import pandas as pd
import numpy as np

# =====================
# LOAD DATA
# =====================
def load_data(path):
    df = pd.read_csv(path)

    if 'time' not in df.columns:
        df.columns = ['time','open','high','low','close','volume']

    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time')
    df.set_index('time', inplace=True)

    return df


# =====================
# INDICATORS
# =====================
def ema(series, period):
    return series.ewm(span=period).mean()

def atr(df, period=14):
    high = df['high']
    low = df['low']
    close = df['close']

    tr = np.maximum(
        high - low,
        np.maximum(abs(high - close.shift()), abs(low - close.shift()))
    )
    return tr.rolling(period).mean()


# =====================
# RESAMPLE H1
# =====================
def to_h1(df_m5):
    return df_m5.resample('1h').agg({
        'open':'first',
        'high':'max',
        'low':'min',
        'close':'last',
        'volume':'sum'
    }).dropna()


# =====================
# BACKTEST HARDCORE
# =====================
def backtest(df_m5):

    df_h1 = to_h1(df_m5)

    # Indicators H1
    df_h1['ema50'] = ema(df_h1['close'], 50)
    df_h1['ema200'] = ema(df_h1['close'], 200)

    # Indicators M5
    df_m5['atr'] = atr(df_m5)
    df_m5['ema20'] = ema(df_m5['close'], 20)
    df_m5['vol_mean'] = df_m5['volume'].rolling(20).mean()
    df_m5['atr_mean'] = df_m5['atr'].rolling(50).mean()

    balance = 1000
    start_balance = balance

    risk_per_trade = 0.01  # โหด
    max_dd = -0.25

    trades = 0
    wins = 0

    peak = balance
    max_drawdown = 0

    logs = []

    daily_trades = 0
    last_day = None

    for i in range(100, len(df_m5)-1):

        row = df_m5.iloc[i]
        time = row.name

        # reset daily trades
        if last_day != time.date():
            daily_trades = 0
            last_day = time.date()

        if daily_trades >= 5:
            continue

        # match H1
        h1 = df_h1[df_h1.index <= time].iloc[-1]

        trend_up = h1['ema50'] > h1['ema200']
        trend_down = h1['ema50'] < h1['ema200']

        trend_strength = abs(h1['ema50'] - h1['ema200']) / h1['close']
        if trend_strength < 0.003:
            logs.append((time,'SKIP','weak_trend'))
            continue

        atr_now = row['atr']
        atr_mean = row['atr_mean']

        if np.isnan(atr_now) or np.isnan(atr_mean):
            continue

        # volatility expansion
        if atr_now < atr_mean * 1.2:
            logs.append((time,'SKIP','no_expansion'))
            continue

        # volume spike
        if row['volume'] < row['vol_mean'] * 1.5:
            logs.append((time,'SKIP','no_volume'))
            continue

        # breakout
        lookback = df_m5.iloc[i-20:i]
        high_range = lookback['high'].max()
        low_range = lookback['low'].min()

        price = row['close']
        open_ = row['open']
        high = row['high']
        low = row['low']

        body = abs(price - open_)
        candle_range = high - low if high != low else 0.0001

        if body / candle_range < 0.6:
            logs.append((time,'SKIP','fake_breakout'))
            continue

        signal = None

        if trend_up and price > high_range:
            signal = "BUY"

        elif trend_down and price < low_range:
            signal = "SELL"

        if not signal:
            continue

        # SL / TP
        sl = 1.5 * atr_now
        tp = 3.5 * atr_now

        risk_amount = balance * risk_per_trade
        qty = risk_amount / sl

        entry = price

        if signal == "BUY":
            sl_price = entry - sl
            tp_price = entry + tp
        else:
            sl_price = entry + sl
            tp_price = entry - tp

        trades += 1
        daily_trades += 1

        # simulate forward
        for j in range(i+1, len(df_m5)):
            future = df_m5.iloc[j]

            if signal == "BUY":
                if future['low'] <= sl_price:
                    balance -= risk_amount
                    logs.append((future.name,'SL','buy_fail'))
                    break

                if future['high'] >= tp_price:
                    profit = risk_amount * (tp/sl)
                    balance += profit
                    wins += 1
                    logs.append((future.name,'TP','buy_win'))
                    break

            else:
                if future['high'] >= sl_price:
                    balance -= risk_amount
                    logs.append((future.name,'SL','sell_fail'))
                    break

                if future['low'] <= tp_price:
                    profit = risk_amount * (tp/sl)
                    balance += profit
                    wins += 1
                    logs.append((future.name,'TP','sell_win'))
                    break

        # drawdown
        peak = max(peak, balance)
        dd = (balance - peak) / peak
        max_drawdown = min(max_drawdown, dd)

        if dd <= max_dd:
            print("STOP: Max DD reached")
            break

    result = {
        "start_balance": start_balance,
        "end_balance": balance,
        "trades": trades,
        "winrate": (wins / trades * 100) if trades > 0 else 0,
        "max_drawdown": max_drawdown,
        "logs": logs[:50]
    }

    return result


# =====================
# RUN
# =====================
if __name__ == "__main__":
    df = load_data("btc_5m.csv")

    result = backtest(df)

    print("\n🔥 RESULT V13 HARDCORE")
    print(result)

    pd.DataFrame(result['logs'], columns=['time','type','reason']) \
        .to_csv("log_v13_hardcore.csv", index=False)