import pandas as pd
import numpy as np

# =====================
# BACKTEST V20 - BALANCED EDGE
# =====================
VERSION = "V20_BALANCED_EDGE"
START_BALANCE = 1000.0

# Risk / Reward
RISK = 0.005
RR = 2.1                 # V20: balanced reward, less overfit than V19
SL_ATR = 1.2
TP_ATR = SL_ATR * RR

# Daily Control
DAILY_TARGET = 0.025
DAILY_STOP = -0.018
MAX_TRADES_PER_DAY = 3   # V20: allow more samples than V19

# Session Control
SESSION_START = 13
SESSION_END = 23

# Hard Stop
MAX_DD = -0.25

# Realistic Cost
FEE_RATE = 0.0004        # 0.04% per side, round trip = x2
SLIPPAGE_ATR = 0.03      # 3% ATR slippage
COOLDOWN_BARS = 5        # V20: balanced cooldown

# Quality Filters
MIN_ATR_PCT = 0.0012
MAX_ATR_PCT = 0.0120
MIN_VOLUME_RATIO = 0.7   # V20: loosen volume filter
EMA_GAP_MIN = 0.0010     # V20: keep trend quality but less strict
MIN_TP_FEE_MULTIPLE = 8  # V20: still avoid tiny TP after fee
MOMENTUM_LOOKBACK = 3
MIN_MOMENTUM_PCT = 0.0005
MIN_EMA20_DISTANCE = 0.0003

# Optional anti-chop filter
MIN_BODY_RATIO = 0.35    # V20: less strict anti-chop filter

# Files
LOG_FILE = "log_v20.csv"
EQUITY_FILE = "equity_v20.csv"
DAILY_FILE = "daily_v20.csv"
MONTHLY_FILE = "monthly_v20.csv"


# =====================
# LOAD DATA
# =====================
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "time" not in df.columns:
        df.columns = ["time", "open", "high", "low", "close", "volume"]

    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time")
    df = df.drop_duplicates(subset=["time"])
    df.set_index("time", inplace=True)

    required = ["open", "high", "low", "close", "volume"]
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required)
    return df


# =====================
# INDICATORS
# =====================
def ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def to_h1(df: pd.DataFrame) -> pd.DataFrame:
    return df.resample("1h").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()


# =====================
# HELPERS
# =====================
def calc_fee(balance: float) -> float:
    return balance * FEE_RATE * 2


def close_trade(trade_logs, time, position, result, exit_price, balance, peak):
    risk_amt = position["risk_amt"]
    gross_pnl = risk_amt * RR if result == "TP" else -risk_amt
    fee = calc_fee(balance)
    pnl = gross_pnl - fee
    balance_after = balance + pnl
    dd = (balance_after - peak) / peak

    trade_logs.append({
        "time": time,
        "entry_time": position["entry_time"],
        "side": position["side"],
        "result": result,
        "entry": position["entry"],
        "exit": exit_price,
        "sl": position["sl"],
        "tp": position["tp"],
        "risk_amt": risk_amt,
        "gross_pnl": gross_pnl,
        "fee": fee,
        "pnl": pnl,
        "balance_after": balance_after,
        "dd": dd,
        "atr_pct": position["atr_pct"],
        "volume_ratio": position["volume_ratio"],
        "momentum_pct": position["momentum_pct"],
        "ema20_distance": position["ema20_distance"],
        "ema_gap": position["ema_gap"],
    })

    return balance_after, pnl


def build_monthly_report(daily_df: pd.DataFrame) -> pd.DataFrame:
    monthly_df = daily_df.copy()
    monthly_df["month"] = pd.to_datetime(monthly_df["date"]).dt.to_period("M").astype(str)
    monthly_df = monthly_df.groupby("month").agg(
        start_balance=("start_balance", "first"),
        end_balance=("end_balance", "last"),
        trades=("trades", "sum"),
    ).reset_index()
    monthly_df["return_pct"] = (
        (monthly_df["end_balance"] - monthly_df["start_balance"])
        / monthly_df["start_balance"]
        * 100
    )
    return monthly_df


# =====================
# BACKTEST V20
# =====================
def backtest(df: pd.DataFrame) -> dict:
    df = df.copy()
    df_h1 = to_h1(df)

    df_h1["ema50"] = ema(df_h1["close"], 50)
    df_h1["ema200"] = ema(df_h1["close"], 200)
    df_h1["ema_gap"] = (df_h1["ema50"] - df_h1["ema200"]).abs() / df_h1["close"]

    df["ema20"] = ema(df["close"], 20)
    df["ema50"] = ema(df["close"], 50)
    df["atr"] = atr(df, 14)
    df["atr_pct"] = df["atr"] / df["close"]
    df["vol_avg"] = df["volume"].rolling(30).mean()
    df["volume_ratio"] = df["volume"] / df["vol_avg"]
    df["momentum"] = df["close"] - df["close"].shift(MOMENTUM_LOOKBACK)
    df["momentum_pct"] = df["momentum"].abs() / df["close"]
    df["ema20_distance"] = (df["close"] - df["ema20"]).abs() / df["close"]
    df["body_ratio"] = (df["close"] - df["open"]).abs() / (df["high"] - df["low"]).replace(0, np.nan)

    balance = START_BALANCE
    peak = balance
    max_dd = 0.0
    trades = 0
    wins = 0

    position = None
    current_day = None
    day_start = balance
    daily_trades = 0
    stop_day = False
    last_exit_i = -999

    trade_logs = []
    equity_logs = []
    daily_logs = []

    for i in range(250, len(df) - 1):
        row = df.iloc[i]
        time = row.name
        price = row["close"]
        day = time.date()

        # New day
        if day != current_day:
            if current_day is not None:
                daily_logs.append({
                    "date": current_day,
                    "start_balance": day_start,
                    "end_balance": balance,
                    "return_pct": (balance - day_start) / day_start * 100,
                    "trades": daily_trades,
                })
            current_day = day
            day_start = balance
            daily_trades = 0
            stop_day = False

        daily_ret = (balance - day_start) / day_start
        if daily_ret >= DAILY_TARGET or daily_ret <= DAILY_STOP:
            stop_day = True

        # Exit first
        if position is not None:
            side = position["side"]

            if side == "BUY":
                if row["low"] <= position["sl"]:
                    balance, _ = close_trade(trade_logs, time, position, "SL", position["sl"], balance, peak)
                    position = None
                    trades += 1
                    last_exit_i = i
                    continue
                if row["high"] >= position["tp"]:
                    balance, _ = close_trade(trade_logs, time, position, "TP", position["tp"], balance, peak)
                    position = None
                    trades += 1
                    wins += 1
                    last_exit_i = i
                    continue

            if side == "SELL":
                if row["high"] >= position["sl"]:
                    balance, _ = close_trade(trade_logs, time, position, "SL", position["sl"], balance, peak)
                    position = None
                    trades += 1
                    last_exit_i = i
                    continue
                if row["low"] <= position["tp"]:
                    balance, _ = close_trade(trade_logs, time, position, "TP", position["tp"], balance, peak)
                    position = None
                    trades += 1
                    wins += 1
                    last_exit_i = i
                    continue

        # Equity / DD
        peak = max(peak, balance)
        dd = (balance - peak) / peak
        max_dd = min(max_dd, dd)
        equity_logs.append({"time": time, "balance": balance, "dd": dd})

        if dd <= MAX_DD:
            break

        # Entry gates
        if position is not None:
            continue
        if stop_day:
            continue
        if daily_trades >= MAX_TRADES_PER_DAY:
            continue
        if i - last_exit_i < COOLDOWN_BARS:
            continue
        if not (SESSION_START <= time.hour <= SESSION_END):
            continue

        required_vals = [
            row["atr"], row["atr_pct"], row["volume_ratio"], row["momentum"],
            row["momentum_pct"], row["ema20_distance"], row["body_ratio"],
        ]
        if any(np.isnan(v) for v in required_vals):
            continue

        # Quality filters: volatility, volume, momentum, candle quality
        if not (MIN_ATR_PCT <= row["atr_pct"] <= MAX_ATR_PCT):
            continue
        if row["volume_ratio"] < MIN_VOLUME_RATIO:
            continue
        if row["momentum_pct"] < MIN_MOMENTUM_PCT:
            continue
        if row["ema20_distance"] < MIN_EMA20_DISTANCE:
            continue
        if row["body_ratio"] < MIN_BODY_RATIO:
            continue

        # H1 trend
        h1_rows = df_h1[df_h1.index <= time]
        if h1_rows.empty:
            continue
        h1 = h1_rows.iloc[-1]

        trend_up = h1["ema50"] > h1["ema200"]
        trend_down = h1["ema50"] < h1["ema200"]

        if h1["ema_gap"] < EMA_GAP_MIN:
            continue
        if not (trend_up or trend_down):
            continue

        prev = df.iloc[i - 1]
        signal = None

        # Core entry from V16/V18 + balanced V20 confirmation
        if trend_up:
            if (
                price < row["ema20"]
                and price > prev["high"]
                and row["momentum"] > 0
                and price > row["ema50"]
            ):
                signal = "BUY"

        if trend_down:
            if (
                price > row["ema20"]
                and price < prev["low"]
                and row["momentum"] < 0
                and price < row["ema50"]
            ):
                signal = "SELL"

        if signal is None:
            continue

        atr_val = row["atr"]
        slip = atr_val * SLIPPAGE_ATR

        if signal == "BUY":
            entry = price + slip
            sl = entry - SL_ATR * atr_val
            tp = entry + TP_ATR * atr_val
        else:
            entry = price - slip
            sl = entry + SL_ATR * atr_val
            tp = entry - TP_ATR * atr_val

        # TP must be wide enough vs fee drag
        tp_distance_pct = abs(tp - entry) / entry
        round_trip_fee_pct = FEE_RATE * 2
        if tp_distance_pct < round_trip_fee_pct * MIN_TP_FEE_MULTIPLE:
            continue

        risk_amt = balance * RISK
        position = {
            "side": signal,
            "entry_time": time,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "risk_amt": risk_amt,
            "atr_pct": row["atr_pct"],
            "volume_ratio": row["volume_ratio"],
            "momentum_pct": row["momentum_pct"],
            "ema20_distance": row["ema20_distance"],
            "ema_gap": h1["ema_gap"],
        }
        daily_trades += 1

    if current_day is not None:
        daily_logs.append({
            "date": current_day,
            "start_balance": day_start,
            "end_balance": balance,
            "return_pct": (balance - day_start) / day_start * 100,
            "trades": daily_trades,
        })

    winrate = wins / trades * 100 if trades else 0.0
    profit_pct = (balance - START_BALANCE) / START_BALANCE * 100

    trade_df = pd.DataFrame(trade_logs)
    equity_df = pd.DataFrame(equity_logs)
    daily_df = pd.DataFrame(daily_logs)

    if not trade_df.empty:
        trade_df.to_csv(LOG_FILE, index=False)
    if not equity_df.empty:
        equity_df.to_csv(EQUITY_FILE, index=False)
    if not daily_df.empty:
        daily_df.to_csv(DAILY_FILE, index=False)
        build_monthly_report(daily_df).to_csv(MONTHLY_FILE, index=False)

    avg_win = trade_df.loc[trade_df["result"] == "TP", "pnl"].mean() if not trade_df.empty and (trade_df["result"] == "TP").any() else 0.0
    avg_loss = trade_df.loc[trade_df["result"] == "SL", "pnl"].mean() if not trade_df.empty and (trade_df["result"] == "SL").any() else 0.0
    profit_factor = 0.0
    if not trade_df.empty:
        gross_profit = trade_df.loc[trade_df["pnl"] > 0, "pnl"].sum()
        gross_loss = abs(trade_df.loc[trade_df["pnl"] < 0, "pnl"].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    return {
        "version": VERSION,
        "start": START_BALANCE,
        "end": balance,
        "profit_pct": profit_pct,
        "trades": trades,
        "wins": wins,
        "winrate": winrate,
        "max_dd_pct": max_dd * 100,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "rr": RR,
        "fee_rate": FEE_RATE,
        "slippage_atr": SLIPPAGE_ATR,
        "cooldown_bars": COOLDOWN_BARS,
        "min_atr_pct": MIN_ATR_PCT,
        "min_volume_ratio": MIN_VOLUME_RATIO,
        "ema_gap_min": EMA_GAP_MIN,
        "min_momentum_pct": MIN_MOMENTUM_PCT,
        "min_ema20_distance": MIN_EMA20_DISTANCE,
        "min_body_ratio": MIN_BODY_RATIO,
        "log_file": LOG_FILE,
        "equity_file": EQUITY_FILE,
        "daily_file": DAILY_FILE,
        "monthly_file": MONTHLY_FILE,
        "sample_logs": trade_logs[:10],
    }


# =====================
# RUN
# =====================
if __name__ == "__main__":
    df = load_data("btc_5m.csv")
    result = backtest(df)

    print("\n🔥 V20 BALANCED EDGE BACKTEST RESULT")
    for key, value in result.items():
        if key != "sample_logs":
            print(f"{key}: {value}")

    print("\n📌 Sample logs:")
    for log in result["sample_logs"]:
        print(log)
