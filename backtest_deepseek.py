import pandas as pd
import numpy as np
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
CSV_FILE = 'bitcoin_365d_15m.csv'   # ตรวจสอบ path ให้ถูกต้อง
INITIAL_CAPITAL = 10_000.0
COMMISSION = 0.001                   # 0.1% ต่อเทรด

ATR_PERIOD = 14
BODY_AVG_PERIOD = 50

# Exit controls
USE_PROFIT_PROTECTION = True
TRAIL_STOP_R = 0.5                  # ระยะ trailing stop เป็น R

# ============== DATA LOADING & M5 RESAMPLING ================
df = pd.read_csv(CSV_FILE, parse_dates=['time'])
df.set_index('time', inplace=True)
df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
df.sort_index(inplace=True)

# Resample to M5 closed candle
m5 = df.resample('5min').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
}).dropna()

# =================== ATR & BODY ============================
m5['tr'] = np.maximum(
    m5['high'] - m5['low'],
    np.maximum(abs(m5['high'] - m5['close'].shift(1)),
               abs(m5['low'] - m5['close'].shift(1)))
)
m5['atr'] = m5['tr'].rolling(ATR_PERIOD).mean().shift(1)
m5['body'] = abs(m5['close'] - m5['open'])
m5['avg_body'] = m5['body'].rolling(BODY_AVG_PERIOD).mean()
m5.dropna(subset=['atr', 'avg_body'], inplace=True)

# ============== 2-HOUR RANGE BUILDER ========================
def build_ranges(data):
    ranges = []
    start = data.index.min().floor('2h')
    end = data.index.max().ceil('2h')
    current = start
    while current < end:
        chunk = data.loc[current : current + pd.Timedelta(hours=2)]
        if len(chunk) > 1:
            range_high = chunk['high'].max()
            range_low = chunk['low'].min()
            valid_start = current + pd.Timedelta(hours=2)
            valid_end = valid_start + pd.Timedelta(hours=2)
            ranges.append({
                'valid_start': valid_start,
                'valid_end': valid_end,
                'high': range_high,
                'low': range_low
            })
        current += pd.Timedelta(hours=2)
    return pd.DataFrame(ranges)

range_df = build_ranges(m5)

# ================ TRADING SIMULATOR =========================
def simulate(data, params):
    """
    params tuple:
        (body_ratio, retest_pct, min_rr, direct_break, exit_mode,
         retest_bars, direct_body_mult)
    """
    body_ratio, retest_pct, min_rr, direct_break, exit_mode, retest_bars, direct_body_mult = params

    capital = INITIAL_CAPITAL
    position = None     # 'long' or None
    units = 0.0
    entry_price = 0.0
    stop_loss = 0.0
    take_profit = 0.0
    entry_r = 0.0      # ระยะ SL ณ จุดเข้า
    initial_stop = 0.0  # SL ตั้งต้น (ไม่เปลี่ยนแปลงโดย protection)
    trades = []
    peak_price = 0.0

    # สำหรับเก็บ equity curve (value after each trade or at bar close)
    equity_curve = [capital]

    for i in range(1, len(data)):
        row = data.iloc[i]
        current_time = data.index[i]
        close = row['close']
        open_ = row['open']
        high = row['high']
        low = row['low']
        atr = row['atr']
        avg_body = row['avg_body']

        # ==================== EXIT LOGIC ====================
        if position == 'long':
            peak_price = max(peak_price, high)
            exit_now = False
            exit_price_val = 0.0

            # Profit protection and trailing stop adjustments (before checking SL/TP)
            if USE_PROFIT_PROTECTION and not exit_now:
                r_profit = 0.0
                if entry_r > 0:
                    r_profit = (close - entry_price) / entry_r
                # Move to -0.2R when profit >= 0.8R
                if r_profit >= 0.8 and stop_loss < entry_price - 0.2 * entry_r:
                    stop_loss = entry_price - 0.2 * entry_r
                # Move to breakeven when profit >= 1R
                if r_profit >= 1.0 and stop_loss < entry_price:
                    stop_loss = entry_price
                # Trailing stop for modes B and C
                if exit_mode in ('B', 'C'):
                    trail_price = peak_price - TRAIL_STOP_R * entry_r
                    stop_loss = max(stop_loss, trail_price)
                # Partial close at 1R (mode C)
                if exit_mode == 'C' and r_profit >= 1.0 and units > 0:
                    # Close half
                    close_amount = units * 0.5 * close * (1 - COMMISSION)
                    capital += close_amount
                    units *= 0.5
                    # Reset stop to entry (breakeven) for remaining half
                    stop_loss = entry_price

            # Check SL/TP
            if low <= stop_loss:
                exit_now = True
                exit_price_val = stop_loss
            elif high >= take_profit:
                exit_now = True
                exit_price_val = take_profit

            if exit_now:
                fill_price = exit_price_val if exit_price_val > 0 else close
                if position == 'long':
                    capital = units * fill_price * (1 - COMMISSION)
                R_mult = (fill_price - entry_price) / entry_r if entry_r != 0 else 0
                trades.append((exit_mode, entry_time, current_time, entry_price, fill_price, R_mult))
                equity_curve.append(capital)   # record equity after trade
                position = None
                units = 0
                peak_price = 0
                continue

        # ==================== ENTRY LOGIC ====================
        if position is None and atr > 0:
            active = range_df[(range_df['valid_start'] <= current_time) &
                              (range_df['valid_end'] > current_time)]
            if active.empty:
                continue
            rng = active.iloc[0]
            range_high = rng['high']
            range_low = rng['low']

            if direct_break:
                # Direct breakout variant
                if row['body'] >= avg_body * direct_body_mult:
                    if row['close'] > range_high and (row['close'] - range_high) >= atr * 0.4:
                        entry_price_val = row['close']
                        sl_distance = body_ratio * atr
                        if sl_distance > 1.2 * atr:
                            continue
                        if sl_distance < 0.5 * atr:
                            sl_distance = 0.5 * atr
                        entry_price = entry_price_val
                        stop_loss = entry_price - sl_distance
                        take_profit = entry_price + min_rr * sl_distance
                        entry_r = sl_distance
                        initial_stop = stop_loss
                        units = capital * (1 - COMMISSION) / entry_price
                        capital = 0
                        position = 'long'
                        entry_time = current_time
                        peak_price = entry_price
            else:
                # Retest entry
                tolerance = retest_pct * range_high   # e.g., 0.002 * 95000 = 190
                touched = False
                lookback = max(0, i - retest_bars)
                for j in range(lookback, i):
                    bar = data.iloc[j]
                    if (bar['low'] <= range_high + tolerance) and (bar['low'] >= range_high - tolerance):
                        touched = True
                        break
                if touched and row['close'] > range_high:
                    entry_price_val = row['close']
                    sl_distance = body_ratio * atr
                    if sl_distance > 1.2 * atr:
                        continue
                    if sl_distance < 0.5 * atr:
                        sl_distance = 0.5 * atr
                    entry_price = entry_price_val
                    stop_loss = entry_price - sl_distance
                    take_profit = entry_price + min_rr * sl_distance
                    entry_r = sl_distance
                    initial_stop = stop_loss
                    units = capital * (1 - COMMISSION) / entry_price
                    capital = 0
                    position = 'long'
                    entry_time = current_time
                    peak_price = entry_price

    # Final liquidation
    if position is not None:
        final_close = data.iloc[-1]['close']
        capital = units * final_close * (1 - COMMISSION)
        R_mult = (final_close - entry_price) / entry_r if entry_r != 0 else 0
        trades.append((exit_mode, entry_time, data.index[-1], entry_price, final_close, R_mult))
        equity_curve.append(capital)

    return capital, trades, equity_curve

# ==================== PARAMETER SWEEP =========================
# Expanded parameter grid
body_ratios = [0.35, 0.45]
retest_pcts = [0.001, 0.002, 0.005, 0.01]     # 0.1% to 1%
min_rrs = [1.2, 1.5]
direct_breaks = [False, True]
exit_modes = ['A', 'B', 'C']
retest_bars_list = [20, 40]
direct_body_mults = [1.2, 1.5]

results = []
for br, rp, rr, db, em, rb, dbm in product(body_ratios, retest_pcts, min_rrs,
                                            direct_breaks, exit_modes,
                                            retest_bars_list, direct_body_mults):
    final_cap, trades, eq_curve = simulate(m5, (br, rp, rr, db, em, rb, dbm))
    if not trades:
        continue
    profit_pct = (final_cap - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    r_vals = [t[5] for t in trades]
    wins = [r for r in r_vals if r > 0]
    losses = [r for r in r_vals if r <= 0]
    n = len(r_vals)
    winrate = len(wins)/n * 100 if n else 0
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    expectancy = np.mean(r_vals) if r_vals else 0
    total_win_r = sum(wins) if wins else 0
    total_loss_r = sum(losses) if losses else 0
    profit_factor = abs(total_win_r / total_loss_r) if total_loss_r != 0 else float('inf')

    # Max drawdown from equity curve
    eq_arr = np.array(eq_curve)
    peak = np.maximum.accumulate(eq_arr)
    drawdown = (eq_arr - peak) / peak * 100
    max_dd_pct = drawdown.min()

    days = (m5.index[-1] - m5.index[0]).days
    if days == 0:
        days = 1
    trades_per_day = n / days

    results.append({
        'params': (br, rp, rr, db, em, rb, dbm),
        'profit_pct': profit_pct,
        'trades': n,
        'trades_per_day': trades_per_day,
        'winrate': winrate,
        'profit_factor': profit_factor,
        'max_drawdown_pct': max_dd_pct,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'expectancy': expectancy,
        'final_cap': final_cap
    })

# ==================== FILTER & RANK ==========================
df_res = pd.DataFrame(results)

# Filter robust configs
filtered = df_res[(df_res['profit_factor'] > 1.2) &
                  (df_res['expectancy'] > 0) &
                  (df_res['trades_per_day'] >= 0.2) &   # relax slightly
                  (df_res['max_drawdown_pct'] > -10)]   # max_dd_pct is negative

if filtered.empty:
    print("⚠️  No config met all strict criteria; showing top 10 by expectancy.")
    filtered = df_res.sort_values('expectancy', ascending=False).head(10)
else:
    filtered = filtered.sort_values(['expectancy', 'profit_factor'], ascending=[False, False])
    filtered = filtered.head(10)

# ==================== OUTPUT =================================
print("\n🏆 Top 10 Robust Day‑Trading Configurations (Corrected)")
print("=" * 95)
for idx, row in filtered.iterrows():
    br, rp, rr, db, em, rb, dbm = row['params']
    print(f"\nConfig: body_ratio={br}, retest_pct={rp}({rp*100:.1f}%), min_rr={rr}, direct={db}, exit={em}, "
          f"retest_bars={rb}, dir_body_mult={dbm}")
    print(f"  Profit: {row['profit_pct']:.2f}%  |  Trades: {row['trades']}  |  Trades/day: {row['trades_per_day']:.2f}")
    print(f"  Winrate: {row['winrate']:.1f}%  |  Profit Factor: {row['profit_factor']:.2f}")
    print(f"  Max DD: {row['max_drawdown_pct']:.2f}%  |  Expectancy: {row['expectancy']:.2f}R")
    print(f"  Avg Win: {row['avg_win']:.2f}R  |  Avg Loss: {row['avg_loss']:.2f}R")