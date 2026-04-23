#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Bitcoin Scalping Strategy V2 (5m + 1h + optional 4h)
- Adjustable Volume filter
- Higher ADX for stronger trend
- Optional 4h trend alignment
"""

import json
import pandas as pd
import numpy as np
import time
from numba import njit
import itertools
import argparse

# ======================== Data Loader ========================
def load_data(filepath):
    df = pd.read_csv(filepath)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    df.sort_index(inplace=True)
    required = ['open', 'high', 'low', 'close']
    if 'volume' in df.columns:
        required.append('volume')
    return df[required]

# ======================== Indicators ========================
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def atr(df, period=14):
    high, low, close = df['high'], df['low'], df['close']
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def adx(df, period=14):
    high, low, close = df['high'], df['low'], df['close']
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    atr_val = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr_val)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr_val)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    return dx.rolling(period).mean()

# ======================== Strategy V2 ========================
def strategy_enhanced_v2(df_5m, df_1h, df_4h=None,
                         fast_ma=10, slow_ma=30,
                         rsi_period=14, rsi_low=35, rsi_high=65,
                         adx_thresh=22,
                         use_volume=True, vol_ma_period=20):
    # 1h trend
    ema50_1h = ema(df_1h['close'], 50)
    trend_1h = (df_1h['close'] > ema50_1h).astype(int)
    trend_5m_1h = trend_1h.reindex(df_5m.index, method='ffill').fillna(0)

    # 4h trend
    if df_4h is not None:
        ema50_4h = ema(df_4h['close'], 50)
        trend_4h = (df_4h['close'] > ema50_4h).astype(int)
        trend_5m_4h = trend_4h.reindex(df_5m.index, method='ffill').fillna(0)
    else:
        trend_5m_4h = pd.Series(1, index=df_5m.index)

    # Volume filter
    if use_volume and 'volume' in df_5m.columns and vol_ma_period > 0:
        vol_ma = df_5m['volume'].rolling(vol_ma_period).mean()
        vol_cond = df_5m['volume'] > vol_ma
    else:
        vol_cond = pd.Series(True, index=df_5m.index)

    # 5m indicators
    fast = ema(df_5m['close'], fast_ma)
    slow = ema(df_5m['close'], slow_ma)
    rsi_vals = rsi(df_5m['close'], rsi_period)
    adx_vals = adx(df_5m, 14)

    signals = pd.Series(0, index=df_5m.index)

    buy_cond = (
        (trend_5m_1h == 1) &
        (trend_5m_4h == 1) &
        (fast > slow) &
        (rsi_vals < rsi_low) &
        (adx_vals > adx_thresh) &
        vol_cond
    )
    sell_cond = (
        (trend_5m_1h == 0) |
        ((fast < slow) & (rsi_vals > rsi_high))
    )
    signals[buy_cond] = 1
    signals[sell_cond] = -1

    return signals, atr(df_5m, 14)

# ======================== Backtester ========================
class ControlledBacktesterV2:
    def __init__(self, df_5m, df_1h, df_4h=None,
                 init_bal=1000, fee=0.0005, risk=0.015, leverage=3,
                 max_dd=0.25, daily_loss_limit=0.20):
        self.df_5m = df_5m
        self.df_1h = df_1h
        self.df_4h = df_4h
        self.init_bal = init_bal
        self.fee = fee
        self.risk = risk
        self.leverage = leverage
        self.max_dd = max_dd
        self.daily_loss_limit = daily_loss_limit

    def run(self, fast_ma=10, slow_ma=30, rsi_period=14, rsi_low=35, rsi_high=65,
            adx_thresh=22, use_volume=True, vol_ma_period=20,
            sl_atr=1.5, tp_atr=3.0):
        signals, atr_vals = strategy_enhanced_v2(
            self.df_5m, self.df_1h, self.df_4h,
            fast_ma=fast_ma, slow_ma=slow_ma,
            rsi_period=rsi_period, rsi_low=rsi_low, rsi_high=rsi_high,
            adx_thresh=adx_thresh,
            use_volume=use_volume, vol_ma_period=vol_ma_period
        )

        close = self.df_5m['close'].values
        dates = self.df_5m.index.normalize()
        n = len(close)
        sig = signals.values.astype(np.int8)
        atr = atr_vals.values

        balance = self.init_bal
        equity = np.zeros(n)
        trades = []
        pos_active = False
        entry_price = 0.0
        position_size = 0.0
        peak_balance = self.init_bal
        daily_start_balance = self.init_bal
        current_date = dates[0]

        for i in range(n):
            price = close[i]
            s = sig[i]
            date = dates[i]

            if date != current_date:
                current_date = date
                daily_start_balance = balance

            if balance < daily_start_balance * (1 - self.daily_loss_limit):
                equity[i] = balance
                continue

            if balance > peak_balance:
                peak_balance = balance
            dd = (peak_balance - balance) / peak_balance if peak_balance > 0 else 0
            if dd >= self.max_dd:
                equity[i] = balance
                continue

            if pos_active:
                sl_price = entry_price - atr[i] * sl_atr
                tp_price = entry_price + atr[i] * tp_atr
                hit_tp = (price >= tp_price)
                hit_sl = (price <= sl_price)
                if hit_tp or hit_sl or s == -1:
                    ret = (price - entry_price) / entry_price * self.leverage - self.fee * 2
                    pnl = position_size * ret
                    balance += pnl
                    trades.append((entry_price, price, ret > 0, pnl))
                    pos_active = False
                    equity[i] = balance
                    continue

            if not pos_active and s == 1:
                pos_active = True
                entry_price = price
                stop_pct = (atr[i] * sl_atr) / price
                if stop_pct > 0:
                    risk_amount = balance * self.risk
                    position_size = risk_amount / stop_pct
                    if position_size > balance:
                        position_size = balance
                else:
                    position_size = balance * 0.05

            equity[i] = balance

        if pos_active:
            ret = (close[-1] - entry_price) / entry_price * self.leverage - self.fee * 2
            pnl = position_size * ret
            balance += pnl
            trades.append((entry_price, close[-1], ret > 0, pnl))
            equity[-1] = balance

        final_bal = equity[-1]
        num_trades = len(trades)
        if num_trades == 0:
            winrate = avg_ret = 0.0
        else:
            wins = sum(1 for t in trades if t[2])
            winrate = wins / num_trades * 100
            avg_ret = np.mean([t[3] for t in trades])
        max_dd = self._max_drawdown(equity)
        daily_rets = self._daily_returns(equity)
        avg_daily_ret = np.mean(daily_rets) * 100 if len(daily_rets) > 0 else 0.0

        return {
            "balance_start": self.init_bal,
            "balance_end": round(final_bal, 2),
            "trades": num_trades,
            "winrate": round(winrate, 2),
            "avg_pnl_per_trade": round(avg_ret, 2),
            "max_drawdown": round(max_dd, 4),
            "avg_daily_return_pct": round(avg_daily_ret, 2),
            "params": {
                "fast_ma": fast_ma, "slow_ma": slow_ma,
                "rsi_period": rsi_period, "rsi_low": rsi_low, "rsi_high": rsi_high,
                "adx_thresh": adx_thresh,
                "use_volume": use_volume, "vol_ma_period": vol_ma_period,
                "sl_atr": sl_atr, "tp_atr": tp_atr
            }
        }

    def _max_drawdown(self, equity):
        peak = self.init_bal
        max_dd = 0.0
        for v in equity:
            if v > peak:
                peak = v
            dd = (peak - v) / peak
            if dd > max_dd:
                max_dd = dd
        return max_dd

    def _daily_returns(self, equity):
        s = pd.Series(equity, index=self.df_5m.index)
        daily = s.resample('D').last().dropna()
        rets = daily.pct_change().dropna()
        return rets.values

# ======================== Grid Search ========================
def grid_search_v2(df_5m, df_1h, df_4h=None,
                   leverage=3, risk=0.015, max_dd=0.25):
    bt = ControlledBacktesterV2(df_5m, df_1h, df_4h,
                                init_bal=1000, fee=0.0005,
                                risk=risk, leverage=leverage, max_dd=max_dd,
                                daily_loss_limit=0.20)
    best = None
    best_params = {}

    # ปรับปรุงพารามิเตอร์
    fast_opts = [8, 10, 12]
    slow_opts = [26, 30, 34]
    rsi_period_opts = [14]
    rsi_low_opts = [30, 35]          # เน้น oversold
    rsi_high_opts = [65, 70]         # เน้น overbought
    adx_opts = [22, 25, 28]          # เพิ่มขึ้นเพื่อกรอง Sideways
    use_volume_opts = [True, False]  # ทดลองเปิด/ปิด Volume
    vol_ma_period_opts = [20]        # ถ้าเปิดใช้ 20
    sl_atr_opts = [1.5, 1.8]
    tp_atr_opts = [3.5, 4.0, 4.5]

    total = (len(fast_opts) * len(slow_opts) * len(rsi_period_opts) *
             len(rsi_low_opts) * len(rsi_high_opts) * len(adx_opts) *
             len(use_volume_opts) * len(vol_ma_period_opts) *
             len(sl_atr_opts) * len(tp_atr_opts))
    count = 0
    print(f"Testing {total} combinations...")

    for fast in fast_opts:
        for slow in slow_opts:
            if fast >= slow: continue
            for rsi_p in rsi_period_opts:
                for rsi_l in rsi_low_opts:
                    for rsi_h in rsi_high_opts:
                        if rsi_l >= rsi_h: continue
                        for adx in adx_opts:
                            for use_vol in use_volume_opts:
                                for vol_mp in vol_ma_period_opts:
                                    for sl in sl_atr_opts:
                                        for tp in tp_atr_opts:
                                            count += 1
                                            res = bt.run(
                                                fast_ma=fast, slow_ma=slow,
                                                rsi_period=rsi_p, rsi_low=rsi_l, rsi_high=rsi_h,
                                                adx_thresh=adx,
                                                use_volume=use_vol, vol_ma_period=vol_mp,
                                                sl_atr=sl, tp_atr=tp
                                            )
                                            # Criteria: max_dd <= 0.25, trades > 150, avg daily return > 0
                                            if (res['max_drawdown'] <= 0.25 and
                                                res['trades'] > 150 and
                                                res['avg_daily_return_pct'] > 0):
                                                if (best is None or
                                                    res['balance_end'] > best['balance_end']):
                                                    best = res
                                                    best_params = res['params']
                                            if count % 500 == 0:
                                                print(f"  Progress: {count}/{total}")

    if best is None:
        print("Fallback: relaxing to max balance with dd <= 0.30 and trades > 100")
        best_bal = -1
        for fast in [10]:
            for slow in [30]:
                for rsi_p in [14]:
                    for rsi_l in [35]:
                        for rsi_h in [65]:
                            for adx in [22]:
                                for use_vol in [False]:
                                    for vol_mp in [20]:
                                        for sl in [1.5]:
                                            for tp in [3.5]:
                                                res = bt.run(
                                                    fast_ma=fast, slow_ma=slow,
                                                    rsi_period=rsi_p, rsi_low=rsi_l, rsi_high=rsi_h,
                                                    adx_thresh=adx,
                                                    use_volume=use_vol, vol_ma_period=vol_mp,
                                                    sl_atr=sl, tp_atr=tp
                                                )
                                                if res['max_drawdown'] <= 0.30 and res['trades'] > 100:
                                                    if res['balance_end'] > best_bal:
                                                        best_bal = res['balance_end']
                                                        best = res
                                                        best_params = res['params']
    if best:
        best['best_params'] = best_params
        best['strategy'] = f"Enhanced V2 (5m/1h/4h) Lev{leverage}x Risk{risk}"
    return best

# ======================== Main ========================
def main():
    parser = argparse.ArgumentParser(description='Enhanced Safe BTC Backtest V2')
    parser.add_argument('--csv5m', type=str, required=True, help='5-minute CSV file')
    parser.add_argument('--csv1h', type=str, required=True, help='1-hour CSV file')
    parser.add_argument('--csv4h', type=str, default=None, help='4-hour CSV file (optional)')
    parser.add_argument('--leverage', type=float, default=3.0)
    parser.add_argument('--risk', type=float, default=0.015)
    parser.add_argument('--maxdd', type=float, default=0.25)
    args = parser.parse_args()

    print("Loading data...")
    df_5m = load_data(args.csv5m)
    print(f"5m shape: {df_5m.shape}")
    df_1h = load_data(args.csv1h)
    print(f"1h shape: {df_1h.shape}")
    df_4h = None
    if args.csv4h:
        df_4h = load_data(args.csv4h)
        print(f"4h shape: {df_4h.shape}")

    print(f"\nLeverage={args.leverage}x, Risk={args.risk*100:.1f}%, MaxDD={args.maxdd*100:.0f}%")
    start_t = time.time()
    result = grid_search_v2(df_5m, df_1h, df_4h,
                            leverage=args.leverage, risk=args.risk, max_dd=args.maxdd)
    elapsed = time.time() - start_t

    print(f"\nCompleted in {elapsed:.1f}s")
    if result:
        out = {k: v for k, v in result.items() if k != 'params'}
        out['best_params'] = result['params']
        print("\n========== BEST RESULT ==========")
        print(json.dumps(out, indent=2))
    else:
        print("No profitable strategy found. Try different --leverage, --risk, or --maxdd.")

if __name__ == "__main__":
    main()