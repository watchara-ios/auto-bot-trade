"""
Mean Reversion Backtest Engine
================================
Logic: fade price extremes เมื่อตลาดอยู่ใน ranging mode (ADX ต่ำ)

Signal timeframe : M15 (Bollinger Band + RSI + ADX)
Exit simulation  : M1  (precise SL/TP fills)

Entry conditions:
  BUY  — close < BB lower  + RSI < rsi_os + ADX < adx_max
  SELL — close > BB upper  + RSI > rsi_ob + ADX < adx_max

SL: entry ± sl_atr_mult × ATR(14)
TP: BB midline (EMA20) at signal time  OR  fixed RR
"""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "reports" / "mean_reversion"


@dataclass
class MRConfig:
    initial_balance: float = 1000.0
    fixed_risk_base: bool = True
    risk_pct: float = 0.01
    fee_rate: float = 0.0005
    spread_atr_mult: float = 0.02
    slippage_min_spread: float = 0.5
    slippage_max_spread: float = 1.5
    max_trades_per_day: int = 3
    daily_loss_limit_pct: float = 0.03
    # Indicators
    bb_period: int = 20
    bb_std: float = 2.0
    rsi_period: int = 14
    adx_period: int = 14
    atr_period: int = 14
    # Regime gate
    adx_max: float = 20.0
    require_adx_falling: bool = False
    # Signal
    rsi_ob: float = 70.0
    rsi_os: float = 30.0
    require_rsi: bool = True
    require_rejection: bool = False
    min_wick_pct: float = 0.30
    require_bb_close: bool = True   # close must be outside BB (vs just wick)
    # SL / TP
    sl_atr_mult: float = 1.5
    tp_type: str = "bb_mid"         # "bb_mid" | "fixed_rr"
    rr: float = 1.5
    # Optional filters
    volume_mult: float = 0.0        # 0 = disabled
    allowed_side: Optional[str] = None
    include_utc_hours: Optional[tuple] = None
    exclude_utc_hours: tuple = ()
    seed: int = 42


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    prev = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"],
         (df["high"] - prev).abs(),
         (df["low"] - prev).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def _adx(df: pd.DataFrame, period: int) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    up = high.diff()
    dn = -low.diff()
    plus_dm  = up.where((up > dn) & (up > 0), 0.0)
    minus_dm = dn.where((dn > up) & (dn > 0), 0.0)
    tr = pd.concat(
        [high - low,
         (high - close.shift(1)).abs(),
         (low  - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    atr_s    = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di  = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_s.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_s.replace(0, np.nan)
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def prepare(m15: pd.DataFrame, config: MRConfig) -> pd.DataFrame:
    df = m15.copy()

    # Bollinger Bands
    mid = df["close"].rolling(config.bb_period).mean()
    std = df["close"].rolling(config.bb_period).std()
    df["bb_upper"] = mid + config.bb_std * std
    df["bb_lower"] = mid - config.bb_std * std
    df["bb_mid"]   = mid
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / mid.replace(0, np.nan)

    # RSI, ATR, ADX
    df["rsi"]     = _rsi(df["close"], config.rsi_period)
    df["atr"]     = _atr(df, config.atr_period)
    df["adx"]     = _adx(df, config.adx_period)
    df["adx_prev"] = df["adx"].shift(1)
    df["adx_falling"] = df["adx"] < df["adx_prev"]

    # Volume ratio
    df["vol_sma20"]   = df["volume"].shift(1).rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["vol_sma20"].replace(0, np.nan)

    # Candle structure (for rejection filter)
    candle_range = (df["high"] - df["low"]).replace(0, np.nan)
    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
    df["upper_wick_pct"] = df["upper_wick"] / candle_range
    df["lower_wick_pct"] = df["lower_wick"] / candle_range

    return df


def signal(row: pd.Series, config: MRConfig) -> Optional[dict]:
    if pd.isna(row.get("bb_upper")) or pd.isna(row.get("rsi")) or pd.isna(row.get("adx")):
        return None

    # Regime gate: must be ranging
    if row["adx"] >= config.adx_max:
        return None
    if config.require_adx_falling and not bool(row.get("adx_falling", False)):
        return None

    # Volume filter
    if config.volume_mult > 0 and row.get("volume_ratio", 0) < config.volume_mult:
        return None

    close = row["close"]
    side = None

    # BUY: oversold fade
    if close < row["bb_lower"] or (not config.require_bb_close and row["low"] < row["bb_lower"]):
        if config.require_rsi and row["rsi"] >= config.rsi_os:
            return None
        if config.require_rejection and row.get("lower_wick_pct", 0) < config.min_wick_pct:
            return None
        side = "BUY"

    # SELL: overbought fade
    elif close > row["bb_upper"] or (not config.require_bb_close and row["high"] > row["bb_upper"]):
        if config.require_rsi and row["rsi"] <= config.rsi_ob:
            return None
        if config.require_rejection and row.get("upper_wick_pct", 0) < config.min_wick_pct:
            return None
        side = "SELL"

    if side is None:
        return None
    if config.allowed_side and side != config.allowed_side:
        return None

    return {"side": side, "bb_mid": float(row["bb_mid"])}


def _simulate_exit(m1: pd.DataFrame, entry_time: pd.Timestamp,
                   side: str, sl: float, tp: float) -> tuple:
    window = m1[m1.index >= entry_time]
    for ts, row in window.iterrows():
        if side == "BUY":
            if row["low"] <= sl:
                return ts, sl, "SL"
            if row["high"] >= tp:
                return ts, tp, "TP"
        else:
            if row["high"] >= sl:
                return ts, sl, "SL"
            if row["low"] <= tp:
                return ts, tp, "TP"
    if window.empty:
        return entry_time, np.nan, "OPEN"
    return window.index[-1], float(window.iloc[-1]["close"]), "OPEN"


def run_backtest(symbol: str, m1: pd.DataFrame, m15: pd.DataFrame,
                 config: MRConfig, variant: str, split: str):
    rng = np.random.default_rng(config.seed)
    balance = config.initial_balance
    peak = balance
    max_dd = 0.0
    trades = []
    equity = []
    daily_trades: dict = {}
    daily_pnl: dict = {}
    unavailable_until = pd.Timestamp.min

    for i in range(len(m15) - 1):
        ts  = m15.index[i]
        row = m15.iloc[i]
        peak   = max(peak, balance)
        max_dd = min(max_dd, (balance - peak) / peak)
        equity.append({
            "time": ts, "symbol": symbol, "variant": variant,
            "split": split, "balance": balance, "drawdown_pct": max_dd * 100,
        })

        if ts <= unavailable_until:
            continue
        if config.include_utc_hours is not None and ts.hour not in config.include_utc_hours:
            continue
        if config.exclude_utc_hours and ts.hour in config.exclude_utc_hours:
            continue

        day = ts.date()
        if daily_trades.get(day, 0) >= config.max_trades_per_day:
            continue
        if daily_pnl.get(day, 0.0) <= -(config.initial_balance * config.daily_loss_limit_pct):
            continue

        sig = signal(row, config)
        if sig is None:
            continue

        # Entry at next M15 open
        next_row   = m15.iloc[i + 1]
        entry_time = m15.index[i + 1]
        entry_base = float(next_row["open"])

        spread   = max(float(row["atr"]) * config.spread_atr_mult,
                       float(row["close"]) * 0.00001)
        slippage = rng.uniform(config.slippage_min_spread,
                               config.slippage_max_spread) * spread

        if sig["side"] == "BUY":
            entry = entry_base + spread / 2 + slippage
        else:
            entry = entry_base - spread / 2 - slippage

        atr_val = float(row["atr"])
        if not np.isfinite(atr_val) or atr_val <= 0:
            continue

        # SL: ATR-based
        sl_dist = config.sl_atr_mult * atr_val
        if sig["side"] == "BUY":
            sl = entry - sl_dist
            if config.tp_type == "bb_mid":
                tp = sig["bb_mid"]
                if tp <= entry:          # midline below entry → skip
                    continue
            else:
                tp = entry + sl_dist * config.rr
        else:
            sl = entry + sl_dist
            if config.tp_type == "bb_mid":
                tp = sig["bb_mid"]
                if tp >= entry:          # midline above entry → skip
                    continue
            else:
                tp = entry - sl_dist * config.rr

        risk_dist = abs(entry - sl)
        if risk_dist <= 0:
            continue

        risk_base = config.initial_balance if config.fixed_risk_base else balance
        qty = (risk_base * config.risk_pct) / risk_dist

        exit_time, exit_price, result = _simulate_exit(m1, entry_time,
                                                        sig["side"], sl, tp)
        unavailable_until = exit_time
        if result == "OPEN" or not np.isfinite(exit_price):
            continue

        gross = (exit_price - entry) * qty
        if sig["side"] == "SELL":
            gross = -gross
        fees = (entry * qty + exit_price * qty) * config.fee_rate
        pnl  = gross - fees

        balance += pnl
        daily_trades[day] = daily_trades.get(day, 0) + 1
        daily_pnl[day]    = daily_pnl.get(day, 0.0) + pnl

        trades.append({
            "symbol": symbol, "variant": variant, "split": split,
            "timestamp": ts, "entry_time": entry_time, "exit_time": exit_time,
            "side": sig["side"], "entry": entry, "sl": sl, "tp": tp,
            "qty": qty, "pnl": pnl, "result": result, "balance": balance,
            "adx": row.get("adx"), "rsi": row.get("rsi"),
            "bb_width": row.get("bb_width"), "atr": atr_val,
        })

    return pd.DataFrame(trades), pd.DataFrame(equity)


def metrics(trades: pd.DataFrame, equity: pd.DataFrame, days: int) -> dict:
    if trades.empty:
        return {"total_trades": 0, "winrate": 0.0, "profit_factor": 0.0,
                "max_drawdown": 0.0, "expectancy": 0.0,
                "trades_per_day": 0.0, "profit_pct": 0.0}
    wins   = trades[trades["pnl"] > 0]
    losses = trades[trades["pnl"] <= 0]
    gp = wins["pnl"].sum()
    gl = -losses["pnl"].sum()
    end = trades["balance"].iloc[-1]
    return {
        "total_trades":  len(trades),
        "winrate":       len(wins) / len(trades) * 100,
        "profit_factor": gp / gl if gl else 0.0,
        "max_drawdown":  float(equity["drawdown_pct"].min()) if not equity.empty else 0.0,
        "expectancy":    trades["pnl"].mean(),
        "trades_per_day": len(trades) / max(days, 1),
        "profit_pct":    (end - 1000.0) / 1000.0 * 100,
    }


def monthly_performance(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    df = trades.copy()
    df["month"] = pd.to_datetime(df["exit_time"]).dt.to_period("M").astype(str)
    return (
        df.groupby(["symbol", "variant", "split", "month"])
        .agg(trades=("pnl", "count"), pnl=("pnl", "sum"),
             wins=("pnl", lambda x: int((x > 0).sum())),
             losses=("pnl", lambda x: int((x <= 0).sum())))
        .reset_index()
    )


def find_symbol_files(symbol: str) -> Optional[dict]:
    prefix_map = {"BTCUSDT": "bitcoin", "ETHUSDT": "ethereum", "SOLUSDT": "solana"}
    prefix = prefix_map.get(symbol, symbol.lower())
    for suffix in ["2022_2025", "365d"]:
        paths = {
            "m1":  DATA_DIR / f"{prefix}_{suffix}_1m.csv",
            "m15": DATA_DIR / f"{prefix}_{suffix}_15m.csv",
        }
        if all(p.exists() for p in paths.values()):
            return paths
    return None


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").drop_duplicates("time").set_index("time")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["open", "high", "low", "close", "volume"]].dropna()


def write_csv(path: Path, df: pd.DataFrame):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
