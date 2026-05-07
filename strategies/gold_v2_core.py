from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


StrategyName = Literal["sweep", "momentum"]


@dataclass(frozen=True)
class GoldV2Config:
    strategy: StrategyName
    variant: str
    initial_balance: float = 10_000.0
    risk_pct: float = 0.0025
    fee_rate: float = 0.0001
    spread_atr_mult: float = 0.03
    slippage_min_spread: float = 0.3
    slippage_max_spread: float = 1.0
    max_trades_per_day: int = 3
    daily_loss_limit_pct: float = 0.03
    max_hold_minutes: int | None = 720
    seed: int = 42

    # Strategy A: liquidity sweep reversal.
    sweep_lookback: int = 20
    wick_body_min: float = 1.5
    atr_percentile_min: float = 30.0
    sweep_rr: float = 2.0
    sl_atr_buffer: float = 0.10
    use_m1_confirmation: bool = False

    # Strategy B: session momentum breakout.
    session: str = "london_ny"
    adx_min: float = 20.0
    momentum_rr: float = 2.0
    use_atr_expansion: bool = True
    body_mult: float = 1.5
    max_wick_pct: float = 0.50
    momentum_sl_atr_mult: float = 1.0
    session_hours_utc: tuple[int, ...] | None = None
    close_extreme_pct: float | None = None
    ema_extension_atr_max: float | None = None
    use_m15_ema_trend: bool = False
    use_m15_ema200_filter: bool = False
    momentum_atr_percentile_min: float | None = None
    momentum_atr_percentile_max: float | None = None
    session_ranges_utc_minutes: tuple[tuple[int, int], ...] | None = None
    entry_mode: str = "next_m5"
    allowed_side: str | None = None
    buy_body_mult: float | None = None
    buy_max_wick_pct: float | None = None
    buy_atr_percentile_min: float | None = None
    buy_close_extreme_pct: float | None = None


def load_csv(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").drop_duplicates("time").set_index("time")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["open", "high", "low", "close", "volume"]].dropna()


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_smooth = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_smooth.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_smooth.replace(0, np.nan)
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def _session_hours(session: str) -> tuple[int, ...]:
    london = (7, 8, 9)          # Thailand 14:00-17:00
    ny = (12, 13, 14, 15)       # Thailand 19:00-23:00
    if session == "london":
        return london
    if session == "ny":
        return ny
    return london + ny


def session_hours(config: GoldV2Config) -> tuple[int, ...]:
    return config.session_hours_utc if config.session_hours_utc is not None else _session_hours(config.session)


def in_session(ts: pd.Timestamp, config: GoldV2Config) -> bool:
    if config.session_ranges_utc_minutes is None:
        return ts.hour in session_hours(config)
    minute = ts.hour * 60 + ts.minute
    return any(start <= minute < end for start, end in config.session_ranges_utc_minutes)


def prepare(m5: pd.DataFrame, m15: pd.DataFrame, config: GoldV2Config) -> pd.DataFrame:
    m15 = m15.copy()
    m15["ema20"] = m15["close"].ewm(span=20, adjust=False).mean()
    m15["ema50"] = m15["close"].ewm(span=50, adjust=False).mean()
    m15["ema200"] = m15["close"].ewm(span=200, adjust=False).mean()
    m15["atr"] = atr(m15)
    m15["atr_sma20"] = m15["atr"].shift(1).rolling(20).mean()
    m15["adx"] = adx(m15)
    m15["adx_rising"] = m15["adx"] > m15["adx"].shift(1)

    m5 = m5.copy()
    m5["atr"] = atr(m5)
    m5["atr_percentile_100"] = m5["atr"].rolling(100).apply(
        lambda values: float((values <= values[-1]).mean() * 100),
        raw=True,
    )
    m5["donchian_high_10"] = m5["high"].shift(1).rolling(10).max()
    m5["donchian_low_10"] = m5["low"].shift(1).rolling(10).min()
    m5["donchian_high_20"] = m5["high"].shift(1).rolling(20).max()
    m5["donchian_low_20"] = m5["low"].shift(1).rolling(20).min()
    m5["donchian_high_30"] = m5["high"].shift(1).rolling(30).max()
    m5["donchian_low_30"] = m5["low"].shift(1).rolling(30).min()
    m5["swing_high"] = m5["high"].shift(1).rolling(8).max()
    m5["swing_low"] = m5["low"].shift(1).rolling(8).min()

    candle_range = (m5["high"] - m5["low"]).replace(0, np.nan)
    m5["body"] = (m5["close"] - m5["open"]).abs()
    m5["ema20"] = m5["close"].ewm(span=20, adjust=False).mean()
    m5["avg_body20"] = m5["body"].shift(1).rolling(20).mean()
    m5["upper_wick"] = m5["high"] - m5[["open", "close"]].max(axis=1)
    m5["lower_wick"] = m5[["open", "close"]].min(axis=1) - m5["low"]
    m5["wick_pct"] = m5[["upper_wick", "lower_wick"]].max(axis=1) / candle_range
    m5["close_location"] = (m5["close"] - m5["low"]) / candle_range

    aligned = m5.copy()
    aligned["m15_ema20"] = m15["ema20"].shift(1).reindex(aligned.index, method="ffill")
    aligned["m15_ema50"] = m15["ema50"].shift(1).reindex(aligned.index, method="ffill")
    aligned["m15_ema200"] = m15["ema200"].shift(1).reindex(aligned.index, method="ffill")
    aligned["m15_adx"] = m15["adx"].shift(1).reindex(aligned.index, method="ffill")
    aligned["m15_adx_rising"] = (
        m15["adx_rising"].shift(1).reindex(aligned.index, method="ffill").fillna(False).astype(bool)
    )
    aligned["m15_atr"] = m15["atr"].shift(1).reindex(aligned.index, method="ffill")
    aligned["m15_atr_sma20"] = m15["atr_sma20"].shift(1).reindex(aligned.index, method="ffill")
    return aligned


def signal(row: pd.Series, config: GoldV2Config) -> dict | None:
    if config.strategy == "sweep":
        return sweep_signal(row, config)
    return momentum_signal(row, config)


def candidate_positions(m5: pd.DataFrame, config: GoldV2Config) -> np.ndarray:
    """Cheap vectorized prefilter; exact rules are still checked by signal()."""
    valid = m5["atr"].notna()
    if config.strategy == "sweep":
        body = m5["body"].clip(lower=m5["close"] * 0.00001)
        high_level = m5[f"donchian_high_{config.sweep_lookback}"]
        low_level = m5[f"donchian_low_{config.sweep_lookback}"]
        lower_ok = m5["lower_wick"] >= body * config.wick_body_min
        upper_ok = m5["upper_wick"] >= body * config.wick_body_min
        long_sweep = (m5["low"] < low_level) & (m5["close"] > low_level) & lower_ok
        short_sweep = (m5["high"] > high_level) & (m5["close"] < high_level) & upper_ok
        valid &= m5["atr_percentile_100"].ge(config.atr_percentile_min)
        valid &= long_sweep | short_sweep
    else:
        if config.session_ranges_utc_minutes is None:
            hours = set(session_hours(config))
            valid &= pd.Series(m5.index.hour, index=m5.index).isin(hours)
        else:
            minutes = pd.Series(m5.index.hour * 60 + m5.index.minute, index=m5.index)
            in_range = pd.Series(False, index=m5.index)
            for start, end in config.session_ranges_utc_minutes:
                in_range |= minutes.ge(start) & minutes.lt(end)
            valid &= in_range
        valid &= m5["m15_adx"].gt(config.adx_min)
        valid &= m5["m15_adx_rising"].astype(bool)
        if config.use_atr_expansion:
            valid &= m5["m15_atr"].gt(m5["m15_atr_sma20"])
        valid &= m5["body"].gt(m5["avg_body20"] * config.body_mult)
        valid &= m5["wick_pct"].le(config.max_wick_pct)
        break_up = m5["close"].gt(m5["donchian_high_20"])
        break_down = m5["close"].lt(m5["donchian_low_20"])
        if config.allowed_side == "BUY":
            break_down &= False
        if config.allowed_side == "SELL":
            break_up &= False
        if config.buy_body_mult is not None:
            break_up &= m5["body"].gt(m5["avg_body20"] * config.buy_body_mult)
        if config.buy_max_wick_pct is not None:
            break_up &= m5["wick_pct"].le(config.buy_max_wick_pct)
        if config.buy_atr_percentile_min is not None:
            break_up &= m5["atr_percentile_100"].ge(config.buy_atr_percentile_min)
        if config.buy_close_extreme_pct is not None:
            break_up &= m5["close_location"].ge(1.0 - config.buy_close_extreme_pct)
        if config.close_extreme_pct is not None:
            break_up &= m5["close_location"].ge(1.0 - config.close_extreme_pct)
            break_down &= m5["close_location"].le(config.close_extreme_pct)
        if config.ema_extension_atr_max is not None:
            distance = (m5["close"] - m5["ema20"]).abs() / m5["atr"].replace(0, np.nan)
            valid &= distance.le(config.ema_extension_atr_max)
        if config.momentum_atr_percentile_min is not None:
            valid &= m5["atr_percentile_100"].ge(config.momentum_atr_percentile_min)
        if config.momentum_atr_percentile_max is not None:
            valid &= m5["atr_percentile_100"].le(config.momentum_atr_percentile_max)
        valid &= break_up | break_down
    positions = np.flatnonzero(valid.fillna(False).to_numpy())
    return positions[positions < len(m5) - 1]


def sweep_signal(row: pd.Series, config: GoldV2Config) -> dict | None:
    if pd.isna(row.get("atr")) or row.get("atr_percentile_100", 0) < config.atr_percentile_min:
        return None
    high_level = row.get(f"donchian_high_{config.sweep_lookback}")
    low_level = row.get(f"donchian_low_{config.sweep_lookback}")
    if pd.isna(high_level) or pd.isna(low_level):
        return None
    body = max(float(row["body"]), float(row["close"]) * 0.00001)
    lower_ok = float(row["lower_wick"]) >= body * config.wick_body_min
    upper_ok = float(row["upper_wick"]) >= body * config.wick_body_min

    if row["low"] < low_level and row["close"] > low_level and lower_ok:
        return {"side": "BUY", "level": low_level, "sweep_extreme": row["low"], "pattern": "sweep_low"}
    if row["high"] > high_level and row["close"] < high_level and upper_ok:
        return {"side": "SELL", "level": high_level, "sweep_extreme": row["high"], "pattern": "sweep_high"}
    return None


def momentum_signal(row: pd.Series, config: GoldV2Config) -> dict | None:
    if not in_session(row.name, config):
        return None
    required = ["atr", "m15_adx", "avg_body20", "donchian_high_20", "donchian_low_20"]
    if any(pd.isna(row.get(col)) for col in required):
        return None
    if row["m15_adx"] <= config.adx_min or not bool(row.get("m15_adx_rising", False)):
        return None
    if config.use_atr_expansion and not (row.get("m15_atr", 0) > row.get("m15_atr_sma20", np.inf)):
        return None
    if row["body"] <= row["avg_body20"] * config.body_mult:
        return None
    if row["wick_pct"] > config.max_wick_pct:
        return None
    if config.ema_extension_atr_max is not None:
        distance = abs(row["close"] - row["ema20"]) / row["atr"]
        if distance > config.ema_extension_atr_max:
            return None
    if config.momentum_atr_percentile_min is not None and row.get("atr_percentile_100", 0) < config.momentum_atr_percentile_min:
        return None
    if config.momentum_atr_percentile_max is not None and row.get("atr_percentile_100", 100) > config.momentum_atr_percentile_max:
        return None
    if row["close"] > row["donchian_high_20"]:
        if config.allowed_side == "SELL":
            return None
        body_mult = config.buy_body_mult if config.buy_body_mult is not None else config.body_mult
        wick_pct = config.buy_max_wick_pct if config.buy_max_wick_pct is not None else config.max_wick_pct
        atr_min = config.buy_atr_percentile_min if config.buy_atr_percentile_min is not None else config.momentum_atr_percentile_min
        close_extreme = config.buy_close_extreme_pct if config.buy_close_extreme_pct is not None else config.close_extreme_pct
        if row["body"] <= row["avg_body20"] * body_mult:
            return None
        if row["wick_pct"] > wick_pct:
            return None
        if atr_min is not None and row.get("atr_percentile_100", 0) < atr_min:
            return None
        if close_extreme is not None and row["close_location"] < 1.0 - close_extreme:
            return None
        if config.close_extreme_pct is not None and row["close_location"] < 1.0 - config.close_extreme_pct:
            return None
        if config.use_m15_ema_trend and not (row["m15_ema20"] > row["m15_ema50"]):
            return None
        if config.use_m15_ema200_filter and not (row["m15_ema50"] > row["m15_ema200"]):
            return None
        return {"side": "BUY", "level": row["donchian_high_20"], "pattern": "session_breakout"}
    if row["close"] < row["donchian_low_20"]:
        if config.allowed_side == "BUY":
            return None
        if config.close_extreme_pct is not None and row["close_location"] > config.close_extreme_pct:
            return None
        if config.use_m15_ema_trend and not (row["m15_ema20"] < row["m15_ema50"]):
            return None
        if config.use_m15_ema200_filter and not (row["m15_ema50"] < row["m15_ema200"]):
            return None
        return {"side": "SELL", "level": row["donchian_low_20"], "pattern": "session_breakout"}
    return None


def m1_confirmation_time(
    m1: pd.DataFrame,
    after: pd.Timestamp,
    side: str,
    window_minutes: int = 3,
) -> pd.Timestamp | None:
    window = m1[(m1.index >= after) & (m1.index < after + pd.Timedelta(minutes=window_minutes))]
    if window.empty:
        return None
    for ts, row in window.iterrows():
        if side == "BUY" and row["close"] > row["open"]:
            return ts + pd.Timedelta(minutes=1)
        if side == "SELL" and row["close"] < row["open"]:
            return ts + pd.Timedelta(minutes=1)
    return None


def trade_levels(row: pd.Series, entry: float, sig: dict, config: GoldV2Config) -> tuple[float, float, float] | None:
    side = sig["side"]
    if config.strategy == "sweep":
        buffer = float(row["atr"]) * config.sl_atr_buffer
        rr = config.sweep_rr
        if side == "BUY":
            sl = float(sig["sweep_extreme"]) - buffer
            risk = entry - sl
            tp = entry + risk * rr
        else:
            sl = float(sig["sweep_extreme"]) + buffer
            risk = sl - entry
            tp = entry - risk * rr
    else:
        rr = config.momentum_rr
        atr_stop = float(row["atr"]) * config.momentum_sl_atr_mult
        if side == "BUY":
            sl = min(float(row.get("swing_low", entry - atr_stop)), entry - atr_stop)
            risk = entry - sl
            tp = entry + risk * rr
        else:
            sl = max(float(row.get("swing_high", entry + atr_stop)), entry + atr_stop)
            risk = sl - entry
            tp = entry - risk * rr
    if not np.isfinite(risk) or risk <= 0:
        return None
    return sl, tp, risk


def simulate_exit(
    exit_bars: pd.DataFrame,
    entry_time: pd.Timestamp,
    side: str,
    sl: float,
    tp: float,
    max_hold_minutes: int | None,
) -> tuple[pd.Timestamp, float, str]:
    start = exit_bars.index.searchsorted(entry_time, side="left")
    if start >= len(exit_bars):
        return entry_time, np.nan, "OPEN"
    if max_hold_minutes:
        cutoff = entry_time + pd.Timedelta(minutes=max_hold_minutes)
        end = exit_bars.index.searchsorted(cutoff, side="left") + 1
        end = min(end, len(exit_bars))
    else:
        cutoff = None
        end = len(exit_bars)

    window = exit_bars.iloc[start:end]
    if side == "BUY":
        sl_hits = np.flatnonzero(window["low"].to_numpy() <= sl)
        tp_hits = np.flatnonzero(window["high"].to_numpy() >= tp)
    else:
        sl_hits = np.flatnonzero(window["high"].to_numpy() >= sl)
        tp_hits = np.flatnonzero(window["low"].to_numpy() <= tp)

    sl_idx = int(sl_hits[0]) if len(sl_hits) else None
    tp_idx = int(tp_hits[0]) if len(tp_hits) else None
    if sl_idx is not None and (tp_idx is None or sl_idx <= tp_idx):
        return window.index[sl_idx], sl, "SL"
    if tp_idx is not None:
        return window.index[tp_idx], tp, "TP"
    if cutoff is not None and not window.empty and window.index[-1] >= cutoff:
        return window.index[-1], float(window.iloc[-1]["close"]), "TIMEOUT"
    return window.index[-1], float(window.iloc[-1]["close"]), "OPEN"


def run_backtest(
    symbol: str,
    m1: pd.DataFrame,
    m5: pd.DataFrame,
    exit_bars: pd.DataFrame,
    config: GoldV2Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(config.seed)
    balance = config.initial_balance
    peak = balance
    max_dd = 0.0
    unavailable_until: pd.Timestamp | None = None
    daily_trades: dict = {}
    daily_pnl: dict = {}
    trades: list[dict] = []
    equity: list[dict] = []

    for i in candidate_positions(m5, config):
        ts = m5.index[i]
        row = m5.iloc[i]
        peak = max(peak, balance)
        max_dd = min(max_dd, (balance - peak) / peak)
        equity.append(
            {
                "time": ts,
                "symbol": symbol,
                "variant": config.variant,
                "strategy": config.strategy,
                "balance": balance,
                "drawdown_pct": max_dd * 100,
            }
        )
        if unavailable_until is not None and ts <= unavailable_until:
            continue
        day = ts.date()
        if daily_trades.get(day, 0) >= config.max_trades_per_day:
            continue
        if daily_pnl.get(day, 0.0) <= -(config.initial_balance * config.daily_loss_limit_pct):
            continue

        sig = signal(row, config)
        if sig is None:
            continue

        entry_time = m5.index[i + 1]
        if config.entry_mode == "next_m1":
            entry_time = ts + pd.Timedelta(minutes=5)
        if config.strategy == "sweep" and config.use_m1_confirmation and not m1.empty:
            confirmed = m1_confirmation_time(m1, ts + pd.Timedelta(minutes=5), sig["side"])
            if confirmed is None:
                continue
            entry_time = confirmed

        if entry_time in m1.index:
            entry_base = float(m1.loc[entry_time, "open"])
        else:
            entry_base = float(m5.iloc[i + 1]["open"])

        spread = max(float(row["atr"]) * config.spread_atr_mult, float(row["close"]) * 0.00001)
        slippage = rng.uniform(config.slippage_min_spread, config.slippage_max_spread) * spread
        if sig["side"] == "BUY":
            entry = entry_base + spread / 2 + slippage
        else:
            entry = entry_base - spread / 2 - slippage

        levels = trade_levels(row, entry, sig, config)
        if levels is None:
            continue
        sl, tp, risk_dist = levels
        qty = (config.initial_balance * config.risk_pct) / risk_dist
        if qty <= 0 or not np.isfinite(qty):
            continue

        exit_time, exit_price, result = simulate_exit(
            exit_bars, entry_time, sig["side"], sl, tp, config.max_hold_minutes
        )
        unavailable_until = exit_time
        if result == "OPEN" or not np.isfinite(exit_price):
            continue

        gross = (exit_price - entry) * qty
        if sig["side"] == "SELL":
            gross = -gross
        fee = (entry * qty + exit_price * qty) * config.fee_rate
        pnl = gross - fee
        balance += pnl
        daily_trades[day] = daily_trades.get(day, 0) + 1
        daily_pnl[day] = daily_pnl.get(day, 0.0) + pnl
        initial_risk = risk_dist * qty
        trades.append(
            {
                "symbol": symbol,
                "variant": config.variant,
                "strategy": config.strategy,
                "timestamp": ts,
                "entry_time": entry_time,
                "exit_time": exit_time,
                "side": sig["side"],
                "pattern": sig.get("pattern", ""),
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "rr": config.sweep_rr if config.strategy == "sweep" else config.momentum_rr,
                "qty": qty,
                "spread": spread,
                "slippage": slippage,
                "fee": fee,
                "gross_pnl": gross,
                "pnl": pnl,
                "r_multiple": pnl / initial_risk if initial_risk else np.nan,
                "result": result,
                "balance": balance,
                "atr": row.get("atr"),
                "atr_percentile_100": row.get("atr_percentile_100"),
                "m15_adx": row.get("m15_adx"),
            }
        )
    return pd.DataFrame(trades), pd.DataFrame(equity)


def profit_factor(pnl: pd.Series) -> float:
    gross_profit = pnl[pnl > 0].sum()
    gross_loss = -pnl[pnl <= 0].sum()
    return float(gross_profit / gross_loss) if gross_loss else 0.0


def metrics(trades: pd.DataFrame, equity: pd.DataFrame, days: int, initial_balance: float) -> dict:
    if trades.empty:
        return {
            "total_trades": 0,
            "winrate": 0.0,
            "profit_factor": 0.0,
            "profit_pct": 0.0,
            "max_drawdown": 0.0,
            "expectancy": 0.0,
            "avg_r": 0.0,
            "trades_per_day": 0.0,
            "worst_month": 0.0,
        }
    wins = trades[trades["pnl"] > 0]
    monthly = monthly_performance(trades)
    return {
        "total_trades": len(trades),
        "winrate": len(wins) / len(trades) * 100,
        "profit_factor": profit_factor(trades["pnl"]),
        "profit_pct": (trades["balance"].iloc[-1] - initial_balance) / initial_balance * 100,
        "max_drawdown": float(equity["drawdown_pct"].min()) if not equity.empty else 0.0,
        "expectancy": trades["pnl"].mean(),
        "avg_r": trades["r_multiple"].mean(),
        "trades_per_day": len(trades) / max(days, 1),
        "worst_month": float(monthly["pnl"].min()) if not monthly.empty else 0.0,
    }


def monthly_performance(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    df = trades.copy()
    df["month"] = pd.to_datetime(df["exit_time"]).dt.to_period("M").astype(str)
    return (
        df.groupby(["symbol", "variant", "strategy", "month"])
        .agg(
            trades=("pnl", "count"),
            pnl=("pnl", "sum"),
            wins=("pnl", lambda x: int((x > 0).sum())),
            losses=("pnl", lambda x: int((x <= 0).sum())),
        )
        .reset_index()
    )
