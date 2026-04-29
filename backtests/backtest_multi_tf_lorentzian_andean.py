import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
REPORT_DIR = ROOT_DIR / "reports"


@dataclass
class Config:
    m1_file: str = str(DATA_DIR / "bitcoin_365d_1m.csv")
    m5_file: str = str(DATA_DIR / "bitcoin_365d_5m.csv")
    m15_file: str = str(DATA_DIR / "bitcoin_365d_15m.csv")
    h1_file: str = str(DATA_DIR / "bitcoin_365d_1h.csv")
    initial_balance: float = 1000.0

    ema_fast: int = 20
    ema_slow: int = 50
    andean_length: int = 50
    andean_signal: int = 9
    lorentzian_k: int = 8
    lorentzian_horizon: int = 8
    lorentzian_threshold: float = 0.15
    lorentzian_train_window: int = 1800
    lorentzian_stride: int = 3
    m15_adx_min: float = 0.0
    require_m15_adx_rising: bool = False
    require_ema50_slope: bool = False
    use_h1_bias: bool = False
    session_name: str = "all"
    use_atr_expansion: bool = False
    atr_ma_period: int = 50
    atr_expansion_ratio_min: float = 1.0
    use_bb_width: bool = False
    bb_length: int = 20
    use_donchian: bool = False
    donchian_n: int = 20
    use_breakout_strength_filter: bool = False
    use_fake_breakout_filter: bool = False
    use_retest_entry: bool = False
    use_zone_quality_score: bool = False
    tier_mode: str = "off"
    tier_b_risk: float = 0.0075
    breakout_body_mult: float = 1.5
    breakout_atr_mult: float = 0.5
    breakout_wick_max_pct: float = 0.5
    breakout_close_location_min: float = 0.6
    volume_spike_mult: float = 0.0
    retest_tolerance_pct: float = 0.001
    max_retest_candles: int = 6
    zone_quality_min: float = 3.0
    chop_max: float = 0.0
    high_vol_day_only: bool = False
    regime_name: str = "baseline"

    atr_period: int = 14
    min_atr_pct: float = 0.001
    swing_lookback: int = 8
    risk_per_trade: float = 0.01
    rr: float = 2.0
    exit_mode: str = "be_1_lock_1_5"
    min_hold_minutes: int = 0
    partial_pct: float = 0.5
    trail_type: str = "prev_candle"
    fee_rate: float = 0.0004
    slippage_atr_mult: float = 0.02
    spread_atr_mult: float = 0.02
    slippage_spread_mult: float = 1.0
    max_spread_atr_mult: float = 0.05
    use_spread_filter: bool = False
    entry_latency_m1_candles: int = 0

    max_m1_confirm_candles: int = 5
    max_trades_per_day: int = 2
    max_losses_per_day: int = 1
    total_days: int = 0

    trade_log: str = str(REPORT_DIR / "multi_tf_lorentzian_andean_trades.csv")
    equity_log: str = str(REPORT_DIR / "multi_tf_lorentzian_andean_equity.csv")
    monthly_log: str = str(REPORT_DIR / "multi_tf_lorentzian_andean_monthly.csv")
    sweep_log: str = str(REPORT_DIR / "multi_tf_lorentzian_andean_sweep.csv")


@dataclass
class Position:
    side: str
    entry_time: pd.Timestamp
    entry: float
    sl: float
    initial_sl: float
    tp: float
    qty: float
    tier: str = "A"
    partial_taken: bool = False
    partial_lock_active: bool = False
    break_even_active: bool = False

    @property
    def one_r(self) -> float:
        return abs(self.entry - self.initial_sl)


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "time" not in df.columns:
        df.columns = ["time", "open", "high", "low", "close", "volume"][: len(df.columns)]
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").drop_duplicates("time").set_index("time")
    required = ["open", "high", "low", "close", "volume"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=required)


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


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


def choppiness_index(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_sum = tr.rolling(period).sum()
    high_low_range = (df["high"].rolling(period).max() - df["low"].rolling(period).min()).replace(0, np.nan)
    return 100 * np.log10(atr_sum / high_low_range) / np.log10(period)


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr_val = atr(df, period)
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(period).sum() / atr_val.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(period).sum() / atr_val.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.rolling(period).mean()


def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    typical = (df["high"] + df["low"] + df["close"]) / 3
    sma = typical.rolling(period).mean()
    mad = (typical - sma).abs().rolling(period).mean()
    return (typical - sma) / (0.015 * mad.replace(0, np.nan))


def macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    macd = ema(close, fast) - ema(close, slow)
    return macd - ema(macd, signal)


def calculate_andean(df: pd.DataFrame, length: int, signal_period: int = 9) -> pd.DataFrame:
    df = df.copy()
    alpha = 2 / (length + 1)
    opens = df["open"].to_numpy()
    closes = df["close"].to_numpy()
    up1 = np.zeros(len(df))
    up2 = np.zeros(len(df))
    dn1 = np.zeros(len(df))
    dn2 = np.zeros(len(df))
    up1[0] = max(opens[0], closes[0])
    up2[0] = max(opens[0] ** 2, closes[0] ** 2)
    dn1[0] = min(opens[0], closes[0])
    dn2[0] = min(opens[0] ** 2, closes[0] ** 2)
    for i in range(1, len(df)):
        c = closes[i]
        o = opens[i]
        up1[i] = max(c, o, up1[i - 1] - (up1[i - 1] - c) * alpha)
        up2[i] = max(c * c, o * o, up2[i - 1] - (up2[i - 1] - c * c) * alpha)
        dn1[i] = min(c, o, dn1[i - 1] + (c - dn1[i - 1]) * alpha)
        dn2[i] = min(c * c, o * o, dn2[i - 1] + (c * c - dn2[i - 1]) * alpha)
    bull = np.sqrt(np.maximum(dn2 - dn1 * dn1, 0))
    bear = np.sqrt(np.maximum(up2 - up1 * up1, 0))
    df["andean_bull"] = bull
    df["andean_bear"] = bear
    df["andean_signal"] = ema(pd.Series(np.maximum(bull, bear), index=df.index), signal_period)
    return df


def add_retest_flags(m5: pd.DataFrame, length: int, tolerance_pct: float, max_candles: int) -> pd.DataFrame:
    m5 = m5.copy()
    buy_flags = np.zeros(len(m5), dtype=bool)
    sell_flags = np.zeros(len(m5), dtype=bool)
    source_body_ratio = np.full(len(m5), np.nan)
    source_move_atr = np.full(len(m5), np.nan)
    source_wick_pct = np.full(len(m5), np.nan)
    source_close_quality = np.full(len(m5), np.nan)
    source_volume_ratio = np.full(len(m5), np.nan)
    pending_side = None
    pending_level = np.nan
    pending_until = -1
    pending_body_ratio = np.nan
    pending_move_atr = np.nan
    pending_wick_pct = np.nan
    pending_close_quality = np.nan
    pending_volume_ratio = np.nan

    high_col = f"donchian_high_{length}"
    low_col = f"donchian_low_{length}"
    for i, (_, row) in enumerate(m5.iterrows()):
        range_high = row.get(high_col)
        range_low = row.get(low_col)
        if pd.isna(range_high) or pd.isna(range_low):
            continue

        if pending_side and i <= pending_until:
            bullish = row["close"] > row["open"]
            bearish = row["close"] < row["open"]
            if pending_side == "BUY":
                near_level = row["low"] <= pending_level * (1 + tolerance_pct)
                reclaimed = row["close"] >= pending_level
                if near_level and reclaimed and bullish:
                    buy_flags[i] = True
                    source_body_ratio[i] = pending_body_ratio
                    source_move_atr[i] = pending_move_atr
                    source_wick_pct[i] = pending_wick_pct
                    source_close_quality[i] = pending_close_quality
                    source_volume_ratio[i] = pending_volume_ratio
                    pending_side = None
            else:
                near_level = row["high"] >= pending_level * (1 - tolerance_pct)
                reclaimed = row["close"] <= pending_level
                if near_level and reclaimed and bearish:
                    sell_flags[i] = True
                    source_body_ratio[i] = pending_body_ratio
                    source_move_atr[i] = pending_move_atr
                    source_wick_pct[i] = pending_wick_pct
                    source_close_quality[i] = pending_close_quality
                    source_volume_ratio[i] = pending_volume_ratio
                    pending_side = None
        elif pending_side and i > pending_until:
            pending_side = None

        if row["close"] > range_high:
            pending_side = "BUY"
            pending_level = range_high
            pending_until = i + max_candles
            pending_body_ratio = row.get("body", np.nan) / row.get("avg_body_20", np.nan)
            pending_move_atr = row.get(f"donchian_buy_move_atr_{length}", np.nan)
            pending_wick_pct = row.get("max_wick_pct", np.nan)
            pending_close_quality = row.get("close_location", np.nan)
            pending_volume_ratio = row.get("volume_ratio", np.nan)
        elif row["close"] < range_low:
            pending_side = "SELL"
            pending_level = range_low
            pending_until = i + max_candles
            pending_body_ratio = row.get("body", np.nan) / row.get("avg_body_20", np.nan)
            pending_move_atr = row.get(f"donchian_sell_move_atr_{length}", np.nan)
            pending_wick_pct = row.get("max_wick_pct", np.nan)
            pending_close_quality = 1 - row.get("close_location", np.nan)
            pending_volume_ratio = row.get("volume_ratio", np.nan)

    m5[f"donchian_retest_buy_{length}"] = buy_flags
    m5[f"donchian_retest_sell_{length}"] = sell_flags
    m5[f"retest_source_body_ratio_{length}"] = source_body_ratio
    m5[f"retest_source_move_atr_{length}"] = source_move_atr
    m5[f"retest_source_wick_pct_{length}"] = source_wick_pct
    m5[f"retest_source_close_quality_{length}"] = source_close_quality
    m5[f"retest_source_volume_ratio_{length}"] = source_volume_ratio
    return m5


def calculate_indicators(m1: pd.DataFrame, m5: pd.DataFrame, m15: pd.DataFrame, config: Config):
    m15 = m15.copy()
    m15["ema_fast"] = ema(m15["close"], config.ema_fast)
    m15["ema_slow"] = ema(m15["close"], config.ema_slow)
    m15["ema50"] = ema(m15["close"], 50)
    m15["ema50_slope"] = m15["ema50"] - m15["ema50"].shift(3)
    m15["adx"] = adx(m15, 14)
    m15["trend"] = np.where(m15["ema_fast"] > m15["ema_slow"], 1, np.where(m15["ema_fast"] < m15["ema_slow"], -1, 0))
    m15["closed_trend"] = m15["trend"].shift(1)
    m15["closed_adx"] = m15["adx"].shift(1)
    m15["closed_adx_rising"] = (m15["adx"].shift(1) > m15["adx"].shift(2)).astype(float)
    m15["closed_ema50_slope"] = m15["ema50_slope"].shift(1)

    m5 = calculate_andean(m5, config.andean_length, config.andean_signal)
    m5["ema20"] = ema(m5["close"], 20)
    m5["atr"] = atr(m5, config.atr_period)
    m5["atr_pct"] = m5["atr"] / m5["close"]
    m5["spread_proxy"] = m5["atr"] * config.spread_atr_mult
    m5["spread_atr_ratio"] = m5["spread_proxy"] / m5["atr"].replace(0, np.nan)
    candle_range = (m5["high"] - m5["low"]).replace(0, np.nan)
    m5["body"] = (m5["close"] - m5["open"]).abs()
    m5["avg_body_20"] = m5["body"].shift(1).rolling(20).mean()
    m5["upper_wick"] = m5["high"] - m5[["open", "close"]].max(axis=1)
    m5["lower_wick"] = m5[["open", "close"]].min(axis=1) - m5["low"]
    m5["max_wick_pct"] = m5[["upper_wick", "lower_wick"]].max(axis=1) / candle_range
    m5["close_location"] = (m5["close"] - m5["low"]) / candle_range
    m5["volume_sma_20"] = m5["volume"].shift(1).rolling(20).mean()
    m5["volume_ratio"] = m5["volume"] / m5["volume_sma_20"].replace(0, np.nan)
    for period in [50, 100]:
        m5[f"atr_sma_{period}"] = m5["atr"].rolling(period).mean()
        m5[f"atr_expansion_{period}"] = m5["atr"] / m5[f"atr_sma_{period}"].replace(0, np.nan)
    for length in [20, 50]:
        basis = m5["close"].rolling(length).mean()
        dev = m5["close"].rolling(length).std()
        width = ((basis + 2 * dev) - (basis - 2 * dev)) / basis.replace(0, np.nan)
        m5[f"bb_width_{length}"] = width
        m5[f"bb_width_expanding_{length}"] = width > width.shift(1)
    for length in [15, 20, 25, 30, 50, 100]:
        m5[f"donchian_high_{length}"] = m5["high"].shift(1).rolling(length).max()
        m5[f"donchian_low_{length}"] = m5["low"].shift(1).rolling(length).min()
        m5[f"donchian_buy_move_atr_{length}"] = (m5["close"] - m5[f"donchian_high_{length}"]) / m5["atr"].replace(0, np.nan)
        m5[f"donchian_sell_move_atr_{length}"] = (m5[f"donchian_low_{length}"] - m5["close"]) / m5["atr"].replace(0, np.nan)
        m5 = add_retest_flags(m5, length, config.retest_tolerance_pct, config.max_retest_candles)
    m5["chop"] = choppiness_index(m5, 14)
    daily = m5.resample("1D").agg({"high": "max", "low": "min", "close": "last"}).dropna()
    daily["range_pct"] = (daily["high"] - daily["low"]) / daily["close"]
    daily["high_vol_day"] = daily["range_pct"].shift(1) > daily["range_pct"].shift(1).rolling(30, min_periods=10).median()
    m5["high_vol_day"] = daily["high_vol_day"].reindex(m5.index, method="ffill").fillna(False)
    m5["rsi"] = rsi(m5["close"], 14)
    m5["adx"] = adx(m5, 14)
    m5["cci"] = cci(m5, 20)
    m5["macd_hist"] = macd_hist(m5["close"])
    m5["swing_low"] = m5["low"].shift(1).rolling(config.swing_lookback).min()
    m5["swing_high"] = m5["high"].shift(1).rolling(config.swing_lookback).max()

    m1 = calculate_andean(m1, config.andean_length, config.andean_signal)
    m1["ema20"] = ema(m1["close"], 20)
    m1_trail = m1.resample("5min").agg({"high": "max", "low": "min", "close": "last"}).dropna()
    m1_trail["ema20_5m"] = ema(m1_trail["close"], 20)
    m1["trail_prev_high"] = m1_trail["high"].shift(1).reindex(m1.index, method="ffill")
    m1["trail_prev_low"] = m1_trail["low"].shift(1).reindex(m1.index, method="ffill")
    m1["trail_ema20"] = m1_trail["ema20_5m"].shift(1).reindex(m1.index, method="ffill")
    return m1, m5, m15


def calculate_h1_indicators(h1: pd.DataFrame) -> pd.DataFrame:
    h1 = h1.copy()
    h1["ema50"] = ema(h1["close"], 50)
    h1["h1_bias"] = np.where(h1["close"] > h1["ema50"], 1, np.where(h1["close"] < h1["ema50"], -1, 0))
    h1["closed_h1_bias"] = h1["h1_bias"].shift(1)
    return h1


def align_timeframes(m5: pd.DataFrame, m15: pd.DataFrame) -> pd.DataFrame:
    m5 = m5.copy()
    m5["m15_trend"] = m15["closed_trend"].reindex(m5.index, method="ffill")
    m5["m15_adx"] = m15["closed_adx"].reindex(m5.index, method="ffill")
    m5["m15_adx_rising"] = m15["closed_adx_rising"].reindex(m5.index, method="ffill")
    m5["m15_ema50_slope"] = m15["closed_ema50_slope"].reindex(m5.index, method="ffill")
    return m5


def align_h1_timeframe(m5: pd.DataFrame, h1: pd.DataFrame) -> pd.DataFrame:
    m5 = m5.copy()
    m5["h1_bias"] = h1["closed_h1_bias"].reindex(m5.index, method="ffill")
    return m5


def _feature_matrix(m5: pd.DataFrame) -> np.ndarray:
    features = m5[["rsi", "adx", "cci", "macd_hist"]].copy()
    features["rsi"] = features["rsi"] / 100
    features["adx"] = features["adx"] / 100
    features["cci"] = features["cci"] / 200
    hist_std = features["macd_hist"].rolling(200, min_periods=50).std().replace(0, np.nan)
    features["macd_hist"] = features["macd_hist"] / hist_std
    return features.replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)


def calculate_lorentzian(m5: pd.DataFrame, config: Config) -> pd.DataFrame:
    m5 = m5.copy()
    x = _feature_matrix(m5)
    future_return = m5["close"].shift(-config.lorentzian_horizon) / m5["close"] - 1
    labels = np.where(future_return > 0, 1, np.where(future_return < 0, -1, 0))
    scores = np.full(len(m5), np.nan)
    min_i = max(250, config.lorentzian_train_window // 4) + config.lorentzian_horizon
    for i in range(min_i, len(m5)):
        if np.isnan(x[i]).any():
            continue
        train_end = i - config.lorentzian_horizon
        train_start = max(0, train_end - config.lorentzian_train_window)
        idx = np.arange(train_start, train_end, config.lorentzian_stride)
        idx = idx[~np.isnan(x[idx]).any(axis=1)]
        if len(idx) < config.lorentzian_k:
            continue
        dist = np.log1p(np.abs(x[idx] - x[i])).sum(axis=1)
        nearest = idx[np.argpartition(dist, config.lorentzian_k - 1)[: config.lorentzian_k]]
        scores[i] = labels[nearest].sum() / config.lorentzian_k
    m5["lorentzian_score"] = scores
    return m5


def breakout_quality(row: pd.Series, side: str, config: Config) -> float:
    length = config.donchian_n
    avg_body = row.get("avg_body_20", np.nan)
    atr_expansion = row.get("atr_expansion_50", np.nan)
    volume_ratio = row.get("volume_ratio", np.nan)
    if side == "BUY":
        move_atr = row.get(f"donchian_buy_move_atr_{length}", np.nan)
        close_quality = row.get("close_location", np.nan)
    else:
        move_atr = row.get(f"donchian_sell_move_atr_{length}", np.nan)
        close_quality = 1 - row.get("close_location", np.nan)
    body_ratio = row.get("body", np.nan) / avg_body if not pd.isna(avg_body) and avg_body > 0 else np.nan
    if config.use_retest_entry:
        body_ratio = row.get(f"retest_source_body_ratio_{length}", body_ratio)
        move_atr = row.get(f"retest_source_move_atr_{length}", move_atr)
        close_quality = row.get(f"retest_source_close_quality_{length}", close_quality)
        volume_ratio = row.get(f"retest_source_volume_ratio_{length}", volume_ratio)

    score = 0.0
    if not pd.isna(body_ratio) and body_ratio >= config.breakout_body_mult:
        score += 1.0
    if not pd.isna(move_atr) and move_atr >= config.breakout_atr_mult:
        score += 1.0
    if not pd.isna(atr_expansion) and atr_expansion >= 1.1:
        score += 1.0
    if bool(row.get("m15_adx_rising", False)):
        score += 1.0
    if config.volume_spike_mult > 0 and not pd.isna(volume_ratio) and volume_ratio >= config.volume_spike_mult:
        score += 1.0
    if not pd.isna(close_quality) and close_quality >= config.breakout_close_location_min:
        score += 1.0
    return score


def high_quality_breakout(row: pd.Series, side: str, config: Config) -> bool:
    length = config.donchian_n
    avg_body = row.get("avg_body_20", np.nan)
    if pd.isna(avg_body) or avg_body <= 0:
        return False
    body_ok = row.get("body", 0) > avg_body * config.breakout_body_mult
    if side == "BUY":
        move_atr = row.get(f"donchian_buy_move_atr_{length}", np.nan)
        close_quality = row.get("close_location", np.nan)
    else:
        move_atr = row.get(f"donchian_sell_move_atr_{length}", np.nan)
        close_quality = 1 - row.get("close_location", np.nan)
    move_ok = not pd.isna(move_atr) and move_atr > config.breakout_atr_mult
    wick_ok = row.get("max_wick_pct", 1) <= config.breakout_wick_max_pct
    close_ok = not pd.isna(close_quality) and close_quality >= config.breakout_close_location_min
    atr_ok = row.get("atr_pct", 0) >= config.min_atr_pct
    return bool(body_ok and move_ok and wick_ok and close_ok and atr_ok)


def donchian_entry_ok(row: pd.Series, side: str, config: Config) -> bool:
    if not config.use_donchian:
        return True

    length = config.donchian_n
    if config.use_retest_entry:
        base_ok = bool(row.get(f"donchian_retest_{side.lower()}_{length}", False))
        move_atr = row.get(f"donchian_{side.lower()}_move_atr_{length}", np.nan)
        body_ratio = row.get(f"retest_source_body_ratio_{length}", np.nan)
        source_wick_pct = row.get(f"retest_source_wick_pct_{length}", np.nan)
        source_close_quality = row.get(f"retest_source_close_quality_{length}", np.nan)
        source_volume_ratio = row.get(f"retest_source_volume_ratio_{length}", np.nan)
        source_move_atr = row.get(f"retest_source_move_atr_{length}", np.nan)
    elif side == "BUY":
        base_ok = row["close"] > row.get(f"donchian_high_{length}", np.inf)
        move_atr = row.get(f"donchian_buy_move_atr_{length}", np.nan)
        avg_body = row.get("avg_body_20", np.nan)
        body_ratio = row.get("body", np.nan) / avg_body if not pd.isna(avg_body) and avg_body > 0 else np.nan
        source_wick_pct = row.get("max_wick_pct", np.nan)
        source_close_quality = row.get("close_location", np.nan)
        source_volume_ratio = row.get("volume_ratio", np.nan)
        source_move_atr = move_atr
    else:
        base_ok = row["close"] < row.get(f"donchian_low_{length}", -np.inf)
        move_atr = row.get(f"donchian_sell_move_atr_{length}", np.nan)
        avg_body = row.get("avg_body_20", np.nan)
        body_ratio = row.get("body", np.nan) / avg_body if not pd.isna(avg_body) and avg_body > 0 else np.nan
        source_wick_pct = row.get("max_wick_pct", np.nan)
        source_close_quality = 1 - row.get("close_location", np.nan)
        source_volume_ratio = row.get("volume_ratio", np.nan)
        source_move_atr = move_atr
    if not base_ok:
        return False

    if config.use_breakout_strength_filter:
        strong_body = not pd.isna(body_ratio) and body_ratio >= config.breakout_body_mult
        strong_move = not pd.isna(source_move_atr) and source_move_atr >= config.breakout_atr_mult
        volume_ok = config.volume_spike_mult <= 0 or source_volume_ratio >= config.volume_spike_mult
        if not (strong_body and strong_move and volume_ok):
            return False

    if config.use_fake_breakout_filter:
        if source_wick_pct > config.breakout_wick_max_pct:
            return False
        if pd.isna(source_close_quality) or source_close_quality < config.breakout_close_location_min:
            return False

    if config.use_zone_quality_score and breakout_quality(row, side, config) < config.zone_quality_min:
        return False
    return True


def generate_signals(row: pd.Series, config: Config) -> Optional[str]:
    required = ["m15_trend", "m15_adx", "m15_adx_rising", "m15_ema50_slope", "lorentzian_score", "andean_bull", "andean_bear", "ema20", "atr", "atr_pct"]
    if config.use_h1_bias:
        required.append("h1_bias")
    if any(pd.isna(row.get(col)) for col in required):
        return None
    if row["atr_pct"] < config.min_atr_pct:
        return None
    if row["m15_adx"] < config.m15_adx_min:
        return None
    if config.require_m15_adx_rising and not bool(row["m15_adx_rising"]):
        return None
    if config.high_vol_day_only and not bool(row.get("high_vol_day", False)):
        return None
    if config.use_atr_expansion and row.get(f"atr_expansion_{config.atr_ma_period}", 0) < config.atr_expansion_ratio_min:
        return None
    if config.use_bb_width and not bool(row.get(f"bb_width_expanding_{config.bb_length}", False)):
        return None
    if config.chop_max > 0 and row.get("chop", np.inf) >= config.chop_max:
        return None
    andean_buy = row["andean_bull"] > row["andean_bear"]
    andean_sell = row["andean_bear"] > row["andean_bull"]
    slope_buy = not config.require_ema50_slope or row["m15_ema50_slope"] > 0
    slope_sell = not config.require_ema50_slope or row["m15_ema50_slope"] < 0
    h1_buy = not config.use_h1_bias or row["h1_bias"] == 1
    h1_sell = not config.use_h1_bias or row["h1_bias"] == -1
    donchian_buy = donchian_entry_ok(row, "BUY", config)
    donchian_sell = donchian_entry_ok(row, "SELL", config)
    if config.tier_mode == "tier_b_only":
        donchian_buy = donchian_buy and high_quality_breakout(row, "BUY", config)
        donchian_sell = donchian_sell and high_quality_breakout(row, "SELL", config)
    if h1_buy and donchian_buy and row["m15_trend"] == 1 and slope_buy and row["lorentzian_score"] > config.lorentzian_threshold and andean_buy and row["close"] > row["ema20"]:
        return "BUY"
    if h1_sell and donchian_sell and row["m15_trend"] == -1 and slope_sell and row["lorentzian_score"] < -config.lorentzian_threshold and andean_sell and row["close"] < row["ema20"]:
        return "SELL"
    return None


def confirm_m1(m1: pd.DataFrame, signal_time: pd.Timestamp, side: str, config: Config):
    window = m1[(m1.index > signal_time) & (m1.index <= signal_time + pd.Timedelta(minutes=config.max_m1_confirm_candles))]
    for ts, row in window.iterrows():
        if side == "BUY" and row["close"] > row["ema20"] and row["andean_bull"] > row["andean_bear"]:
            entry_ts = ts + pd.Timedelta(minutes=config.entry_latency_m1_candles)
            if config.entry_latency_m1_candles > 0 and entry_ts in m1.index:
                return entry_ts, float(m1.loc[entry_ts, "close"])
            if config.entry_latency_m1_candles > 0:
                delayed = m1[m1.index > ts].head(config.entry_latency_m1_candles)
                if len(delayed) == config.entry_latency_m1_candles:
                    return delayed.index[-1], float(delayed.iloc[-1]["close"])
                return None, None
            return ts, float(row["close"])
        if side == "SELL" and row["close"] < row["ema20"] and row["andean_bear"] > row["andean_bull"]:
            entry_ts = ts + pd.Timedelta(minutes=config.entry_latency_m1_candles)
            if config.entry_latency_m1_candles > 0 and entry_ts in m1.index:
                return entry_ts, float(m1.loc[entry_ts, "close"])
            if config.entry_latency_m1_candles > 0:
                delayed = m1[m1.index > ts].head(config.entry_latency_m1_candles)
                if len(delayed) == config.entry_latency_m1_candles:
                    return delayed.index[-1], float(delayed.iloc[-1]["close"])
                return None, None
            return ts, float(row["close"])
    return None, None


def position_size(balance: float, entry: float, sl: float, config: Config, risk_per_trade: Optional[float] = None) -> float:
    risk = config.risk_per_trade if risk_per_trade is None else risk_per_trade
    risk_amount = balance * risk
    stop = abs(entry - sl)
    fee_per_unit_at_stop = config.fee_rate * (entry + sl)
    if stop <= 0:
        return 0.0
    return risk_amount / (stop + fee_per_unit_at_stop)


def close_pnl(position: Position, exit_price: float, config: Config, qty: Optional[float] = None):
    close_qty = position.qty if qty is None else qty
    gross = (exit_price - position.entry) * close_qty
    if position.side == "SELL":
        gross = -gross
    fee = (position.entry * close_qty + exit_price * close_qty) * config.fee_rate
    return gross - fee, fee


class CsvLogger:
    def __init__(self, config: Config, enabled=True):
        self.config = config
        self.enabled = enabled
        if not enabled:
            return
        for path in [config.trade_log, config.equity_log, config.monthly_log]:
            if Path(path).exists():
                Path(path).unlink()

    def append(self, path: str, row: dict):
        if not self.enabled:
            return
        exists = Path(path).exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not exists:
                writer.writeheader()
            writer.writerow(row)


def in_session(ts: pd.Timestamp, session_name: str) -> bool:
    if session_name == "all":
        return True
    minutes = ts.hour * 60 + ts.minute
    sessions = {
        "14_20": (14 * 60, 20 * 60),
        "20_24": (20 * 60, 24 * 60),
        "00_03": (0, 3 * 60),
    }
    start, end = sessions[session_name]
    return start <= minutes < end


def _exit_position(position: Position, row: pd.Series, config: Config):
    if position.side == "BUY":
        if row["low"] <= position.sl:
            return "SL", float(position.sl), position.qty, True
        if row["high"] >= position.tp:
            return "TP", float(position.tp), position.qty, True
        if config.exit_mode == "fixed":
            return None, None, None, False
        if config.exit_mode == "partial_1r" and not position.partial_taken and row["high"] >= position.entry + position.one_r:
            position.partial_taken = True
            return "PARTIAL_1R", float(position.entry + position.one_r), position.qty * 0.5, False
        if config.exit_mode == "be_1_2" and row["high"] >= position.entry + 1.2 * position.one_r:
            position.sl = max(position.sl, position.entry)
            position.break_even_active = True
        elif config.exit_mode == "be_1_lock_1_5" and row["high"] >= position.entry + 1.5 * position.one_r:
            position.sl = max(position.sl, position.entry + 0.5 * position.one_r)
            position.partial_lock_active = True
        elif config.exit_mode in {"be_1_lock_1_5", "be_1_no_lock", "partial_1r"} and row["high"] >= position.entry + position.one_r:
            position.sl = max(position.sl, position.entry)
            position.break_even_active = True
    else:
        if row["high"] >= position.sl:
            return "SL", float(position.sl), position.qty, True
        if row["low"] <= position.tp:
            return "TP", float(position.tp), position.qty, True
        if config.exit_mode == "fixed":
            return None, None, None, False
        if config.exit_mode == "partial_1r" and not position.partial_taken and row["low"] <= position.entry - position.one_r:
            position.partial_taken = True
            return "PARTIAL_1R", float(position.entry - position.one_r), position.qty * 0.5, False
        if config.exit_mode == "be_1_2" and row["low"] <= position.entry - 1.2 * position.one_r:
            position.sl = min(position.sl, position.entry)
            position.break_even_active = True
        elif config.exit_mode == "be_1_lock_1_5" and row["low"] <= position.entry - 1.5 * position.one_r:
            position.sl = min(position.sl, position.entry - 0.5 * position.one_r)
            position.partial_lock_active = True
        elif config.exit_mode in {"be_1_lock_1_5", "be_1_no_lock", "partial_1r"} and row["low"] <= position.entry - position.one_r:
            position.sl = min(position.sl, position.entry)
            position.break_even_active = True
    return None, None, None, False


def simulate_exit(position: Position, m1_cache: dict, config: Config):
    rows = []
    index = m1_cache["index"]
    high = m1_cache["high"]
    low = m1_cache["low"]
    close = m1_cache["close"]
    trail_prev_high = m1_cache["trail_prev_high"]
    trail_prev_low = m1_cache["trail_prev_low"]
    trail_ema20 = m1_cache["trail_ema20"]
    hold_until = position.entry_time + pd.Timedelta(minutes=config.min_hold_minutes)
    start = index.searchsorted(position.entry_time, side="right")
    for i in range(start, len(index)):
        ts = index[i]
        h = high[i]
        l = low[i]
        c = close[i]
        reason = None
        exit_price = None
        close_qty = None
        should_close = False
        can_manage = ts >= hold_until
        if position.side == "BUY":
            if l <= position.sl:
                reason, exit_price, close_qty, should_close = "SL", position.sl, position.qty, True
            elif config.exit_mode.startswith("fixed") and h >= position.tp:
                reason, exit_price, close_qty, should_close = "TP", position.tp, position.qty, True
            elif can_manage and config.exit_mode != "fixed":
                if config.exit_mode == "partial_1r" and not position.partial_taken and h >= position.entry + position.one_r:
                    position.partial_taken = True
                    reason, exit_price, close_qty, should_close = "PARTIAL_1R", position.entry + position.one_r, position.qty * 0.5, False
                elif config.exit_mode == "partial_trailing" and not position.partial_taken and h >= position.entry + position.one_r:
                    position.partial_taken = True
                    reason, exit_price, close_qty, should_close = "PARTIAL_1R", position.entry + position.one_r, position.qty * config.partial_pct, False
                elif config.exit_mode == "be_1_2" and h >= position.entry + 1.2 * position.one_r:
                    position.sl = max(position.sl, position.entry)
                elif config.exit_mode == "be_1_lock_1_5" and h >= position.entry + 1.5 * position.one_r:
                    position.sl = max(position.sl, position.entry + 0.5 * position.one_r)
                elif config.exit_mode in {"be_1_lock_1_5", "be_1_no_lock", "partial_1r"} and h >= position.entry + position.one_r:
                    position.sl = max(position.sl, position.entry)
                if config.exit_mode in {"trailing_only", "partial_trailing"} and i > 0:
                    trail = trail_prev_low[i] if config.trail_type == "prev_candle" else trail_ema20[i]
                    if not np.isnan(trail) and trail < c:
                        position.sl = max(position.sl, trail)
        else:
            if h >= position.sl:
                reason, exit_price, close_qty, should_close = "SL", position.sl, position.qty, True
            elif config.exit_mode.startswith("fixed") and l <= position.tp:
                reason, exit_price, close_qty, should_close = "TP", position.tp, position.qty, True
            elif can_manage and config.exit_mode != "fixed":
                if config.exit_mode == "partial_1r" and not position.partial_taken and l <= position.entry - position.one_r:
                    position.partial_taken = True
                    reason, exit_price, close_qty, should_close = "PARTIAL_1R", position.entry - position.one_r, position.qty * 0.5, False
                elif config.exit_mode == "partial_trailing" and not position.partial_taken and l <= position.entry - position.one_r:
                    position.partial_taken = True
                    reason, exit_price, close_qty, should_close = "PARTIAL_1R", position.entry - position.one_r, position.qty * config.partial_pct, False
                elif config.exit_mode == "be_1_2" and l <= position.entry - 1.2 * position.one_r:
                    position.sl = min(position.sl, position.entry)
                elif config.exit_mode == "be_1_lock_1_5" and l <= position.entry - 1.5 * position.one_r:
                    position.sl = min(position.sl, position.entry - 0.5 * position.one_r)
                elif config.exit_mode in {"be_1_lock_1_5", "be_1_no_lock", "partial_1r"} and l <= position.entry - position.one_r:
                    position.sl = min(position.sl, position.entry)
                if config.exit_mode in {"trailing_only", "partial_trailing"} and i > 0:
                    trail = trail_prev_high[i] if config.trail_type == "prev_candle" else trail_ema20[i]
                    if not np.isnan(trail) and trail > c:
                        position.sl = min(position.sl, trail)
        if not reason:
            continue
        pnl, fee = close_pnl(position, exit_price, config, close_qty)
        rows.append(
            {
                "entry_time": position.entry_time,
                "exit_time": ts,
                "side": position.side,
                "entry": position.entry,
                "sl": position.initial_sl,
                "tp": position.tp,
                "exit": exit_price,
                "qty": close_qty,
                "tier": position.tier,
                "pnl": pnl,
                "fee": fee,
                "result": "WIN" if pnl > 0 else "LOSS",
                "reason": reason,
            }
        )
        if should_close:
            return rows, ts
        position.qty -= close_qty
        if position.qty <= 0:
            return rows, ts
    if position.qty > 0:
        ts = index[-1]
        exit_price = float(close[-1])
        pnl, fee = close_pnl(position, exit_price, config)
        rows.append(
            {
                "entry_time": position.entry_time,
                "exit_time": ts,
                "side": position.side,
                "entry": position.entry,
                "sl": position.initial_sl,
                "tp": position.tp,
                "exit": exit_price,
                "qty": position.qty,
                "tier": position.tier,
                "pnl": pnl,
                "fee": fee,
                "result": "WIN" if pnl > 0 else "LOSS",
                "reason": "EOD",
            }
        )
        return rows, ts
    return rows, position.entry_time


def run_backtest(m1: pd.DataFrame, m5: pd.DataFrame, config: Config, write_logs=True):
    logger = CsvLogger(config, enabled=write_logs)
    balance = config.initial_balance
    peak = balance
    max_dd = 0.0
    wins = 0
    losses = 0
    gross_profit = 0.0
    gross_loss = 0.0
    daily_trades = {}
    daily_losses = {}
    trade_rows = []
    equity_rows = []
    next_m5_allowed_time = m5.index[0]
    unavailable_until = m5.index[0]
    m1_cache = {
        "index": m1.index,
        "high": m1["high"].to_numpy(dtype=float),
        "low": m1["low"].to_numpy(dtype=float),
        "close": m1["close"].to_numpy(dtype=float),
        "trail_prev_high": m1["trail_prev_high"].to_numpy(dtype=float),
        "trail_prev_low": m1["trail_prev_low"].to_numpy(dtype=float),
        "trail_ema20": m1["trail_ema20"].to_numpy(dtype=float),
    }

    for ts, row in m5.iterrows():
        peak = max(peak, balance)
        dd = (balance - peak) / peak
        max_dd = min(max_dd, dd)
        equity = {"time": ts, "balance": balance, "drawdown_pct": dd * 100}
        equity_rows.append(equity)
        logger.append(config.equity_log, equity)

        if ts <= unavailable_until or ts < next_m5_allowed_time:
            continue
        day = ts.date()
        if daily_trades.get(day, 0) >= config.max_trades_per_day or daily_losses.get(day, 0) >= config.max_losses_per_day:
            continue
        if not in_session(ts, config.session_name):
            continue
        side = generate_signals(row, config)
        if side is None:
            continue
        entry_time, entry_price = confirm_m1(m1, ts, side, config)
        if entry_time is None:
            continue
        spread = float(row.get("spread_proxy", row["atr"] * config.spread_atr_mult))
        if config.use_spread_filter and spread > float(row["atr"] * config.max_spread_atr_mult):
            continue
        slip = float(row["atr"] * config.slippage_atr_mult + spread * config.slippage_spread_mult)
        entry = entry_price + slip if side == "BUY" else entry_price - slip
        atr_sl = float(row["atr"])
        if config.exit_mode in {"trailing_only", "partial_trailing"}:
            if side == "BUY":
                sl = entry - atr_sl
                risk = atr_sl
                tp = np.inf
            else:
                sl = entry + atr_sl
                risk = atr_sl
                tp = -np.inf
        elif side == "BUY":
            swing_sl = float(row["swing_low"]) if not pd.isna(row["swing_low"]) else entry - atr_sl
            sl = min(entry - atr_sl, swing_sl)
            risk = abs(entry - sl)
            tp = entry + risk * config.rr
        else:
            swing_sl = float(row["swing_high"]) if not pd.isna(row["swing_high"]) else entry + atr_sl
            sl = max(entry + atr_sl, swing_sl)
            risk = abs(entry - sl)
            tp = entry - risk * config.rr
        tier = "A"
        risk_for_trade = config.risk_per_trade
        if config.tier_mode in {"hybrid", "tier_b_only"} and high_quality_breakout(row, side, config):
            tier = "B"
            risk_for_trade = config.tier_b_risk
        qty = position_size(balance, entry, sl, config, risk_for_trade)
        if qty <= 0:
            continue
        position = Position(side, entry_time, entry, sl, sl, tp, qty, tier=tier)
        daily_trades[day] = daily_trades.get(day, 0) + 1
        next_m5_allowed_time = ts + pd.Timedelta(minutes=5)
        exits, exit_time = simulate_exit(position, m1_cache, config)
        unavailable_until = exit_time
        for trade in exits:
            balance += trade["pnl"]
            trade["balance"] = balance
            trade["spread_proxy"] = spread
            trade["slippage_price"] = slip
            trade["slippage_cost"] = slip * trade["qty"]
            trade["latency_m1_candles"] = config.entry_latency_m1_candles
            wins += int(trade["pnl"] > 0)
            losses += int(trade["pnl"] <= 0)
            gross_profit += max(trade["pnl"], 0)
            gross_loss += abs(min(trade["pnl"], 0))
            if trade["pnl"] <= 0:
                daily_losses[day] = daily_losses.get(day, 0) + 1
            trade_rows.append(trade)
            logger.append(config.trade_log, trade)

    trades = wins + losses
    first_day = m5.index[0].date()
    last_day = m5.index[-1].date()
    days = config.total_days or max((last_day - first_day).days + 1, 1)
    avg_win = gross_profit / wins if wins else 0.0
    avg_loss = -gross_loss / losses if losses else 0.0
    metrics = {
        "ema": f"{config.ema_fast}/{config.ema_slow}",
        "lorentzian_k": config.lorentzian_k,
        "lorentzian_horizon": config.lorentzian_horizon,
        "andean_length": config.andean_length,
        "min_atr_pct": config.min_atr_pct,
        "risk": config.risk_per_trade,
        "rr": config.rr,
        "exit_mode": config.exit_mode,
        "m15_adx_min": config.m15_adx_min,
        "require_m15_adx_rising": config.require_m15_adx_rising,
        "require_ema50_slope": config.require_ema50_slope,
        "use_h1_bias": config.use_h1_bias,
        "session": config.session_name,
        "min_hold_minutes": config.min_hold_minutes,
        "partial_pct": config.partial_pct,
        "trail_type": config.trail_type,
        "regime": config.regime_name,
        "use_atr_expansion": config.use_atr_expansion,
        "atr_ma_period": config.atr_ma_period,
        "atr_expansion_ratio_min": config.atr_expansion_ratio_min,
        "use_bb_width": config.use_bb_width,
        "bb_length": config.bb_length,
        "use_donchian": config.use_donchian,
        "donchian_n": config.donchian_n,
        "use_breakout_strength_filter": config.use_breakout_strength_filter,
        "use_fake_breakout_filter": config.use_fake_breakout_filter,
        "use_retest_entry": config.use_retest_entry,
        "use_zone_quality_score": config.use_zone_quality_score,
        "tier_mode": config.tier_mode,
        "tier_b_risk": config.tier_b_risk,
        "spread_atr_mult": config.spread_atr_mult,
        "slippage_spread_mult": config.slippage_spread_mult,
        "max_spread_atr_mult": config.max_spread_atr_mult,
        "use_spread_filter": config.use_spread_filter,
        "entry_latency_m1_candles": config.entry_latency_m1_candles,
        "breakout_body_mult": config.breakout_body_mult,
        "breakout_atr_mult": config.breakout_atr_mult,
        "volume_spike_mult": config.volume_spike_mult,
        "zone_quality_min": config.zone_quality_min,
        "chop_max": config.chop_max,
        "high_vol_day_only": config.high_vol_day_only,
        "profit_pct": (balance - config.initial_balance) / config.initial_balance * 100,
        "final_balance": balance,
        "total_trades": trades,
        "trades_per_day": trades / days,
        "winrate": wins / trades * 100 if trades else 0.0,
        "profit_factor": gross_profit / gross_loss if gross_loss else 0.0,
        "max_drawdown_pct": max_dd * 100,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "expectancy": (gross_profit - gross_loss) / trades if trades else 0.0,
    }
    trade_df = pd.DataFrame(trade_rows)
    if not trade_df.empty:
        metrics["avg_slippage_cost"] = float(trade_df["slippage_cost"].mean()) if "slippage_cost" in trade_df.columns else 0.0
        metrics["total_slippage_cost"] = float(trade_df["slippage_cost"].sum()) if "slippage_cost" in trade_df.columns else 0.0
        metrics["slippage_cost_per_trade"] = metrics["avg_slippage_cost"]
    else:
        metrics["avg_slippage_cost"] = 0.0
        metrics["total_slippage_cost"] = 0.0
        metrics["slippage_cost_per_trade"] = 0.0
    for tier in ["A", "B"]:
        tier_df = trade_df[trade_df["tier"] == tier] if not trade_df.empty and "tier" in trade_df.columns else pd.DataFrame()
        tier_wins = int((tier_df["pnl"] > 0).sum()) if not tier_df.empty else 0
        tier_losses = int((tier_df["pnl"] <= 0).sum()) if not tier_df.empty else 0
        tier_profit = float(tier_df.loc[tier_df["pnl"] > 0, "pnl"].sum()) if not tier_df.empty else 0.0
        tier_loss = float(-tier_df.loc[tier_df["pnl"] <= 0, "pnl"].sum()) if not tier_df.empty else 0.0
        tier_trades = tier_wins + tier_losses
        metrics[f"tier_{tier.lower()}_trades"] = tier_trades
        metrics[f"tier_{tier.lower()}_winrate"] = tier_wins / tier_trades * 100 if tier_trades else 0.0
        metrics[f"tier_{tier.lower()}_profit_factor"] = tier_profit / tier_loss if tier_loss else 0.0
        metrics[f"tier_{tier.lower()}_expectancy"] = (tier_profit - tier_loss) / tier_trades if tier_trades else 0.0
    monthly = monthly_stats(pd.DataFrame(trade_rows), pd.DataFrame(equity_rows))
    metrics["negative_months"] = int((monthly.get("monthly_profit", pd.Series(dtype=float)) < 0).sum()) if not monthly.empty else 0
    if write_logs:
        monthly.to_csv(config.monthly_log, index=False)
    return metrics, monthly


def monthly_stats(trades: pd.DataFrame, equity: pd.DataFrame) -> pd.DataFrame:
    if equity.empty:
        return pd.DataFrame()
    eq = equity.copy()
    eq["month"] = pd.to_datetime(eq["time"]).dt.to_period("M").astype(str)
    eq["month_peak"] = eq.groupby("month")["balance"].cummax()
    eq["monthly_dd"] = (eq["balance"] - eq["month_peak"]) / eq["month_peak"] * 100
    monthly = eq.groupby("month").agg(
        start_balance=("balance", "first"),
        end_balance=("balance", "last"),
        monthly_max_drawdown=("monthly_dd", "min"),
    ).reset_index()
    monthly["monthly_profit"] = monthly["end_balance"] - monthly["start_balance"]
    monthly["monthly_profit_pct"] = monthly["monthly_profit"] / monthly["start_balance"] * 100
    if trades.empty:
        monthly["monthly_trades"] = 0
        monthly["monthly_winrate"] = 0.0
        return monthly
    tr = trades.copy()
    tr["month"] = pd.to_datetime(tr["exit_time"]).dt.to_period("M").astype(str)
    stats = tr.groupby("month").agg(
        monthly_trades=("pnl", "count"),
        monthly_winrate=("pnl", lambda x: (x > 0).mean() * 100),
    ).reset_index()
    monthly = monthly.merge(stats, on="month", how="left")
    monthly["monthly_trades"] = monthly["monthly_trades"].fillna(0).astype(int)
    monthly["monthly_winrate"] = monthly["monthly_winrate"].fillna(0.0)
    return monthly


def save_results(rows: list[dict], config: Config):
    df = pd.DataFrame(rows)
    df["passes_robust_filter"] = (
        (df["expectancy"] > 0)
        & (df["profit_factor"] > 1.2)
        & (df["max_drawdown_pct"] > -10)
        & (df["trades_per_day"] >= 0.3)
        & (df["negative_months"] <= 4)
    )
    df = df.sort_values(
        ["passes_robust_filter", "profit_factor", "expectancy", "max_drawdown_pct", "trades_per_day", "negative_months"],
        ascending=[False, False, False, False, False, True],
    )
    df.to_csv(config.sweep_log, index=False)
    return df


def sweep(m1_raw: pd.DataFrame, m5_raw: pd.DataFrame, m15_raw: pd.DataFrame, base: Config):
    rows = []
    indicator_cache = {}
    lorentzian_cache = {}
    for ema_pair in [(20, 50), (50, 200)]:
        for k in [8, 16]:
            for horizon in [4, 8, 12]:
                for andean_length in [20, 50]:
                    for atr_threshold in [0.0005, 0.001, 0.002]:
                        for risk in [0.005, 0.01]:
                            cfg = Config(**{**base.__dict__})
                            cfg.ema_fast, cfg.ema_slow = ema_pair
                            cfg.lorentzian_k = k
                            cfg.lorentzian_horizon = horizon
                            cfg.andean_length = andean_length
                            cfg.min_atr_pct = atr_threshold
                            cfg.risk_per_trade = risk
                            indicator_key = (ema_pair, andean_length)
                            if indicator_key not in indicator_cache:
                                m1_cached, m5_cached, m15_cached = calculate_indicators(m1_raw, m5_raw, m15_raw, cfg)
                                indicator_cache[indicator_key] = (m1_cached, align_timeframes(m5_cached, m15_cached))
                            m1, m5_base = indicator_cache[indicator_key]
                            m5 = m5_base.copy()
                            lorentzian_key = (k, horizon)
                            if lorentzian_key not in lorentzian_cache:
                                m5_lorentzian = calculate_lorentzian(m5_base, cfg)
                                lorentzian_cache[lorentzian_key] = m5_lorentzian["lorentzian_score"].copy()
                            m5["lorentzian_score"] = lorentzian_cache[lorentzian_key].reindex(m5.index)
                            metrics, _ = run_backtest(m1, m5, cfg, write_logs=False)
                            rows.append(metrics)
    return save_results(rows, base)


def robustness_sweep(m1_raw: pd.DataFrame, m5_raw: pd.DataFrame, m15_raw: pd.DataFrame, base: Config):
    rows = []
    cfg = Config(**{**base.__dict__})
    cfg.ema_fast = 50
    cfg.ema_slow = 200
    cfg.lorentzian_k = 8
    cfg.lorentzian_horizon = 8
    cfg.andean_length = 50
    cfg.min_atr_pct = 0.002
    cfg.risk_per_trade = 0.005

    m1, m5_base, m15 = calculate_indicators(m1_raw, m5_raw, m15_raw, cfg)
    m5_base = align_timeframes(m5_base, m15)
    m5_base = calculate_lorentzian(m5_base, cfg)
    cfg.total_days = max((m5_base.index[-1].date() - m5_base.index[0].date()).days + 1, 1)

    exit_modes = ["fixed", "be_1_2", "be_1_no_lock", "partial_1r"]
    adx_filters = [18, 20, 25]
    sessions = ["14_20", "20_24", "00_03", "all"]
    rr_values = [1.5, 2.0, 2.5]
    for exit_mode in exit_modes:
        for adx_min in adx_filters:
            for session_name in sessions:
                for rr in rr_values:
                    for require_slope in [False, True]:
                        test_cfg = Config(**{**cfg.__dict__})
                        test_cfg.exit_mode = exit_mode
                        test_cfg.m15_adx_min = adx_min
                        test_cfg.session_name = session_name
                        test_cfg.rr = rr
                        test_cfg.require_ema50_slope = require_slope
                        if test_cfg.require_ema50_slope:
                            buy_slope_ok = m5_base["m15_ema50_slope"] > 0
                            sell_slope_ok = m5_base["m15_ema50_slope"] < 0
                        else:
                            buy_slope_ok = pd.Series(True, index=m5_base.index)
                            sell_slope_ok = pd.Series(True, index=m5_base.index)
                        m5_test = m5_base[
                            (m5_base["atr_pct"] >= test_cfg.min_atr_pct)
                            & (m5_base["m15_adx"] >= test_cfg.m15_adx_min)
                            & (m5_base.index.map(lambda ts: in_session(ts, test_cfg.session_name)))
                            & (
                                (
                                    (m5_base["m15_trend"] == 1)
                                    & (m5_base["lorentzian_score"] > test_cfg.lorentzian_threshold)
                                    & (m5_base["andean_bull"] > m5_base["andean_bear"])
                                    & (m5_base["close"] > m5_base["ema20"])
                                    & buy_slope_ok
                                )
                                | (
                                    (m5_base["m15_trend"] == -1)
                                    & (m5_base["lorentzian_score"] < -test_cfg.lorentzian_threshold)
                                    & (m5_base["andean_bear"] > m5_base["andean_bull"])
                                    & (m5_base["close"] < m5_base["ema20"])
                                    & sell_slope_ok
                                )
                            )
                        ]
                        if m5_test.empty:
                            rows.append({**run_backtest(m1, m5_base.iloc[:1], test_cfg, write_logs=False)[0], "signal_candidates": 0})
                            continue
                        metrics, _ = run_backtest(m1, m5_test, test_cfg, write_logs=False)
                        metrics["signal_candidates"] = len(m5_test)
                        rows.append(metrics)

    base.sweep_log = str(REPORT_DIR / "multi_tf_lorentzian_andean_robustness_sweep.csv")
    return save_results(rows, base)


def signal_candidates(m5: pd.DataFrame, config: Config) -> pd.DataFrame:
    if config.require_ema50_slope:
        buy_slope_ok = m5["m15_ema50_slope"] > 0
        sell_slope_ok = m5["m15_ema50_slope"] < 0
    else:
        buy_slope_ok = pd.Series(True, index=m5.index)
        sell_slope_ok = pd.Series(True, index=m5.index)
    if config.use_h1_bias:
        buy_h1_ok = m5["h1_bias"] == 1
        sell_h1_ok = m5["h1_bias"] == -1
    else:
        buy_h1_ok = pd.Series(True, index=m5.index)
        sell_h1_ok = pd.Series(True, index=m5.index)
    adx_rising_ok = (m5["m15_adx_rising"].astype(bool)) if config.require_m15_adx_rising else pd.Series(True, index=m5.index)
    atr_expansion_ok = (
        m5[f"atr_expansion_{config.atr_ma_period}"] >= config.atr_expansion_ratio_min
        if config.use_atr_expansion
        else pd.Series(True, index=m5.index)
    )
    bb_width_ok = (
        m5[f"bb_width_expanding_{config.bb_length}"].astype(bool)
        if config.use_bb_width
        else pd.Series(True, index=m5.index)
    )
    chop_ok = m5["chop"] < config.chop_max if config.chop_max > 0 else pd.Series(True, index=m5.index)
    high_vol_ok = m5["high_vol_day"].astype(bool) if config.high_vol_day_only else pd.Series(True, index=m5.index)
    if config.use_donchian:
        buy_donchian_ok = m5.apply(lambda row: donchian_entry_ok(row, "BUY", config), axis=1)
        sell_donchian_ok = m5.apply(lambda row: donchian_entry_ok(row, "SELL", config), axis=1)
        if config.tier_mode == "tier_b_only":
            buy_donchian_ok = buy_donchian_ok & m5.apply(lambda row: high_quality_breakout(row, "BUY", config), axis=1)
            sell_donchian_ok = sell_donchian_ok & m5.apply(lambda row: high_quality_breakout(row, "SELL", config), axis=1)
    else:
        buy_donchian_ok = pd.Series(True, index=m5.index)
        sell_donchian_ok = pd.Series(True, index=m5.index)
    return m5[
        (m5["atr_pct"] >= config.min_atr_pct)
        & (m5["m15_adx"] >= config.m15_adx_min)
        & adx_rising_ok
        & atr_expansion_ok
        & bb_width_ok
        & chop_ok
        & high_vol_ok
        & (m5.index.map(lambda ts: in_session(ts, config.session_name)))
        & (
            (
                buy_h1_ok
                & buy_donchian_ok
                & (m5["m15_trend"] == 1)
                & (m5["lorentzian_score"] > config.lorentzian_threshold)
                & (m5["andean_bull"] > m5["andean_bear"])
                & (m5["close"] > m5["ema20"])
                & buy_slope_ok
            )
            | (
                sell_h1_ok
                & sell_donchian_ok
                & (m5["m15_trend"] == -1)
                & (m5["lorentzian_score"] < -config.lorentzian_threshold)
                & (m5["andean_bear"] > m5["andean_bull"])
                & (m5["close"] < m5["ema20"])
                & sell_slope_ok
            )
        )
    ]


def exit_optimization_sweep(m1_raw: pd.DataFrame, m5_raw: pd.DataFrame, m15_raw: pd.DataFrame, h1_raw: pd.DataFrame, base: Config):
    rows = []
    cfg = Config(**{**base.__dict__})
    cfg.ema_fast = 50
    cfg.ema_slow = 200
    cfg.lorentzian_k = 8
    cfg.lorentzian_horizon = 8
    cfg.andean_length = 50
    cfg.min_atr_pct = 0.002
    cfg.risk_per_trade = 0.005
    cfg.m15_adx_min = 20
    cfg.require_m15_adx_rising = True

    m1, m5_base, m15 = calculate_indicators(m1_raw, m5_raw, m15_raw, cfg)
    h1 = calculate_h1_indicators(h1_raw)
    m5_base = align_timeframes(m5_base, m15)
    m5_base = align_h1_timeframe(m5_base, h1)
    m5_base = calculate_lorentzian(m5_base, cfg)
    cfg.total_days = max((m5_base.index[-1].date() - m5_base.index[0].date()).days + 1, 1)

    tests = []
    for rr in [1.5, 2.0]:
        tests.append({"exit_mode": "fixed", "rr": rr, "min_hold_minutes": 0, "partial_pct": 0.0, "trail_type": "none"})
    for trail_type in ["prev_candle", "ema20"]:
        for hold in [15, 25]:
            tests.append({"exit_mode": "trailing_only", "rr": 0.0, "min_hold_minutes": hold, "partial_pct": 0.0, "trail_type": trail_type})
            for partial_pct in [0.3, 0.5]:
                tests.append({"exit_mode": "partial_trailing", "rr": 0.0, "min_hold_minutes": hold, "partial_pct": partial_pct, "trail_type": trail_type})

    for session_name in ["14_20", "all"]:
        for use_h1_bias in [False, True]:
            for test in tests:
                test_cfg = Config(**{**cfg.__dict__})
                test_cfg.session_name = session_name
                test_cfg.use_h1_bias = use_h1_bias
                for key, value in test.items():
                    setattr(test_cfg, key, value)
                m5_test = signal_candidates(m5_base, test_cfg)
                if m5_test.empty:
                    metrics, _ = run_backtest(m1, m5_base.iloc[:1], test_cfg, write_logs=False)
                    metrics["signal_candidates"] = 0
                    rows.append(metrics)
                    continue
                metrics, _ = run_backtest(m1, m5_test, test_cfg, write_logs=False)
                metrics["signal_candidates"] = len(m5_test)
                rows.append(metrics)

    base.sweep_log = str(REPORT_DIR / "multi_tf_lorentzian_andean_exit_sweep.csv")
    return save_results(rows, base)


def regime_filter_sweep(m1_raw: pd.DataFrame, m5_raw: pd.DataFrame, m15_raw: pd.DataFrame, h1_raw: pd.DataFrame, base: Config):
    rows = []
    cfg = Config(**{**base.__dict__})
    cfg.ema_fast = 50
    cfg.ema_slow = 200
    cfg.lorentzian_k = 8
    cfg.lorentzian_horizon = 8
    cfg.andean_length = 50
    cfg.min_atr_pct = 0.002
    cfg.risk_per_trade = 0.005
    cfg.m15_adx_min = 20
    cfg.require_m15_adx_rising = True

    m1, m5_base, m15 = calculate_indicators(m1_raw, m5_raw, m15_raw, cfg)
    h1 = calculate_h1_indicators(h1_raw)
    m5_base = align_timeframes(m5_base, m15)
    m5_base = align_h1_timeframe(m5_base, h1)
    m5_base = calculate_lorentzian(m5_base, cfg)
    cfg.total_days = max((m5_base.index[-1].date() - m5_base.index[0].date()).days + 1, 1)

    regimes = [{"regime_name": "baseline"}]
    for period in [50, 100]:
        for ratio in [1.1, 1.2, 1.3]:
            regimes.append(
                {
                    "regime_name": f"atr_expansion_{period}_{ratio}",
                    "use_atr_expansion": True,
                    "atr_ma_period": period,
                    "atr_expansion_ratio_min": ratio,
                }
            )
    for length in [20, 50]:
        regimes.append({"regime_name": f"bb_width_{length}", "use_bb_width": True, "bb_length": length})
    for length in [20, 50, 100]:
        regimes.append({"regime_name": f"donchian_{length}", "use_donchian": True, "donchian_n": length})
    for threshold in [45, 50, 55]:
        regimes.append({"regime_name": f"chop_lt_{threshold}", "chop_max": threshold})
    regimes.append({"regime_name": "high_vol_day", "high_vol_day_only": True})

    combo_templates = [
        {
            "regime_name": "atr_50_1.1_plus_bb20",
            "use_atr_expansion": True,
            "atr_ma_period": 50,
            "atr_expansion_ratio_min": 1.1,
            "use_bb_width": True,
            "bb_length": 20,
        },
        {
            "regime_name": "atr_50_1.1_plus_donchian20",
            "use_atr_expansion": True,
            "atr_ma_period": 50,
            "atr_expansion_ratio_min": 1.1,
            "use_donchian": True,
            "donchian_n": 20,
        },
        {
            "regime_name": "atr_50_1.2_plus_chop50",
            "use_atr_expansion": True,
            "atr_ma_period": 50,
            "atr_expansion_ratio_min": 1.2,
            "chop_max": 50,
        },
        {
            "regime_name": "bb20_plus_chop50",
            "use_bb_width": True,
            "bb_length": 20,
            "chop_max": 50,
        },
        {
            "regime_name": "donchian20_plus_chop50",
            "use_donchian": True,
            "donchian_n": 20,
            "chop_max": 50,
        },
        {
            "regime_name": "atr_bb_donchian",
            "use_atr_expansion": True,
            "atr_ma_period": 50,
            "atr_expansion_ratio_min": 1.1,
            "use_bb_width": True,
            "bb_length": 20,
            "use_donchian": True,
            "donchian_n": 20,
        },
    ]
    regimes.extend(combo_templates)

    exit_tests = [
        {"exit_mode": "fixed", "rr": 1.5},
        {"exit_mode": "fixed", "rr": 2.0},
        {"exit_mode": "be_1_2", "rr": 1.5},
        {"exit_mode": "be_1_2", "rr": 2.0},
    ]
    for session_name in ["all", "14_20"]:
        for high_vol_only in [False, True]:
            for regime in regimes:
                for exit_test in exit_tests:
                    test_cfg = Config(**{**cfg.__dict__})
                    test_cfg.session_name = session_name
                    test_cfg.high_vol_day_only = high_vol_only
                    for key, value in regime.items():
                        setattr(test_cfg, key, value)
                    if high_vol_only:
                        test_cfg.regime_name = f"{test_cfg.regime_name}_high_vol"
                    for key, value in exit_test.items():
                        setattr(test_cfg, key, value)
                    m5_test = signal_candidates(m5_base, test_cfg)
                    if m5_test.empty:
                        metrics, _ = run_backtest(m1, m5_base.iloc[:1], test_cfg, write_logs=False)
                        metrics["signal_candidates"] = 0
                    else:
                        metrics, _ = run_backtest(m1, m5_test, test_cfg, write_logs=False)
                        metrics["signal_candidates"] = len(m5_test)
                    rows.append(metrics)

    base.sweep_log = str(REPORT_DIR / "multi_tf_lorentzian_andean_regime_sweep.csv")
    return save_results(rows, base)


def supply_demand_donchian_sweep(m1_raw: pd.DataFrame, m5_raw: pd.DataFrame, m15_raw: pd.DataFrame, h1_raw: pd.DataFrame, base: Config):
    rows = []
    cfg = Config(**{**base.__dict__})
    cfg.ema_fast = 50
    cfg.ema_slow = 200
    cfg.lorentzian_k = 8
    cfg.lorentzian_horizon = 8
    cfg.andean_length = 50
    cfg.min_atr_pct = 0.002
    cfg.risk_per_trade = 0.005
    cfg.rr = 2.0
    cfg.exit_mode = "fixed"
    cfg.m15_adx_min = 20
    cfg.require_m15_adx_rising = True
    cfg.use_donchian = True
    cfg.donchian_n = 20

    m1, m5_base, m15 = calculate_indicators(m1_raw, m5_raw, m15_raw, cfg)
    h1 = calculate_h1_indicators(h1_raw)
    m5_base = align_timeframes(m5_base, m15)
    m5_base = align_h1_timeframe(m5_base, h1)
    m5_base = calculate_lorentzian(m5_base, cfg)
    cfg.total_days = max((m5_base.index[-1].date() - m5_base.index[0].date()).days + 1, 1)

    variants = [
        {
            "regime_name": "baseline_donchian",
            "use_breakout_strength_filter": False,
            "use_fake_breakout_filter": False,
            "use_retest_entry": False,
            "use_zone_quality_score": False,
        },
        {
            "regime_name": "donchian_strength",
            "use_breakout_strength_filter": True,
            "use_fake_breakout_filter": True,
            "use_retest_entry": False,
            "use_zone_quality_score": True,
        },
        {
            "regime_name": "donchian_retest",
            "use_breakout_strength_filter": False,
            "use_fake_breakout_filter": True,
            "use_retest_entry": True,
            "use_zone_quality_score": False,
        },
        {
            "regime_name": "donchian_strength_retest",
            "use_breakout_strength_filter": True,
            "use_fake_breakout_filter": True,
            "use_retest_entry": True,
            "use_zone_quality_score": True,
        },
    ]

    for body_mult in [1.3, 1.5]:
        for atr_mult in [0.4, 0.5]:
            for quality_min in [3.0, 4.0]:
                for variant in variants:
                    test_cfg = Config(**{**cfg.__dict__})
                    test_cfg.breakout_body_mult = body_mult
                    test_cfg.breakout_atr_mult = atr_mult
                    test_cfg.zone_quality_min = quality_min
                    for key, value in variant.items():
                        setattr(test_cfg, key, value)
                    m5_test = signal_candidates(m5_base, test_cfg)
                    if m5_test.empty:
                        metrics, _ = run_backtest(m1, m5_base.iloc[:1], test_cfg, write_logs=False)
                        metrics["signal_candidates"] = 0
                    else:
                        metrics, _ = run_backtest(m1, m5_test, test_cfg, write_logs=False)
                        metrics["signal_candidates"] = len(m5_test)
                    rows.append(metrics)

    base.sweep_log = str(REPORT_DIR / "multi_tf_lorentzian_andean_supply_demand_sweep.csv")
    return save_results(rows, base)


def hybrid_donchian_tier_sweep(m1_raw: pd.DataFrame, m5_raw: pd.DataFrame, m15_raw: pd.DataFrame, h1_raw: pd.DataFrame, base: Config):
    rows = []
    cfg = Config(**{**base.__dict__})
    cfg.ema_fast = 50
    cfg.ema_slow = 200
    cfg.lorentzian_k = 8
    cfg.lorentzian_horizon = 8
    cfg.andean_length = 50
    cfg.min_atr_pct = 0.002
    cfg.risk_per_trade = 0.005
    cfg.rr = 2.0
    cfg.exit_mode = "fixed"
    cfg.m15_adx_min = 20
    cfg.require_m15_adx_rising = True
    cfg.use_donchian = True
    cfg.donchian_n = 20
    cfg.use_retest_entry = False
    cfg.breakout_body_mult = 1.5
    cfg.breakout_atr_mult = 0.5
    cfg.breakout_wick_max_pct = 0.5
    cfg.breakout_close_location_min = 0.6

    m1, m5_base, m15 = calculate_indicators(m1_raw, m5_raw, m15_raw, cfg)
    h1 = calculate_h1_indicators(h1_raw)
    m5_base = align_timeframes(m5_base, m15)
    m5_base = align_h1_timeframe(m5_base, h1)
    m5_base = calculate_lorentzian(m5_base, cfg)
    cfg.total_days = max((m5_base.index[-1].date() - m5_base.index[0].date()).days + 1, 1)

    tests = [
        {"regime_name": "tier_a_only_risk_0.0025", "tier_mode": "tier_a_only", "risk_per_trade": 0.0025, "tier_b_risk": 0.0025},
        {"regime_name": "tier_a_only_risk_0.0050", "tier_mode": "tier_a_only", "risk_per_trade": 0.005, "tier_b_risk": 0.005},
        {"regime_name": "tier_b_only_risk_0.0075", "tier_mode": "tier_b_only", "risk_per_trade": 0.005, "tier_b_risk": 0.0075},
        {"regime_name": "tier_b_only_risk_0.0100", "tier_mode": "tier_b_only", "risk_per_trade": 0.005, "tier_b_risk": 0.01},
    ]
    for tier_a_risk in [0.0025, 0.005]:
        for tier_b_risk in [0.0075, 0.01]:
            tests.append(
                {
                    "regime_name": f"hybrid_a_{tier_a_risk:.4f}_b_{tier_b_risk:.4f}",
                    "tier_mode": "hybrid",
                    "risk_per_trade": tier_a_risk,
                    "tier_b_risk": tier_b_risk,
                }
            )

    for test in tests:
        test_cfg = Config(**{**cfg.__dict__})
        for key, value in test.items():
            setattr(test_cfg, key, value)
        m5_test = signal_candidates(m5_base, test_cfg)
        if m5_test.empty:
            metrics, _ = run_backtest(m1, m5_base.iloc[:1], test_cfg, write_logs=False)
            metrics["signal_candidates"] = 0
        else:
            metrics, _ = run_backtest(m1, m5_test, test_cfg, write_logs=False)
            metrics["signal_candidates"] = len(m5_test)
        rows.append(metrics)

    base.sweep_log = str(REPORT_DIR / "multi_tf_lorentzian_andean_hybrid_tiers.csv")
    return save_results(rows, base)


def execution_realism_sweep(m1_raw: pd.DataFrame, m5_raw: pd.DataFrame, m15_raw: pd.DataFrame, h1_raw: pd.DataFrame, base: Config):
    rows = []
    cfg = Config(**{**base.__dict__})
    cfg.ema_fast = 50
    cfg.ema_slow = 200
    cfg.lorentzian_k = 8
    cfg.lorentzian_horizon = 8
    cfg.andean_length = 50
    cfg.min_atr_pct = 0.002
    cfg.risk_per_trade = 0.0025
    cfg.tier_b_risk = 0.01
    cfg.tier_mode = "hybrid"
    cfg.rr = 2.0
    cfg.exit_mode = "fixed"
    cfg.m15_adx_min = 20
    cfg.require_m15_adx_rising = True
    cfg.use_donchian = True
    cfg.donchian_n = 20
    cfg.use_retest_entry = False
    cfg.breakout_body_mult = 1.5
    cfg.breakout_atr_mult = 0.5
    cfg.breakout_wick_max_pct = 0.5
    cfg.breakout_close_location_min = 0.6
    cfg.spread_atr_mult = 0.02
    cfg.slippage_atr_mult = 0.0

    m1, m5_base, m15 = calculate_indicators(m1_raw, m5_raw, m15_raw, cfg)
    h1 = calculate_h1_indicators(h1_raw)
    m5_base = align_timeframes(m5_base, m15)
    m5_base = align_h1_timeframe(m5_base, h1)
    m5_base = calculate_lorentzian(m5_base, cfg)
    cfg.total_days = max((m5_base.index[-1].date() - m5_base.index[0].date()).days + 1, 1)

    for spread_mult in [0.5, 1.0, 2.0]:
        for latency in [0, 1]:
            for use_spread_filter in [False, True]:
                test_cfg = Config(**{**cfg.__dict__})
                test_cfg.regime_name = f"exec_spread_{spread_mult:g}x_latency_{latency}_filter_{int(use_spread_filter)}"
                test_cfg.slippage_spread_mult = spread_mult
                test_cfg.entry_latency_m1_candles = latency
                test_cfg.use_spread_filter = use_spread_filter
                test_cfg.max_spread_atr_mult = 0.03
                m5_test = signal_candidates(m5_base, test_cfg)
                if m5_test.empty:
                    metrics, _ = run_backtest(m1, m5_base.iloc[:1], test_cfg, write_logs=False)
                    metrics["signal_candidates"] = 0
                else:
                    metrics, _ = run_backtest(m1, m5_test, test_cfg, write_logs=False)
                    metrics["signal_candidates"] = len(m5_test)
                rows.append(metrics)

    base.sweep_log = str(REPORT_DIR / "multi_tf_lorentzian_andean_execution_realism.csv")
    return save_results(rows, base)


def print_warnings(m1: pd.DataFrame, m5: pd.DataFrame, m15: pd.DataFrame):
    print(f"Loaded M1={len(m1)} rows {m1.index.min()} -> {m1.index.max()}")
    print(f"Loaded M5={len(m5)} rows {m5.index.min()} -> {m5.index.max()}")
    print(f"Loaded M15={len(m15)} rows {m15.index.min()} -> {m15.index.max()}")
    if m1.index.min() > m5.index.min() or m1.index.max() < m5.index.max():
        print("WARNING: M1 range does not fully cover M5 range; confirmations may be skipped.")
    if m15.index.min() > m5.index.min() or m15.index.max() < m5.index.max():
        print("WARNING: M15 range does not fully cover M5 range; trend alignment may be incomplete.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["backtest", "sweep", "robustness", "exit_sweep", "regime_sweep", "supply_demand_sweep", "hybrid_tier_sweep", "execution_sweep"], default="sweep")
    parser.add_argument("--m1", default=str(DATA_DIR / "bitcoin_365d_1m.csv"))
    parser.add_argument("--m5", default=str(DATA_DIR / "bitcoin_365d_5m.csv"))
    parser.add_argument("--m15", default=str(DATA_DIR / "bitcoin_365d_15m.csv"))
    parser.add_argument("--h1", default=str(DATA_DIR / "bitcoin_365d_1h.csv"))
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--andean-length", type=int, default=50)
    parser.add_argument("--ema-fast", type=int, default=20)
    parser.add_argument("--ema-slow", type=int, default=50)
    parser.add_argument("--atr-threshold", type=float, default=0.001)
    parser.add_argument("--risk", type=float, default=0.01)
    parser.add_argument("--rr", type=float, default=2.0)
    parser.add_argument("--exit-mode", choices=["fixed", "be_1_2", "be_1_no_lock", "be_1_lock_1_5", "partial_1r", "trailing_only", "partial_trailing"], default="be_1_lock_1_5")
    parser.add_argument("--m15-adx-min", type=float, default=0.0)
    parser.add_argument("--require-m15-adx-rising", action="store_true")
    parser.add_argument("--require-ema50-slope", action="store_true")
    parser.add_argument("--use-h1-bias", action="store_true")
    parser.add_argument("--min-hold-minutes", type=int, default=0)
    parser.add_argument("--partial-pct", type=float, default=0.5)
    parser.add_argument("--trail-type", choices=["prev_candle", "ema20", "none"], default="prev_candle")
    parser.add_argument("--session", choices=["14_20", "20_24", "00_03", "all"], default="all")
    args = parser.parse_args()

    cfg = Config(
        m1_file=args.m1,
        m5_file=args.m5,
        m15_file=args.m15,
        h1_file=args.h1,
        lorentzian_k=args.k,
        lorentzian_horizon=args.horizon,
        andean_length=args.andean_length,
        ema_fast=args.ema_fast,
        ema_slow=args.ema_slow,
        min_atr_pct=args.atr_threshold,
        risk_per_trade=args.risk,
        rr=args.rr,
        exit_mode=args.exit_mode,
        m15_adx_min=args.m15_adx_min,
        require_m15_adx_rising=args.require_m15_adx_rising,
        require_ema50_slope=args.require_ema50_slope,
        use_h1_bias=args.use_h1_bias,
        min_hold_minutes=args.min_hold_minutes,
        partial_pct=args.partial_pct,
        trail_type=args.trail_type,
        session_name=args.session,
    )
    m1_raw = load_csv(cfg.m1_file)
    m5_raw = load_csv(cfg.m5_file)
    m15_raw = load_csv(cfg.m15_file)
    h1_raw = load_csv(cfg.h1_file)
    print_warnings(m1_raw, m5_raw, m15_raw)

    if args.mode in {"sweep", "robustness", "exit_sweep", "regime_sweep", "supply_demand_sweep", "hybrid_tier_sweep", "execution_sweep"}:
        if args.mode == "sweep":
            results = sweep(m1_raw, m5_raw, m15_raw, cfg)
        elif args.mode == "robustness":
            results = robustness_sweep(m1_raw, m5_raw, m15_raw, cfg)
        elif args.mode == "exit_sweep":
            results = exit_optimization_sweep(m1_raw, m5_raw, m15_raw, h1_raw, cfg)
        elif args.mode == "regime_sweep":
            results = regime_filter_sweep(m1_raw, m5_raw, m15_raw, h1_raw, cfg)
        elif args.mode == "supply_demand_sweep":
            results = supply_demand_donchian_sweep(m1_raw, m5_raw, m15_raw, h1_raw, cfg)
        elif args.mode == "hybrid_tier_sweep":
            results = hybrid_donchian_tier_sweep(m1_raw, m5_raw, m15_raw, h1_raw, cfg)
        else:
            results = execution_realism_sweep(m1_raw, m5_raw, m15_raw, h1_raw, cfg)
        columns = [
            "passes_robust_filter",
            "profit_pct",
            "final_balance",
            "total_trades",
            "trades_per_day",
            "winrate",
            "profit_factor",
            "max_drawdown_pct",
            "avg_win",
            "avg_loss",
            "expectancy",
            "negative_months",
            "ema",
            "lorentzian_k",
            "lorentzian_horizon",
            "andean_length",
            "min_atr_pct",
            "risk",
            "rr",
            "exit_mode",
            "m15_adx_min",
            "require_m15_adx_rising",
            "require_ema50_slope",
            "use_h1_bias",
            "session",
            "min_hold_minutes",
            "partial_pct",
            "trail_type",
            "regime",
            "use_atr_expansion",
            "atr_ma_period",
            "atr_expansion_ratio_min",
            "use_bb_width",
            "bb_length",
            "use_donchian",
            "donchian_n",
            "use_breakout_strength_filter",
            "use_fake_breakout_filter",
            "use_retest_entry",
            "use_zone_quality_score",
            "tier_mode",
            "tier_b_risk",
            "spread_atr_mult",
            "slippage_spread_mult",
            "max_spread_atr_mult",
            "use_spread_filter",
            "entry_latency_m1_candles",
            "breakout_body_mult",
            "breakout_atr_mult",
            "volume_spike_mult",
            "zone_quality_min",
            "chop_max",
            "high_vol_day_only",
            "tier_a_trades",
            "tier_a_winrate",
            "tier_a_profit_factor",
            "tier_a_expectancy",
            "tier_b_trades",
            "tier_b_winrate",
            "tier_b_profit_factor",
            "tier_b_expectancy",
            "avg_slippage_cost",
            "total_slippage_cost",
            "slippage_cost_per_trade",
        ]
        print("TOP MULTI-TF LORENTZIAN + ANDEAN CONFIGS")
        print(results[columns].head(20).to_string(index=False))
        print(f"Saved sweep results to {cfg.sweep_log}")
        return

    m1, m5, m15 = calculate_indicators(m1_raw, m5_raw, m15_raw, cfg)
    m5 = align_timeframes(m5, m15)
    if cfg.use_h1_bias:
        h1 = calculate_h1_indicators(h1_raw)
        m5 = align_h1_timeframe(m5, h1)
    m5 = calculate_lorentzian(m5, cfg)
    metrics, monthly = run_backtest(m1, m5, cfg, write_logs=True)
    print("MULTI-TF LORENTZIAN + ANDEAN BACKTEST")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print("\nMONTHLY STATS")
    print(monthly.to_string(index=False))


if __name__ == "__main__":
    main()
