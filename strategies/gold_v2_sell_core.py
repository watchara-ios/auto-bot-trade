from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GoldV2SellConfig:
    symbol: str = "XAUUSDm"
    rr: float = 1.8
    body_mult: float = 2.0
    max_wick_pct: float = 0.40
    atr_percentile_min: float = 40.0
    donchian_n: int = 20
    swing_lookback: int = 8
    atr_period: int = 14
    avg_body_period: int = 20
    session_start_utc_minute: int = 12 * 60
    session_end_utc_minute: int = 14 * 60
    max_signals_per_day: int = 1


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


def prepare_m5(m5: pd.DataFrame, config: GoldV2SellConfig) -> pd.DataFrame:
    out = m5.copy()
    out["atr"] = atr(out, config.atr_period)
    out["atr_percentile_100"] = out["atr"].rolling(100).apply(
        lambda values: float((values <= values[-1]).mean() * 100),
        raw=True,
    )
    out["donchian_low"] = out["low"].shift(1).rolling(config.donchian_n).min()
    out["swing_high"] = out["high"].shift(1).rolling(config.swing_lookback).max()
    candle_range = (out["high"] - out["low"]).replace(0, np.nan)
    out["body"] = (out["close"] - out["open"]).abs()
    out["avg_body"] = out["body"].shift(1).rolling(config.avg_body_period).mean()
    out["upper_wick"] = out["high"] - out[["open", "close"]].max(axis=1)
    out["lower_wick"] = out[["open", "close"]].min(axis=1) - out["low"]
    out["wick_pct"] = out[["upper_wick", "lower_wick"]].max(axis=1) / candle_range
    return out


def is_in_session(ts: pd.Timestamp, config: GoldV2SellConfig) -> bool:
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    minute = ts.hour * 60 + ts.minute
    return config.session_start_utc_minute <= minute < config.session_end_utc_minute


def latest_closed_m5(prepared: pd.DataFrame) -> pd.Series:
    if len(prepared) < 2:
        raise ValueError("not enough M5 bars")
    return prepared.iloc[-2]


def evaluate_sell_signal(row: pd.Series, config: GoldV2SellConfig) -> tuple[dict | None, str]:
    if not is_in_session(row.name, config):
        return None, "outside_session"
    required = ["atr", "atr_percentile_100", "donchian_low", "swing_high", "avg_body", "wick_pct"]
    missing = [col for col in required if pd.isna(row.get(col))]
    if missing:
        return None, "warmup_missing_" + ",".join(missing)
    if row["close"] >= row["donchian_low"]:
        return None, f"no_breakdown close={row['close']:.3f} >= donchian_low={row['donchian_low']:.3f}"
    if row["body"] <= row["avg_body"] * config.body_mult:
        return None, f"body_too_small body={row['body']:.3f} avg_body={row['avg_body']:.3f} mult={config.body_mult:g}"
    if row["wick_pct"] > config.max_wick_pct:
        return None, f"wick_too_large wick_pct={row['wick_pct']:.3f} max={config.max_wick_pct:g}"
    if row["atr_percentile_100"] < config.atr_percentile_min:
        return None, f"atr_percentile_low atr_pct={row['atr_percentile_100']:.1f} min={config.atr_percentile_min:g}"
    return {
        "side": "SELL",
        "pattern": "gold_v2_sell_momentum",
        "level": float(row["donchian_low"]),
        "candle_time": row.name,
    }, "signal_ok"


def sell_levels(row: pd.Series, entry_bid: float, config: GoldV2SellConfig) -> tuple[dict | None, str]:
    atr_stop = entry_bid + float(row["atr"])
    swing_stop = float(row["swing_high"])
    sl = max(swing_stop, atr_stop)
    risk = sl - entry_bid
    if not np.isfinite(risk) or risk <= 0:
        return None, f"invalid_risk risk={risk}"
    tp = entry_bid - risk * config.rr
    return {
        "entry": float(entry_bid),
        "sl": float(sl),
        "tp": float(tp),
        "risk": float(risk),
        "rr": config.rr,
    }, "levels_ok"


def classify_exit_since(
    bars: pd.DataFrame,
    entry_time: pd.Timestamp,
    sl: float,
    tp: float,
) -> tuple[str, pd.Timestamp | None, float | None]:
    if bars.empty:
        return "OPEN", None, None
    if entry_time.tzinfo is None and bars.index.tz is not None:
        entry_time = entry_time.tz_localize(bars.index.tz)
    window = bars[bars.index >= entry_time]
    for ts, row in window.iterrows():
        if row["high"] >= sl:
            return "SL", ts, float(sl)
        if row["low"] <= tp:
            return "TP", ts, float(tp)
    return "OPEN", None, None

