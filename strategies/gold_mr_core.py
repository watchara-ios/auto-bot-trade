from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GoldMRConfig:
    """Keltner Channel mean-reversion strategy configuration."""
    keltner_period: int = 20
    keltner_atr_mult: float = 2.0
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    ema_trend_period: int = 50              # same-timeframe trend guard (EMA50 on M5)
    require_confirmation_candle: bool = True
    max_atr_distance_mult: float = 3.0     # block falling-knife: |close-ema20|/atr > this
    sl_atr_mult: float = 1.0               # SL = band extreme ± sl_atr_mult × ATR
    rr: float = 1.5
    risk_pct: float = 0.0025
    max_trades_per_day: int = 5
    max_hold_minutes: int = 240
    allowed_side: Optional[str] = None     # "BUY" / "SELL" / None = both
    session_ranges_utc_minutes: tuple[tuple[int, int], ...] = ((7 * 60, 20 * 60),)


# ─────────────────────────────────────────────────────────────────────────────
# Indicator helpers
# ─────────────────────────────────────────────────────────────────────────────

def _atr(df: pd.DataFrame, period: int = 20) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


# ─────────────────────────────────────────────────────────────────────────────
# prepare
# ─────────────────────────────────────────────────────────────────────────────

def prepare(df_m5: pd.DataFrame, df_m15: pd.DataFrame, cfg: GoldMRConfig) -> pd.DataFrame:
    """
    Compute all M5 indicators and merge M15 EMA50.
    Returns M5 DataFrame with all columns added; index = UTC timestamp.
    """
    m5 = df_m5.copy()
    if "time" in m5.columns:
        m5["time"] = pd.to_datetime(m5["time"])
        m5 = m5.set_index("time")
    m5 = m5.sort_index()
    for col in ("open", "high", "low", "close", "volume"):
        if col in m5.columns:
            m5[col] = pd.to_numeric(m5[col], errors="coerce")

    m15 = df_m15.copy()
    if "time" in m15.columns:
        m15["time"] = pd.to_datetime(m15["time"])
        m15 = m15.set_index("time")
    m15 = m15.sort_index()
    for col in ("open", "high", "low", "close", "volume"):
        if col in m15.columns:
            m15[col] = pd.to_numeric(m15[col], errors="coerce")

    # ── M5 indicators ────────────────────────────────────────────────────────
    m5["atr"]       = _atr(m5, cfg.keltner_period)
    m5["ema20"]     = m5["close"].ewm(span=cfg.keltner_period, adjust=False).mean()
    m5["kc_upper"]  = m5["ema20"] + m5["atr"] * cfg.keltner_atr_mult
    m5["kc_lower"]  = m5["ema20"] - m5["atr"] * cfg.keltner_atr_mult
    m5["rsi"]       = _rsi(m5["close"], cfg.rsi_period)
    m5["ema_trend"] = m5["close"].ewm(span=cfg.ema_trend_period, adjust=False).mean()
    m5["distance_from_middle"] = (
        (m5["close"] - m5["ema20"]) / m5["atr"].replace(0, np.nan)
    )
    m5["body"]        = (m5["close"] - m5["open"]).abs()
    m5["range_"]      = m5["high"] - m5["low"]
    m5["hour_minute"] = m5.index.hour * 60 + m5.index.minute

    # ── M15 EMA50 merged into M5 via shift(1)/ffill (no look-ahead) ──────────
    m15["ema50_m15"] = m15["close"].ewm(span=50, adjust=False).mean()
    m5["ema50_m15"]  = m15["ema50_m15"].shift(1).reindex(m5.index, method="ffill")

    return m5


# ─────────────────────────────────────────────────────────────────────────────
# Session helper
# ─────────────────────────────────────────────────────────────────────────────

def _in_session(hour_minute: int, cfg: GoldMRConfig) -> bool:
    return any(start <= hour_minute < end for start, end in cfg.session_ranges_utc_minutes)


# ─────────────────────────────────────────────────────────────────────────────
# mr_signal
# ─────────────────────────────────────────────────────────────────────────────

def mr_signal(
    row_prev: pd.Series,
    row_curr: pd.Series,
    cfg: GoldMRConfig,
) -> Optional[dict]:
    """
    Evaluate MR signal on two consecutive completed M5 bars.
      row_prev = breakout candle  (iloc[-3] in live bot)
      row_curr = confirmation candle (iloc[-2] in live bot, NEVER iloc[-1])

    Returns signal dict or None.
    """
    # Session gate — apply to confirmation candle time
    hm = row_curr.get("hour_minute")
    if pd.isna(hm) or not _in_session(int(hm), cfg):
        return None

    # NaN guards
    _req_prev = ("low", "high", "kc_lower", "kc_upper", "rsi", "atr", "close")
    _req_curr = ("close", "ema_trend", "distance_from_middle", "kc_lower", "kc_upper")
    if any(pd.isna(row_prev.get(c)) for c in _req_prev):
        return None
    if any(pd.isna(row_curr.get(c)) for c in _req_curr):
        return None

    allowed = cfg.allowed_side  # None = both sides

    # ── BUY: dipped below lower KC + RSI oversold → closed back inside ───────
    if allowed in (None, "BUY"):
        buy_breakout = float(row_prev["low"]) <= float(row_prev["kc_lower"])
        buy_oversold = float(row_prev["rsi"]) < cfg.rsi_oversold
        if buy_breakout and buy_oversold:
            if not cfg.require_confirmation_candle:
                return {
                    "side": "BUY",
                    "pattern": "kc_reversal",
                    "breakout_low": float(row_prev["low"]),
                }
            back_inside  = float(row_curr["close"]) > float(row_prev["kc_lower"])
            bullish_bar  = float(row_curr["close"]) > float(row_prev["close"])
            not_knife    = float(row_curr["close"]) >= float(row_curr["ema_trend"]) * 0.98
            not_extreme  = abs(float(row_curr["distance_from_middle"])) <= cfg.max_atr_distance_mult
            if back_inside and bullish_bar and not_knife and not_extreme:
                return {
                    "side": "BUY",
                    "pattern": "kc_reversal",
                    "breakout_low": float(row_prev["low"]),
                }

    # ── SELL: spiked above upper KC + RSI overbought → closed back inside ────
    if allowed in (None, "SELL"):
        sell_breakout   = float(row_prev["high"]) >= float(row_prev["kc_upper"])
        sell_overbought = float(row_prev["rsi"]) > cfg.rsi_overbought
        if sell_breakout and sell_overbought:
            if not cfg.require_confirmation_candle:
                return {
                    "side": "SELL",
                    "pattern": "kc_reversal",
                    "breakout_high": float(row_prev["high"]),
                }
            back_inside  = float(row_curr["close"]) < float(row_prev["kc_upper"])
            bearish_bar  = float(row_curr["close"]) < float(row_prev["close"])
            not_knife    = float(row_curr["close"]) <= float(row_curr["ema_trend"]) * 1.02
            not_extreme  = abs(float(row_curr["distance_from_middle"])) <= cfg.max_atr_distance_mult
            if back_inside and bearish_bar and not_knife and not_extreme:
                return {
                    "side": "SELL",
                    "pattern": "kc_reversal",
                    "breakout_high": float(row_prev["high"]),
                }

    return None


# ─────────────────────────────────────────────────────────────────────────────
# trade_levels
# ─────────────────────────────────────────────────────────────────────────────

def trade_levels(
    row_prev: pd.Series,
    row_curr: pd.Series,  # noqa: ARG001 — reserved for future confirmation-candle SL logic
    entry_price: float,
    signal: dict,
    cfg: GoldMRConfig,
) -> Optional[tuple[float, float, float]]:
    """Return (sl, tp, risk) or None if risk <= 0."""
    atr_val = float(row_prev["atr"])
    side    = signal["side"]

    if side == "BUY":
        sl   = float(row_prev["low"])  - atr_val * cfg.sl_atr_mult
        risk = entry_price - sl
        tp   = entry_price + risk * cfg.rr
    else:
        sl   = float(row_prev["high"]) + atr_val * cfg.sl_atr_mult
        risk = sl - entry_price
        tp   = entry_price - risk * cfg.rr

    if not np.isfinite(risk) or risk <= 0:
        return None
    return sl, tp, risk
