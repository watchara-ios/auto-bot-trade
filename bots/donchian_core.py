import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Telemetry: count how often each filter rejects a signal.
# Read this from forex_bot.py or any caller to find the bottleneck.
_REJECT_STATS: Counter = Counter()

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "backtests"))

import backtest_multi_tf_lorentzian_andean as bt


@dataclass
class DonchianCoreConfig:
    allowed_side: str = ""
    ema_fast: int = 50
    ema_slow: int = 200
    donchian_n: int = 20
    adx_min: float = 20.0
    adx_max: Optional[float] = None
    require_adx_rising: bool = True
    session_hours_utc: tuple[int, ...] = ()
    atr_percentile_min: Optional[float] = None
    volume_mult: float = 0.0
    require_atr_expansion: bool = False
    atr_expansion_period: int = 50
    swing_lookback: int = 8
    rr: float = 2.0
    tier_a_risk: float = 0.0025
    tier_b_risk: float = 0.01
    breakout_body_mult: float = 1.2        # loosened from 1.5
    breakout_atr_mult: float = 0.5
    max_wick_pct: float = 0.5
    close_quality_min: float = 0.5         # loosened from 0.6
    min_atr_pct: float = 0.0005
    max_m1_confirm_candles: int = 10       # loosened from 5
    entry_latency_m1_candles: int = 1
    max_trades_per_day: int = 5
    max_losses_per_day: int = 2
    # Pro: Minervini — ADX must accelerate ≥ N consecutive bars before entry
    adx_bars_rising: int = 1


def to_indexed_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "time" in out.columns:
        out["time"] = pd.to_datetime(out["time"])
        out = out.set_index("time")
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out[["open", "high", "low", "close", "volume"]].dropna()


def bt_config(config: DonchianCoreConfig) -> bt.Config:
    return bt.Config(
        ema_fast=config.ema_fast,
        ema_slow=config.ema_slow,
        atr_period=14,
        min_atr_pct=config.min_atr_pct,
        swing_lookback=config.swing_lookback,
        rr=config.rr,
        risk_per_trade=config.tier_a_risk,
        tier_b_risk=config.tier_b_risk,
        tier_mode="hybrid",
        exit_mode="fixed",
        m15_adx_min=config.adx_min,
        require_m15_adx_rising=config.require_adx_rising,
        use_donchian=True,
        donchian_n=config.donchian_n,
        breakout_body_mult=config.breakout_body_mult,
        breakout_atr_mult=config.breakout_atr_mult,
        breakout_wick_max_pct=config.max_wick_pct,
        breakout_close_location_min=config.close_quality_min,
        max_m1_confirm_candles=config.max_m1_confirm_candles,
        entry_latency_m1_candles=config.entry_latency_m1_candles,
        max_trades_per_day=config.max_trades_per_day,
        max_losses_per_day=config.max_losses_per_day,
    )


def prepare_timeframes(m1: pd.DataFrame, m5: pd.DataFrame, m15: pd.DataFrame, config: DonchianCoreConfig):
    cfg = bt_config(config)
    m1 = to_indexed_ohlcv(m1)
    m5 = to_indexed_ohlcv(m5)
    m15 = to_indexed_ohlcv(m15)
    m1, m5, m15 = bt.calculate_indicators(m1, m5, m15, cfg)
    m5 = bt.align_timeframes(m5, m15)
    m5["m15_ema_fast"] = m15["ema_fast"].shift(1).reindex(m5.index, method="ffill")
    m5["m15_ema_slow"] = m15["ema_slow"].shift(1).reindex(m5.index, method="ffill")

    # Minervini: consecutive bars M15 ADX is rising
    if "adx" in m15.columns:
        _diff = m15["adx"].diff()
        _cnt, _vals = 0, []
        for d in _diff:
            if pd.isna(d) or d <= 0:
                _cnt = 0
            else:
                _cnt += 1
            _vals.append(_cnt)
        m15["adx_consec_rising"] = _vals
        m5["m15_adx_consec_rising"] = (
            m15["adx_consec_rising"].shift(1)
            .reindex(m5.index, method="ffill")
            .fillna(0)
        )

    m5["atr_percentile_100"] = m5["atr"].rolling(100).apply(
        lambda values: pd.Series(values).rank(pct=True).iloc[-1] * 100,
        raw=False,
    )
    return m1, m5, cfg


def donchian_side(row: pd.Series, cfg: bt.Config) -> Optional[str]:
    high_col = f"donchian_high_{cfg.donchian_n}"
    low_col = f"donchian_low_{cfg.donchian_n}"
    if pd.isna(row.get(high_col)) or pd.isna(row.get(low_col)):
        return None
    if row["close"] > row[high_col]:
        return "BUY"
    if row["close"] < row[low_col]:
        return "SELL"
    return None


def bos_confirmed(row: pd.Series, side: str) -> bool:
    if side == "BUY":
        return not pd.isna(row.get("swing_high")) and row["high"] > row["swing_high"]
    return not pd.isna(row.get("swing_low")) and row["low"] < row["swing_low"]


def breakout_strength_score(row: pd.Series, side: str, cfg: bt.Config) -> int:
    score = 0
    avg_body = row.get("avg_body_20", np.nan)
    if not pd.isna(avg_body) and avg_body > 0 and row.get("body", 0) > avg_body * cfg.breakout_body_mult:
        score += 1
    if side == "BUY":
        move_atr = row.get(f"donchian_buy_move_atr_{cfg.donchian_n}", np.nan)
        close_quality = row.get("close_location", np.nan)
    else:
        move_atr = row.get(f"donchian_sell_move_atr_{cfg.donchian_n}", np.nan)
        close_quality = 1 - row.get("close_location", np.nan)
    if not pd.isna(move_atr) and move_atr > cfg.breakout_atr_mult:
        score += 1
    if row.get("max_wick_pct", 1) <= cfg.breakout_wick_max_pct:
        score += 1
    if not pd.isna(close_quality) and close_quality >= cfg.breakout_close_location_min:
        score += 1
    return score


def signal_from_closed_row(symbol: str, row: pd.Series, cfg: bt.Config) -> tuple[Optional[dict], str]:
    side = donchian_side(row, cfg)
    if side is None:
        _REJECT_STATS["no_donchian_breakout"] += 1
        return None, "no Donchian breakout"
    if pd.isna(row.get("atr")) or pd.isna(row.get("atr_pct")):
        _REJECT_STATS["atr_not_ready"] += 1
        return None, "M5 ATR not ready"
    if row["atr_pct"] < cfg.min_atr_pct:
        _REJECT_STATS["atr_too_low"] += 1
        return None, f"ATR too low atr_pct={row['atr_pct']:.5f}"
    if row.get("m15_adx", 0) < cfg.m15_adx_min:
        _REJECT_STATS["adx_too_low"] += 1
        return None, f"M15 ADX too low {row.get('m15_adx', 0):.2f} < {cfg.m15_adx_min}"
    if cfg.require_m15_adx_rising and not bool(row.get("m15_adx_rising", False)):
        _REJECT_STATS["adx_not_rising"] += 1
        return None, "M15 ADX not rising"
    trend = row.get("m15_trend")
    if (side == "BUY" and trend != 1) or (side == "SELL" and trend != -1):
        _REJECT_STATS["trend_not_aligned"] += 1
        return None, f"M15 trend not aligned side={side} trend={trend}"
    if not bos_confirmed(row, side):
        _REJECT_STATS["bos_not_confirmed"] += 1
        return None, "BOS not confirmed"

    tier = "B" if bt.high_quality_breakout(row, side, cfg) else "A"
    strength = breakout_strength_score(row, side, cfg)
    return {
        "symbol": symbol,
        "side": side,
        "exit_side": "SELL" if side == "BUY" else "BUY",
        "tier": tier,
        "risk_pct": cfg.tier_b_risk if tier == "B" else cfg.risk_per_trade,
        "entry": float(row["close"]),
        "atr": float(row["atr"]),
        "atr_pct": float(row["atr_pct"]),
        "adx": float(row["m15_adx"]),
        "adx_rising": bool(row["m15_adx_rising"]),
        "donchian_breakout": True,
        "bos_confirmed": True,
        "breakout_strength_score": strength,
        "score": 80 if tier == "B" else 62,
        "type": f"donchian_tier_{tier.lower()}",
        "reason": (
            f"Donchian{cfg.donchian_n} {side} + M15 trend + ADX {row['m15_adx']:.1f} rising "
            f"+ BOS + Tier {tier}"
        ),
    }, "signal"


def apply_micro_edge_filters(row: pd.Series, side: str, config: DonchianCoreConfig) -> Optional[str]:
    fails: list[str] = []

    if config.allowed_side and config.allowed_side.upper() != "BOTH" and side != config.allowed_side:
        _REJECT_STATS["side_blocked"] += 1
        fails.append(f"side blocked allowed={config.allowed_side} got={side}")

    if config.session_hours_utc and row.name.hour not in config.session_hours_utc:
        _REJECT_STATS["outside_session"] += 1
        hrs = f"{min(config.session_hours_utc):02d}-{max(config.session_hours_utc):02d}"
        fails.append(f"outside session ({hrs} UTC) hour={row.name.hour}")

    adx = float(row.get("m15_adx", 0) or 0)
    if config.adx_max is not None and adx >= config.adx_max:
        _REJECT_STATS["adx_too_high"] += 1
        fails.append(f"M15 ADX too high {adx:.2f} >= {config.adx_max}")

    atr_pctile = row.get("atr_percentile_100", np.nan)
    if config.atr_percentile_min is not None:
        if pd.isna(atr_pctile) or atr_pctile < config.atr_percentile_min:
            _REJECT_STATS["atr_percentile_low"] += 1
            fails.append(f"ATR percentile below {config.atr_percentile_min:g}: {atr_pctile:.1f}")

    volume_ratio = row.get("volume_ratio", np.nan)
    if config.volume_mult > 0:
        if pd.isna(volume_ratio) or volume_ratio < config.volume_mult:
            _REJECT_STATS["volume_low"] += 1
            fails.append(f"volume ratio below {config.volume_mult:g}: {volume_ratio:.2f}")

    if config.require_atr_expansion:
        expansion_col = f"atr_expansion_{config.atr_expansion_period}"
        expansion = row.get(expansion_col, np.nan)
        if pd.isna(expansion) or expansion <= 1.0:
            _REJECT_STATS["atr_not_expanding"] += 1
            fails.append(f"ATR not expanding {expansion_col}={expansion:.2f}")

    if config.adx_bars_rising > 1:
        consec = float(row.get("m15_adx_consec_rising", 0) or 0)
        if consec < config.adx_bars_rising:
            _REJECT_STATS["adx_consec_rising_low"] += 1
            fails.append(f"M15 ADX consec_rising {int(consec)} < {config.adx_bars_rising} required")

    return fails[0] if fails else None


def add_entry_sl_tp(signal: dict, row: pd.Series, entry: float, rr: float) -> dict:
    atr_value = float(row["atr"])
    if signal["side"] == "BUY":
        swing_sl = float(row["swing_low"]) if not pd.isna(row.get("swing_low")) else entry - atr_value
        sl = min(entry - atr_value, swing_sl)
        risk = abs(entry - sl)
        tp = entry + risk * rr
    else:
        swing_sl = float(row["swing_high"]) if not pd.isna(row.get("swing_high")) else entry + atr_value
        sl = max(entry + atr_value, swing_sl)
        risk = abs(entry - sl)
        tp = entry - risk * rr
    signal = signal.copy()
    signal.update({"entry": float(entry), "sl": float(sl), "tp": float(tp), "risk": float(risk), "rr": rr})
    return signal


def latest_signal(symbol: str, m1: pd.DataFrame, m5: pd.DataFrame, m15: pd.DataFrame, config: DonchianCoreConfig):
    _REJECT_STATS["_total_attempts"] += 1
    m1_ready, m5_ready, cfg = prepare_timeframes(m1, m5, m15, config)
    if len(m5_ready) < config.donchian_n + 5:
        _REJECT_STATS["insufficient_data"] += 1
        return None, "M5 data not enough", None

    row_time = m5_ready.index[-2]
    signal_time = row_time + pd.Timedelta(minutes=5)
    row = m5_ready.loc[row_time]
    signal, reason = signal_from_closed_row(symbol, row, cfg)
    if signal is None:
        return None, reason, row_time
    block_reason = apply_micro_edge_filters(row, signal["side"], config)
    if block_reason:
        return None, block_reason, row_time

    entry_time, entry = bt.confirm_m1(m1_ready, signal_time, signal["side"], cfg)
    if entry_time is None:
        _REJECT_STATS["m1_confirm_failed"] += 1
        return None, "M1 confirmation/latency not ready", row_time

    signal = add_entry_sl_tp(signal, row, entry, config.rr)
    signal["signal_time"] = str(signal_time)
    signal["entry_time"] = str(entry_time)
    signal["candle_time"] = str(row_time)
    return signal, reason, row_time
