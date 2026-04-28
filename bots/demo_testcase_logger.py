import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from donchian_core import (
    DonchianCoreConfig,
    add_entry_sl_tp,
    bos_confirmed,
    breakout_strength_score,
    bt,
    prepare_timeframes,
)


ROOT_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT_DIR / "logs"
TESTCASE_CSV = LOG_DIR / "demo_signal_testcases.csv"
SUMMARY_CSV = LOG_DIR / "demo_testcase_summary.csv"
STATE_FILE = LOG_DIR / "demo_testcase_state.json"

TESTCASE_FIELDS = [
    "logged_at",
    "market",
    "symbol",
    "candle_time",
    "signal_time",
    "close",
    "side_candidate",
    "m15_trend",
    "ema50",
    "ema200",
    "adx",
    "adx_rising",
    "donchian_high",
    "donchian_low",
    "donchian_breakout_buy",
    "donchian_breakout_sell",
    "bos_buy",
    "bos_sell",
    "m1_confirm_buy",
    "m1_confirm_sell",
    "tier",
    "risk_pct",
    "breakout_body_ratio",
    "breakout_move_atr_ratio",
    "wick_pct",
    "close_quality",
    "atr_pct",
    "entry",
    "sl",
    "tp",
    "rr",
    "spread",
    "spread_points",
    "spread_pct",
    "final_signal",
    "blocked_by",
    "block_reason",
    "external_blocked_by",
    "external_block_reason",
]

SUMMARY_FIELDS = ["updated_at", "market", "symbol", "blocked_by", "count", "pct"]


def _safe_float(value, default=""):
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _append_csv(path: Path, row: dict, fields: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fields})


def _write_csv(path: Path, rows: list[dict], fields: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_state(state: dict):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _confirm_m1(m1: pd.DataFrame, signal_time: pd.Timestamp, cfg) -> tuple[bool, bool, dict]:
    buy_time, buy_entry = bt.confirm_m1(m1, signal_time, "BUY", cfg)
    sell_time, sell_entry = bt.confirm_m1(m1, signal_time, "SELL", cfg)
    return buy_time is not None, sell_time is not None, {
        "BUY": (buy_time, buy_entry),
        "SELL": (sell_time, sell_entry),
    }


def _side(row: pd.Series, cfg) -> Optional[str]:
    high_col = f"donchian_high_{cfg.donchian_n}"
    low_col = f"donchian_low_{cfg.donchian_n}"
    if pd.isna(row.get(high_col)) or pd.isna(row.get(low_col)):
        return None
    if row["close"] > row[high_col]:
        return "BUY"
    if row["close"] < row[low_col]:
        return "SELL"
    return None


def _quality(row: pd.Series, side: Optional[str], cfg) -> dict:
    avg_body = row.get("avg_body_20", np.nan)
    body_ratio = row.get("body", np.nan) / avg_body if not pd.isna(avg_body) and avg_body else np.nan
    if side == "BUY":
        move_atr = row.get(f"donchian_buy_move_atr_{cfg.donchian_n}", np.nan)
        close_quality = row.get("close_location", np.nan)
    elif side == "SELL":
        move_atr = row.get(f"donchian_sell_move_atr_{cfg.donchian_n}", np.nan)
        close_quality = 1 - row.get("close_location", np.nan)
    else:
        move_atr = np.nan
        close_quality = np.nan
    return {
        "body_ratio": _safe_float(body_ratio),
        "move_atr": _safe_float(move_atr),
        "wick_pct": _safe_float(row.get("max_wick_pct")),
        "close_quality": _safe_float(close_quality),
    }


def _tier(row: pd.Series, side: Optional[str], cfg) -> str:
    if not side:
        return ""
    return "B" if bt.high_quality_breakout(row, side, cfg) else "A"


def _trade_levels(row: pd.Series, side: str, entry: float, tier: str, core_config: DonchianCoreConfig) -> tuple[dict, str]:
    signal = {
        "side": side,
        "tier": tier,
        "risk_pct": core_config.tier_b_risk if tier == "B" else core_config.tier_a_risk,
        "entry": entry,
    }
    try:
        trade = add_entry_sl_tp(signal, row, entry, core_config.rr)
    except Exception as exc:
        return {}, f"SL/TP error: {exc}"
    values = [trade.get("entry"), trade.get("sl"), trade.get("tp"), trade.get("risk")]
    if any(value is None or not np.isfinite(value) for value in values):
        return trade, "SL/TP non-finite"
    if trade["risk"] <= 0:
        return trade, "risk distance <= 0"
    if side == "BUY" and not (trade["sl"] < trade["entry"] < trade["tp"]):
        return trade, "BUY SL/TP invalid"
    if side == "SELL" and not (trade["tp"] < trade["entry"] < trade["sl"]):
        return trade, "SELL SL/TP invalid"
    return trade, ""


def _update_summary():
    if not TESTCASE_CSV.exists():
        return
    df = pd.read_csv(TESTCASE_CSV)
    if df.empty:
        return
    rows = []
    updated_at = datetime.now().isoformat(timespec="seconds")
    for (market, symbol), group in df.groupby(["market", "symbol"], dropna=False):
        total = max(len(group), 1)
        counts = group["blocked_by"].value_counts(dropna=False)
        for blocked_by, count in counts.items():
            rows.append({
                "updated_at": updated_at,
                "market": market,
                "symbol": symbol,
                "blocked_by": blocked_by,
                "count": int(count),
                "pct": float(count) / total * 100,
            })
    _write_csv(SUMMARY_CSV, rows, SUMMARY_FIELDS)


def log_demo_testcase(
    market: str,
    symbol: str,
    m1_raw: pd.DataFrame,
    m5_raw: pd.DataFrame,
    m15_raw: pd.DataFrame,
    core_config: DonchianCoreConfig,
    external_blocked_by: str = "",
    external_block_reason: str = "",
    spread: Optional[dict] = None,
    force_update=False,
) -> Optional[dict]:
    try:
        m1, m5, cfg = prepare_timeframes(m1_raw, m5_raw, m15_raw, core_config)
        if len(m5) < core_config.donchian_n + 5:
            return None
        row_time = pd.Timestamp(m5.index[-2])
        signal_time = row_time + pd.Timedelta(minutes=5)
        key = f"{market}:{symbol}:{row_time.isoformat()}"
        state = _load_state()
        if not force_update and state.get(key):
            return None

        row = m5.loc[row_time]
        side = _side(row, cfg)
        m1_buy, m1_sell, entries = _confirm_m1(m1, signal_time, cfg)
        trend = row.get("m15_trend", np.nan)
        adx = row.get("m15_adx", np.nan)
        adx_rising = bool(row.get("m15_adx_rising", False))
        bos_buy = not pd.isna(row.get("swing_high")) and row["high"] > row["swing_high"]
        bos_sell = not pd.isna(row.get("swing_low")) and row["low"] < row["swing_low"]
        tier = _tier(row, side, cfg)
        quality = _quality(row, side, cfg)

        final_signal = side or "NONE"
        blocked_by = "VALID_SIGNAL"
        block_reason = "valid"
        trade = {}

        if not side:
            blocked_by = "NO_DONCHIAN_BREAKOUT"
            block_reason = "M5 close did not break Donchian high/low"
            final_signal = "NONE"
        elif (side == "BUY" and trend != 1) or (side == "SELL" and trend != -1):
            blocked_by = "NO_M15_TREND"
            block_reason = f"M15 trend={trend} not aligned with {side}"
            final_signal = "NONE"
        elif pd.isna(adx) or adx < cfg.m15_adx_min:
            blocked_by = "ADX_TOO_LOW"
            block_reason = f"M15 ADX {_safe_float(adx, 0):.2f} < {cfg.m15_adx_min}"
            final_signal = "NONE"
        elif cfg.require_m15_adx_rising and not adx_rising:
            blocked_by = "ADX_NOT_RISING"
            block_reason = "M15 ADX is not rising"
            final_signal = "NONE"
        elif not bos_confirmed(row, side):
            blocked_by = "NO_BOS"
            block_reason = f"{side} did not break previous swing structure"
            final_signal = "NONE"
        elif not (m1_buy if side == "BUY" else m1_sell):
            blocked_by = "NO_M1_CONFIRM"
            block_reason = f"no M1 {side} confirmation within {cfg.max_m1_confirm_candles} candles"
            final_signal = "NONE"
        elif external_blocked_by:
            blocked_by = external_blocked_by
            block_reason = external_block_reason
            final_signal = "NONE"
        else:
            _, entry = entries.get(side, (None, None))
            if entry is None:
                entry = float(row["close"])
            trade, invalid_reason = _trade_levels(row, side, entry, tier, core_config)
            if invalid_reason:
                blocked_by = "INVALID_SL_TP"
                block_reason = invalid_reason
                final_signal = "NONE"

        spread = spread or {}
        row_out = {
            "logged_at": datetime.now().isoformat(timespec="seconds"),
            "market": market,
            "symbol": symbol,
            "candle_time": row_time,
            "signal_time": signal_time,
            "close": _safe_float(row.get("close")),
            "side_candidate": side or "NONE",
            "m15_trend": "UP" if trend == 1 else "DOWN" if trend == -1 else "NONE",
            "ema50": _safe_float(row.get("m15_ema_fast")),
            "ema200": _safe_float(row.get("m15_ema_slow")),
            "adx": _safe_float(adx),
            "adx_rising": adx_rising,
            "donchian_high": _safe_float(row.get(f"donchian_high_{cfg.donchian_n}")),
            "donchian_low": _safe_float(row.get(f"donchian_low_{cfg.donchian_n}")),
            "donchian_breakout_buy": side == "BUY",
            "donchian_breakout_sell": side == "SELL",
            "bos_buy": bos_buy,
            "bos_sell": bos_sell,
            "m1_confirm_buy": m1_buy,
            "m1_confirm_sell": m1_sell,
            "tier": tier,
            "risk_pct": core_config.tier_b_risk if tier == "B" else core_config.tier_a_risk if tier == "A" else "",
            "breakout_body_ratio": quality["body_ratio"],
            "breakout_move_atr_ratio": quality["move_atr"],
            "wick_pct": quality["wick_pct"],
            "close_quality": quality["close_quality"],
            "atr_pct": _safe_float(row.get("atr_pct")),
            "entry": _safe_float(trade.get("entry")),
            "sl": _safe_float(trade.get("sl")),
            "tp": _safe_float(trade.get("tp")),
            "rr": core_config.rr,
            "spread": spread.get("spread", ""),
            "spread_points": spread.get("spread_points", ""),
            "spread_pct": spread.get("spread_pct", ""),
            "final_signal": final_signal,
            "blocked_by": blocked_by,
            "block_reason": block_reason,
            "external_blocked_by": external_blocked_by,
            "external_block_reason": external_block_reason,
        }
        _append_csv(TESTCASE_CSV, row_out, TESTCASE_FIELDS)
        state[key] = row_out["logged_at"]
        _save_state(state)
        _update_summary()
        return row_out
    except Exception as exc:
        _append_csv(
            TESTCASE_CSV,
            {
                "logged_at": datetime.now().isoformat(timespec="seconds"),
                "market": market,
                "symbol": symbol,
                "blocked_by": "DIAGNOSTIC_ERROR",
                "block_reason": str(exc),
                "external_blocked_by": external_blocked_by,
                "external_block_reason": external_block_reason,
            },
            TESTCASE_FIELDS,
        )
        return None
