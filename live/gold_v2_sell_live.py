from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from strategies.gold_v2_sell_core import (  # noqa: E402
    GoldV2SellConfig,
    evaluate_sell_signal,
    is_in_session,
    latest_closed_m5,
    prepare_m5,
    sell_levels,
)


class Config:
    SYMBOL = os.getenv("GOLD_V2_SYMBOL", "XAUUSDm")
    BARS_M5 = int(os.getenv("GOLD_V2_BARS_M5", "300"))
    BARS_M1 = int(os.getenv("GOLD_V2_BARS_M1", "500"))
    CHECK_INTERVAL_SECONDS = int(os.getenv("GOLD_V2_CHECK_INTERVAL_SECONDS", "60"))

    # Two explicit gates are required before this script sends real MT5 orders.
    LIVE_ENABLED = os.getenv("GOLD_V2_LIVE_ENABLED", "false").lower() == "true"
    LIVE_CONFIRM = os.getenv("GOLD_V2_LIVE_CONFIRM", "").strip()
    RISK_PCT = float(os.getenv("GOLD_V2_RISK_PCT", "0.0025"))
    MAGIC_NUMBER = int(os.getenv("GOLD_V2_MAGIC_NUMBER", "20260508"))
    MAX_SPREAD_POINTS = float(os.getenv("GOLD_V2_MAX_SPREAD_POINTS", "600"))
    DEVIATION = int(os.getenv("GOLD_V2_DEVIATION", "30"))

    MT5_LOGIN = os.getenv("MT5_LOGIN", "").strip()
    MT5_PASSWORD = os.getenv("MT5_PASSWORD", "").strip()
    MT5_SERVER = os.getenv("MT5_SERVER", "").strip()
    MT5_EXPLICIT_LOGIN = os.getenv("GOLD_V2_MT5_EXPLICIT_LOGIN", "true").lower() == "true"

    LOG_DIR = ROOT / "logs"
    OUT_DIR = ROOT / "outputs" / "gold_v2_sell_live"
    STATE_FILE = LOG_DIR / "gold_v2_sell_live_state.json"
    KILL_FILE = LOG_DIR / "gold_v2_sell_live_STOP"


STRATEGY = GoldV2SellConfig(symbol=Config.SYMBOL)


def log(message: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def load_state() -> dict:
    defaults = {
        "last_checked_candle": "",
        "last_signal_candle": "",
        "signals_today_date": "",
        "signals_today_count": 0,
    }
    try:
        if Config.STATE_FILE.exists():
            defaults.update(json.loads(Config.STATE_FILE.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError):
        pass
    return defaults


def save_state(state: dict) -> None:
    Config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    Config.STATE_FILE.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")


def require_live_confirmation() -> None:
    if not Config.LIVE_ENABLED or Config.LIVE_CONFIRM != "I_UNDERSTAND_REAL_ORDERS":
        raise RuntimeError(
            "LIVE trading is locked. Set GOLD_V2_LIVE_ENABLED=true and "
            "GOLD_V2_LIVE_CONFIRM=I_UNDERSTAND_REAL_ORDERS to allow real orders."
        )


def connect_mt5() -> None:
    if mt5 is None:
        raise RuntimeError("MetaTrader5 package is not installed")
    if not mt5.initialize():
        raise RuntimeError(f"mt5.initialize() failed: {mt5.last_error()}")
    if Config.MT5_EXPLICIT_LOGIN and Config.MT5_LOGIN:
        if not mt5.login(int(Config.MT5_LOGIN), Config.MT5_PASSWORD, Config.MT5_SERVER):
            raise RuntimeError(f"mt5.login() failed: {mt5.last_error()}")
    account = mt5.account_info()
    if account is None:
        raise RuntimeError(f"MT5 account unavailable: {mt5.last_error()}")
    if not mt5.symbol_select(Config.SYMBOL, True):
        raise RuntimeError(f"Cannot select {Config.SYMBOL}: {mt5.last_error()}")
    log(
        f"Connected MT5 | account={account.login} server={account.server} "
        f"symbol={Config.SYMBOL} balance={account.balance:.2f}"
    )


def mt5_timeframe(name: str):
    return getattr(mt5, f"TIMEFRAME_{name}")


def get_ohlcv(timeframe, bars: int) -> pd.DataFrame:
    rates = mt5.copy_rates_from_pos(Config.SYMBOL, timeframe, 0, bars)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"No MT5 rates for {Config.SYMBOL} timeframe={timeframe}: {mt5.last_error()}")
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.sort_values("time").drop_duplicates("time").set_index("time")
    df.rename(columns={"tick_volume": "volume"}, inplace=True)
    return df[["open", "high", "low", "close", "volume"]].dropna()


def spread_snapshot() -> dict:
    tick = mt5.symbol_info_tick(Config.SYMBOL)
    info = mt5.symbol_info(Config.SYMBOL)
    if tick is None or info is None or info.point <= 0:
        return {"ok": False, "reason": "missing_tick_or_info"}
    spread = float(tick.ask - tick.bid)
    return {
        "ok": True,
        "bid": float(tick.bid),
        "ask": float(tick.ask),
        "spread": spread,
        "spread_points": spread / float(info.point),
        "point": float(info.point),
        "digits": int(info.digits),
    }


def today_utc() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def reset_daily_counter_if_needed(state: dict) -> None:
    today = today_utc()
    if state.get("signals_today_date") != today:
        state["signals_today_date"] = today
        state["signals_today_count"] = 0


def has_open_gold_v2_position() -> bool:
    positions = mt5.positions_get(symbol=Config.SYMBOL) or []
    return any(int(getattr(pos, "magic", 0)) == Config.MAGIC_NUMBER for pos in positions)


def calculate_lot(entry: float, sl: float) -> tuple[float, str]:
    account = mt5.account_info()
    info = mt5.symbol_info(Config.SYMBOL)
    if account is None or info is None:
        return 0.0, "missing_account_or_symbol_info"
    risk_money = float(account.balance) * Config.RISK_PCT
    stop_dist = abs(float(sl) - float(entry))
    if stop_dist <= 0 or info.trade_tick_size <= 0 or info.trade_tick_value <= 0:
        return 0.0, "invalid_stop_or_tick_value"
    loss_per_lot = (stop_dist / info.trade_tick_size) * info.trade_tick_value
    if loss_per_lot <= 0:
        return 0.0, "invalid_loss_per_lot"

    step = float(info.volume_step or 0.01)
    min_lot = float(info.volume_min or step)
    max_lot = float(info.volume_max or (risk_money / loss_per_lot))
    lot = np.floor((risk_money / loss_per_lot) / step) * step
    lot = min(max(lot, min_lot), max_lot)

    margin = mt5.order_calc_margin(mt5.ORDER_TYPE_SELL, Config.SYMBOL, lot, entry)
    free = float(getattr(account, "margin_free", 0))
    if margin is not None and free > 0 and margin > free * 0.9:
        lot = np.floor((free * 0.9 / margin * lot) / step) * step
        lot = min(max(lot, min_lot), max_lot)
    digits = max(0, int(round(-np.log10(step)))) if step < 1 else 0
    lot = round(float(lot), digits)
    if lot < min_lot:
        return 0.0, f"lot_below_min lot={lot} min={min_lot}"
    return lot, "lot_ok"


def check_stops(price: float, sl: float, tp: float) -> tuple[bool, str]:
    info = mt5.symbol_info(Config.SYMBOL)
    if info is None:
        return False, "missing_symbol_info"
    stops_level = float(getattr(info, "stops_level", 0) or 0)
    if stops_level <= 0:
        return True, "stops_ok"
    min_dist = stops_level * float(info.point)
    if abs(sl - price) < min_dist:
        return False, f"sl_too_close dist={abs(sl-price):.5f} min={min_dist:.5f}"
    if abs(price - tp) < min_dist:
        return False, f"tp_too_close dist={abs(price-tp):.5f} min={min_dist:.5f}"
    return True, "stops_ok"


def order_row(candle, levels: dict, spread: dict, lot: float, status: str, result=None, reason: str = "") -> dict:
    retcode = getattr(result, "retcode", None) if result is not None else None
    order = getattr(result, "order", None) if result is not None else None
    deal = getattr(result, "deal", None) if result is not None else None
    return {
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "symbol": Config.SYMBOL,
        "side": "SELL",
        "status": status,
        "reason": reason,
        "candle_time": candle.name.isoformat(),
        "volume": lot,
        "price": levels["entry"],
        "sl": levels["sl"],
        "tp": levels["tp"],
        "risk": levels["risk"],
        "rr": levels["rr"],
        "bid": spread.get("bid"),
        "ask": spread.get("ask"),
        "spread_points": spread.get("spread_points"),
        "retcode": retcode,
        "order": order,
        "deal": deal,
        "magic": Config.MAGIC_NUMBER,
        "body": float(candle.get("body", 0)),
        "avg_body": float(candle.get("avg_body", 0)),
        "wick_pct": float(candle.get("wick_pct", 0)),
        "atr_percentile": float(candle.get("atr_percentile_100", 0)),
        "donchian_low": float(candle.get("donchian_low", 0)),
    }


def send_live_sell(candle, levels: dict, spread: dict) -> tuple[bool, str]:
    price = float(spread["bid"])
    info = mt5.symbol_info(Config.SYMBOL)
    if info is None:
        return False, "missing_symbol_info"
    digits = int(info.digits)
    levels = {
        **levels,
        "entry": round(price, digits),
        "sl": round(float(levels["sl"]), digits),
        "tp": round(float(levels["tp"]), digits),
    }
    lot, lot_reason = calculate_lot(levels["entry"], levels["sl"])
    if lot <= 0:
        append_csv(Config.OUT_DIR / "orders.csv", order_row(candle, levels, spread, lot, "REJECTED", reason=lot_reason))
        return False, lot_reason
    stops_ok, stops_reason = check_stops(levels["entry"], levels["sl"], levels["tp"])
    if not stops_ok:
        append_csv(Config.OUT_DIR / "orders.csv", order_row(candle, levels, spread, lot, "REJECTED", reason=stops_reason))
        return False, stops_reason

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": Config.SYMBOL,
        "volume": lot,
        "type": mt5.ORDER_TYPE_SELL,
        "price": levels["entry"],
        "sl": levels["sl"],
        "tp": levels["tp"],
        "deviation": Config.DEVIATION,
        "magic": Config.MAGIC_NUMBER,
        "comment": "gold_v2_sell_live",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    ok = result is not None and result.retcode == mt5.TRADE_RETCODE_DONE
    status = "ORDER_DONE" if ok else "ORDER_FAILED"
    reason = "sent" if ok else f"order_send_failed result={result} last_error={mt5.last_error()}"
    append_csv(Config.OUT_DIR / "orders.csv", order_row(candle, levels, spread, lot, status, result=result, reason=reason))
    return ok, reason


def reject(candle, reason: str, spread: dict, category: str = "reject") -> None:
    append_csv(
        Config.OUT_DIR / "rejects_live.csv",
        {
            "logged_at": datetime.now(timezone.utc).isoformat(),
            "category": category,
            "symbol": Config.SYMBOL,
            "candle_time": candle.name.isoformat(),
            "reason": reason,
            "close": float(candle["close"]),
            "body": float(candle.get("body", 0)),
            "avg_body": float(candle.get("avg_body", 0)),
            "wick_pct": float(candle.get("wick_pct", 0)),
            "atr_percentile": float(candle.get("atr_percentile_100", 0)),
            "donchian_low": float(candle.get("donchian_low", 0)),
            "spread_points": spread.get("spread_points"),
        },
    )


def process_once(state: dict) -> None:
    reset_daily_counter_if_needed(state)
    m5 = get_ohlcv(mt5_timeframe("M5"), Config.BARS_M5)
    prepared = prepare_m5(m5, STRATEGY)
    candle = latest_closed_m5(prepared)
    spread = spread_snapshot()
    inside = is_in_session(candle.name, STRATEGY)
    candle_key = candle.name.isoformat()
    latest = (
        f"latest_closed={candle_key} close={float(candle['close']):.3f} "
        f"spread_pts={spread.get('spread_points')} inside_session={inside}"
    )

    if state.get("last_checked_candle") == candle_key:
        log(f"Status | {latest} | same_candle")
        return
    state["last_checked_candle"] = candle_key

    signal, reason = evaluate_sell_signal(candle, STRATEGY)
    if signal is None:
        reject(candle, reason, spread)
        log(f"Reject | {latest} | reason={reason}")
        return
    if not spread.get("ok"):
        reject(candle, spread.get("reason", "spread_unavailable"), spread)
        log(f"Reject | {latest} | reason=spread_unavailable")
        return
    if float(spread["spread_points"]) > Config.MAX_SPREAD_POINTS:
        reason = f"spread_too_high {spread['spread_points']:.1f}>{Config.MAX_SPREAD_POINTS:g}"
        reject(candle, reason, spread)
        log(f"Reject | {latest} | reason={reason}")
        return
    if has_open_gold_v2_position():
        reject(candle, "open_gold_v2_position", spread, "missed_trade")
        log(f"Missed | {latest} | reason=open_gold_v2_position")
        return
    if state.get("signals_today_count", 0) >= STRATEGY.max_signals_per_day:
        reject(candle, "daily_signal_limit", spread, "missed_trade")
        log(f"Missed | {latest} | reason=daily_signal_limit")
        return
    if state.get("last_signal_candle") == candle_key:
        reject(candle, "duplicate_signal_candle", spread, "missed_trade")
        log(f"Missed | {latest} | reason=duplicate_signal_candle")
        return

    levels, level_reason = sell_levels(candle, float(spread["bid"]), STRATEGY)
    if levels is None:
        reject(candle, level_reason, spread)
        log(f"Reject | {latest} | reason={level_reason}")
        return
    ok, order_reason = send_live_sell(candle, levels, spread)
    if ok:
        state["last_signal_candle"] = candle_key
        state["signals_today_count"] = int(state.get("signals_today_count", 0)) + 1
        log(
            f"LIVE SELL SENT | {latest} | price={levels['entry']:.3f} "
            f"sl={levels['sl']:.3f} tp={levels['tp']:.3f}"
        )
    else:
        log(f"Order blocked/failed | {latest} | reason={order_reason}")


def run(once: bool = False) -> None:
    Config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    Config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    require_live_confirmation()
    log("Gold V2 SELL-only LIVE bot starting")
    connect_mt5()
    state = load_state()
    try:
        while True:
            if Config.KILL_FILE.exists():
                log(f"Kill file detected: {Config.KILL_FILE}")
                save_state(state)
                break
            try:
                process_once(state)
                save_state(state)
            except Exception as exc:
                log(f"Loop error: {exc}")
            if once:
                break
            time.sleep(Config.CHECK_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        log("Stopped by user")
        save_state(state)
    finally:
        if mt5 is not None:
            mt5.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="process one closed M5 candle and exit")
    args = parser.parse_args()
    try:
        run(once=args.once)
    except RuntimeError as exc:
        log(f"Startup blocked: {exc}")
        raise SystemExit(1) from exc
