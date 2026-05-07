from __future__ import annotations

import csv
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError:  # lets py_compile run on machines without MT5 installed
    mt5 = None

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from strategies.gold_v2_sell_core import (  # noqa: E402
    GoldV2SellConfig,
    classify_exit_since,
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
    MT5_LOGIN = os.getenv("MT5_LOGIN", "").strip()
    MT5_PASSWORD = os.getenv("MT5_PASSWORD", "").strip()
    MT5_SERVER = os.getenv("MT5_SERVER", "").strip()
    MT5_EXPLICIT_LOGIN = os.getenv("GOLD_V2_MT5_EXPLICIT_LOGIN", "true").lower() == "true"

    LOG_DIR = ROOT / "logs"
    OUT_DIR = ROOT / "outputs" / "gold_v2_sell_live"
    STATE_FILE = LOG_DIR / "gold_v2_sell_state.json"
    KILL_FILE = LOG_DIR / "gold_v2_sell_STOP"


STRATEGY = GoldV2SellConfig(symbol=Config.SYMBOL)


def log(message: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def _json_default(value):
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def load_state() -> dict:
    defaults = {
        "last_checked_candle": "",
        "last_signal_candle": "",
        "signals_today_date": "",
        "signals_today_count": 0,
        "open_trade": None,
        "closed_trade_ids": [],
    }
    try:
        if Config.STATE_FILE.exists():
            saved = json.loads(Config.STATE_FILE.read_text(encoding="utf-8"))
            defaults.update(saved)
    except (json.JSONDecodeError, OSError):
        pass
    return defaults


def save_state(state: dict) -> None:
    Config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    Config.STATE_FILE.write_text(json.dumps(state, indent=2, default=_json_default), encoding="utf-8")


def append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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
    log(f"Connected MT5 | account={account.login} server={account.server} symbol={Config.SYMBOL}")


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
        return {
            "bid": None,
            "ask": None,
            "spread": None,
            "spread_points": None,
            "point": None,
            "digits": None,
        }
    spread = float(tick.ask - tick.bid)
    return {
        "bid": float(tick.bid),
        "ask": float(tick.ask),
        "spread": spread,
        "spread_points": spread / float(info.point),
        "point": float(info.point),
        "digits": int(info.digits),
    }


def m15_context_snapshot(m15: pd.DataFrame) -> dict:
    if len(m15) < 2:
        return {"m15_time": "", "m15_close": None, "m15_ema20": None, "m15_ema50": None}
    closed = m15.iloc[:-1].copy()
    close = closed["close"]
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    last = closed.iloc[-1]
    return {
        "m15_time": closed.index[-1].isoformat(),
        "m15_close": float(last["close"]),
        "m15_ema20": float(ema20.iloc[-1]),
        "m15_ema50": float(ema50.iloc[-1]),
        "m15_trend": "DOWN" if ema20.iloc[-1] < ema50.iloc[-1] else "UP",
    }


def today_utc() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def reset_daily_counter_if_needed(state: dict) -> None:
    today = today_utc()
    if state.get("signals_today_date") != today:
        state["signals_today_date"] = today
        state["signals_today_count"] = 0


def reject_row(candle, reason: str, spread: dict, context: dict, category: str = "reject") -> dict:
    return {
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "category": category,
        "symbol": Config.SYMBOL,
        "candle_time": candle.name.isoformat(),
        "reason": reason,
        "open": float(candle["open"]),
        "high": float(candle["high"]),
        "low": float(candle["low"]),
        "close": float(candle["close"]),
        "body": float(candle.get("body", 0)),
        "avg_body": float(candle.get("avg_body", 0)),
        "wick_pct": float(candle.get("wick_pct", 0)),
        "atr": float(candle.get("atr", 0)),
        "atr_percentile": float(candle.get("atr_percentile_100", 0)),
        "donchian_low": float(candle.get("donchian_low", 0)),
        "spread_points": spread.get("spread_points"),
        **context,
    }


def signal_row(candle, levels: dict, spread: dict, context: dict, trade_id: str) -> dict:
    spread_cost = float(spread["ask"] - spread["bid"]) if spread["ask"] is not None else 0.0
    return {
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "trade_id": trade_id,
        "symbol": Config.SYMBOL,
        "side": "SELL",
        "candle_time": candle.name.isoformat(),
        "entry_time": datetime.now(timezone.utc).isoformat(),
        "expected_entry": spread.get("bid"),
        "simulated_fill": levels["entry"],
        "sl": levels["sl"],
        "tp": levels["tp"],
        "risk": levels["risk"],
        "rr": levels["rr"],
        "r_size": levels["risk"],
        "bid": spread.get("bid"),
        "ask": spread.get("ask"),
        "spread": spread.get("spread"),
        "spread_points": spread.get("spread_points"),
        "spread_cost": spread_cost,
        "body": float(candle.get("body", 0)),
        "avg_body": float(candle.get("avg_body", 0)),
        "wick_pct": float(candle.get("wick_pct", 0)),
        "atr_percentile": float(candle.get("atr_percentile_100", 0)),
        "donchian_low": float(candle.get("donchian_low", 0)),
        **context,
    }


def open_trade_from_signal(row: dict) -> dict:
    return {
        "trade_id": row["trade_id"],
        "symbol": row["symbol"],
        "side": row["side"],
        "signal_candle": row["candle_time"],
        "entry_time": row["entry_time"],
        "entry": row["simulated_fill"],
        "sl": row["sl"],
        "tp": row["tp"],
        "risk": row["risk"],
        "rr": row["rr"],
        "status": "OPEN",
        "last_checked_at": row["entry_time"],
    }


def update_open_trade(state: dict, m1: pd.DataFrame, m5: pd.DataFrame) -> None:
    trade = state.get("open_trade")
    if not trade:
        return
    bars = m1 if not m1.empty else m5
    entry_time = pd.Timestamp(trade["entry_time"])
    result, exit_time, exit_price = classify_exit_since(
        bars,
        entry_time,
        float(trade["sl"]),
        float(trade["tp"]),
    )
    trade["last_checked_at"] = datetime.now(timezone.utc).isoformat()
    if result == "OPEN":
        state["open_trade"] = trade
        return
    risk = float(trade["risk"])
    entry = float(trade["entry"])
    pnl_r = (entry - float(exit_price)) / risk if risk else 0.0
    closed = {
        **trade,
        "status": result,
        "exit_time": exit_time.isoformat() if exit_time is not None else "",
        "exit_price": exit_price,
        "pnl_r": pnl_r,
        "closed_at": datetime.now(timezone.utc).isoformat(),
    }
    append_csv(Config.OUT_DIR / "sim_trades.csv", closed)
    state.setdefault("closed_trade_ids", []).append(trade["trade_id"])
    state["open_trade"] = None
    log(f"Sim trade closed | {trade['trade_id']} result={result} pnl_r={pnl_r:.2f}")


def refresh_daily_summary() -> None:
    paths = {
        "signals": Config.OUT_DIR / "signals.csv",
        "rejects": Config.OUT_DIR / "rejects.csv",
        "trades": Config.OUT_DIR / "sim_trades.csv",
    }
    rows = []
    days = set()
    frames = {}
    for key, path in paths.items():
        if path.exists() and path.stat().st_size > 0:
            frames[key] = pd.read_csv(path)
            date_col = "logged_at" if key != "trades" else "closed_at"
            if date_col in frames[key]:
                days.update(pd.to_datetime(frames[key][date_col], errors="coerce", utc=True).dt.date.dropna())
        else:
            frames[key] = pd.DataFrame()
    for day in sorted(days):
        day_s = day.isoformat()
        signals = frames["signals"]
        rejects = frames["rejects"]
        trades = frames["trades"]
        sig_day = signals[pd.to_datetime(signals.get("logged_at", pd.Series(dtype=str)), errors="coerce", utc=True).dt.date == day] if not signals.empty else signals
        rej_day = rejects[pd.to_datetime(rejects.get("logged_at", pd.Series(dtype=str)), errors="coerce", utc=True).dt.date == day] if not rejects.empty else rejects
        tr_day = trades[pd.to_datetime(trades.get("closed_at", pd.Series(dtype=str)), errors="coerce", utc=True).dt.date == day] if not trades.empty else trades
        rows.append(
            {
                "date": day_s,
                "signals": len(sig_day),
                "rejects": len(rej_day),
                "closed_trades": len(tr_day),
                "tp": int((tr_day.get("status", pd.Series(dtype=str)) == "TP").sum()) if not tr_day.empty else 0,
                "sl": int((tr_day.get("status", pd.Series(dtype=str)) == "SL").sum()) if not tr_day.empty else 0,
                "pnl_r": float(tr_day.get("pnl_r", pd.Series(dtype=float)).sum()) if not tr_day.empty else 0.0,
            }
        )
    write_csv(Config.OUT_DIR / "daily_summary.csv", rows)


def process_once(state: dict) -> None:
    reset_daily_counter_if_needed(state)
    m5 = get_ohlcv(mt5_timeframe("M5"), Config.BARS_M5)
    m1 = get_ohlcv(mt5_timeframe("M1"), Config.BARS_M1)
    m15 = get_ohlcv(mt5_timeframe("M15"), 200)
    prepared = prepare_m5(m5, STRATEGY)
    candle = latest_closed_m5(prepared)
    spread = spread_snapshot()
    context = m15_context_snapshot(m15)
    inside = is_in_session(candle.name, STRATEGY)

    update_open_trade(state, m1, m5)
    candle_key = candle.name.isoformat()
    latest = (
        f"latest_closed={candle_key} close={float(candle['close']):.3f} "
        f"spread_pts={spread.get('spread_points')} inside_session={inside}"
    )

    if state.get("last_checked_candle") == candle_key:
        open_status = "open_trade=yes" if state.get("open_trade") else "open_trade=no"
        log(f"Status | {latest} | same_candle | {open_status}")
        return
    state["last_checked_candle"] = candle_key

    signal, reason = evaluate_sell_signal(candle, STRATEGY)
    if signal is None:
        append_csv(Config.OUT_DIR / "rejects.csv", reject_row(candle, reason, spread, context))
        log(f"Reject | {latest} | m15_trend={context.get('m15_trend')} | reason={reason}")
        return

    if state.get("open_trade"):
        append_csv(Config.OUT_DIR / "rejects.csv", reject_row(candle, "missed_open_sim_trade", spread, context, "missed_trade"))
        log(f"Missed | {latest} | reason=open_sim_trade")
        return
    if state.get("signals_today_count", 0) >= STRATEGY.max_signals_per_day:
        append_csv(Config.OUT_DIR / "rejects.csv", reject_row(candle, "missed_daily_signal_limit", spread, context, "missed_trade"))
        log(f"Missed | {latest} | reason=daily_signal_limit")
        return
    if state.get("last_signal_candle") == candle_key:
        append_csv(Config.OUT_DIR / "rejects.csv", reject_row(candle, "duplicate_signal_candle", spread, context, "missed_trade"))
        log(f"Missed | {latest} | reason=duplicate_signal_candle")
        return
    if spread.get("bid") is None:
        append_csv(Config.OUT_DIR / "rejects.csv", reject_row(candle, "missing_bid_ask", spread, context))
        log(f"Reject | {latest} | reason=missing_bid_ask")
        return

    levels, level_reason = sell_levels(candle, float(spread["bid"]), STRATEGY)
    if levels is None:
        append_csv(Config.OUT_DIR / "rejects.csv", reject_row(candle, level_reason, spread, context))
        log(f"Reject | {latest} | reason={level_reason}")
        return

    trade_id = f"{Config.SYMBOL}_{candle.name:%Y%m%d_%H%M}"
    row = signal_row(candle, levels, spread, context, trade_id)
    append_csv(Config.OUT_DIR / "signals.csv", row)
    state["open_trade"] = open_trade_from_signal(row)
    state["last_signal_candle"] = candle_key
    state["signals_today_count"] = int(state.get("signals_today_count", 0)) + 1
    log(
        "SIGNAL | "
        f"{latest} | SELL fill={levels['entry']:.3f} sl={levels['sl']:.3f} "
        f"tp={levels['tp']:.3f} risk={levels['risk']:.3f}"
    )


def run(once: bool = False) -> None:
    Config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    Config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    log("Gold V2 SELL-only dry-run validator starting")
    log("DRY_RUN only: this script never sends real orders")
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
                refresh_daily_summary()
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
    run(once=args.once)
