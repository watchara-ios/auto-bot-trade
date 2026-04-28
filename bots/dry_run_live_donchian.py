import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
LOG_DIR = ROOT_DIR / "logs"
sys.path.insert(0, str(ROOT_DIR / "backtests"))

import backtest_multi_tf_lorentzian_andean as bt


@dataclass
class DryRunConfig:
    symbol: str = os.getenv("DRY_RUN_SYMBOL", "BTCUSDT")
    base_url: str = os.getenv("BINANCE_FAPI_BASE_URL", "https://demo-fapi.binance.com")
    poll_seconds: int = int(os.getenv("DRY_RUN_POLL_SECONDS", "30"))
    kline_limit: int = int(os.getenv("DRY_RUN_KLINE_LIMIT", "1200"))
    initial_balance: float = float(os.getenv("DRY_RUN_BALANCE", "1000"))
    tier_a_risk: float = 0.0025
    tier_b_risk: float = 0.01
    rr: float = 2.0
    donchian_n: int = 20
    adx_min: float = 20.0
    max_trades_per_day: int = 2
    max_losses_per_day: int = 1
    max_m1_confirm_candles: int = 5
    latency_candles: int = 1
    swing_lookback: int = 8
    breakout_body_mult: float = 1.5
    breakout_atr_mult: float = 0.5
    max_wick_pct: float = 0.5
    close_quality_min: float = 0.6
    signal_csv: Path = LOG_DIR / "dry_run_signals.csv"
    trade_csv: Path = LOG_DIR / "dry_run_trades.csv"
    spread_csv: Path = LOG_DIR / "dry_run_spread_log.csv"
    daily_csv: Path = LOG_DIR / "dry_run_daily_summary.csv"
    state_file: Path = LOG_DIR / "dry_run_state.json"


SIGNAL_FIELDS = [
    "signal_id",
    "symbol",
    "signal_time",
    "side",
    "tier",
    "status",
    "spread",
    "spread_pct",
    "estimated_slippage",
    "entry",
    "sl",
    "tp",
    "rr",
    "adx",
    "adx_rising",
    "donchian_breakout",
    "bos_confirmed",
    "breakout_strength_score",
]

TRADE_FIELDS = [
    "signal_id",
    "symbol",
    "side",
    "tier",
    "signal_time",
    "entry_time",
    "exit_time",
    "entry",
    "sl",
    "tp",
    "exit",
    "qty",
    "pnl",
    "result",
    "reason",
    "spread",
    "spread_pct",
    "estimated_slippage",
    "latency_candles",
    "live_vs_theoretical_entry_diff",
]

SPREAD_FIELDS = ["time", "symbol", "bid", "ask", "mid", "spread", "spread_pct"]

DAILY_FIELDS = [
    "date",
    "signals",
    "tier_a_signals",
    "tier_b_signals",
    "wins",
    "losses",
    "estimated_pnl",
    "avg_spread",
    "max_spread",
    "avg_spread_pct",
    "avg_slippage",
    "missed_trades",
    "tier_a_pf",
    "tier_b_pf",
]


def log(message: str):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def append_csv(path: Path, row: dict, fields: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fields})


def rewrite_csv(path: Path, rows: list[dict], fields: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def ensure_files(config: DryRunConfig):
    for path, fields in [
        (config.signal_csv, SIGNAL_FIELDS),
        (config.trade_csv, TRADE_FIELDS),
        (config.spread_csv, SPREAD_FIELDS),
        (config.daily_csv, DAILY_FIELDS),
    ]:
        if path.exists():
            try:
                existing = pd.read_csv(path, nrows=0).columns.tolist()
                if any(field not in existing for field in fields):
                    backup = path.with_suffix(path.suffix + f".bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                    path.rename(backup)
                    log(f"🗂️ Rotated old log schema {path.name} -> {backup.name}")
                    rewrite_csv(path, [], fields)
            except Exception:
                backup = path.with_suffix(path.suffix + f".bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                path.rename(backup)
                rewrite_csv(path, [], fields)
        else:
            rewrite_csv(path, [], fields)


def load_state(config: DryRunConfig) -> dict:
    if config.state_file.exists():
        with open(config.state_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "balance": config.initial_balance,
        "processed_m5": {},
        "open_trade": None,
        "closed_signal_ids": [],
        "daily_trades": {},
        "daily_losses": {},
    }


def save_state(config: DryRunConfig, state: dict):
    config.state_file.parent.mkdir(parents=True, exist_ok=True)
    with open(config.state_file, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, default=str)


class BinancePublic:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def get(self, path: str, params=None):
        resp = requests.get(f"{self.base_url}{path}", params=params or {}, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def server_time_ms(self) -> int:
        try:
            return int(self.get("/fapi/v1/time")["serverTime"])
        except Exception:
            return int(time.time() * 1000)

    def klines(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        now_ms = self.server_time_ms()
        data = self.get("/fapi/v1/klines", {"symbol": symbol, "interval": interval, "limit": limit})
        df = pd.DataFrame(
            data,
            columns=[
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_asset_volume",
                "num_trades",
                "taker_buy_base",
                "taker_buy_quote",
                "ignore",
            ],
        )
        df = df[df["close_time"].astype(np.int64) < now_ms]
        df["time"] = pd.to_datetime(df["open_time"], unit="ms")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df[["time", "open", "high", "low", "close", "volume"]].dropna().set_index("time")

    def book_ticker(self, symbol: str) -> dict:
        data = self.get("/fapi/v1/ticker/bookTicker", {"symbol": symbol})
        return spread_from_bid_ask(symbol, float(data["bidPrice"]), float(data["askPrice"]), pd.Timestamp.utcnow())


def spread_from_bid_ask(symbol: str, bid: float, ask: float, ts) -> dict:
    mid = (bid + ask) / 2
    spread = ask - bid
    return {
        "time": ts,
        "symbol": symbol,
        "bid": bid,
        "ask": ask,
        "mid": mid,
        "spread": spread,
        "spread_pct": spread / mid if mid else 0.0,
    }


def synthetic_spread(symbol: str, row: pd.Series, ts) -> dict:
    mid = float(row["close"])
    spread = float(row["atr"] * 0.02) if not pd.isna(row.get("atr")) else mid * 0.0001
    return spread_from_bid_ask(symbol, mid - spread / 2, mid + spread / 2, ts)


def strategy_config(config: DryRunConfig) -> bt.Config:
    return bt.Config(
        initial_balance=config.initial_balance,
        ema_fast=50,
        ema_slow=200,
        atr_period=14,
        swing_lookback=config.swing_lookback,
        rr=config.rr,
        risk_per_trade=config.tier_a_risk,
        tier_b_risk=config.tier_b_risk,
        tier_mode="hybrid",
        exit_mode="fixed",
        m15_adx_min=config.adx_min,
        require_m15_adx_rising=True,
        use_donchian=True,
        donchian_n=config.donchian_n,
        breakout_body_mult=config.breakout_body_mult,
        breakout_atr_mult=config.breakout_atr_mult,
        breakout_wick_max_pct=config.max_wick_pct,
        breakout_close_location_min=config.close_quality_min,
        max_m1_confirm_candles=config.max_m1_confirm_candles,
        entry_latency_m1_candles=config.latency_candles,
        max_trades_per_day=config.max_trades_per_day,
        max_losses_per_day=config.max_losses_per_day,
    )


def prepare_data(m1: pd.DataFrame, m5: pd.DataFrame, m15: pd.DataFrame, cfg: bt.Config):
    m1, m5, m15 = bt.calculate_indicators(m1, m5, m15, cfg)
    m5 = bt.align_timeframes(m5, m15)
    return m1, m5


def load_csv(path: Path) -> pd.DataFrame:
    return bt.load_csv(str(path))


def m15_trend_ok(row: pd.Series, side: str, cfg: bt.Config) -> bool:
    if pd.isna(row.get("m15_trend")):
        return False
    return (side == "BUY" and row["m15_trend"] == 1) or (side == "SELL" and row["m15_trend"] == -1)


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


def high_quality(row: pd.Series, side: str, cfg: bt.Config) -> bool:
    return bt.high_quality_breakout(row, side, cfg)


def signal_from_row(row: pd.Series, cfg: bt.Config) -> Optional[dict]:
    side = donchian_side(row, cfg)
    if side is None:
        return None
    if row.get("m15_adx", 0) < cfg.m15_adx_min or not bool(row.get("m15_adx_rising", False)):
        return None
    if not m15_trend_ok(row, side, cfg):
        return None
    if not bos_confirmed(row, side):
        return None
    tier = "B" if high_quality(row, side, cfg) else "A"
    return {
        "side": side,
        "tier": tier,
        "adx": float(row["m15_adx"]),
        "adx_rising": bool(row["m15_adx_rising"]),
        "donchian_breakout": True,
        "bos_confirmed": True,
        "breakout_strength_score": breakout_strength_score(row, side, cfg),
    }


def confirm_entry(m1: pd.DataFrame, signal_time: pd.Timestamp, side: str, cfg: bt.Config):
    return bt.confirm_m1(m1, signal_time, side, cfg)


def build_trade(config: DryRunConfig, cfg: bt.Config, row: pd.Series, signal: dict, signal_time, entry_time, theoretical_entry, spread: dict, balance: float):
    side = signal["side"]
    half_spread = spread["spread"] / 2
    live_entry = spread["ask"] if side == "BUY" else spread["bid"]
    live_vs_theoretical = live_entry - theoretical_entry
    estimated_slippage = abs(live_vs_theoretical)
    atr_value = float(row["atr"])
    if side == "BUY":
        swing_sl = float(row["swing_low"]) if not pd.isna(row.get("swing_low")) else live_entry - atr_value
        sl = min(live_entry - atr_value, swing_sl)
        risk = abs(live_entry - sl)
        tp = live_entry + risk * config.rr
    else:
        swing_sl = float(row["swing_high"]) if not pd.isna(row.get("swing_high")) else live_entry + atr_value
        sl = max(live_entry + atr_value, swing_sl)
        risk = abs(live_entry - sl)
        tp = live_entry - risk * config.rr
    risk_pct = config.tier_b_risk if signal["tier"] == "B" else config.tier_a_risk
    qty = (balance * risk_pct) / risk if risk > 0 else 0.0
    signal_id = f"{config.symbol}-{pd.Timestamp(signal_time).isoformat()}-{side}"
    return {
        "signal_id": signal_id,
        "symbol": config.symbol,
        "side": side,
        "tier": signal["tier"],
        "signal_time": str(signal_time),
        "entry_time": str(entry_time),
        "entry": live_entry,
        "sl": sl,
        "tp": tp,
        "qty": qty,
        "spread": spread["spread"],
        "spread_pct": spread["spread_pct"],
        "estimated_slippage": estimated_slippage if estimated_slippage > 0 else half_spread,
        "latency_candles": config.latency_candles,
        "live_vs_theoretical_entry_diff": live_vs_theoretical,
        "last_checked_time": str(entry_time),
    }


def can_trade_today(config: DryRunConfig, state: dict, day: str) -> bool:
    if state.get("open_trade"):
        return False
    if state.get("daily_trades", {}).get(day, 0) >= config.max_trades_per_day:
        return False
    if state.get("daily_losses", {}).get(day, 0) >= config.max_losses_per_day:
        return False
    return True


def log_signal(config: DryRunConfig, signal: dict, signal_id: str, signal_time, status: str, spread: dict, trade: Optional[dict], row: pd.Series):
    append_csv(
        config.signal_csv,
        {
            "signal_id": signal_id,
            "symbol": config.symbol,
            "signal_time": signal_time,
            "side": signal["side"],
            "tier": signal["tier"],
            "status": status,
            "spread": spread["spread"],
            "spread_pct": spread["spread_pct"],
            "estimated_slippage": trade.get("estimated_slippage", 0.0) if trade else 0.0,
            "entry": trade.get("entry", "") if trade else "",
            "sl": trade.get("sl", "") if trade else "",
            "tp": trade.get("tp", "") if trade else "",
            "rr": config.rr,
            "adx": signal["adx"],
            "adx_rising": signal["adx_rising"],
            "donchian_breakout": signal["donchian_breakout"],
            "bos_confirmed": signal["bos_confirmed"],
            "breakout_strength_score": signal["breakout_strength_score"],
        },
        SIGNAL_FIELDS,
    )


def check_exit(trade: dict, m1_window: pd.DataFrame):
    last_checked = pd.Timestamp(trade["last_checked_time"])
    for ts, row in m1_window[m1_window.index > last_checked].iterrows():
        if trade["side"] == "BUY":
            if row["low"] <= trade["sl"]:
                return "SL", float(trade["sl"]), ts
            if row["high"] >= trade["tp"]:
                return "TP", float(trade["tp"]), ts
        else:
            if row["high"] >= trade["sl"]:
                return "SL", float(trade["sl"]), ts
            if row["low"] <= trade["tp"]:
                return "TP", float(trade["tp"]), ts
    if not m1_window.empty:
        trade["last_checked_time"] = str(m1_window.index[-1])
    return None, None, None


def close_trade(config: DryRunConfig, state: dict, reason: str, exit_price: float, exit_time):
    trade = state["open_trade"]
    pnl = (exit_price - trade["entry"]) * trade["qty"]
    if trade["side"] == "SELL":
        pnl = -pnl
    result = "WIN" if pnl > 0 else "LOSS"
    append_csv(
        config.trade_csv,
        {
            **trade,
            "exit_time": exit_time,
            "exit": exit_price,
            "pnl": pnl,
            "result": result,
            "reason": reason,
        },
        TRADE_FIELDS,
    )
    state["balance"] = state.get("balance", config.initial_balance) + pnl
    day = str(pd.Timestamp(exit_time).date())
    if pnl <= 0:
        state.setdefault("daily_losses", {})[day] = state.setdefault("daily_losses", {}).get(day, 0) + 1
    state.setdefault("closed_signal_ids", []).append(trade["signal_id"])
    state["open_trade"] = None
    log(f"🏁 {trade['symbol']} {trade['side']} {trade['tier']} {result} {reason} pnl={pnl:.2f}")


def update_open_trade(config: DryRunConfig, state: dict, m1: pd.DataFrame, until_time=None):
    if not state.get("open_trade"):
        return
    window = m1 if until_time is None else m1[m1.index <= pd.Timestamp(until_time)]
    reason, exit_price, exit_time = check_exit(state["open_trade"], window)
    if reason:
        close_trade(config, state, reason, exit_price, exit_time)


def update_daily_summary(config: DryRunConfig):
    signals = pd.read_csv(config.signal_csv) if config.signal_csv.exists() else pd.DataFrame()
    trades = pd.read_csv(config.trade_csv) if config.trade_csv.exists() else pd.DataFrame()
    spreads = pd.read_csv(config.spread_csv) if config.spread_csv.exists() else pd.DataFrame()
    days = set()
    if not signals.empty:
        days.update(pd.to_datetime(signals["signal_time"], errors="coerce").dt.date.dropna())
    if not trades.empty:
        days.update(pd.to_datetime(trades["exit_time"], errors="coerce").dt.date.dropna())
    rows = []
    for day in sorted(days):
        day_signals = signals[pd.to_datetime(signals["signal_time"], errors="coerce").dt.date == day] if not signals.empty else pd.DataFrame()
        day_trades = trades[pd.to_datetime(trades["exit_time"], errors="coerce").dt.date == day] if not trades.empty else pd.DataFrame()
        day_spreads = spreads[pd.to_datetime(spreads["time"], errors="coerce").dt.date == day] if not spreads.empty else pd.DataFrame()
        tier_pf = {}
        for tier in ["A", "B"]:
            tier_trades = day_trades[day_trades["tier"] == tier] if not day_trades.empty and "tier" in day_trades.columns else pd.DataFrame()
            gp = tier_trades.loc[tier_trades["pnl"] > 0, "pnl"].sum() if not tier_trades.empty else 0.0
            gl = -tier_trades.loc[tier_trades["pnl"] <= 0, "pnl"].sum() if not tier_trades.empty else 0.0
            tier_pf[tier] = gp / gl if gl else 0.0
        rows.append(
            {
                "date": day,
                "signals": len(day_signals),
                "tier_a_signals": int((day_signals.get("tier", pd.Series(dtype=str)) == "A").sum()),
                "tier_b_signals": int((day_signals.get("tier", pd.Series(dtype=str)) == "B").sum()),
                "wins": int((day_trades.get("result", pd.Series(dtype=str)) == "WIN").sum()),
                "losses": int((day_trades.get("result", pd.Series(dtype=str)) == "LOSS").sum()),
                "estimated_pnl": float(day_trades.get("pnl", pd.Series(dtype=float)).sum()) if not day_trades.empty else 0.0,
                "avg_spread": float(day_spreads.get("spread", pd.Series(dtype=float)).mean()) if not day_spreads.empty else 0.0,
                "max_spread": float(day_spreads.get("spread", pd.Series(dtype=float)).max()) if not day_spreads.empty else 0.0,
                "avg_spread_pct": float(day_spreads.get("spread_pct", pd.Series(dtype=float)).mean()) if not day_spreads.empty else 0.0,
                "avg_slippage": float(day_signals.get("estimated_slippage", pd.Series(dtype=float)).mean()) if not day_signals.empty else 0.0,
                "missed_trades": int((day_signals.get("status", pd.Series(dtype=str)).astype(str).str.startswith("MISSED", na=False)).sum()) if not day_signals.empty else 0,
                "tier_a_pf": tier_pf["A"],
                "tier_b_pf": tier_pf["B"],
            }
        )
    rewrite_csv(config.daily_csv, rows, DAILY_FIELDS)


def process_closed_m5(config: DryRunConfig, state: dict, cfg: bt.Config, m1: pd.DataFrame, m5: pd.DataFrame, signal_time, spread: dict):
    row_time = pd.Timestamp(signal_time) - pd.Timedelta(minutes=5)
    if row_time not in m5.index:
        return
    processed_key = str(row_time)
    if state.setdefault("processed_m5", {}).get(config.symbol) == processed_key:
        return
    state["processed_m5"][config.symbol] = processed_key

    update_open_trade(config, state, m1, signal_time)
    row = m5.loc[row_time]
    append_csv(config.spread_csv, spread, SPREAD_FIELDS)
    signal = signal_from_row(row, cfg)
    if signal is None:
        return

    signal_id = f"{config.symbol}-{pd.Timestamp(signal_time).isoformat()}-{signal['side']}"
    if signal_id in state.get("closed_signal_ids", []):
        return
    day = str(pd.Timestamp(signal_time).date())
    if not can_trade_today(config, state, day):
        log_signal(config, signal, signal_id, signal_time, "MISSED_RISK_LIMIT_OR_OPEN_TRADE", spread, None, row)
        return

    entry_time, theoretical_entry = confirm_entry(m1, pd.Timestamp(signal_time), signal["side"], cfg)
    if entry_time is None:
        log_signal(config, signal, signal_id, signal_time, "MISSED_NO_M1_CONFIRM", spread, None, row)
        return

    trade = build_trade(config, cfg, row, signal, signal_time, entry_time, theoretical_entry, spread, state.get("balance", config.initial_balance))
    if trade["qty"] <= 0:
        log_signal(config, signal, signal_id, signal_time, "MISSED_INVALID_SIZE", spread, None, row)
        return

    state["open_trade"] = trade
    state.setdefault("daily_trades", {})[day] = state.setdefault("daily_trades", {}).get(day, 0) + 1
    log_signal(config, signal, signal_id, signal_time, "OPENED_DRY_RUN", spread, trade, row)
    log(f"🧪 {config.symbol} {signal['side']} Tier {signal['tier']} entry={trade['entry']:.2f} sl={trade['sl']:.2f} tp={trade['tp']:.2f} spread={spread['spread']:.2f}")


def live_dry_run(config: DryRunConfig):
    ensure_files(config)
    state = load_state(config)
    api = BinancePublic(config.base_url)
    cfg = strategy_config(config)
    log(f"🧪 Binance live DRY_RUN started symbol={config.symbol} base={config.base_url}")
    log("🔒 Real orders are disabled")
    while True:
        try:
            m1 = api.klines(config.symbol, "1m", config.kline_limit)
            m5 = api.klines(config.symbol, "5m", config.kline_limit)
            m15 = api.klines(config.symbol, "15m", config.kline_limit)
            m1, m5 = prepare_data(m1, m5, m15, cfg)
            if m5.empty:
                time.sleep(config.poll_seconds)
                continue
            latest_open = m5.index[-1]
            signal_time = latest_open + pd.Timedelta(minutes=5)
            spread = api.book_ticker(config.symbol)
            process_closed_m5(config, state, cfg, m1, m5, signal_time, spread)
            update_open_trade(config, state, m1)
            update_daily_summary(config)
            save_state(config, state)
        except KeyboardInterrupt:
            save_state(config, state)
            log("⏹️ stopped by user")
            break
        except Exception as exc:
            save_state(config, state)
            log(f"💥 loop error: {exc}")
        time.sleep(config.poll_seconds)


def speed_delay(speed: str) -> float:
    if speed == "max":
        return 0.0
    value = max(float(speed), 1.0)
    return 300.0 / value


def replay_dry_run(config: DryRunConfig, speed: str, max_candles: int = 0):
    ensure_files(config)
    state = load_state(config)
    cfg = strategy_config(config)
    m1 = load_csv(DATA_DIR / "bitcoin_365d_1m.csv")
    m5 = load_csv(DATA_DIR / "bitcoin_365d_5m.csv")
    m15 = load_csv(DATA_DIR / "bitcoin_365d_15m.csv")
    m1, m5 = prepare_data(m1, m5, m15, cfg)
    delay = speed_delay(speed)
    log(f"🎞️ Replay DRY_RUN started symbol={config.symbol} speed={speed}")
    for idx, row_time in enumerate(m5.index):
        if max_candles and idx >= max_candles:
            break
        signal_time = row_time + pd.Timedelta(minutes=5)
        current_m1 = m1[m1.index <= signal_time + pd.Timedelta(minutes=config.max_m1_confirm_candles + config.latency_candles)]
        current_m5 = m5[m5.index <= row_time]
        if current_m1.empty:
            continue
        spread = synthetic_spread(config.symbol, m5.loc[row_time], signal_time)
        process_closed_m5(config, state, cfg, current_m1, current_m5, signal_time, spread)
        if idx % 500 == 0:
            save_state(config, state)
        if delay > 0:
            time.sleep(delay)
    update_open_trade(config, state, m1)
    update_daily_summary(config)
    save_state(config, state)
    log("✅ replay finished")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["live_dry_run", "replay_dry_run"], default="live_dry_run")
    parser.add_argument("--speed", default="max", help="Replay speed: 1, 10, 100, or max")
    parser.add_argument("--symbol", default=os.getenv("DRY_RUN_SYMBOL", "BTCUSDT"))
    parser.add_argument("--reset-state", action="store_true")
    parser.add_argument("--max-candles", type=int, default=0, help="Replay only the first N M5 candles; 0 means all")
    return parser.parse_args()


def main():
    args = parse_args()
    config = DryRunConfig(symbol=args.symbol.upper())
    if args.reset_state and config.state_file.exists():
        config.state_file.unlink()
    if args.mode == "live_dry_run":
        live_dry_run(config)
    else:
        replay_dry_run(config, args.speed, args.max_candles)


if __name__ == "__main__":
    main()
