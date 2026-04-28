import csv
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "backtests"))

import backtest_multi_tf_lorentzian_andean as strategy


@dataclass
class LiveConfig:
    base_url: str = os.getenv("BINANCE_FAPI_BASE_URL", "https://demo-fapi.binance.com")
    symbols: tuple[str, ...] = tuple(os.getenv("DRY_RUN_SYMBOLS", "BTCUSDT").split(","))
    poll_seconds: int = int(os.getenv("DRY_RUN_POLL_SECONDS", "30"))
    kline_limit: int = int(os.getenv("DRY_RUN_KLINE_LIMIT", "1200"))
    initial_balance: float = float(os.getenv("DRY_RUN_BALANCE", "1000"))
    risk_per_trade: float = 0.005
    rr: float = 2.0
    max_m1_confirm_candles: int = 5
    max_trades_per_day: int = 2
    log_dir: Path = Path("logs")
    state_file: Path = Path("logs/dry_run_donchian_state.json")
    signal_csv: Path = Path("logs/dry_run_signals.csv")
    trade_csv: Path = Path("logs/dry_run_trades.csv")
    daily_csv: Path = Path("logs/dry_run_daily_summary.csv")


def log(msg: str):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


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
        bid = float(data["bidPrice"])
        ask = float(data["askPrice"])
        mid = (bid + ask) / 2
        spread = ask - bid
        return {
            "bid": bid,
            "ask": ask,
            "mid": mid,
            "spread": spread,
            "spread_pct": spread / mid if mid else 0.0,
        }


class CsvStore:
    @staticmethod
    def append(path: Path, row: dict):
        path.parent.mkdir(parents=True, exist_ok=True)
        exists = path.exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    @staticmethod
    def rewrite(path: Path, rows: list[dict]):
        path.parent.mkdir(parents=True, exist_ok=True)
        if not rows:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


SIGNAL_FIELDS = [
    "signal_id",
    "symbol",
    "signal_time",
    "side",
    "status",
    "spread",
    "spread_pct",
    "estimated_slippage",
    "entry",
    "sl",
    "tp",
    "backtest_entry",
    "backtest_sl",
    "backtest_tp",
]

TRADE_FIELDS = [
    "signal_id",
    "symbol",
    "side",
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
    "backtest_entry",
    "backtest_sl",
    "backtest_tp",
    "live_vs_backtest_entry_diff",
]

DAILY_FIELDS = [
    "date",
    "signals",
    "wins",
    "losses",
    "estimated_pnl",
    "avg_spread",
    "avg_spread_pct",
    "avg_slippage",
    "missed_trades",
]


def ensure_csv(path: Path, fields: list[str]):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()


def ensure_output_files(config: LiveConfig):
    ensure_csv(config.signal_csv, SIGNAL_FIELDS)
    ensure_csv(config.trade_csv, TRADE_FIELDS)
    ensure_csv(config.daily_csv, DAILY_FIELDS)


def strategy_config(symbol: str, live: LiveConfig) -> strategy.Config:
    return strategy.Config(
        initial_balance=live.initial_balance,
        ema_fast=50,
        ema_slow=200,
        lorentzian_k=8,
        lorentzian_horizon=8,
        andean_length=50,
        min_atr_pct=0.002,
        risk_per_trade=live.risk_per_trade,
        rr=live.rr,
        exit_mode="fixed",
        m15_adx_min=20,
        require_m15_adx_rising=True,
        use_donchian=True,
        donchian_n=20,
        regime_name=f"{symbol}_dry_run_donchian_20",
        max_m1_confirm_candles=live.max_m1_confirm_candles,
    )


def load_state(config: LiveConfig) -> dict:
    if config.state_file.exists():
        with open(config.state_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"open_trades": [], "closed_trade_ids": [], "processed_m5": {}, "pending_signals": [], "balance": config.initial_balance}


def save_state(config: LiveConfig, state: dict):
    config.state_file.parent.mkdir(parents=True, exist_ok=True)
    with open(config.state_file, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, default=str)


def prepare_market_data(api: BinancePublic, symbol: str, live: LiveConfig, cfg: strategy.Config):
    m1 = api.klines(symbol, "1m", live.kline_limit)
    m5 = api.klines(symbol, "5m", live.kline_limit)
    m15 = api.klines(symbol, "15m", live.kline_limit)
    h1 = api.klines(symbol, "1h", min(live.kline_limit, 1000))
    m1, m5, m15 = strategy.calculate_indicators(m1, m5, m15, cfg)
    h1 = strategy.calculate_h1_indicators(h1)
    m5 = strategy.align_timeframes(m5, m15)
    m5 = strategy.align_h1_timeframe(m5, h1)
    m5 = strategy.calculate_lorentzian(m5, cfg)
    return m1, m5


def signal_side(row: pd.Series, cfg: strategy.Config) -> Optional[str]:
    return strategy.generate_signals(row, cfg)


def confirm_entry(m1: pd.DataFrame, signal_time: pd.Timestamp, side: str, cfg: strategy.Config):
    return strategy.confirm_m1(m1, signal_time, side, cfg)


def theoretical_trade(row: pd.Series, entry_price: float, side: str, cfg: strategy.Config, spread: dict):
    atr = float(row["atr"])
    half_spread = spread["spread"] / 2
    estimated_slippage = atr * cfg.slippage_atr_mult + half_spread
    live_entry = entry_price + estimated_slippage if side == "BUY" else entry_price - estimated_slippage
    backtest_entry = entry_price + atr * cfg.slippage_atr_mult if side == "BUY" else entry_price - atr * cfg.slippage_atr_mult

    if side == "BUY":
        swing_sl = float(row["swing_low"]) if not pd.isna(row["swing_low"]) else live_entry - atr
        live_sl = min(live_entry - atr, swing_sl)
        live_risk = abs(live_entry - live_sl)
        live_tp = live_entry + live_risk * cfg.rr
        bt_sl = min(backtest_entry - atr, swing_sl)
        bt_risk = abs(backtest_entry - bt_sl)
        bt_tp = backtest_entry + bt_risk * cfg.rr
    else:
        swing_sl = float(row["swing_high"]) if not pd.isna(row["swing_high"]) else live_entry + atr
        live_sl = max(live_entry + atr, swing_sl)
        live_risk = abs(live_entry - live_sl)
        live_tp = live_entry - live_risk * cfg.rr
        bt_sl = max(backtest_entry + atr, swing_sl)
        bt_risk = abs(backtest_entry - bt_sl)
        bt_tp = backtest_entry - bt_risk * cfg.rr

    risk_amount = cfg.initial_balance * cfg.risk_per_trade
    qty = risk_amount / live_risk if live_risk > 0 else 0.0
    return {
        "entry": live_entry,
        "sl": live_sl,
        "tp": live_tp,
        "qty": qty,
        "risk": live_risk,
        "estimated_slippage": estimated_slippage,
        "backtest_entry": backtest_entry,
        "backtest_sl": bt_sl,
        "backtest_tp": bt_tp,
    }


def check_trade_exit(trade: dict, m1: pd.DataFrame):
    entry_time = pd.Timestamp(trade["entry_time"])
    future = m1[m1.index > entry_time]
    for ts, row in future.iterrows():
        if ts <= pd.Timestamp(trade.get("last_checked_time", trade["entry_time"])):
            continue
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
    if not future.empty:
        trade["last_checked_time"] = str(future.index[-1])
    return None, None, None


def pnl_for_exit(trade: dict, exit_price: float) -> float:
    gross = (exit_price - trade["entry"]) * trade["qty"]
    if trade["side"] == "SELL":
        gross = -gross
    return gross


def update_daily_summary(config: LiveConfig):
    if not config.trade_csv.exists() and not config.signal_csv.exists():
        return
    trades = pd.read_csv(config.trade_csv) if config.trade_csv.exists() else pd.DataFrame()
    signals = pd.read_csv(config.signal_csv) if config.signal_csv.exists() else pd.DataFrame()
    rows = []
    days = sorted(set(pd.to_datetime(signals.get("signal_time", pd.Series(dtype=str)), errors="coerce").dt.date.dropna()))
    days += [d for d in pd.to_datetime(trades.get("exit_time", pd.Series(dtype=str)), errors="coerce").dt.date.dropna() if d not in days]
    for day in sorted(set(days)):
        day_signals = signals[pd.to_datetime(signals["signal_time"], errors="coerce").dt.date == day] if not signals.empty else pd.DataFrame()
        day_trades = trades[pd.to_datetime(trades["exit_time"], errors="coerce").dt.date == day] if not trades.empty else pd.DataFrame()
        rows.append(
            {
                "date": day,
                "signals": len(day_signals),
                "wins": int((day_trades.get("result", pd.Series(dtype=str)) == "WIN").sum()),
                "losses": int((day_trades.get("result", pd.Series(dtype=str)) == "LOSS").sum()),
                "estimated_pnl": float(day_trades.get("pnl", pd.Series(dtype=float)).sum()) if not day_trades.empty else 0.0,
                "avg_spread": float(day_signals.get("spread", pd.Series(dtype=float)).mean()) if not day_signals.empty else 0.0,
                "avg_spread_pct": float(day_signals.get("spread_pct", pd.Series(dtype=float)).mean()) if not day_signals.empty else 0.0,
                "avg_slippage": float(day_signals.get("estimated_slippage", pd.Series(dtype=float)).mean()) if not day_signals.empty else 0.0,
                "missed_trades": int((day_signals.get("status", pd.Series(dtype=str)) == "MISSED_NO_M1_CONFIRM").sum()) if not day_signals.empty else 0,
            }
        )
    CsvStore.rewrite(config.daily_csv, rows)


def process_symbol(api: BinancePublic, config: LiveConfig, state: dict, symbol: str):
    cfg = strategy_config(symbol, config)
    m1, m5 = prepare_market_data(api, symbol, config, cfg)
    if len(m5) < 500 or len(m1) < 100:
        log(f"⚠️ {symbol} not enough closed candles m1={len(m1)} m5={len(m5)}")
        return

    spread = api.book_ticker(symbol)
    if len(m5) < 3:
        log(f"⚠️ {symbol} not enough M5 candles after filtering")
        return

    latest_time = m5.index[-2]
    signal_close_time = latest_time + pd.Timedelta(minutes=5)
    processed_key = str(latest_time)
    if state["processed_m5"].get(symbol) == processed_key:
        update_open_trades(config, state, symbol, m1)
        return

    row = m5.loc[latest_time]
    side = signal_side(row, cfg)
    state["processed_m5"][symbol] = processed_key
    update_open_trades(config, state, symbol, m1)

    if side is None:
        log(f"🕯️ {symbol} {signal_close_time} closed: no signal")
        return

    entry_time, entry_price = confirm_entry(m1, signal_close_time, side, cfg)
    signal_id = f"{symbol}-{signal_close_time.isoformat()}-{side}"
    if entry_time is None:
        CsvStore.append(
            config.signal_csv,
            {
                "signal_id": signal_id,
                "symbol": symbol,
                "signal_time": signal_close_time,
                "side": side,
                "status": "MISSED_NO_M1_CONFIRM",
                "spread": spread["spread"],
                "spread_pct": spread["spread_pct"],
                "estimated_slippage": 0.0,
                "entry": "",
                "sl": "",
                "tp": "",
                "backtest_entry": "",
                "backtest_sl": "",
                "backtest_tp": "",
            },
        )
        log(f"⏭️ {symbol} {side} missed: no M1 confirmation")
        return

    trade = theoretical_trade(row, entry_price, side, cfg, spread)
    open_trade = {
        "signal_id": signal_id,
        "symbol": symbol,
        "side": side,
        "signal_time": str(signal_close_time),
        "entry_time": str(entry_time),
        **trade,
        "spread": spread["spread"],
        "spread_pct": spread["spread_pct"],
        "last_checked_time": str(entry_time),
    }
    state["open_trades"].append(open_trade)
    CsvStore.append(
        config.signal_csv,
        {
            "signal_id": signal_id,
            "symbol": symbol,
            "signal_time": signal_close_time,
            "side": side,
            "status": "OPENED_DRY_RUN",
            "spread": spread["spread"],
            "spread_pct": spread["spread_pct"],
            "estimated_slippage": trade["estimated_slippage"],
            "entry": trade["entry"],
            "sl": trade["sl"],
            "tp": trade["tp"],
            "backtest_entry": trade["backtest_entry"],
            "backtest_sl": trade["backtest_sl"],
            "backtest_tp": trade["backtest_tp"],
        },
    )
    log(f"🧪 {symbol} {side} DRY_RUN entry={trade['entry']:.2f} sl={trade['sl']:.2f} tp={trade['tp']:.2f} spread={spread['spread']:.2f}")


def update_open_trades(config: LiveConfig, state: dict, symbol: str, m1: pd.DataFrame):
    still_open = []
    for trade in state["open_trades"]:
        if trade["symbol"] != symbol:
            still_open.append(trade)
            continue
        reason, exit_price, exit_time = check_trade_exit(trade, m1)
        if reason is None:
            still_open.append(trade)
            continue
        pnl = pnl_for_exit(trade, exit_price)
        result = "WIN" if pnl > 0 else "LOSS"
        CsvStore.append(
            config.trade_csv,
            {
                "signal_id": trade["signal_id"],
                "symbol": trade["symbol"],
                "side": trade["side"],
                "signal_time": trade["signal_time"],
                "entry_time": trade["entry_time"],
                "exit_time": exit_time,
                "entry": trade["entry"],
                "sl": trade["sl"],
                "tp": trade["tp"],
                "exit": exit_price,
                "qty": trade["qty"],
                "pnl": pnl,
                "result": result,
                "reason": reason,
                "spread": trade["spread"],
                "spread_pct": trade["spread_pct"],
                "estimated_slippage": trade["estimated_slippage"],
                "backtest_entry": trade["backtest_entry"],
                "backtest_sl": trade["backtest_sl"],
                "backtest_tp": trade["backtest_tp"],
                "live_vs_backtest_entry_diff": trade["entry"] - trade["backtest_entry"],
            },
        )
        log(f"🏁 {trade['symbol']} {trade['side']} {result} {reason} pnl={pnl:.2f}")
    state["open_trades"] = still_open


def main():
    config = LiveConfig()
    config.log_dir.mkdir(parents=True, exist_ok=True)
    ensure_output_files(config)
    api = BinancePublic(config.base_url)
    state = load_state(config)
    log(f"🧪 Donchian DRY_RUN live validator started symbols={','.join(config.symbols)} base={config.base_url}")
    log("🔒 Real orders are disabled in this module")

    while True:
        try:
            for symbol in config.symbols:
                process_symbol(api, config, state, symbol.strip().upper())
            update_daily_summary(config)
            save_state(config, state)
        except KeyboardInterrupt:
            save_state(config, state)
            log("⏹️ stopped by user")
            break
        except Exception as exc:
            log(f"💥 loop error: {exc}")
            save_state(config, state)
        time.sleep(config.poll_seconds)


if __name__ == "__main__":
    main()
