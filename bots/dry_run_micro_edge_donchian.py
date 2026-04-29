import argparse
import csv
import hashlib
import hmac
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs"
sys.path.insert(0, str(ROOT / "backtests"))
load_dotenv(ROOT / ".env")

from realistic_donchian_backtest import Config as BacktestConfig
from realistic_donchian_backtest import prepare


@dataclass
class DryRunConfig:
    symbol: str = os.getenv("MICRO_EDGE_SYMBOL", "BTCUSDT")
    base_url: str = (
        os.getenv("BINANCE_BASE_URL")
        or os.getenv("BINANCE_FAPI_BASE_URL")
        or "https://demo-fapi.binance.com"
    )
    api_key: str = os.getenv("BINANCE_API_KEY") or os.getenv("BINANCE2_API_KEY") or ""
    api_secret: str = os.getenv("BINANCE_SECRET") or os.getenv("BINANCE2_SECRET") or ""
    live_trading: bool = os.getenv("MICRO_EDGE_LIVE_TRADING", "false").lower() == "true"
    dry_run: bool = os.getenv(
        "MICRO_EDGE_DRY_RUN",
        "false" if os.getenv("MICRO_EDGE_LIVE_TRADING", "false").lower() == "true" else "true",
    ).lower() == "true"
    live_confirm: str = os.getenv("LIVE_CONFIRM") or os.getenv("MICRO_EDGE_CONFIRM_LIVE", "")
    leverage: int = int(os.getenv("MICRO_EDGE_LEVERAGE", "1"))
    poll_seconds: int = int(os.getenv("MICRO_EDGE_POLL_SECONDS", "30"))
    kline_limit: int = int(os.getenv("MICRO_EDGE_KLINE_LIMIT", "1200"))
    initial_balance: float = float(os.getenv("MICRO_EDGE_BALANCE", "1000"))
    max_trades_per_day: int = 2
    rr: float = 2.0
    tier_a_risk: float = float(os.getenv("MICRO_EDGE_TIER_A_RISK", "0.001"))
    tier_b_risk: float = float(os.getenv("MICRO_EDGE_TIER_B_RISK", "0.0025"))
    fee_rate: float = 0.0005
    session_hours_utc: tuple[int, ...] = (8, 9, 10, 11, 12)
    adx_min: float = 20.0
    adx_max: float = 30.0
    atr_percentile_min: float = 65.0
    max_spread_pct: float = float(os.getenv("MICRO_EDGE_MAX_SPREAD_PCT", "0.0005"))
    slippage_min_spread: float = 0.5
    slippage_max_spread: float = 1.5
    seed: int = 42
    signal_csv: Path = LOG_DIR / "live_signals.csv"
    trade_csv: Path = LOG_DIR / "live_trades.csv"
    missed_csv: Path = LOG_DIR / "live_missed_trades.csv"
    daily_summary_csv: Path = LOG_DIR / "daily_summary.csv"
    state_file: Path = LOG_DIR / "live_state.json"


SIGNAL_FIELDS = [
    "cycle_time",
    "utc_time",
    "symbol",
    "m5_time",
    "session_allowed",
    "adx",
    "adx_pass",
    "atr_percentile",
    "atr_pass",
    "donchian_high",
    "donchian_breakout",
    "bos_confirmed",
    "tier",
    "tier_score",
    "final_signal",
    "mode",
    "status",
    "block_reason",
    "bid",
    "ask",
    "spread",
    "spread_pct",
    "expected_entry",
    "expected_sl",
    "expected_tp",
    "expected_r",
    "actual_entry",
    "actual_sl",
    "actual_tp",
    "actual_r",
    "slippage",
    "quality_score",
    "live_entry_order_id",
    "live_sl_order_id",
    "live_tp_order_id",
    "rr",
]

TRADE_FIELDS = [
    "signal_id",
    "symbol",
    "side",
    "tier",
    "m5_time",
    "entry_time",
    "exit_time",
    "expected_entry",
    "expected_sl",
    "expected_tp",
    "expected_r",
    "actual_entry",
    "actual_sl",
    "actual_tp",
    "actual_r",
    "exit",
    "qty",
    "fee",
    "pnl",
    "pnl_r",
    "expected_r_result",
    "actual_r_result",
    "r_deviation_pct",
    "result",
    "reason",
    "bid",
    "ask",
    "spread",
    "spread_pct",
    "slippage",
    "mode",
    "live_entry_order_id",
    "live_sl_order_id",
    "live_tp_order_id",
]

MISSED_FIELDS = [
    "cycle_time",
    "utc_time",
    "symbol",
    "m5_time",
    "side",
    "tier",
    "status",
    "reason",
    "session_allowed",
    "adx",
    "atr_percentile",
    "donchian_breakout",
    "bos_confirmed",
    "bid",
    "ask",
    "spread",
    "spread_pct",
    "expected_entry",
    "expected_sl",
    "expected_tp",
    "expected_r",
    "quality_score",
]

DAILY_FIELDS = [
    "date",
    "total_signals",
    "executed_trades",
    "missed_trades",
    "wins",
    "losses",
    "avg_spread",
    "avg_slippage",
    "total_pnl",
    "total_pnl_r",
]


def log(message: str):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def append_csv(path: Path, row: dict, fields: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    if exists:
        try:
            existing = pd.read_csv(path, nrows=0).columns.tolist()
            if any(field not in existing for field in fields):
                backup = path.with_suffix(path.suffix + f".bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                path.rename(backup)
                exists = False
                log(f"Rotated old schema: {path.name} -> {backup.name}")
        except Exception:
            backup = path.with_suffix(path.suffix + f".bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            path.rename(backup)
            exists = False
            log(f"Rotated unreadable csv: {path.name} -> {backup.name}")

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fields})


def load_state(config: DryRunConfig) -> dict:
    if config.state_file.exists():
        with open(config.state_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "balance": config.initial_balance,
        "processed_m5": None,
        "open_trade": None,
        "daily_trades": {},
    }


def save_state(config: DryRunConfig, state: dict):
    config.state_file.parent.mkdir(parents=True, exist_ok=True)
    with open(config.state_file, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, default=str)


class BinancePublic:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def get(self, path: str, params=None):
        response = self.session.get(f"{self.base_url}{path}", params=params or {}, timeout=15)
        response.raise_for_status()
        return response.json()

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
                "trades",
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

    def signed(self, method: str, path: str, params=None, config: Optional[DryRunConfig] = None):
        if config is None or not config.api_key or not config.api_secret:
            raise RuntimeError("Missing BINANCE_API_KEY/BINANCE_SECRET for live trading")
        params = dict(params or {})
        params["timestamp"] = self.server_time_ms()
        params["recvWindow"] = 10000
        query = urlencode(params)
        signature = hmac.new(config.api_secret.encode(), query.encode(), hashlib.sha256).hexdigest()
        url = f"{self.base_url}{path}?{query}&signature={signature}"
        headers = {"X-MBX-APIKEY": config.api_key}
        if method == "GET":
            response = self.session.get(url, headers=headers, timeout=15)
        elif method == "POST":
            response = self.session.post(url, headers=headers, timeout=15)
        elif method == "DELETE":
            response = self.session.delete(url, headers=headers, timeout=15)
        else:
            raise ValueError(method)
        try:
            data = response.json()
        except Exception:
            data = {"raw": response.text}
        if response.status_code >= 400:
            raise RuntimeError(f"Binance error {response.status_code}: {data}")
        return data

    def exchange_info(self, symbol: str) -> dict:
        data = self.get("/fapi/v1/exchangeInfo")
        for item in data.get("symbols", []):
            if item.get("symbol") == symbol:
                return item
        raise RuntimeError(f"Symbol not found: {symbol}")

    @staticmethod
    def _filter(symbol_info: dict, filter_type: str) -> dict:
        for item in symbol_info.get("filters", []):
            if item.get("filterType") == filter_type:
                return item
        return {}

    @staticmethod
    def round_step(value: float, step: str) -> float:
        if not step:
            return value
        return float(Decimal(str(value)).quantize(Decimal(step), rounding=ROUND_DOWN))

    def normalize_qty(self, symbol: str, qty: float) -> float:
        info = self.exchange_info(symbol)
        lot_filter = self._filter(info, "LOT_SIZE")
        step = lot_filter.get("stepSize", "0.001")
        min_qty = float(lot_filter.get("minQty", 0))
        normalized = self.round_step(qty, step)
        if normalized < min_qty:
            raise RuntimeError(f"Quantity {normalized} below minQty {min_qty}")
        return normalized

    def min_qty(self, symbol: str) -> float:
        info = self.exchange_info(symbol)
        lot_filter = self._filter(info, "LOT_SIZE")
        return float(lot_filter.get("minQty", 0))

    def normalize_price(self, symbol: str, price: float) -> float:
        info = self.exchange_info(symbol)
        price_filter = self._filter(info, "PRICE_FILTER")
        tick = price_filter.get("tickSize", "0.10")
        return self.round_step(price, tick)

    def balance_usdt(self, config: DryRunConfig) -> float:
        data = self.signed("GET", "/fapi/v2/balance", {}, config)
        for item in data:
            if item.get("asset") == "USDT":
                return float(item.get("availableBalance") or item.get("balance") or 0)
        return config.initial_balance

    def position_amount(self, symbol: str, config: DryRunConfig) -> float:
        data = self.signed("GET", "/fapi/v2/positionRisk", {"symbol": symbol}, config)
        if isinstance(data, list):
            for item in data:
                if item.get("symbol") == symbol:
                    return float(item.get("positionAmt", 0))
        return 0.0

    def cancel_all_orders(self, symbol: str, config: DryRunConfig):
        return self.signed("DELETE", "/fapi/v1/allOpenOrders", {"symbol": symbol}, config)

    def set_leverage(self, symbol: str, config: DryRunConfig):
        return self.signed("POST", "/fapi/v1/leverage", {"symbol": symbol, "leverage": config.leverage}, config)

    def market_order(self, symbol: str, side: str, qty: float, config: DryRunConfig, reduce_only: bool = False):
        params = {
            "symbol": symbol,
            "side": side,
            "type": "MARKET",
            "quantity": qty,
            "newOrderRespType": "RESULT",
        }
        if reduce_only:
            params["reduceOnly"] = "true"
        return self.signed("POST", "/fapi/v1/order", params, config)

    def protective_order(self, symbol: str, side: str, order_type: str, stop_price: float, qty: float, config: DryRunConfig):
        trigger = self.normalize_price(symbol, stop_price)
        params = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "stopPrice": trigger,
            "quantity": qty,
            "reduceOnly": "true",
            "workingType": "MARK_PRICE",
        }
        try:
            return self.signed("POST", "/fapi/v1/order", params, config)
        except Exception as exc:
            if "-4120" not in str(exc):
                raise
            algo_params = {
                "algoType": "CONDITIONAL",
                "symbol": symbol,
                "side": side,
                "type": order_type,
                "triggerPrice": trigger,
                "quantity": qty,
                "reduceOnly": "true",
                "workingType": "MARK_PRICE",
            }
            return self.signed("POST", "/fapi/v1/algoOrder", algo_params, config)


def strategy_config(config: DryRunConfig) -> BacktestConfig:
    return BacktestConfig(
        use_volume_filter=True,
        volume_mult=1.2,
        use_atr_expansion=True,
        allowed_side="BUY",
        adx_min=config.adx_min,
        adx_max=config.adx_max,
        atr_percentile_min=config.atr_percentile_min,
        include_utc_hours=config.session_hours_utc,
        rr=config.rr,
        tier_a_risk=config.tier_a_risk,
        tier_b_risk=config.tier_b_risk,
        fee_rate=config.fee_rate,
    )


def tier_score(row: pd.Series, cfg: BacktestConfig) -> tuple[str, int]:
    score = 0
    avg_body = row.get("avg_body20", np.nan)
    if not pd.isna(avg_body) and avg_body > 0 and row.get("body", 0) > avg_body * cfg.breakout_body_mult:
        score += 1
    move_atr = (row["close"] - row["donchian_high"]) / row["atr"] if row.get("atr", 0) else np.nan
    if not pd.isna(move_atr) and move_atr > cfg.breakout_atr_mult:
        score += 1
    if row.get("wick_pct", 1) <= cfg.max_wick_pct:
        score += 1
    if row.get("close_location", 0) >= cfg.close_quality_min:
        score += 1
    return ("B" if score == 4 else "A"), score


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def signal_quality_score(evaluation: dict) -> int:
    adx = float(evaluation.get("adx", 0) or 0)
    atr_pct = float(evaluation.get("atr_percentile", 0) or 0)
    tier = evaluation.get("tier", "")
    tier_points = float(evaluation.get("tier_score", 0) or 0)

    adx_score = clamp(1 - abs(adx - 25.0) / 5.0, 0, 1) * 25
    atr_score = clamp((atr_pct - 65.0) / 35.0, 0, 1) * 25
    candle_score = clamp(tier_points / 4.0, 0, 1) * 30
    tier_bonus = 20 if tier == "B" else 10 if tier == "A" else 0
    return int(round(clamp(adx_score + atr_score + candle_score + tier_bonus, 0, 100)))


def evaluate_cycle(row: pd.Series, config: DryRunConfig, cfg: BacktestConfig) -> dict:
    session_allowed = row.name.hour in config.session_hours_utc
    adx = float(row.get("m15_adx", 0) or 0)
    adx_pass = config.adx_min < adx < config.adx_max
    atr_pct = float(row.get("atr_percentile_100", 0) or 0)
    atr_pass = atr_pct >= config.atr_percentile_min
    donchian_breakout = bool(row.get("close", 0) > row.get("donchian_high", np.inf))
    bos_confirmed = bool(row.get("high", 0) > row.get("swing_high", np.inf))
    volume_pass = bool(row.get("volume_ratio", 0) >= cfg.volume_mult)
    atr_expansion_pass = bool(row.get("atr", 0) > row.get("atr_sma50", np.inf))
    tier, score = tier_score(row, cfg) if donchian_breakout else ("", 0)

    checks = [
        (session_allowed, "outside 08-13 UTC session"),
        (adx_pass, "ADX not between 20 and 30"),
        (atr_pass, "ATR percentile below 65"),
        (donchian_breakout, "no M5 Donchian BUY breakout"),
        (bos_confirmed, "no bullish BOS"),
        (volume_pass, "volume ratio below 1.2"),
        (atr_expansion_pass, "ATR not above SMA50"),
    ]
    for ok, reason in checks:
        if not ok:
            return {
                "final_signal": False,
                "block_reason": reason,
                "tier": tier,
                "tier_score": score,
                "session_allowed": session_allowed,
                "adx": adx,
                "adx_pass": adx_pass,
                "atr_percentile": atr_pct,
                "atr_pass": atr_pass,
                "donchian_breakout": donchian_breakout,
                "bos_confirmed": bos_confirmed,
            }

    return {
        "final_signal": True,
        "block_reason": "",
        "tier": tier,
        "tier_score": score,
        "session_allowed": session_allowed,
        "adx": adx,
        "adx_pass": adx_pass,
        "atr_percentile": atr_pct,
        "atr_pass": atr_pass,
        "donchian_breakout": donchian_breakout,
        "bos_confirmed": bos_confirmed,
    }


def trade_levels(row: pd.Series, entry: float, tier: str, config: DryRunConfig, balance: Optional[float] = None):
    sl = min(entry - row["atr"], row["swing_low"])
    risk = entry - sl
    if not np.isfinite(risk) or risk <= 0:
        return None
    tp = entry + risk * config.rr
    risk_pct = config.tier_b_risk if tier == "B" else config.tier_a_risk
    risk_balance = balance if balance is not None and balance > 0 else config.initial_balance
    qty = (risk_balance * risk_pct) / risk
    return sl, tp, qty


def live_enabled(config: DryRunConfig) -> bool:
    return not config.dry_run


def is_mainnet(config: DryRunConfig) -> bool:
    return config.base_url.rstrip("/") == "https://fapi.binance.com"


def live_confirmed(config: DryRunConfig) -> bool:
    return str(config.live_confirm).lower() in {"true", "yes", "yes_i_understand"}


def response_order_id(response: dict) -> str:
    return str(response.get("orderId") or response.get("clientAlgoId") or response.get("algoId") or "")


def response_avg_price(response: dict, fallback: float) -> float:
    for key in ("avgPrice", "price"):
        try:
            value = float(response.get(key, 0))
            if value > 0:
                return value
        except Exception:
            pass
    return fallback


def summarize_daily(config: DryRunConfig):
    today = pd.Timestamp.now(tz="UTC").date().isoformat()
    rows = []
    if config.signal_csv.exists():
        signals = pd.read_csv(config.signal_csv)
        signals["date"] = pd.to_datetime(signals["cycle_time"], errors="coerce", utc=True).dt.date.astype(str)
        day_signals = signals[signals["date"] == today]
    else:
        day_signals = pd.DataFrame()

    if config.trade_csv.exists():
        trades = pd.read_csv(config.trade_csv)
        trades["date"] = pd.to_datetime(trades["entry_time"], errors="coerce", utc=True).dt.date.astype(str)
        day_trades = trades[trades["date"] == today]
    else:
        day_trades = pd.DataFrame()

    if config.missed_csv.exists():
        missed = pd.read_csv(config.missed_csv)
        missed["date"] = pd.to_datetime(missed["cycle_time"], errors="coerce", utc=True).dt.date.astype(str)
        day_missed = missed[missed["date"] == today]
    else:
        day_missed = pd.DataFrame()

    rows.append(
        {
            "date": today,
            "total_signals": int(day_signals["final_signal"].astype(str).str.lower().eq("true").sum())
            if not day_signals.empty and "final_signal" in day_signals
            else 0,
            "executed_trades": int(len(day_trades)),
            "missed_trades": int(len(day_missed)),
            "wins": int(day_trades["result"].astype(str).eq("WIN").sum()) if not day_trades.empty else 0,
            "losses": int(day_trades["result"].astype(str).eq("LOSS").sum()) if not day_trades.empty else 0,
            "avg_spread": float(day_signals["spread"].mean()) if not day_signals.empty else 0.0,
            "avg_slippage": float(day_trades["slippage"].mean()) if not day_trades.empty else 0.0,
            "total_pnl": float(day_trades["pnl"].sum()) if not day_trades.empty else 0.0,
            "total_pnl_r": float(day_trades["pnl_r"].sum()) if not day_trades.empty else 0.0,
        }
    )

    config.daily_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(config.daily_summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DAILY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def close_open_trade(config: DryRunConfig, state: dict, m1: pd.DataFrame):
    trade = state.get("open_trade")
    if not trade:
        return
    last_checked = pd.Timestamp(trade["last_checked_time"])
    for ts, row in m1[m1.index > last_checked].iterrows():
        exit_price = None
        reason = None
        if row["low"] <= trade["actual_sl"]:
            exit_price = trade["actual_sl"]
            reason = "SL"
        elif row["high"] >= trade["actual_tp"]:
            exit_price = trade["actual_tp"]
            reason = "TP"
        if reason:
            entry = trade["actual_entry"]
            gross = (exit_price - entry) * trade["qty"]
            fee = (entry * trade["qty"] + exit_price * trade["qty"]) * config.fee_rate
            pnl = gross - fee
            result = "WIN" if pnl > 0 else "LOSS"
            actual_r = float(trade.get("actual_r", 0) or 0)
            expected_r = float(trade.get("expected_r", 0) or 0)
            actual_r_result = (exit_price - entry) / actual_r if actual_r else 0.0
            expected_r_result = (exit_price - trade["expected_entry"]) / expected_r if expected_r else 0.0
            r_deviation_pct = (
                (actual_r_result - expected_r_result) / abs(expected_r_result) * 100
                if expected_r_result
                else 0.0
            )
            append_csv(
                config.trade_csv,
                {
                    **trade,
                    "exit_time": ts,
                    "exit": exit_price,
                    "fee": fee,
                    "pnl": pnl,
                    "pnl_r": actual_r_result,
                    "expected_r_result": expected_r_result,
                    "actual_r_result": actual_r_result,
                    "r_deviation_pct": r_deviation_pct,
                    "result": result,
                    "reason": reason,
                },
                TRADE_FIELDS,
            )
            state["balance"] = state.get("balance", config.initial_balance) + pnl
            state["open_trade"] = None
            log(f"Closed {trade.get('mode', 'DRY_RUN')} trade {result} {reason} pnl={pnl:.2f}")
            return
        trade["last_checked_time"] = str(ts)


def process_once(config: DryRunConfig, api: BinancePublic, state: dict, rng: np.random.Generator) -> bool:
    cfg = strategy_config(config)
    mode = "LIVE" if live_enabled(config) else "DRY_RUN"
    m1 = api.klines(config.symbol, "1m", config.kline_limit)
    m5_raw = api.klines(config.symbol, "5m", config.kline_limit)
    m15 = api.klines(config.symbol, "15m", config.kline_limit)
    m5 = prepare(m5_raw, m15, cfg)
    if m5.empty:
        return False
    close_open_trade(config, state, m1)

    row = m5.iloc[-1]
    m5_time = m5.index[-1]
    if state.get("processed_m5") == str(m5_time):
        return False
    state["processed_m5"] = str(m5_time)

    book = api.book_ticker(config.symbol)
    evaluation = evaluate_cycle(row, config, cfg)
    quality_score = signal_quality_score(evaluation)
    expected_entry = ""
    expected_sl = ""
    expected_tp = ""
    expected_r = ""
    actual_entry = ""
    actual_sl = ""
    actual_tp = ""
    actual_r = ""
    slippage = ""
    live_entry_order_id = ""
    live_sl_order_id = ""
    live_tp_order_id = ""
    status = "NO_SIGNAL"

    if evaluation["final_signal"]:
        day = str(m5_time.date())
        daily_trades = state.setdefault("daily_trades", {})
        account_balance = config.initial_balance
        if live_enabled(config):
            account_balance = api.balance_usdt(config)
        expected_entry = book["ask"]
        expected_levels = trade_levels(row, expected_entry, evaluation["tier"], config, account_balance)
        if expected_levels:
            expected_sl, expected_tp, _ = expected_levels
            expected_r = expected_entry - expected_sl
        if state.get("open_trade"):
            status = "BLOCKED_OPEN_TRADE"
        elif book["spread_pct"] > config.max_spread_pct:
            status = "HIGH_SPREAD"
        elif live_enabled(config) and abs(api.position_amount(config.symbol, config)) > 0:
            status = "BLOCKED_EXCHANGE_POSITION"
        elif daily_trades.get(day, 0) >= config.max_trades_per_day:
            status = "BLOCKED_DAILY_LIMIT"
        else:
            if live_enabled(config):
                actual_entry = expected_entry
                slippage = 0.0
            else:
                slippage = rng.uniform(config.slippage_min_spread, config.slippage_max_spread) * book["spread"]
                actual_entry = book["ask"] + slippage
            levels = trade_levels(row, actual_entry, evaluation["tier"], config, account_balance)
            if levels:
                actual_sl, actual_tp, raw_qty = levels
                actual_r = actual_entry - actual_sl
                try:
                    min_qty = api.min_qty(config.symbol)
                    if raw_qty <= min_qty:
                        raise RuntimeError(f"raw qty {raw_qty:.8f} <= minQty {min_qty}")
                    qty = api.normalize_qty(config.symbol, raw_qty) if live_enabled(config) else raw_qty
                except Exception as exc:
                    qty = 0
                    status = f"QTY_TOO_SMALL: {exc}"
                signal_id = f"{config.symbol}-{m5_time.isoformat()}-BUY"
                if qty > 0 and status == "NO_SIGNAL":
                    try:
                        if live_enabled(config):
                            entry_response = api.market_order(config.symbol, "BUY", qty, config)
                            live_entry_order_id = response_order_id(entry_response)
                            actual_entry = response_avg_price(entry_response, expected_entry)
                            slippage = actual_entry - expected_entry
                            live_actual_levels = trade_levels(row, actual_entry, evaluation["tier"], config, account_balance)
                            if not live_actual_levels:
                                raise RuntimeError("Invalid SL/TP after live fill")
                            actual_sl, actual_tp, qty = live_actual_levels[0], live_actual_levels[1], qty
                            actual_r = actual_entry - actual_sl
                            sl_response = api.protective_order(
                                config.symbol,
                                "SELL",
                                "STOP_MARKET",
                                actual_sl,
                                qty,
                                config,
                            )
                            tp_response = api.protective_order(
                                config.symbol,
                                "SELL",
                                "TAKE_PROFIT_MARKET",
                                actual_tp,
                                qty,
                                config,
                            )
                            live_sl_order_id = response_order_id(sl_response)
                            live_tp_order_id = response_order_id(tp_response)
                            status = "OPENED_LIVE"
                        else:
                            status = "OPENED_DRY_RUN"

                        state["open_trade"] = {
                            "signal_id": signal_id,
                            "symbol": config.symbol,
                            "side": "BUY",
                            "tier": evaluation["tier"],
                            "m5_time": str(m5_time),
                            "entry_time": str(pd.Timestamp.now(tz="UTC")),
                            "expected_entry": expected_entry,
                            "expected_sl": expected_sl,
                            "expected_tp": expected_tp,
                            "expected_r": expected_r,
                            "actual_entry": actual_entry,
                            "actual_sl": actual_sl,
                            "actual_tp": actual_tp,
                            "actual_r": actual_r,
                            "qty": qty,
                            "bid": book["bid"],
                            "ask": book["ask"],
                            "spread": book["spread"],
                            "spread_pct": book["spread_pct"],
                            "slippage": slippage,
                            "mode": mode,
                            "live_entry_order_id": live_entry_order_id,
                            "live_sl_order_id": live_sl_order_id,
                            "live_tp_order_id": live_tp_order_id,
                            "last_checked_time": str(m1.index[-1]),
                        }
                        daily_trades[day] = daily_trades.get(day, 0) + 1
                        log(
                            f"{mode} signal BUY Tier {evaluation['tier']} "
                            f"expected={expected_entry:.2f} actual={actual_entry:.2f} "
                            f"sl={actual_sl:.2f} tp={actual_tp:.2f} qty={qty} quality={quality_score}"
                        )
                    except Exception as exc:
                        status = f"LIVE_ORDER_FAILED: {exc}" if live_enabled(config) else f"OPEN_FAILED: {exc}"
                        if live_enabled(config):
                            try:
                                api.cancel_all_orders(config.symbol, config)
                                api.market_order(config.symbol, "SELL", qty, config, reduce_only=True)
                                status = f"{status}; EMERGENCY_CLOSE_SENT"
                            except Exception as close_exc:
                                status = f"{status}; EMERGENCY_CLOSE_FAILED: {close_exc}"
            else:
                status = "INVALID_SL_TP"
        if status not in ("OPENED_DRY_RUN", "OPENED_LIVE"):
            append_csv(
                config.missed_csv,
                {
                    "cycle_time": pd.Timestamp.now(tz="UTC"),
                    "utc_time": pd.Timestamp.now(tz="UTC").isoformat(),
                    "symbol": config.symbol,
                    "m5_time": m5_time,
                    "side": "BUY",
                    "tier": evaluation["tier"],
                    "status": status,
                    "reason": status,
                    "session_allowed": evaluation["session_allowed"],
                    "adx": evaluation["adx"],
                    "atr_percentile": evaluation["atr_percentile"],
                    "donchian_breakout": evaluation["donchian_breakout"],
                    "bos_confirmed": evaluation["bos_confirmed"],
                    "bid": book["bid"],
                    "ask": book["ask"],
                    "spread": book["spread"],
                    "spread_pct": book["spread_pct"],
                    "expected_entry": expected_entry,
                    "expected_sl": expected_sl,
                    "expected_tp": expected_tp,
                    "expected_r": expected_r,
                    "quality_score": quality_score,
                },
                MISSED_FIELDS,
            )

    append_csv(
        config.signal_csv,
        {
            "cycle_time": pd.Timestamp.now(tz="UTC"),
            "utc_time": pd.Timestamp.now(tz="UTC").isoformat(),
            "symbol": config.symbol,
            "m5_time": m5_time,
            "session_allowed": evaluation["session_allowed"],
            "adx": evaluation["adx"],
            "adx_pass": evaluation["adx_pass"],
            "atr_percentile": evaluation["atr_percentile"],
            "atr_pass": evaluation["atr_pass"],
            "donchian_high": row.get("donchian_high", ""),
            "donchian_breakout": evaluation["donchian_breakout"],
            "bos_confirmed": evaluation["bos_confirmed"],
            "tier": evaluation["tier"],
            "tier_score": evaluation["tier_score"],
            "final_signal": evaluation["final_signal"],
            "mode": mode,
            "status": status,
            "block_reason": evaluation["block_reason"],
            "bid": book["bid"],
            "ask": book["ask"],
            "spread": book["spread"],
            "spread_pct": book["spread_pct"],
            "expected_entry": expected_entry,
            "expected_sl": expected_sl,
            "expected_tp": expected_tp,
            "expected_r": expected_r,
            "actual_entry": actual_entry,
            "actual_sl": actual_sl,
            "actual_tp": actual_tp,
            "actual_r": actual_r,
            "slippage": slippage,
            "quality_score": quality_score,
            "live_entry_order_id": live_entry_order_id,
            "live_sl_order_id": live_sl_order_id,
            "live_tp_order_id": live_tp_order_id,
            "rr": config.rr,
        },
        SIGNAL_FIELDS,
    )
    summarize_daily(config)
    save_state(config, state)
    log(
        f"Cycle {m5_time} status={status} adx={evaluation['adx']:.2f} "
        f"atr_pct={evaluation['atr_percentile']:.1f} spread={book['spread']:.2f} "
        f"reason={evaluation['block_reason'] or '-'}"
    )
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Run one closed-candle cycle and exit")
    args = parser.parse_args()

    config = DryRunConfig()
    api = BinancePublic(config.base_url)
    state = load_state(config)
    rng = np.random.default_rng(config.seed)

    log(
        "Startup "
        f"DRY_RUN={config.dry_run} BASE_URL={config.base_url} LIVE_CONFIRM={config.live_confirm or '-'} "
        f"tier_a_risk={config.tier_a_risk:.4f} tier_b_risk={config.tier_b_risk:.4f} "
        f"session_hours_utc={','.join(map(str, config.session_hours_utc))} "
        f"max_spread_pct={config.max_spread_pct:.6f}"
    )

    if live_enabled(config):
        if is_mainnet(config) and not live_confirmed(config):
            raise RuntimeError("Mainnet live mode requires LIVE_CONFIRM=true")
        if not config.api_key or not config.api_secret:
            raise RuntimeError("LIVE mode requires BINANCE_API_KEY/BINANCE_SECRET in .env")
        api.set_leverage(config.symbol, config)
        log(f"Micro-edge Donchian LIVE started symbol={config.symbol} leverage={config.leverage}x")
        log("REAL ORDERS ENABLED. Market entry + reduce-only SL/TP will be sent on valid signals.")
    else:
        log(f"Micro-edge Donchian DRY_RUN started symbol={config.symbol}")
        log("No real orders will be placed.")

    while True:
        try:
            process_once(config, api, state, rng)
            save_state(config, state)
        except KeyboardInterrupt:
            save_state(config, state)
            log("Stopped by user")
            break
        except Exception as exc:
            save_state(config, state)
            log(f"Loop error: {exc}")
        if args.once:
            break
        time.sleep(config.poll_seconds)


if __name__ == "__main__":
    main()
