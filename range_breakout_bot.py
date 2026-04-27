import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


"""
BTC/ETH/SOL day-trading backtester.

Core rules:
- Entry timeframe = M5 closed candles.
- Trend timeframe = M15 closed candles, shifted before mapping to M5.
- Range breakout + retest remains the primary setup.
- Strong breakouts can enter directly when enabled.
- Only one open position is allowed across all symbols.
- Signals run only on closed M5 candles.
"""


@dataclass
class BotConfig:
    start_balance: float = 1000.0
    risk_per_trade: float = 0.005
    max_concurrent_positions: int = 1
    max_trades_per_day: int = 3
    max_consecutive_losses: int = 1
    daily_profit_target: float = 0.02
    max_daily_loss: float = 0.02

    sessions: tuple[tuple[str, int, int], ...] = (
        ("14-20", 14, 20),
        ("20-24", 20, 24),
        ("00-03", 0, 3),
    )
    timezone_offset_hours: int = 7

    trend_ema_fast: int = 20
    trend_ema_slow: int = 50
    min_range_candles: int = 12
    range_window_minutes: int = 120

    use_retest_entry: bool = True
    retest_tolerance_pct: float = 0.001
    max_retest_candles: int = 6
    allow_direct_breakout: bool = True
    direct_breakout_atr_mult: float = 0.5
    direct_body_avg_mult: float = 1.5
    fake_breakout_atr_mult: float = 0.2
    cooldown_candles: int = 3

    volume_lookback: int = 30
    min_volume_ratio: float = 1.05
    min_body_ratio: float = 0.45

    atr_period: int = 14
    atr_percentile_lookback: int = 100
    atr_percentile_low: float = 0.30
    atr_percentile_high: float = 0.80
    sl_atr_mult: float = 0.5
    min_sl_atr_mult: float = 0.5
    max_sl_atr_mult: float = 1.2
    min_rr: float = 1.2
    tp_r: float = 1.5
    exit_mode: str = "trail"

    macd_fast: int = 4
    macd_slow: int = 9
    macd_signal: int = 4

    fee_rate: float = 0.0004
    max_fee_to_r: float = 0.35
    slippage_atr_mult: float = 0.03
    dry_run: bool = True

    trade_log: str = "range_breakout_trades.csv"
    equity_log: str = "range_breakout_equity.csv"
    monthly_log: str = "range_breakout_monthly.csv"
    results_log: str = "range_breakout_results.csv"


@dataclass
class BreakoutState:
    symbol: str
    side: str
    level: float
    range_high: float
    range_low: float
    detected_time: pd.Timestamp
    session: str
    candles_waited: int = 0


@dataclass
class Signal:
    symbol: str
    side: str
    entry: float
    sl: float
    target: float
    range_high: float
    range_low: float
    volume_ratio: float
    rr: float
    session: str
    reason: str


@dataclass
class Position:
    symbol: str
    side: str
    entry_time: pd.Timestamp
    entry: float
    sl: float
    initial_sl: float
    target: float
    quantity: float
    risk_amount: float
    session: str
    reason: str
    break_even_active: bool = False
    partial_closed: bool = False
    realized_pnl: float = 0.0
    realized_fee: float = 0.0

    @property
    def one_r(self) -> float:
        return abs(self.entry - self.initial_sl)


class BotLogger:
    def __init__(self, config: BotConfig, enabled: bool = True):
        self.config = config
        self.enabled = enabled
        if not enabled:
            return
        for path in [config.trade_log, config.equity_log, config.monthly_log, config.results_log]:
            if Path(path).exists():
                Path(path).unlink(missing_ok=True)

    def trade(self, row: dict):
        if self.enabled:
            self._append_csv(self.config.trade_log, row)

    def equity(self, row: dict):
        if self.enabled:
            self._append_csv(self.config.equity_log, row)

    def monthly(self, df: pd.DataFrame):
        if self.enabled:
            df.to_csv(self.config.monthly_log, index=False)

    def results(self, row: dict):
        if self.enabled:
            self._append_csv(self.config.results_log, row)

    @staticmethod
    def _append_csv(path: str, row: dict):
        exists = Path(path).exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not exists:
                writer.writeheader()
            writer.writerow(row)


class CsvDataFeed:
    SYMBOL_FILE_PREFIXES = {
        "BTCUSDT": ["bitcoin", "btc"],
        "ETHUSDT": ["ethereum", "eth"],
        "SOLUSDT": ["solana", "sol"],
    }

    @staticmethod
    def load(path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        if "time" not in df.columns:
            df.columns = ["time", "open", "high", "low", "close", "volume"]
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time").drop_duplicates("time").set_index("time")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.dropna(subset=["open", "high", "low", "close", "volume"])

    @classmethod
    def find_symbol_files(cls, symbol: str, data_dir: str) -> tuple[Optional[str], Optional[str]]:
        prefixes = cls.SYMBOL_FILE_PREFIXES.get(symbol, [symbol.lower().replace("usdt", "")])
        root = Path(data_dir)
        for prefix in prefixes:
            m5 = root / f"{prefix}_365d_5m.csv"
            m15 = root / f"{prefix}_365d_15m.csv"
            if m5.exists() and m15.exists():
                return str(m5), str(m15)
        return None, None

    @classmethod
    def load_symbols(cls, symbols: list[str], data_dir: str) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], list[str]]:
        m5_by_symbol = {}
        m15_by_symbol = {}
        missing = []
        for symbol in symbols:
            m5_path, m15_path = cls.find_symbol_files(symbol, data_dir)
            if not m5_path or not m15_path:
                missing.append(symbol)
                continue
            m5_by_symbol[symbol] = cls.load(m5_path)
            m15_by_symbol[symbol] = cls.load(m15_path)
        return m5_by_symbol, m15_by_symbol, missing


class Indicators:
    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def atr(df: pd.DataFrame, period: int) -> pd.Series:
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

    @staticmethod
    def to_m15(df: pd.DataFrame) -> pd.DataFrame:
        return df.resample("15min").agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        ).dropna()


class SessionHelper:
    @staticmethod
    def bangkok_time(now: pd.Timestamp, config: BotConfig) -> pd.Timestamp:
        return now + pd.Timedelta(hours=config.timezone_offset_hours)

    @staticmethod
    def session_name(now: pd.Timestamp, config: BotConfig) -> Optional[str]:
        hour = SessionHelper.bangkok_time(now, config).hour
        for name, start, end in config.sessions:
            if end == 24 and start <= hour <= 23:
                return name
            if start <= end and start <= hour < end:
                return name
            if start > end and (hour >= start or hour < end):
                return name
        return None

    @staticmethod
    def bkk_date(now: pd.Timestamp, config: BotConfig):
        return SessionHelper.bangkok_time(now, config).date()


class RangeBreakoutStrategy:
    def __init__(self, config: BotConfig):
        self.config = config

    def add_features(self, symbol: str, m5: pd.DataFrame, m15: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        df = m5.copy()
        df["symbol"] = symbol
        df["session"] = [SessionHelper.session_name(ts, self.config) for ts in df.index]
        df["session_key"] = [
            f"{SessionHelper.bkk_date(ts, self.config)}_{sess}" if sess else None
            for ts, sess in zip(df.index, df["session"])
        ]

        trend_df = m15.copy() if m15 is not None else Indicators.to_m15(df)
        trend_df["ema_fast"] = Indicators.ema(trend_df["close"], self.config.trend_ema_fast)
        trend_df["ema_slow"] = Indicators.ema(trend_df["close"], self.config.trend_ema_slow)
        trend_df["trend"] = np.where(trend_df["ema_fast"] > trend_df["ema_slow"], "UP", "DOWN")
        trend_df["closed_trend"] = trend_df["trend"].shift(1)

        df["atr"] = Indicators.atr(df, self.config.atr_period)
        atr_ref = df["atr"].shift(1).rolling(self.config.atr_percentile_lookback)
        df["atr_low"] = atr_ref.quantile(self.config.atr_percentile_low)
        df["atr_high"] = atr_ref.quantile(self.config.atr_percentile_high)
        df["macd"] = Indicators.ema(df["close"], self.config.macd_fast) - Indicators.ema(df["close"], self.config.macd_slow)
        df["macd_signal"] = Indicators.ema(df["macd"], self.config.macd_signal)
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        df["volume_avg"] = df["volume"].shift(1).rolling(self.config.volume_lookback).mean()
        df["volume_ratio"] = df["volume"] / df["volume_avg"]
        df["body_size"] = (df["close"] - df["open"]).abs()
        df["avg_body"] = df["body_size"].shift(1).rolling(self.config.volume_lookback).mean()
        df["body_ratio"] = df["body_size"] / (df["high"] - df["low"]).replace(0, np.nan)
        df["m15_trend"] = trend_df["closed_trend"].reindex(df.index, method="ffill")

        bkk_time = df.index + pd.Timedelta(hours=self.config.timezone_offset_hours)
        minutes = bkk_time.hour * 60 + bkk_time.minute
        slot = minutes // self.config.range_window_minutes
        df["range_slot_key"] = [
            f"{SessionHelper.bkk_date(ts, self.config)}_{int(s)}" if sess else None
            for ts, s, sess in zip(df.index, slot, df["session"])
        ]
        df["previous_range_slot_key"] = [
            f"{SessionHelper.bkk_date(ts, self.config)}_{int(s) - 1}" if sess and int(s) > 0 else None
            for ts, s, sess in zip(df.index, slot, df["session"])
        ]
        range_by_slot = df[df["range_slot_key"].notna()].groupby("range_slot_key").agg(
            slot_high=("high", "max"),
            slot_low=("low", "min"),
            slot_bars=("close", "count"),
        )
        df["session_bar"] = df.groupby("range_slot_key", dropna=False).cumcount()
        df["range_high"] = np.nan
        df["range_low"] = np.nan
        for slot_key, values in range_by_slot.iterrows():
            if values["slot_bars"] < self.config.min_range_candles:
                continue
            slot_index = df.index[df["range_slot_key"] == slot_key]
            initial_index = slot_index[: self.config.min_range_candles]
            trade_index = slot_index[self.config.min_range_candles :]
            df.loc[trade_index, "range_high"] = df.loc[initial_index, "high"].max()
            df.loc[trade_index, "range_low"] = df.loc[initial_index, "low"].min()
        return df

    def _filters_ok(self, row: pd.Series) -> bool:
        required = [
            "atr",
            "atr_low",
            "atr_high",
            "range_high",
            "range_low",
            "m15_trend",
            "session",
            "volume_ratio",
            "body_ratio",
            "macd",
            "macd_signal",
            "macd_hist",
            "body_size",
            "avg_body",
        ]
        if any(pd.isna(row.get(col)) for col in required):
            return False
        if row["session"] is None:
            return False
        if row["atr"] <= 0 or not (row["atr_low"] <= row["atr"] <= row["atr_high"]):
            return False
        if row["volume_ratio"] < self.config.min_volume_ratio:
            return False
        if row["body_ratio"] < self.config.min_body_ratio:
            return False
        return True

    def breakout_state(self, row: pd.Series) -> Optional[BreakoutState]:
        if not self._filters_ok(row):
            return None
        atr_value = float(row["atr"])
        buy_break = row["close"] > row["range_high"] + atr_value * self.config.fake_breakout_atr_mult
        sell_break = row["close"] < row["range_low"] - atr_value * self.config.fake_breakout_atr_mult
        if row["m15_trend"] == "UP" and buy_break:
            return BreakoutState(row["symbol"], "BUY", float(row["range_high"]), float(row["range_high"]), float(row["range_low"]), row.name, row["session"])
        if row["m15_trend"] == "DOWN" and sell_break:
            return BreakoutState(row["symbol"], "SELL", float(row["range_low"]), float(row["range_high"]), float(row["range_low"]), row.name, row["session"])
        return None

    def direct_signal(self, row: pd.Series, state: BreakoutState) -> Optional[Signal]:
        if not self.config.allow_direct_breakout or not self._filters_ok(row):
            return None
        atr_value = float(row["atr"])
        strong_buy = state.side == "BUY" and row["close"] > state.range_high + atr_value * self.config.direct_breakout_atr_mult
        strong_sell = state.side == "SELL" and row["close"] < state.range_low - atr_value * self.config.direct_breakout_atr_mult
        if not (strong_buy or strong_sell):
            return None
        if row["body_size"] < row["avg_body"] * self.config.direct_body_avg_mult:
            return None
        if state.side == "BUY":
            if not (row["close"] > row["open"] and row["macd"] > row["macd_signal"] and row["macd_hist"] > 0):
                return None
        else:
            if not (row["close"] < row["open"] and row["macd"] < row["macd_signal"] and row["macd_hist"] < 0):
                return None
        return self._build_signal(row, state, direct=True)

    def retest_signal(self, row: pd.Series, state: BreakoutState) -> Optional[Signal]:
        if not self._filters_ok(row):
            return None
        level = state.level
        tolerance = level * self.config.retest_tolerance_pct
        if state.side == "BUY":
            near_level = row["low"] <= level + tolerance and row["close"] >= level - tolerance
            candle_ok = row["close"] > row["open"]
            macd_ok = row["macd"] > row["macd_signal"] and row["macd_hist"] > 0
            if not (near_level and candle_ok and macd_ok):
                return None
        else:
            near_level = row["high"] >= level - tolerance and row["close"] <= level + tolerance
            candle_ok = row["close"] < row["open"]
            macd_ok = row["macd"] < row["macd_signal"] and row["macd_hist"] < 0
            if not (near_level and candle_ok and macd_ok):
                return None
        return self._build_signal(row, state, direct=False)

    def _build_signal(self, row: pd.Series, state: BreakoutState, direct: bool) -> Optional[Signal]:
        atr_value = float(row["atr"])
        slip = atr_value * self.config.slippage_atr_mult

        if state.side == "BUY":
            entry = float(row["close"] + slip)
            raw_sl = float(min(row["low"], state.range_high - atr_value * self.config.sl_atr_mult))
            sl = self._normalized_sl(entry, raw_sl, atr_value, "BUY")
            if sl is None:
                return None
            risk = abs(entry - sl)
            target = entry + risk * self.config.tp_r
            rr = self.config.tp_r if risk > 0 else 0
        else:
            entry = float(row["close"] - slip)
            raw_sl = float(max(row["high"], state.range_low + atr_value * self.config.sl_atr_mult))
            sl = self._normalized_sl(entry, raw_sl, atr_value, "SELL")
            if sl is None:
                return None
            risk = abs(entry - sl)
            target = entry - risk * self.config.tp_r
            rr = self.config.tp_r if risk > 0 else 0
        fee_to_r = self.config.fee_rate * (entry + target) / risk if risk > 0 else 999
        if fee_to_r > self.config.max_fee_to_r:
            return None
        if rr < self.config.min_rr:
            return None
        reason = "direct_breakout" if direct else "breakout_retest"
        return Signal(row["symbol"], state.side, entry, sl, target, state.range_high, state.range_low, row["volume_ratio"], rr, state.session, reason)

    def _normalized_sl(self, entry: float, raw_sl: float, atr_value: float, side: str) -> Optional[float]:
        distance = abs(entry - raw_sl)
        max_distance = atr_value * self.config.max_sl_atr_mult
        min_distance = atr_value * self.config.min_sl_atr_mult
        if distance > max_distance:
            return None
        distance = max(distance, min_distance)
        if side == "BUY":
            return entry - distance
        return entry + distance


class RiskManager:
    def __init__(self, config: BotConfig):
        self.config = config
        self.daily_trades = 0
        self.current_day = None
        self.consecutive_losses = 0
        self.day_start_balance = None

    def reset_day_if_needed(self, now: pd.Timestamp, balance: float):
        bkk_date = SessionHelper.bkk_date(now, self.config)
        if self.current_day != bkk_date:
            self.current_day = bkk_date
            self.daily_trades = 0
            self.consecutive_losses = 0
            self.day_start_balance = balance

    def can_trade(self, now: pd.Timestamp, position: Optional[Position], balance: float, cooldown_until: Optional[pd.Timestamp]) -> bool:
        self.reset_day_if_needed(now, balance)
        daily_ret = (balance - self.day_start_balance) / self.day_start_balance
        if position is not None:
            return False
        if cooldown_until is not None and now <= cooldown_until:
            return False
        if daily_ret >= self.config.daily_profit_target:
            return False
        if daily_ret <= -self.config.max_daily_loss:
            return False
        if self.daily_trades >= self.config.max_trades_per_day:
            return False
        if self.consecutive_losses >= self.config.max_consecutive_losses:
            return False
        return True

    def size_position(self, balance: float, entry: float, sl: float) -> tuple[float, float]:
        risk_amount = balance * self.config.risk_per_trade
        stop_distance = abs(entry - sl)
        if stop_distance <= 0:
            return 0.0, risk_amount
        fee_per_unit_at_stop = self.config.fee_rate * (entry + sl)
        return risk_amount / (stop_distance + fee_per_unit_at_stop), risk_amount

    def record_entry(self):
        self.daily_trades += 1

    def record_exit(self, pnl: float):
        self.consecutive_losses = self.consecutive_losses + 1 if pnl < 0 else 0


class TradeManager:
    def __init__(self, config: BotConfig):
        self.config = config

    def open_position(self, signal: Signal, balance: float, risk: RiskManager, now: pd.Timestamp) -> Optional[Position]:
        qty, risk_amount = risk.size_position(balance, signal.entry, signal.sl)
        if qty <= 0:
            return None
        risk.record_entry()
        return Position(
            symbol=signal.symbol,
            side=signal.side,
            entry_time=now,
            entry=signal.entry,
            sl=signal.sl,
            initial_sl=signal.sl,
            target=signal.target,
            quantity=qty,
            risk_amount=risk_amount,
            session=signal.session,
            reason=signal.reason,
        )

    def manage_position(self, position: Position, row: pd.Series, prev_row: pd.Series) -> tuple[Optional[str], Optional[float], float, float]:
        partial_pnl = 0.0
        partial_fee = 0.0
        if position.side == "BUY":
            if row["low"] <= position.sl:
                return "SL", float(position.sl), partial_pnl, partial_fee
            if row["high"] >= position.target:
                return "TP", float(position.target), partial_pnl, partial_fee
            if self.config.exit_mode in {"trail", "partial_trail"}:
                if row["high"] >= position.entry + 0.8 * position.one_r:
                    position.sl = max(position.sl, position.entry - 0.2 * position.one_r)
                if row["high"] >= position.entry + position.one_r:
                    if self.config.exit_mode == "partial_trail" and not position.partial_closed:
                        partial_pnl, partial_fee = self._partial_close(position, position.entry + position.one_r)
                    position.sl = max(position.sl, position.entry)
                    position.break_even_active = True
                if position.break_even_active:
                    position.sl = max(position.sl, float(prev_row["low"]))

        if position.side == "SELL":
            if row["high"] >= position.sl:
                return "SL", float(position.sl), partial_pnl, partial_fee
            if row["low"] <= position.target:
                return "TP", float(position.target), partial_pnl, partial_fee
            if self.config.exit_mode in {"trail", "partial_trail"}:
                if row["low"] <= position.entry - 0.8 * position.one_r:
                    position.sl = min(position.sl, position.entry + 0.2 * position.one_r)
                if row["low"] <= position.entry - position.one_r:
                    if self.config.exit_mode == "partial_trail" and not position.partial_closed:
                        partial_pnl, partial_fee = self._partial_close(position, position.entry - position.one_r)
                    position.sl = min(position.sl, position.entry)
                    position.break_even_active = True
                if position.break_even_active:
                    position.sl = min(position.sl, float(prev_row["high"]))

        return None, None, partial_pnl, partial_fee

    def _partial_close(self, position: Position, exit_price: float) -> tuple[float, float]:
        close_qty = position.quantity * 0.5
        gross = (exit_price - position.entry) * close_qty
        if position.side == "SELL":
            gross = -gross
        fee = (position.entry * close_qty + exit_price * close_qty) * self.config.fee_rate
        pnl = gross - fee
        position.quantity -= close_qty
        position.realized_pnl += pnl
        position.realized_fee += fee
        position.partial_closed = True
        return pnl, fee

    def pnl(self, position: Position, exit_price: float) -> tuple[float, float]:
        gross = (exit_price - position.entry) * position.quantity
        if position.side == "SELL":
            gross = -gross
        fee = (position.entry * position.quantity + exit_price * position.quantity) * self.config.fee_rate
        return gross - fee + position.realized_pnl, fee + position.realized_fee


class Backtester:
    def __init__(self, config: BotConfig, write_logs: bool = True):
        self.config = config
        self.strategy = RangeBreakoutStrategy(config)
        self.risk = RiskManager(config)
        self.trades = TradeManager(config)
        self.logger = BotLogger(config, enabled=write_logs)

    def prepare(self, m5_by_symbol: dict[str, pd.DataFrame], m15_by_symbol: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[tuple[str, pd.Timestamp], pd.Series]]:
        frames = []
        row_lookup = {}
        for symbol, m5 in m5_by_symbol.items():
            features = self.strategy.add_features(symbol, m5, m15_by_symbol.get(symbol))
            features["prev_low"] = features["low"].shift(1)
            features["prev_high"] = features["high"].shift(1)
            frames.append(features)
            for ts, row in features.iterrows():
                row_lookup[(symbol, ts)] = row
        events = pd.concat(frames).sort_index(kind="mergesort")
        return events, row_lookup

    def run(self, m5_by_symbol: dict[str, pd.DataFrame], m15_by_symbol: dict[str, pd.DataFrame]) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        events, row_lookup = self.prepare(m5_by_symbol, m15_by_symbol)
        balance = self.config.start_balance
        peak = balance
        max_dd = 0.0
        position = None
        cooldown_until = None
        pending: dict[str, BreakoutState] = {}
        trade_rows = []
        equity_rows = []

        for now, row in events.iterrows():
            symbol = row["symbol"]

            if position is not None and position.symbol == symbol:
                prev_row = row_lookup.get((symbol, row.name - pd.Timedelta(minutes=5)))
                if prev_row is None:
                    prev_row = row
                result, exit_price, _, _ = self.trades.manage_position(position, row, prev_row)
                if result:
                    pnl, fee = self.trades.pnl(position, exit_price)
                    balance += pnl
                    self.risk.record_exit(pnl)
                    cooldown_until = now + pd.Timedelta(minutes=5 * self.config.cooldown_candles)
                    trade_row = {
                        "exit_time": now,
                        "entry_time": position.entry_time,
                        "symbol": position.symbol,
                        "side": position.side,
                        "result": "WIN" if pnl > 0 else "LOSS",
                        "entry": position.entry,
                        "exit": exit_price,
                        "initial_sl": position.initial_sl,
                        "target": position.target,
                        "final_sl": position.sl,
                        "quantity": position.quantity,
                        "pnl": pnl,
                        "fee": fee,
                        "balance": balance,
                        "break_even": position.break_even_active,
                        "session": position.session,
                        "reason": position.reason,
                    }
                    trade_rows.append(trade_row)
                    self.logger.trade(trade_row)
                    position = None

            peak = max(peak, balance)
            dd = (balance - peak) / peak
            max_dd = min(max_dd, dd)
            equity_row = {"time": now, "balance": balance, "drawdown_pct": dd * 100}
            equity_rows.append(equity_row)
            self.logger.equity(equity_row)

            if not self.risk.can_trade(now, position, balance, cooldown_until):
                continue

            state = pending.get(symbol)
            if state is not None:
                if row["session"] != state.session:
                    pending.pop(symbol, None)
                    state = None
                else:
                    state.candles_waited += 1
                    if state.candles_waited > self.config.max_retest_candles:
                        pending.pop(symbol, None)
                        state = None

            if state is not None:
                signal = self.strategy.retest_signal(row, state)
                if signal is not None:
                    position = self.trades.open_position(signal, balance, self.risk, now)
                    pending.pop(symbol, None)
                    continue

            new_state = self.strategy.breakout_state(row)
            if new_state is None:
                continue
            direct_signal = self.strategy.direct_signal(row, new_state)
            if direct_signal is not None:
                position = self.trades.open_position(direct_signal, balance, self.risk, now)
                pending.pop(symbol, None)
            elif self.config.use_retest_entry:
                pending[symbol] = new_state

        trade_df = pd.DataFrame(trade_rows)
        equity_df = pd.DataFrame(equity_rows)
        monthly_df = self.monthly_report(trade_df, equity_df)
        metrics = self.metrics(trade_df, equity_df, balance, max_dd, m5_by_symbol)
        self.logger.monthly(monthly_df)
        self.logger.results(metrics)
        return metrics, trade_df, equity_df, monthly_df

    def metrics(self, trades: pd.DataFrame, equity: pd.DataFrame, end_balance: float, max_dd: float, m5_by_symbol: dict[str, pd.DataFrame]) -> dict:
        total_trades = len(trades)
        wins = trades[trades["pnl"] > 0] if total_trades else pd.DataFrame()
        losses = trades[trades["pnl"] <= 0] if total_trades else pd.DataFrame()
        gross_profit = wins["pnl"].sum() if total_trades else 0.0
        gross_loss = abs(losses["pnl"].sum()) if total_trades else 0.0
        first_time = min(df.index.min() for df in m5_by_symbol.values())
        last_time = max(df.index.max() for df in m5_by_symbol.values())
        test_days = max((SessionHelper.bkk_date(last_time, self.config) - SessionHelper.bkk_date(first_time, self.config)).days + 1, 1)
        avg_win = wins["pnl"].mean() if len(wins) else 0.0
        avg_loss = losses["pnl"].mean() if len(losses) else 0.0
        expectancy = trades["pnl"].mean() if total_trades else 0.0
        return {
            "symbols": ",".join(m5_by_symbol.keys()),
            "sessions": ",".join(name for name, _, _ in self.config.sessions),
            "ema": f"{self.config.trend_ema_fast}/{self.config.trend_ema_slow}",
            "body_ratio": self.config.min_body_ratio,
            "retest_tolerance_pct": self.config.retest_tolerance_pct,
            "min_rr": self.config.min_rr,
            "exit_mode": self.config.exit_mode,
            "allow_direct_breakout": self.config.allow_direct_breakout,
            "risk_per_trade": self.config.risk_per_trade,
            "start_balance": self.config.start_balance,
            "end_balance": end_balance,
            "profit_pct": (end_balance - self.config.start_balance) / self.config.start_balance * 100,
            "trades": total_trades,
            "trades_per_day": total_trades / test_days,
            "winrate": len(wins) / total_trades * 100 if total_trades else 0.0,
            "profit_factor": gross_profit / gross_loss if gross_loss > 0 else 0.0,
            "max_drawdown_pct": max_dd * 100,
            "avg_profit_per_day": (end_balance - self.config.start_balance) / test_days,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "expectancy": expectancy,
            "trade_log": self.config.trade_log,
            "equity_log": self.config.equity_log,
            "monthly_log": self.config.monthly_log,
        }

    @staticmethod
    def monthly_report(trades: pd.DataFrame, equity: pd.DataFrame) -> pd.DataFrame:
        if equity.empty:
            return pd.DataFrame()
        eq = equity.drop_duplicates("time").copy()
        eq["month"] = pd.to_datetime(eq["time"]).dt.to_period("M").astype(str)
        eq["month_peak"] = eq.groupby("month")["balance"].cummax()
        eq["monthly_drawdown_pct"] = (eq["balance"] - eq["month_peak"]) / eq["month_peak"] * 100
        monthly = eq.groupby("month").agg(
            start_balance=("balance", "first"),
            end_balance=("balance", "last"),
            monthly_max_drawdown=("monthly_drawdown_pct", "min"),
        ).reset_index()
        monthly["monthly_profit"] = monthly["end_balance"] - monthly["start_balance"]
        monthly["monthly_profit_pct"] = monthly["monthly_profit"] / monthly["start_balance"] * 100

        if trades.empty:
            monthly["monthly_trades"] = 0
            monthly["monthly_winrate"] = 0.0
            monthly["monthly_profit_factor"] = 0.0
            return monthly

        tr = trades.copy()
        tr["month"] = pd.to_datetime(tr["exit_time"]).dt.to_period("M").astype(str)
        stats = tr.groupby("month").agg(
            monthly_trades=("pnl", "count"),
            monthly_winrate=("pnl", lambda x: (x > 0).mean() * 100),
            gross_profit=("pnl", lambda x: x[x > 0].sum()),
            gross_loss=("pnl", lambda x: abs(x[x <= 0].sum())),
        ).reset_index()
        stats["monthly_profit_factor"] = np.where(stats["gross_loss"] > 0, stats["gross_profit"] / stats["gross_loss"], 0.0)
        monthly = monthly.merge(stats[["month", "monthly_trades", "monthly_winrate", "monthly_profit_factor"]], on="month", how="left")
        monthly["monthly_trades"] = monthly["monthly_trades"].fillna(0).astype(int)
        monthly["monthly_winrate"] = monthly["monthly_winrate"].fillna(0.0)
        monthly["monthly_profit_factor"] = monthly["monthly_profit_factor"].fillna(0.0)
        return monthly


def parse_symbols(raw: str) -> list[str]:
    return [part.strip().upper() for part in raw.split(",") if part.strip()]


def build_config(args) -> BotConfig:
    return BotConfig(
        risk_per_trade=args.risk,
        max_trades_per_day=args.max_trades_per_day,
        min_volume_ratio=args.volume_ratio,
        min_body_ratio=args.body_ratio,
        min_range_candles=args.min_range_candles,
        retest_tolerance_pct=args.retest_tolerance,
        max_retest_candles=args.max_retest_candles,
        min_rr=args.min_rr,
        tp_r=args.tp_r,
        exit_mode=args.exit_mode,
        trend_ema_fast=args.ema_fast,
        trend_ema_slow=args.ema_slow,
        allow_direct_breakout=not args.no_direct_breakout,
        cooldown_candles=args.cooldown_candles,
    )


def clone_config(config: BotConfig, **updates) -> BotConfig:
    values = {**config.__dict__, **updates}
    return BotConfig(**values)


def robustness_score(row: dict) -> tuple:
    passes = (
        row["profit_factor"] > 1.2
        and row["expectancy"] > 0
        and row["trades_per_day"] >= 0.5
        and row["max_drawdown_pct"] > -10
    )
    enough_trades = min(row["trades_per_day"], 3.0)
    dd_quality = row["max_drawdown_pct"]
    return (passes, row["expectancy"], row["profit_factor"], enough_trades, dd_quality, row["profit_pct"])


def run_sweep(base_config: BotConfig, m5_by_symbol: dict[str, pd.DataFrame], m15_by_symbol: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for exit_mode in ["fixed", "trail", "partial_trail"]:
        for body_ratio in [0.30, 0.35, 0.45]:
            for tolerance in [0.001, 0.002, 0.003]:
                for min_rr in [1.0, 1.2, 1.5]:
                    for direct in [False, True]:
                        cfg = clone_config(
                            base_config,
                            exit_mode=exit_mode,
                            min_body_ratio=body_ratio,
                            retest_tolerance_pct=tolerance,
                            min_rr=min_rr,
                            allow_direct_breakout=direct,
                        )
                        metrics, _, _, _ = Backtester(cfg, write_logs=False).run(m5_by_symbol, m15_by_symbol)
                        rows.append(metrics)
    df = pd.DataFrame(rows)
    df["passes_target"] = (
        (df["profit_factor"] > 1.2)
        & (df["expectancy"] > 0)
        & (df["trades_per_day"] >= 0.5)
        & (df["max_drawdown_pct"] > -10)
    )
    df = df.sort_values(
        ["passes_target", "expectancy", "profit_factor", "trades_per_day", "max_drawdown_pct", "profit_pct"],
        ascending=[False, False, False, False, False, False],
    )
    df.to_csv(base_config.results_log, index=False)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["backtest", "sweep"], default="backtest")
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT")
    parser.add_argument("--risk", type=float, default=0.005)
    parser.add_argument("--max-trades-per-day", type=int, default=3)
    parser.add_argument("--volume-ratio", type=float, default=1.05)
    parser.add_argument("--body-ratio", type=float, default=0.45)
    parser.add_argument("--min-range-candles", type=int, default=12)
    parser.add_argument("--retest-tolerance", type=float, default=0.001)
    parser.add_argument("--max-retest-candles", type=int, default=6)
    parser.add_argument("--min-rr", type=float, default=1.2)
    parser.add_argument("--tp-r", type=float, default=1.5)
    parser.add_argument("--exit-mode", choices=["fixed", "trail", "partial_trail"], default="trail")
    parser.add_argument("--ema-fast", type=int, default=20)
    parser.add_argument("--ema-slow", type=int, default=50)
    parser.add_argument("--cooldown-candles", type=int, default=3)
    parser.add_argument("--no-direct-breakout", action="store_true")
    args = parser.parse_args()

    config = build_config(args)
    symbols = parse_symbols(args.symbols)
    m5_by_symbol, m15_by_symbol, missing = CsvDataFeed.load_symbols(symbols, args.data_dir)
    if missing:
        print(f"Missing data for: {', '.join(missing)}")
    if not m5_by_symbol:
        raise RuntimeError("No symbol data found. Expected files like bitcoin_365d_5m.csv and bitcoin_365d_15m.csv.")

    if args.mode == "sweep":
        results = run_sweep(config, m5_by_symbol, m15_by_symbol)
        columns = [
            "passes_target",
            "profit_pct",
            "trades",
            "trades_per_day",
            "winrate",
            "profit_factor",
            "max_drawdown_pct",
            "avg_win",
            "avg_loss",
            "expectancy",
            "exit_mode",
            "body_ratio",
            "retest_tolerance_pct",
            "min_rr",
            "allow_direct_breakout",
        ]
        print("TOP 10 ROBUST CONFIGS")
        print(results[columns].head(10).to_string(index=False))
        print(f"\nSaved sweep results to {config.results_log}")
        return

    metrics, trades, equity, monthly = Backtester(config).run(m5_by_symbol, m15_by_symbol)
    print("RANGE BREAKOUT DAY-TRADING BACKTEST RESULT")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print("\nMONTHLY BREAKDOWN")
    print(monthly.to_string(index=False))


if __name__ == "__main__":
    main()
