import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# =========================================================
# CONFIG
# =========================================================
@dataclass
class Config:
    data_file: str = "btc_5m.csv"
    start_balance: float = 1000.0
    dry_run: bool = True

    # Strategy
    range_lookback: int = 120
    ema_fast: int = 50
    ema_slow: int = 200
    macd_fast: int = 4
    macd_slow: int = 9
    macd_signal: int = 4
    atr_period: int = 14
    atr_stop_mult: float = 2.0
    min_volume_ratio: float = 2.0
    min_body_ratio: float = 0.55
    max_spread_pct: float = 0.0004

    # Risk
    risk_per_trade: float = 0.001
    max_trades_per_day: int = 3
    max_consecutive_losses: int = 2
    max_daily_loss_pct: float = 0.02
    daily_profit_target_pct: float = 0.02
    fee_rate: float = 0.0004
    slippage_atr_mult: float = 0.03

    # Runtime
    session_start_hour: int = 13
    session_end_hour: int = 23
    reconnect_attempts: int = 3
    reconnect_sleep_seconds: float = 1.0

    # Logs
    decision_log: str = "rb_macd_decisions.csv"
    trade_log: str = "rb_macd_trades.csv"
    equity_log: str = "rb_macd_equity.csv"


@dataclass
class Signal:
    side: str
    entry: float
    sl: float
    reason: str


@dataclass
class Position:
    side: str
    entry_time: pd.Timestamp
    entry: float
    sl: float
    initial_sl: float
    quantity: float
    risk_amount: float
    break_even_active: bool = False

    @property
    def one_r(self) -> float:
        return abs(self.entry - self.initial_sl)


# =========================================================
# CSV LOGGER
# =========================================================
class CsvLogger:
    def __init__(self, config: Config):
        self.config = config
        for path in [config.decision_log, config.trade_log, config.equity_log]:
            if Path(path).exists():
                Path(path).unlink()

    def decision(self, row: dict):
        self._append(self.config.decision_log, row)

    def trade(self, row: dict):
        self._append(self.config.trade_log, row)

    def equity(self, row: dict):
        self._append(self.config.equity_log, row)

    @staticmethod
    def _append(path: str, row: dict):
        exists = Path(path).exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not exists:
                writer.writeheader()
            writer.writerow(row)


# =========================================================
# BROKER ADAPTER
# =========================================================
class BrokerAdapter:
    def reconnect(self):
        return True

    def place_market_order(self, signal: Signal, quantity: float, dry_run: bool):
        if dry_run:
            return {"status": "DRY_RUN", "side": signal.side, "qty": quantity, "entry": signal.entry}
        return {"status": "FILLED", "side": signal.side, "qty": quantity, "entry": signal.entry}


class SafeBrokerAdapter:
    def __init__(self, broker: BrokerAdapter, config: Config):
        self.broker = broker
        self.config = config

    def place_market_order(self, signal: Signal, quantity: float):
        last_error = None
        for _ in range(self.config.reconnect_attempts):
            try:
                return self.broker.place_market_order(signal, quantity, self.config.dry_run)
            except Exception as exc:
                last_error = exc
                self.broker.reconnect()
                time.sleep(self.config.reconnect_sleep_seconds)
        raise RuntimeError(f"Broker order failed after retries: {last_error}")


# =========================================================
# MARKET DATA LOADER
# =========================================================
class MarketDataLoader:
    @staticmethod
    def load_csv(path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        if "time" not in df.columns:
            df.columns = ["time", "open", "high", "low", "close", "volume"]
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time").drop_duplicates("time").set_index("time")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.dropna(subset=["open", "high", "low", "close", "volume"])


# =========================================================
# INDICATOR CALCULATOR
# =========================================================
class IndicatorCalculator:
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
    def add(df: pd.DataFrame, config: Config) -> pd.DataFrame:
        df = df.copy()
        df["ema_fast"] = IndicatorCalculator.ema(df["close"], config.ema_fast)
        df["ema_slow"] = IndicatorCalculator.ema(df["close"], config.ema_slow)
        df["macd"] = IndicatorCalculator.ema(df["close"], config.macd_fast) - IndicatorCalculator.ema(df["close"], config.macd_slow)
        df["macd_signal"] = IndicatorCalculator.ema(df["macd"], config.macd_signal)
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        df["atr"] = IndicatorCalculator.atr(df, config.atr_period)
        df["volume_avg"] = df["volume"].shift(1).rolling(30).mean()
        df["volume_ratio"] = df["volume"] / df["volume_avg"]
        df["body_ratio"] = (df["close"] - df["open"]).abs() / (df["high"] - df["low"]).replace(0, np.nan)
        return df


# =========================================================
# RANGE BUILDER
# =========================================================
class RangeBuilder:
    @staticmethod
    def add(df: pd.DataFrame, config: Config) -> pd.DataFrame:
        df = df.copy()
        df["range_high"] = df["high"].shift(1).rolling(config.range_lookback).max()
        df["range_low"] = df["low"].shift(1).rolling(config.range_lookback).min()
        return df


# =========================================================
# POSITION SIZE CALCULATOR
# =========================================================
class PositionSizeCalculator:
    @staticmethod
    def calculate(balance: float, entry: float, sl: float, config: Config) -> tuple[float, float]:
        risk_amount = balance * config.risk_per_trade
        stop_distance = abs(entry - sl)
        if stop_distance <= 0:
            return 0.0, risk_amount
        return risk_amount / stop_distance, risk_amount


# =========================================================
# RISK MANAGER
# =========================================================
class RiskManager:
    def __init__(self, config: Config):
        self.config = config
        self.current_day = None
        self.day_start_balance = None
        self.daily_trades = 0
        self.consecutive_losses = 0

    def reset_day(self, now: pd.Timestamp, balance: float):
        if self.current_day != now.date():
            self.current_day = now.date()
            self.day_start_balance = balance
            self.daily_trades = 0
            self.consecutive_losses = 0

    def can_open(self, now: pd.Timestamp, balance: float, has_position: bool) -> tuple[bool, str]:
        self.reset_day(now, balance)
        if has_position:
            return False, "duplicate_position_blocked"
        if not (self.config.session_start_hour <= now.hour <= self.config.session_end_hour):
            return False, "outside_session"
        daily_ret = (balance - self.day_start_balance) / self.day_start_balance
        if daily_ret <= -self.config.max_daily_loss_pct:
            return False, "daily_loss_stop"
        if daily_ret >= self.config.daily_profit_target_pct:
            return False, "daily_profit_target"
        if self.daily_trades >= self.config.max_trades_per_day:
            return False, "max_trades_per_day"
        if self.consecutive_losses >= self.config.max_consecutive_losses:
            return False, "two_consecutive_losses"
        return True, "ok"

    def record_entry(self):
        self.daily_trades += 1

    def record_exit(self, pnl: float):
        self.consecutive_losses = self.consecutive_losses + 1 if pnl < 0 else 0


# =========================================================
# ENTRY SIGNAL GENERATOR
# =========================================================
class EntrySignalGenerator:
    def __init__(self, config: Config):
        self.config = config

    def signal(self, row: pd.Series, spread_pct: float) -> tuple[Optional[Signal], str]:
        required = [
            "ema_fast", "ema_slow", "macd", "macd_signal", "macd_hist",
            "atr", "range_high", "range_low", "volume_ratio", "body_ratio",
        ]
        if any(pd.isna(row.get(col)) for col in required):
            return None, "indicator_not_ready"
        if spread_pct > self.config.max_spread_pct:
            return None, "spread_too_wide"
        if row["volume_ratio"] < self.config.min_volume_ratio:
            return None, "volume_too_low"
        if row["body_ratio"] < self.config.min_body_ratio:
            return None, "weak_candle_body"

        slip = float(row["atr"] * self.config.slippage_atr_mult)
        uptrend = row["ema_fast"] > row["ema_slow"]
        downtrend = row["ema_fast"] < row["ema_slow"]
        macd_buy = row["macd"] > row["macd_signal"] and row["macd_hist"] > 0
        macd_sell = row["macd"] < row["macd_signal"] and row["macd_hist"] < 0

        if uptrend and macd_buy and row["close"] > row["range_high"]:
            entry = float(row["close"] + slip)
            sl = float(entry - row["atr"] * self.config.atr_stop_mult)
            return Signal("BUY", entry, sl, "range_breakout_buy"), "signal_buy"

        if downtrend and macd_sell and row["close"] < row["range_low"]:
            entry = float(row["close"] - slip)
            sl = float(entry + row["atr"] * self.config.atr_stop_mult)
            return Signal("SELL", entry, sl, "range_breakout_sell"), "signal_sell"

        return None, "no_breakout"


# =========================================================
# ORDER MANAGER
# =========================================================
class OrderManager:
    def __init__(self, config: Config, broker: SafeBrokerAdapter):
        self.config = config
        self.broker = broker

    def open_position(self, signal: Signal, balance: float, now: pd.Timestamp, risk: RiskManager) -> Optional[Position]:
        qty, risk_amount = PositionSizeCalculator.calculate(balance, signal.entry, signal.sl, self.config)
        if qty <= 0:
            return None
        result = self.broker.place_market_order(signal, qty)
        if result["status"] not in {"DRY_RUN", "FILLED"}:
            return None
        risk.record_entry()
        return Position(signal.side, now, signal.entry, signal.sl, signal.sl, qty, risk_amount)


# =========================================================
# POSITION MANAGER
# =========================================================
class PositionManager:
    def __init__(self, config: Config):
        self.config = config

    def update(self, position: Position, row: pd.Series, prev_row: pd.Series) -> tuple[Optional[str], Optional[float]]:
        if position.side == "BUY":
            if not position.break_even_active and row["high"] >= position.entry + position.one_r:
                position.sl = max(position.sl, position.entry)
                position.break_even_active = True
            if position.break_even_active:
                position.sl = max(position.sl, float(prev_row["low"]))
            if row["low"] <= position.sl:
                return "EXIT", float(position.sl)

        if position.side == "SELL":
            if not position.break_even_active and row["low"] <= position.entry - position.one_r:
                position.sl = min(position.sl, position.entry)
                position.break_even_active = True
            if position.break_even_active:
                position.sl = min(position.sl, float(prev_row["high"]))
            if row["high"] >= position.sl:
                return "EXIT", float(position.sl)

        return None, None

    def pnl(self, position: Position, exit_price: float) -> tuple[float, float]:
        gross = (exit_price - position.entry) * position.quantity
        if position.side == "SELL":
            gross = -gross
        fee = (position.entry * position.quantity + exit_price * position.quantity) * self.config.fee_rate
        return gross - fee, fee


# =========================================================
# BACKTESTER
# =========================================================
class Backtester:
    def __init__(self, config: Config):
        self.config = config
        self.logger = CsvLogger(config)
        self.signal_generator = EntrySignalGenerator(config)
        self.risk = RiskManager(config)
        self.order_manager = OrderManager(config, SafeBrokerAdapter(BrokerAdapter(), config))
        self.position_manager = PositionManager(config)

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = IndicatorCalculator.add(df, self.config)
        df = RangeBuilder.add(df, self.config)
        return df

    def run(self, df: pd.DataFrame) -> dict:
        df = self.prepare(df)
        balance = self.config.start_balance
        peak = balance
        max_dd = 0.0
        wins = 0
        losses = 0
        position = None

        for i in range(max(self.config.ema_slow, self.config.range_lookback) + 2, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i - 1]
            now = row.name

            # Backtest uses a synthetic spread proxy. Live mode should read broker spread.
            spread_pct = (row["high"] - row["low"]) / row["close"] * 0.05

            if position is not None:
                exit_reason, exit_price = self.position_manager.update(position, row, prev_row)
                if exit_reason:
                    pnl, fee = self.position_manager.pnl(position, exit_price)
                    balance += pnl
                    self.risk.record_exit(pnl)
                    if pnl > 0:
                        wins += 1
                    else:
                        losses += 1
                    self.logger.trade({
                        "exit_time": now,
                        "entry_time": position.entry_time,
                        "side": position.side,
                        "entry": position.entry,
                        "exit": exit_price,
                        "initial_sl": position.initial_sl,
                        "final_sl": position.sl,
                        "qty": position.quantity,
                        "pnl": pnl,
                        "fee": fee,
                        "balance": balance,
                        "break_even": position.break_even_active,
                    })
                    position = None

            peak = max(peak, balance)
            dd = (balance - peak) / peak
            max_dd = min(max_dd, dd)
            self.logger.equity({"time": now, "balance": balance, "drawdown_pct": dd * 100})

            allowed, risk_reason = self.risk.can_open(now, balance, position is not None)
            signal, signal_reason = self.signal_generator.signal(row, spread_pct)
            self.logger.decision({
                "time": now,
                "allowed": allowed,
                "risk_reason": risk_reason,
                "signal_reason": signal_reason,
                "side": signal.side if signal else "",
                "close": row["close"],
                "range_high": row.get("range_high"),
                "range_low": row.get("range_low"),
                "volume_ratio": row.get("volume_ratio"),
                "body_ratio": row.get("body_ratio"),
                "spread_pct": spread_pct,
                "balance": balance,
            })
            if not allowed or signal is None:
                continue

            position = self.order_manager.open_position(signal, balance, now, self.risk)

        trades = wins + losses
        return {
            "start_balance": self.config.start_balance,
            "end_balance": balance,
            "profit_pct": (balance - self.config.start_balance) / self.config.start_balance * 100,
            "trades": trades,
            "wins": wins,
            "losses": losses,
            "winrate": wins / trades * 100 if trades else 0.0,
            "max_drawdown_pct": max_dd * 100,
            "decision_log": self.config.decision_log,
            "trade_log": self.config.trade_log,
            "equity_log": self.config.equity_log,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="btc_5m.csv")
    parser.add_argument("--risk", type=float, default=0.001)
    parser.add_argument("--range-lookback", type=int, default=120)
    parser.add_argument("--volume-ratio", type=float, default=2.0)
    parser.add_argument("--body-ratio", type=float, default=0.55)
    parser.add_argument("--atr-stop", type=float, default=2.0)
    parser.add_argument("--daily-target", type=float, default=0.02)
    parser.add_argument("--daily-loss", type=float, default=0.02)
    args = parser.parse_args()

    config = Config(
        data_file=args.data,
        risk_per_trade=args.risk,
        range_lookback=args.range_lookback,
        min_volume_ratio=args.volume_ratio,
        min_body_ratio=args.body_ratio,
        atr_stop_mult=args.atr_stop,
        daily_profit_target_pct=args.daily_target,
        max_daily_loss_pct=args.daily_loss,
    )

    df = MarketDataLoader.load_csv(config.data_file)
    result = Backtester(config).run(df)
    print("RANGE BREAKOUT MACD 4-9-4 BACKTEST")
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
