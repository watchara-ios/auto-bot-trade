import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
REPORT_DIR = ROOT_DIR / "reports"


@dataclass
class Config:
    data_file: str = str(DATA_DIR / "bitcoin_365d_5m.csv")
    start_balance: float = 1000.0
    risk_per_trade: float = 0.005
    fee_rate: float = 0.0004
    slippage_atr_mult: float = 0.03

    andean_length: int = 50
    andean_signal: int = 9
    atr_period: int = 14
    atr_stop_mult: float = 1.5
    rr: float = 2.0

    ema_trend_period: int = 200
    use_trend_filter: bool = True
    min_atr_pct: float = 0.0005
    max_atr_pct: float = 0.012
    min_volume_ratio: float = 0.7
    cooldown_candles: int = 3

    max_trades_per_day: int = 3
    max_consecutive_losses: int = 2
    max_daily_loss_pct: float = 0.03
    daily_profit_target_pct: float = 0.05

    trade_log: str = str(REPORT_DIR / "andean_trades.csv")
    equity_log: str = str(REPORT_DIR / "andean_equity.csv")
    results_log: str = str(REPORT_DIR / "andean_results.csv")


@dataclass
class Position:
    side: str
    entry_time: pd.Timestamp
    entry: float
    sl: float
    tp: float
    qty: float


class CsvLogger:
    def __init__(self, config: Config, enabled=True):
        self.config = config
        self.enabled = enabled
        if not enabled:
            return
        for path in [config.trade_log, config.equity_log]:
            if Path(path).exists():
                Path(path).unlink()

    def trade(self, row: dict):
        if self.enabled:
            self._append(self.config.trade_log, row)

    def equity(self, row: dict):
        if self.enabled:
            self._append(self.config.equity_log, row)

    @staticmethod
    def write_results(path: str, rows: list[dict]):
        pd.DataFrame(rows).to_csv(path, index=False)

    @staticmethod
    def _append(path: str, row: dict):
        exists = Path(path).exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not exists:
                writer.writeheader()
            writer.writerow(row)


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "time" not in df.columns:
        df.columns = ["time", "open", "high", "low", "close", "volume"]
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").drop_duplicates("time").set_index("time")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["open", "high", "low", "close", "volume"])


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


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


def add_andean(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    df = df.copy()
    alpha = 2 / (config.andean_length + 1)
    opens = df["open"].to_numpy()
    closes = df["close"].to_numpy()

    up1 = np.zeros(len(df))
    up2 = np.zeros(len(df))
    dn1 = np.zeros(len(df))
    dn2 = np.zeros(len(df))

    up1[0] = max(opens[0], closes[0])
    up2[0] = max(opens[0] ** 2, closes[0] ** 2)
    dn1[0] = min(opens[0], closes[0])
    dn2[0] = min(opens[0] ** 2, closes[0] ** 2)

    for i in range(1, len(df)):
        c = closes[i]
        o = opens[i]
        up1[i] = max(c, o, up1[i - 1] - (up1[i - 1] - c) * alpha)
        up2[i] = max(c * c, o * o, up2[i - 1] - (up2[i - 1] - c * c) * alpha)
        dn1[i] = min(c, o, dn1[i - 1] + (c - dn1[i - 1]) * alpha)
        dn2[i] = min(c * c, o * o, dn2[i - 1] + (c * c - dn2[i - 1]) * alpha)

    bull = np.sqrt(np.maximum(dn2 - dn1 * dn1, 0))
    bear = np.sqrt(np.maximum(up2 - up1 * up1, 0))
    df["andean_bull"] = bull
    df["andean_bear"] = bear
    df["andean_signal"] = ema(pd.Series(np.maximum(bull, bear), index=df.index), config.andean_signal)
    return df


def prepare(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    df = add_andean(df, config)
    df["atr"] = atr(df, config.atr_period)
    df["atr_pct"] = df["atr"] / df["close"]
    df["ema_trend"] = ema(df["close"], config.ema_trend_period)
    df["volume_avg"] = df["volume"].shift(1).rolling(30).mean()
    df["volume_ratio"] = df["volume"] / df["volume_avg"]
    df["bull_cross"] = (df["andean_bull"].shift(1) <= df["andean_bear"].shift(1)) & (df["andean_bull"] > df["andean_bear"])
    df["bear_cross"] = (df["andean_bear"].shift(1) <= df["andean_bull"].shift(1)) & (df["andean_bear"] > df["andean_bull"])
    return df


class RiskState:
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

    def can_open(self, now: pd.Timestamp, balance: float, has_position: bool, cooldown_until: int, index: int) -> bool:
        self.reset_day(now, balance)
        if has_position or index <= cooldown_until:
            return False
        daily_ret = (balance - self.day_start_balance) / self.day_start_balance
        if daily_ret <= -self.config.max_daily_loss_pct:
            return False
        if daily_ret >= self.config.daily_profit_target_pct:
            return False
        if self.daily_trades >= self.config.max_trades_per_day:
            return False
        if self.consecutive_losses >= self.config.max_consecutive_losses:
            return False
        return True

    def record_entry(self):
        self.daily_trades += 1

    def record_exit(self, pnl: float):
        self.consecutive_losses = self.consecutive_losses + 1 if pnl < 0 else 0


def build_signal(row: pd.Series, config: Config):
    if any(pd.isna(row.get(col)) for col in ["atr", "atr_pct", "volume_ratio", "ema_trend", "andean_signal"]):
        return None
    if not (config.min_atr_pct <= row["atr_pct"] <= config.max_atr_pct):
        return None
    if row["volume_ratio"] < config.min_volume_ratio:
        return None

    bullish = row["andean_bull"] > row["andean_bear"] and row["andean_bull"] > row["andean_signal"]
    bearish = row["andean_bear"] > row["andean_bull"] and row["andean_bear"] > row["andean_signal"]

    if config.use_trend_filter:
        bullish = bullish and row["close"] > row["ema_trend"]
        bearish = bearish and row["close"] < row["ema_trend"]

    slip = row["atr"] * config.slippage_atr_mult
    stop_distance = row["atr"] * config.atr_stop_mult
    if row["bull_cross"] and bullish:
        entry = float(row["close"] + slip)
        sl = entry - stop_distance
        tp = entry + stop_distance * config.rr
        return "BUY", entry, sl, tp
    if row["bear_cross"] and bearish:
        entry = float(row["close"] - slip)
        sl = entry + stop_distance
        tp = entry - stop_distance * config.rr
        return "SELL", entry, sl, tp
    return None


def position_size(balance: float, entry: float, sl: float, config: Config) -> float:
    risk_amount = balance * config.risk_per_trade
    stop_distance = abs(entry - sl)
    fee_per_unit = config.fee_rate * (entry + sl)
    if stop_distance <= 0:
        return 0.0
    return risk_amount / (stop_distance + fee_per_unit)


def close_pnl(position: Position, exit_price: float, config: Config):
    gross = (exit_price - position.entry) * position.qty
    if position.side == "SELL":
        gross = -gross
    fee = (position.entry * position.qty + exit_price * position.qty) * config.fee_rate
    return gross - fee, fee


def backtest(df: pd.DataFrame, config: Config, write_logs=True) -> dict:
    df = prepare(df, config)
    logger = CsvLogger(config, enabled=write_logs)
    risk = RiskState(config)
    balance = config.start_balance
    peak = balance
    max_dd = 0.0
    wins = 0
    losses = 0
    gross_profit = 0.0
    gross_loss = 0.0
    position = None
    cooldown_until = -1

    start_i = max(config.andean_length, config.ema_trend_period, 60) + 2
    for i in range(start_i, len(df)):
        now = df.index[i]
        row = df.iloc[i]

        if position is not None:
            if position.side == "BUY":
                if row["low"] <= position.sl:
                    exit_price = position.sl
                elif row["high"] >= position.tp:
                    exit_price = position.tp
                else:
                    exit_price = None
            else:
                if row["high"] >= position.sl:
                    exit_price = position.sl
                elif row["low"] <= position.tp:
                    exit_price = position.tp
                else:
                    exit_price = None

            if exit_price is not None:
                pnl, fee = close_pnl(position, exit_price, config)
                balance += pnl
                risk.record_exit(pnl)
                cooldown_until = i + config.cooldown_candles
                if pnl > 0:
                    wins += 1
                    gross_profit += pnl
                else:
                    losses += 1
                    gross_loss += abs(pnl)
                logger.trade({
                    "exit_time": now,
                    "entry_time": position.entry_time,
                    "side": position.side,
                    "entry": position.entry,
                    "sl": position.sl,
                    "tp": position.tp,
                    "exit": exit_price,
                    "qty": position.qty,
                    "pnl": pnl,
                    "fee": fee,
                    "balance": balance,
                })
                position = None

        peak = max(peak, balance)
        dd = (balance - peak) / peak
        max_dd = min(max_dd, dd)
        logger.equity({"time": now, "balance": balance, "drawdown_pct": dd * 100})

        if not risk.can_open(now, balance, position is not None, cooldown_until, i):
            continue

        signal = build_signal(row, config)
        if signal is None:
            continue
        side, entry, sl, tp = signal
        qty = position_size(balance, entry, sl, config)
        if qty <= 0:
            continue
        risk.record_entry()
        position = Position(side, now, entry, sl, tp, qty)

    trades = wins + losses
    days = max((df.index[-1].date() - df.index[0].date()).days + 1, 1)
    return {
        "length": config.andean_length,
        "signal": config.andean_signal,
        "atr_stop": config.atr_stop_mult,
        "trend_filter": config.use_trend_filter,
        "rr": config.rr,
        "start_balance": config.start_balance,
        "end_balance": balance,
        "profit_pct": (balance - config.start_balance) / config.start_balance * 100,
        "trades": trades,
        "trades_per_day": trades / days,
        "wins": wins,
        "losses": losses,
        "winrate": wins / trades * 100 if trades else 0.0,
        "profit_factor": gross_profit / gross_loss if gross_loss else 0.0,
        "max_drawdown_pct": max_dd * 100,
        "expectancy": (gross_profit - gross_loss) / trades if trades else 0.0,
        "trade_log": config.trade_log,
        "equity_log": config.equity_log,
    }


def sweep(df: pd.DataFrame, base: Config) -> pd.DataFrame:
    rows = []
    for length in [20, 34, 50, 89]:
        for sig in [9, 13]:
            for atr_mult in [1.0, 1.5, 2.0]:
                for trend in [True, False]:
                    cfg = Config(**{**base.__dict__})
                    cfg.andean_length = length
                    cfg.andean_signal = sig
                    cfg.atr_stop_mult = atr_mult
                    cfg.use_trend_filter = trend
                    rows.append(backtest(df, cfg, write_logs=False))
    result = pd.DataFrame(rows).sort_values(
        ["profit_factor", "expectancy", "max_drawdown_pct", "trades"],
        ascending=[False, False, False, False],
    )
    CsvLogger.write_results(base.results_log, result.to_dict("records"))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA_DIR / "bitcoin_365d_5m.csv"))
    parser.add_argument("--mode", choices=["backtest", "sweep"], default="sweep")
    parser.add_argument("--risk", type=float, default=0.005)
    parser.add_argument("--rr", type=float, default=2.0)
    parser.add_argument("--length", type=int, default=50)
    parser.add_argument("--signal", type=int, default=9)
    parser.add_argument("--atr-stop", type=float, default=1.5)
    parser.add_argument("--no-trend-filter", action="store_true")
    args = parser.parse_args()

    config = Config(
        data_file=args.data,
        risk_per_trade=args.risk,
        rr=args.rr,
        andean_length=args.length,
        andean_signal=args.signal,
        atr_stop_mult=args.atr_stop,
        use_trend_filter=not args.no_trend_filter,
    )
    df = load_csv(config.data_file)

    if args.mode == "sweep":
        result = sweep(df, config)
        columns = [
            "profit_pct",
            "trades",
            "trades_per_day",
            "winrate",
            "profit_factor",
            "max_drawdown_pct",
            "expectancy",
            "length",
            "signal",
            "atr_stop",
            "trend_filter",
        ]
        print("TOP ANDEAN RR 1:2 CONFIGS")
        print(result[columns].head(10).to_string(index=False))
        print(f"Saved full results to {config.results_log}")
        return

    result = backtest(df, config)
    print("ANDEAN OSCILLATOR RR 1:2 BACKTEST")
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
