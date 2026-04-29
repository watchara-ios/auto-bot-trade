import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "reports" / "pullback_structure"


@dataclass
class Config:
    initial_balance: float = 1000.0
    risk_pct: float = 0.005
    fee_rate: float = 0.0005
    spread_atr_mult: float = 0.02
    slippage_min_spread: float = 0.5
    slippage_max_spread: float = 1.5
    rr: float = 2.0
    max_trades_per_day: int = 2
    daily_loss_limit_pct: float = 0.03
    adx_min: float = 20.0
    atr_period: int = 14
    adx_period: int = 14
    structure_lookback_m15: int = 8
    structure_compare_lag: int = 4
    support_lookback_m5: int = 24
    rejection_tolerance_atr: float = 0.35
    wick_body_min: float = 1.0
    close_quality_long: float = 0.6
    close_quality_short: float = 0.4
    sl_atr_buffer: float = 0.25
    atr_median_lookback: int = 100
    use_volume_spike: bool = False
    volume_mult: float = 1.2
    seed: int = 42


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").drop_duplicates("time").set_index("time")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["open", "high", "low", "close", "volume"]].dropna()


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


def adx(df: pd.DataFrame, period: int) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_smooth = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_smooth.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_smooth.replace(0, np.nan)
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def prepare(m5: pd.DataFrame, m15: pd.DataFrame, config: Config) -> pd.DataFrame:
    m15 = m15.copy()
    m15["adx"] = adx(m15, config.adx_period)
    m15["structure_high"] = m15["high"].shift(1).rolling(config.structure_lookback_m15).max()
    m15["structure_low"] = m15["low"].shift(1).rolling(config.structure_lookback_m15).min()
    m15["higher_high"] = m15["structure_high"] > m15["structure_high"].shift(config.structure_compare_lag)
    m15["higher_low"] = m15["structure_low"] > m15["structure_low"].shift(config.structure_compare_lag)
    m15["lower_high"] = m15["structure_high"] < m15["structure_high"].shift(config.structure_compare_lag)
    m15["lower_low"] = m15["structure_low"] < m15["structure_low"].shift(config.structure_compare_lag)
    m15["trend"] = np.where(
        m15["higher_high"] & m15["higher_low"],
        1,
        np.where(m15["lower_high"] & m15["lower_low"], -1, 0),
    )

    m5 = m5.copy()
    m5["atr"] = atr(m5, config.atr_period)
    m5["atr_median"] = m5["atr"].shift(1).rolling(config.atr_median_lookback).median()
    m5["support"] = m5["low"].shift(1).rolling(config.support_lookback_m5).min()
    m5["resistance"] = m5["high"].shift(1).rolling(config.support_lookback_m5).max()
    candle_range = (m5["high"] - m5["low"]).replace(0, np.nan)
    m5["body"] = (m5["close"] - m5["open"]).abs()
    m5["lower_wick"] = m5[["open", "close"]].min(axis=1) - m5["low"]
    m5["upper_wick"] = m5["high"] - m5[["open", "close"]].max(axis=1)
    m5["close_location"] = (m5["close"] - m5["low"]) / candle_range
    m5["volume_sma20"] = m5["volume"].shift(1).rolling(20).mean()
    m5["volume_ratio"] = m5["volume"] / m5["volume_sma20"].replace(0, np.nan)

    aligned = m5.copy()
    aligned["m15_trend"] = m15["trend"].shift(1).reindex(aligned.index, method="ffill")
    aligned["m15_adx"] = m15["adx"].shift(1).reindex(aligned.index, method="ffill")
    return aligned


def signal(row: pd.Series, config: Config):
    required = ["atr", "atr_median", "support", "resistance", "m15_adx", "m15_trend"]
    if any(pd.isna(row.get(col)) for col in required):
        return None
    if row["m15_adx"] <= config.adx_min:
        return None
    if row["atr"] <= row["atr_median"]:
        return None
    if config.use_volume_spike and row.get("volume_ratio", 0) < config.volume_mult:
        return None

    tolerance = row["atr"] * config.rejection_tolerance_atr
    body = max(row["body"], row["atr"] * 0.02)

    if row["m15_trend"] == 1:
        touched_support = row["low"] <= row["support"] + tolerance
        bullish_reject = row["close"] > row["open"] and row["lower_wick"] >= body * config.wick_body_min
        close_quality = row["close_location"] >= config.close_quality_long
        reclaimed_level = row["close"] > row["support"]
        if touched_support and bullish_reject and close_quality and reclaimed_level:
            return {"side": "BUY", "structure": row["support"]}

    if row["m15_trend"] == -1:
        touched_resistance = row["high"] >= row["resistance"] - tolerance
        bearish_reject = row["close"] < row["open"] and row["upper_wick"] >= body * config.wick_body_min
        close_quality = row["close_location"] <= config.close_quality_short
        rejected_level = row["close"] < row["resistance"]
        if touched_resistance and bearish_reject and close_quality and rejected_level:
            return {"side": "SELL", "structure": row["resistance"]}

    return None


def trade_levels(row: pd.Series, entry: float, sig: dict, config: Config):
    side = sig["side"]
    if side == "BUY":
        sl = min(row["low"], sig["structure"]) - row["atr"] * config.sl_atr_buffer
        risk = entry - sl
        tp = entry + risk * config.rr
    else:
        sl = max(row["high"], sig["structure"]) + row["atr"] * config.sl_atr_buffer
        risk = sl - entry
        tp = entry - risk * config.rr
    if not np.isfinite(risk) or risk <= 0:
        return None
    return sl, tp, risk


def simulate_exit(m1: pd.DataFrame, entry_time: pd.Timestamp, side: str, sl: float, tp: float):
    window = m1[m1.index >= entry_time]
    for ts, row in window.iterrows():
        if side == "BUY":
            if row["low"] <= sl:
                return ts, sl, "SL"
            if row["high"] >= tp:
                return ts, tp, "TP"
        else:
            if row["high"] >= sl:
                return ts, sl, "SL"
            if row["low"] <= tp:
                return ts, tp, "TP"
    if window.empty:
        return entry_time, np.nan, "OPEN"
    return window.index[-1], float(window.iloc[-1]["close"]), "OPEN"


def run_backtest(symbol: str, m1: pd.DataFrame, m5: pd.DataFrame, config: Config, variant: str):
    rng = np.random.default_rng(config.seed)
    balance = config.initial_balance
    peak = balance
    max_dd = 0.0
    unavailable_until = pd.Timestamp.min
    daily_trades = {}
    daily_pnl = {}
    trades = []
    equity = []

    for i in range(len(m5) - 1):
        ts = m5.index[i]
        row = m5.iloc[i]
        peak = max(peak, balance)
        max_dd = min(max_dd, (balance - peak) / peak)
        equity.append({"time": ts, "symbol": symbol, "variant": variant, "balance": balance, "drawdown_pct": max_dd * 100})

        if ts <= unavailable_until:
            continue
        day = ts.date()
        if daily_trades.get(day, 0) >= config.max_trades_per_day:
            continue
        if daily_pnl.get(day, 0.0) <= -(config.initial_balance * config.daily_loss_limit_pct):
            continue

        sig = signal(row, config)
        if sig is None:
            continue

        next_row = m5.iloc[i + 1]
        entry_time = m5.index[i + 1]
        spread = max(float(row["atr"]) * config.spread_atr_mult, float(row["close"]) * 0.00001)
        slippage = rng.uniform(config.slippage_min_spread, config.slippage_max_spread) * spread
        if sig["side"] == "BUY":
            entry = float(next_row["open"]) + spread / 2 + slippage
        else:
            entry = float(next_row["open"]) - spread / 2 - slippage

        levels = trade_levels(row, entry, sig, config)
        if levels is None:
            continue
        sl, tp, risk_dist = levels
        qty = (config.initial_balance * config.risk_pct) / risk_dist
        if qty <= 0 or not np.isfinite(qty):
            continue

        exit_time, exit_price, result = simulate_exit(m1, entry_time, sig["side"], sl, tp)
        unavailable_until = exit_time
        if result == "OPEN" or not np.isfinite(exit_price):
            continue

        gross = (exit_price - entry) * qty
        if sig["side"] == "SELL":
            gross = -gross
        fee = (entry * qty + exit_price * qty) * config.fee_rate
        pnl = gross - fee
        balance += pnl
        daily_trades[day] = daily_trades.get(day, 0) + 1
        daily_pnl[day] = daily_pnl.get(day, 0.0) + pnl

        initial_risk = risk_dist * qty
        trades.append(
            {
                "symbol": symbol,
                "variant": variant,
                "timestamp": ts,
                "entry_time": entry_time,
                "exit_time": exit_time,
                "side": sig["side"],
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "rr": config.rr,
                "qty": qty,
                "spread": spread,
                "slippage": slippage,
                "fee": fee,
                "gross_pnl": gross,
                "pnl": pnl,
                "r_multiple": pnl / initial_risk if initial_risk else np.nan,
                "result": result,
                "balance": balance,
                "m15_adx": row["m15_adx"],
                "atr": row["atr"],
                "volume_ratio": row["volume_ratio"],
            }
        )

    return pd.DataFrame(trades), pd.DataFrame(equity)


def metrics(trades: pd.DataFrame, equity: pd.DataFrame, days: int) -> dict:
    if trades.empty:
        return {
            "total_trades": 0,
            "winrate": 0.0,
            "profit_factor": 0.0,
            "profit_pct": 0.0,
            "max_drawdown": 0.0,
            "expectancy": 0.0,
            "trades_per_day": 0.0,
            "avg_r": 0.0,
        }
    wins = trades[trades["pnl"] > 0]
    losses = trades[trades["pnl"] <= 0]
    gross_profit = wins["pnl"].sum()
    gross_loss = -losses["pnl"].sum()
    return {
        "total_trades": len(trades),
        "winrate": len(wins) / len(trades) * 100,
        "profit_factor": gross_profit / gross_loss if gross_loss else 0.0,
        "profit_pct": (trades["balance"].iloc[-1] - 1000.0) / 1000.0 * 100,
        "max_drawdown": float(equity["drawdown_pct"].min()) if not equity.empty else 0.0,
        "expectancy": trades["pnl"].mean(),
        "trades_per_day": len(trades) / max(days, 1),
        "avg_r": trades["r_multiple"].mean(),
    }


def monthly_performance(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    df = trades.copy()
    df["month"] = pd.to_datetime(df["exit_time"]).dt.to_period("M").astype(str)
    return df.groupby(["symbol", "variant", "month"]).agg(
        trades=("pnl", "count"),
        pnl=("pnl", "sum"),
        wins=("pnl", lambda x: int((x > 0).sum())),
        losses=("pnl", lambda x: int((x <= 0).sum())),
    ).reset_index()


def write_csv(path: Path, df: pd.DataFrame):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def variants() -> list[tuple[str, Config]]:
    base = Config()
    return [
        ("pullback_base", base),
        ("pullback_volume_1_2", Config(use_volume_spike=True, volume_mult=1.2)),
        ("pullback_volume_1_5", Config(use_volume_spike=True, volume_mult=1.5)),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m1", default=str(DATA_DIR / "bitcoin_2022_2025_1m.csv"))
    parser.add_argument("--m5", default=str(DATA_DIR / "bitcoin_2022_2025_5m.csv"))
    parser.add_argument("--m15", default=str(DATA_DIR / "bitcoin_2022_2025_15m.csv"))
    args = parser.parse_args()

    m1 = load_csv(Path(args.m1))
    m5_raw = load_csv(Path(args.m5))
    m15 = load_csv(Path(args.m15))
    days = max((m5_raw.index[-1].date() - m5_raw.index[0].date()).days + 1, 1)

    summary_rows = []
    trade_frames = []
    equity_frames = []
    monthly_frames = []
    for name, config in variants():
        m5 = prepare(m5_raw, m15, config)
        trades, equity = run_backtest("BTCUSDT", m1, m5, config, name)
        summary_rows.append({"symbol": "BTCUSDT", "variant": name, "days": days, **metrics(trades, equity, days)})
        if not trades.empty:
            trade_frames.append(trades)
            monthly_frames.append(monthly_performance(trades))
        if not equity.empty:
            equity_frames.append(equity)

    summary = pd.DataFrame(summary_rows).sort_values(["profit_factor", "expectancy"], ascending=[False, False])
    trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    equity = pd.concat(equity_frames, ignore_index=True) if equity_frames else pd.DataFrame()
    monthly = pd.concat(monthly_frames, ignore_index=True) if monthly_frames else pd.DataFrame()

    write_csv(REPORT_DIR / "summary.csv", summary)
    write_csv(REPORT_DIR / "trades.csv", trades)
    write_csv(REPORT_DIR / "equity_curve.csv", equity)
    write_csv(REPORT_DIR / "monthly_performance.csv", monthly)

    cols = [
        "variant",
        "total_trades",
        "winrate",
        "profit_factor",
        "profit_pct",
        "max_drawdown",
        "expectancy",
        "trades_per_day",
        "avg_r",
    ]
    print("=== Pullback Structure Backtest ===")
    print(summary[cols].to_string(index=False))
    print(f"Outputs: {REPORT_DIR}")


if __name__ == "__main__":
    main()
