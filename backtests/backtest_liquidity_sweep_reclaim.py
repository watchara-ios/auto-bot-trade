from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "reports" / "liquidity_sweep_reclaim"


@dataclass
class Config:
    initial_balance: float = 1000.0
    risk_pct: float = 0.005
    fee_rate: float = 0.0005
    spread_atr_mult: float = 0.02
    slippage_min_spread: float = 0.5
    slippage_max_spread: float = 1.5
    max_trades_per_day: int = 2
    daily_loss_limit_pct: float = 0.03
    atr_period: int = 14
    adx_period: int = 14
    swing_left: int = 3
    swing_right: int = 3
    body_min_pct: float = 0.40
    wick_min_pct: float = 0.30
    sl_atr_buffer: float = 0.20
    rr: float = 2.0
    volume_mult: float = 0.0
    require_m15_adx_below_30: bool = False
    require_atr_pct_gt_30: bool = False
    seed: int = 42


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").drop_duplicates("time").set_index("time")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["open", "high", "low", "close", "volume"]].dropna()


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


def confirmed_swings(df: pd.DataFrame, left: int, right: int) -> pd.DataFrame:
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    swing_high = np.full(len(df), np.nan)
    swing_low = np.full(len(df), np.nan)

    for center in range(left, len(df) - right):
        high_window = highs[center - left : center + right + 1]
        low_window = lows[center - left : center + right + 1]
        confirmed_at = center + right
        if highs[center] == np.max(high_window) and np.sum(high_window == highs[center]) == 1:
            swing_high[confirmed_at] = highs[center]
        if lows[center] == np.min(low_window) and np.sum(low_window == lows[center]) == 1:
            swing_low[confirmed_at] = lows[center]

    out = df.copy()
    out["confirmed_swing_high"] = pd.Series(swing_high, index=df.index).ffill()
    out["confirmed_swing_low"] = pd.Series(swing_low, index=df.index).ffill()
    return out


def prepare(m5: pd.DataFrame, m15: pd.DataFrame, config: Config) -> pd.DataFrame:
    m15 = m15.copy()
    m15["ema50"] = ema(m15["close"], 50)
    m15["ema200"] = ema(m15["close"], 200)
    m15["adx"] = adx(m15, config.adx_period)
    m15["strong_bullish"] = (m15["ema50"] > m15["ema200"]) & (m15["adx"] > 30)
    m15["strong_bearish"] = (m15["ema50"] < m15["ema200"]) & (m15["adx"] > 30)

    m5 = confirmed_swings(m5, config.swing_left, config.swing_right)
    m5["atr"] = atr(m5, config.atr_period)
    m5["atr_pct"] = m5["atr"].rolling(100).apply(lambda values: float((values <= values[-1]).mean() * 100), raw=True)
    m5["volume_sma20"] = m5["volume"].shift(1).rolling(20).mean()
    m5["volume_ratio"] = m5["volume"] / m5["volume_sma20"].replace(0, np.nan)

    candle_range = (m5["high"] - m5["low"]).replace(0, np.nan)
    m5["body_pct"] = (m5["close"] - m5["open"]).abs() / candle_range
    m5["lower_wick_pct"] = (m5[["open", "close"]].min(axis=1) - m5["low"]) / candle_range
    m5["upper_wick_pct"] = (m5["high"] - m5[["open", "close"]].max(axis=1)) / candle_range

    aligned = m5.copy()
    aligned["m15_adx"] = m15["adx"].shift(1).reindex(aligned.index, method="ffill")
    aligned["m15_strong_bullish"] = (
        m15["strong_bullish"]
        .shift(1)
        .reindex(aligned.index, method="ffill")
        .astype("boolean")
        .fillna(False)
        .astype(bool)
    )
    aligned["m15_strong_bearish"] = (
        m15["strong_bearish"]
        .shift(1)
        .reindex(aligned.index, method="ffill")
        .astype("boolean")
        .fillna(False)
        .astype(bool)
    )
    return aligned


def signal(row: pd.Series, config: Config):
    required = [
        "atr",
        "confirmed_swing_high",
        "confirmed_swing_low",
        "m15_adx",
        "body_pct",
        "lower_wick_pct",
        "upper_wick_pct",
    ]
    if any(pd.isna(row.get(col)) for col in required):
        return None
    if config.require_m15_adx_below_30 and row["m15_adx"] >= 30:
        return None
    if config.require_atr_pct_gt_30 and row.get("atr_pct", 0) <= 30:
        return None
    if config.volume_mult and row.get("volume_ratio", 0) < config.volume_mult:
        return None

    swing_low = row["confirmed_swing_low"]
    long_sweep = (
        not bool(row["m15_strong_bearish"])
        and row["low"] < swing_low
        and row["close"] > swing_low
        and row["body_pct"] >= config.body_min_pct
        and row["lower_wick_pct"] >= config.wick_min_pct
    )
    if long_sweep:
        return {"side": "BUY", "level": swing_low, "sweep_extreme": row["low"]}

    swing_high = row["confirmed_swing_high"]
    short_sweep = (
        not bool(row["m15_strong_bullish"])
        and row["high"] > swing_high
        and row["close"] < swing_high
        and row["body_pct"] >= config.body_min_pct
        and row["upper_wick_pct"] >= config.wick_min_pct
    )
    if short_sweep:
        return {"side": "SELL", "level": swing_high, "sweep_extreme": row["high"]}

    return None


def m1_confirms(m1_slice: pd.DataFrame, side: str, reclaim_high: float, reclaim_low: float):
    highs = m1_slice["high"].shift(1).rolling(3).max()
    lows = m1_slice["low"].shift(1).rolling(3).min()
    for idx, (_, row) in enumerate(m1_slice.iterrows()):
        if side == "BUY" and (row["close"] > reclaim_high or row["close"] > highs.iloc[idx]):
            return m1_slice.index[idx]
        if side == "SELL" and (row["close"] < reclaim_low or row["close"] < lows.iloc[idx]):
            return m1_slice.index[idx]
    return None


def trade_levels(row: pd.Series, entry: float, sig: dict, config: Config):
    if sig["side"] == "BUY":
        sl = sig["sweep_extreme"] - row["atr"] * config.sl_atr_buffer
        risk = entry - sl
        tp = entry + risk * config.rr
    else:
        sl = sig["sweep_extreme"] + row["atr"] * config.sl_atr_buffer
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

        m5_close_time = ts + pd.Timedelta(minutes=5)
        confirm_window = m1[(m1.index >= m5_close_time) & (m1.index < m5_close_time + pd.Timedelta(minutes=3))]
        confirm_time = m1_confirms(confirm_window, sig["side"], row["high"], row["low"])
        if confirm_time is None:
            continue

        entry_time = confirm_time + pd.Timedelta(minutes=1)
        if entry_time not in m1.index:
            continue
        entry_base = float(m1.loc[entry_time, "open"])
        spread = max(float(row["atr"]) * config.spread_atr_mult, float(row["close"]) * 0.00001)
        slippage = rng.uniform(config.slippage_min_spread, config.slippage_max_spread) * spread
        if sig["side"] == "BUY":
            entry = entry_base + spread / 2 + slippage
        else:
            entry = entry_base - spread / 2 - slippage

        levels = trade_levels(row, entry, sig, config)
        if levels is None:
            continue
        sl, tp, risk_dist = levels
        qty = (config.initial_balance * 0.005) / risk_dist
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
                "confirm_time": confirm_time,
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
                "atr_pct": row["atr_pct"],
                "volume_ratio": row["volume_ratio"],
            }
        )
    return pd.DataFrame(trades), pd.DataFrame(equity)


def profit_factor(pnl: pd.Series) -> float:
    gross_profit = pnl[pnl > 0].sum()
    gross_loss = -pnl[pnl <= 0].sum()
    return float(gross_profit / gross_loss) if gross_loss else 0.0


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
            "rejected": True,
        }
    wins = trades[trades["pnl"] > 0]
    max_dd = float(equity["drawdown_pct"].min()) if not equity.empty else 0.0
    pf = profit_factor(trades["pnl"])
    profit_pct = (trades["balance"].iloc[-1] - 1000.0) / 1000.0 * 100
    return {
        "total_trades": len(trades),
        "winrate": len(wins) / len(trades) * 100,
        "profit_factor": pf,
        "profit_pct": profit_pct,
        "max_drawdown": max_dd,
        "expectancy": trades["pnl"].mean(),
        "trades_per_day": len(trades) / max(days, 1),
        "avg_r": trades["r_multiple"].mean(),
        "rejected": bool(pf < 1.05 or max_dd < -25 or len(trades) < 100),
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


def breakdown(trades: pd.DataFrame, column: str) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows = []
    for keys, group in trades.groupby(["variant", column], dropna=False):
        variant, bucket = keys
        wins = group[group["pnl"] > 0]
        rows.append(
            {
                "variant": variant,
                column: bucket,
                "trades": len(group),
                "winrate": len(wins) / len(group) * 100 if len(group) else 0.0,
                "profit_factor": profit_factor(group["pnl"]),
                "profit_pct": group["pnl"].sum() / 1000 * 100,
                "expectancy": group["pnl"].mean() if len(group) else 0.0,
                "avg_r": group["r_multiple"].mean() if "r_multiple" in group else 0.0,
            }
        )
    return pd.DataFrame(rows)


def write_csv(path: Path, df: pd.DataFrame):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def variants() -> list[tuple[str, Config]]:
    configs = []
    for rr in [1.5, 2.0]:
        for volume_mult in [0.0, 1.2, 1.5]:
            for adx_filter in [False, True]:
                for atr_filter in [False, True]:
                    name = f"rr{rr:g}_vol{volume_mult:g}_adxlt30{int(adx_filter)}_atrgt30{int(atr_filter)}"
                    configs.append(
                        (
                            name,
                            Config(
                                rr=rr,
                                volume_mult=volume_mult,
                                require_m15_adx_below_30=adx_filter,
                                require_atr_pct_gt_30=atr_filter,
                            ),
                        )
                    )
    return configs


def main():
    m1 = load_csv(DATA_DIR / "bitcoin_2022_2025_1m.csv")
    m5_raw = load_csv(DATA_DIR / "bitcoin_2022_2025_5m.csv")
    m15 = load_csv(DATA_DIR / "bitcoin_2022_2025_15m.csv")
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
            trades["session_utc"] = pd.to_datetime(trades["timestamp"]).dt.strftime("%H:00")
            trade_frames.append(trades)
            monthly_frames.append(monthly_performance(trades))
        if not equity.empty:
            equity_frames.append(equity)

    summary = pd.DataFrame(summary_rows).sort_values(["profit_factor", "expectancy"], ascending=[False, False])
    trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    monthly = pd.concat(monthly_frames, ignore_index=True) if monthly_frames else pd.DataFrame()
    equity = pd.concat(equity_frames, ignore_index=True) if equity_frames else pd.DataFrame()

    write_csv(REPORT_DIR / "summary.csv", summary)
    write_csv(REPORT_DIR / "trades.csv", trades)
    write_csv(REPORT_DIR / "monthly_performance.csv", monthly)
    write_csv(REPORT_DIR / "equity_curve.csv", equity)
    write_csv(REPORT_DIR / "side_breakdown.csv", breakdown(trades, "side"))
    write_csv(REPORT_DIR / "session_breakdown.csv", breakdown(trades, "session_utc"))

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
        "rejected",
    ]
    print("=== Liquidity Sweep + Reclaim Backtest ===")
    print(summary[cols].head(20).to_string(index=False))
    print(f"Outputs: {REPORT_DIR}")


if __name__ == "__main__":
    main()
