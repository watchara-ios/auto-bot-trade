import argparse
import csv
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "reports" / "realistic_donchian"


@dataclass
class Config:
    initial_balance: float = 1000.0
    fixed_risk_base: bool = True
    fee_rate: float = 0.0005
    spread_atr_mult: float = 0.02
    slippage_min_spread: float = 0.5
    slippage_max_spread: float = 1.5
    entry_delay_minutes: int = 0
    daily_loss_limit_pct: float = 0.03
    max_trades_per_day: int = 2
    donchian_n: int = 20
    swing_lookback: int = 8
    adx_period: int = 14
    atr_period: int = 14
    adx_min: float = 20.0
    rr: float = 2.0
    tier_a_risk: float = 0.0025
    tier_b_risk: float = 0.01
    breakout_body_mult: float = 1.5
    breakout_atr_mult: float = 0.5
    max_wick_pct: float = 0.5
    close_quality_min: float = 0.6
    use_volume_filter: bool = False
    volume_mult: float = 1.2
    use_atr_expansion: bool = False
    use_bb_width_expansion: bool = False
    tier_b_bb_close: bool = False
    tier_b_volume_required: bool = False
    disable_tier_b: bool = False
    require_tier_b: bool = False
    improve_tier_a: bool = False
    tier_a_volume_mult: float = 1.5
    tier_a_body_mult: float = 1.2
    allowed_side: Optional[str] = None
    exclude_utc_hours: tuple[int, ...] = ()
    include_utc_hours: Optional[tuple[int, ...]] = None
    adx_max: Optional[float] = None
    use_adaptive_adx_cap: bool = False
    atr_percentile_min: Optional[float] = None
    exclude_weekdays: tuple[str, ...] = ()
    max_bars_after_donchian_expansion: Optional[int] = None
    seed: int = 42
    max_hold_minutes: Optional[int] = None
    use_regime_filter: bool = False
    regime_adx_min: float = 20.0
    use_weekly_regime: bool = False
    weekly_adx_min: float = 25.0
    weekly_swing_lookback: int = 1


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
    plus_dm = (high.diff()).where((high.diff() > -low.diff()) & (high.diff() > 0), 0.0)
    minus_dm = (-low.diff()).where((-low.diff() > high.diff()) & (-low.diff() > 0), 0.0)
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
    m15["ema50"] = ema(m15["close"], 50)
    m15["ema200"] = ema(m15["close"], 200)
    m15["trend"] = np.where(m15["ema50"] > m15["ema200"], 1, np.where(m15["ema50"] < m15["ema200"], -1, 0))
    m15["adx"] = adx(m15, config.adx_period)
    m15["adx_rising"] = m15["adx"] > m15["adx"].shift(1)
    m15["adx_rolling_max50"] = m15["adx"].shift(1).rolling(50).max()

    m5 = m5.copy()
    m5["atr"] = atr(m5, config.atr_period)
    m5["atr_sma50"] = m5["atr"].shift(1).rolling(50).mean()
    m5["atr_percentile_100"] = m5["atr"].rolling(100).apply(
        lambda values: float((values <= values[-1]).mean() * 100),
        raw=True,
    )
    m5["donchian_high"] = m5["high"].shift(1).rolling(config.donchian_n).max()
    m5["donchian_low"] = m5["low"].shift(1).rolling(config.donchian_n).min()
    m5["donchian_width"] = m5["donchian_high"] - m5["donchian_low"]
    expansion = m5["donchian_width"] > m5["donchian_width"].shift(1)
    bars_since_expansion = []
    count = np.nan
    for is_expanding in expansion.fillna(False):
        if is_expanding:
            count = 0
        elif not pd.isna(count):
            count += 1
        bars_since_expansion.append(count)
    m5["bars_since_donchian_expansion"] = bars_since_expansion
    m5["swing_high"] = m5["high"].shift(1).rolling(config.swing_lookback).max()
    m5["swing_low"] = m5["low"].shift(1).rolling(config.swing_lookback).min()
    candle_range = (m5["high"] - m5["low"]).replace(0, np.nan)
    m5["body"] = (m5["close"] - m5["open"]).abs()
    m5["avg_body20"] = m5["body"].shift(1).rolling(20).mean()
    m5["upper_wick"] = m5["high"] - m5[["open", "close"]].max(axis=1)
    m5["lower_wick"] = m5[["open", "close"]].min(axis=1) - m5["low"]
    m5["wick_pct"] = m5[["upper_wick", "lower_wick"]].max(axis=1) / candle_range
    m5["close_location"] = (m5["close"] - m5["low"]) / candle_range
    m5["volume_sma20"] = m5["volume"].shift(1).rolling(20).mean()
    m5["volume_ratio"] = m5["volume"] / m5["volume_sma20"].replace(0, np.nan)
    basis = m5["close"].rolling(20).mean()
    dev = m5["close"].rolling(20).std()
    m5["bb_upper"] = basis + 2 * dev
    m5["bb_lower"] = basis - 2 * dev
    m5["bb_width"] = (m5["bb_upper"] - m5["bb_lower"]) / basis.replace(0, np.nan)
    m5["bb_width_expanding"] = m5["bb_width"] > m5["bb_width"].shift(1)

    aligned = m5.copy()
    aligned["m15_trend"] = m15["trend"].shift(1).reindex(aligned.index, method="ffill")
    aligned["m15_ema50"] = m15["ema50"].shift(1).reindex(aligned.index, method="ffill")
    aligned["m15_ema200"] = m15["ema200"].shift(1).reindex(aligned.index, method="ffill")
    aligned["m15_adx"] = m15["adx"].shift(1).reindex(aligned.index, method="ffill")
    aligned["m15_adx_rolling_max50"] = m15["adx_rolling_max50"].shift(1).reindex(aligned.index, method="ffill")
    aligned["m15_adx_rising"] = (
        m15["adx_rising"]
        .shift(1)
        .reindex(aligned.index, method="ffill")
        .astype("boolean")
        .fillna(False)
        .astype(bool)
    )

    # Daily regime: resample M15 → 1D for higher-timeframe trend/ADX filter
    d1 = m15.resample("1D").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna(subset=["open", "close"])
    d1["d1_adx"] = adx(d1, 14)
    d1_ema50 = ema(d1["close"], 50)
    d1_ema200 = ema(d1["close"], 200)
    d1["d1_trend"] = np.where(d1_ema50 > d1_ema200, 1, np.where(d1_ema50 < d1_ema200, -1, 0))
    aligned["d1_adx"] = d1["d1_adx"].shift(1).reindex(aligned.index, method="ffill")
    aligned["d1_trend"] = d1["d1_trend"].shift(1).reindex(aligned.index, method="ffill")

    # Weekly regime: resample M15 → W1 for swing structure (HH+HL / LH+LL) + ADX
    w1 = m15.resample("W").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna(subset=["open", "close"])
    w1["w1_adx"] = adx(w1, 14)
    w1_hh = w1["high"] > w1["high"].shift(1)
    w1_hl = w1["low"] > w1["low"].shift(1)
    w1_lh = w1["high"] < w1["high"].shift(1)
    w1_ll = w1["low"] < w1["low"].shift(1)
    w1["w1_bull_1"] = w1_hh & w1_hl
    w1["w1_bear_1"] = w1_lh & w1_ll
    # Stricter: 2 consecutive weeks of same structure
    w1["w1_bull_2"] = w1["w1_bull_1"] & w1["w1_bull_1"].shift(1)
    w1["w1_bear_2"] = w1["w1_bear_1"] & w1["w1_bear_1"].shift(1)
    for col in ["w1_adx", "w1_bull_1", "w1_bear_1", "w1_bull_2", "w1_bear_2"]:
        aligned[col] = (
            w1[col].shift(1).reindex(aligned.index, method="ffill").fillna(False)
        )

    return aligned


def signal(row: pd.Series, config: Config) -> Optional[dict]:
    if pd.isna(row.get("donchian_high")) or pd.isna(row.get("m15_adx")) or pd.isna(row.get("atr")):
        return None
    side = None
    if row["close"] > row["donchian_high"]:
        side = "BUY"
    elif row["close"] < row["donchian_low"]:
        side = "SELL"
    if side is None:
        return None
    if config.allowed_side and side != config.allowed_side:
        return None
    if side == "BUY" and row["m15_trend"] != 1:
        return None
    if side == "SELL" and row["m15_trend"] != -1:
        return None
    if config.use_regime_filter:
        if row.get("d1_adx", 0) <= config.regime_adx_min:
            return None
        d1_trend = row.get("d1_trend", 0)
        if side == "BUY" and d1_trend != 1:
            return None
        if side == "SELL" and d1_trend != -1:
            return None
    if config.use_weekly_regime:
        lb = min(max(config.weekly_swing_lookback, 1), 2)
        if row.get("w1_adx", 0) <= config.weekly_adx_min:
            return None
        if side == "BUY" and not bool(row.get(f"w1_bull_{lb}", False)):
            return None
        if side == "SELL" and not bool(row.get(f"w1_bear_{lb}", False)):
            return None
    if row["m15_adx"] <= config.adx_min or not bool(row["m15_adx_rising"]):
        return None
    if config.adx_max is not None and row["m15_adx"] >= config.adx_max:
        return None
    if config.use_adaptive_adx_cap and row["m15_adx"] >= row.get("m15_adx_rolling_max50", np.inf):
        return None
    if config.atr_percentile_min is not None and row.get("atr_percentile_100", 0) <= config.atr_percentile_min:
        return None
    if (
        config.max_bars_after_donchian_expansion is not None
        and row.get("bars_since_donchian_expansion", np.inf) > config.max_bars_after_donchian_expansion
    ):
        return None
    if side == "BUY" and not (row["high"] > row["swing_high"]):
        return None
    if side == "SELL" and not (row["low"] < row["swing_low"]):
        return None
    if config.use_volume_filter and row.get("volume_ratio", 0) < config.volume_mult:
        return None
    if config.use_atr_expansion and not (row["atr"] > row["atr_sma50"]):
        return None
    if config.use_bb_width_expansion and not bool(row.get("bb_width_expanding", False)):
        return None

    if side == "BUY":
        move_atr = (row["close"] - row["donchian_high"]) / row["atr"]
        close_quality = row["close_location"]
        bb_close = row["close"] > row["bb_upper"]
    else:
        move_atr = (row["donchian_low"] - row["close"]) / row["atr"]
        close_quality = 1 - row["close_location"]
        bb_close = row["close"] < row["bb_lower"]
    body_ok = row["body"] > row["avg_body20"] * config.breakout_body_mult
    move_ok = move_atr > config.breakout_atr_mult
    wick_ok = row["wick_pct"] <= config.max_wick_pct
    close_ok = close_quality >= config.close_quality_min
    volume_ok = row.get("volume_ratio", 0) >= config.volume_mult
    tier_b = body_ok and move_ok and wick_ok and close_ok
    if config.tier_b_bb_close:
        tier_b = tier_b and bb_close
    if config.tier_b_volume_required:
        tier_b = tier_b and volume_ok
    if config.disable_tier_b:
        tier_b = False
    if config.require_tier_b and not tier_b:
        return None
    if config.improve_tier_a and not tier_b:
        tier_a_ok = (
            row.get("volume_ratio", 0) >= config.tier_a_volume_mult
            and row["body"] > row["avg_body20"] * config.tier_a_body_mult
            and row["atr"] > row["atr_sma50"]
        )
        if not tier_a_ok:
            return None
    return {"side": side, "tier": "B" if tier_b else "A"}


def trade_levels(row: pd.Series, entry: float, side: str, config: Config):
    if side == "BUY":
        sl = min(entry - row["atr"], row["swing_low"])
        risk = entry - sl
        tp = entry + risk * config.rr
    else:
        sl = max(entry + row["atr"], row["swing_high"])
        risk = sl - entry
        tp = entry - risk * config.rr
    if not np.isfinite(risk) or risk <= 0:
        return None
    return sl, tp, risk


def simulate_exit(m1: pd.DataFrame, entry_time: pd.Timestamp, side: str, sl: float, tp: float, max_hold_minutes: Optional[int] = None):
    cutoff = entry_time + pd.Timedelta(minutes=max_hold_minutes) if max_hold_minutes is not None else None
    window = m1[m1.index >= entry_time]
    for ts, row in window.iterrows():
        if cutoff is not None and ts >= cutoff:
            return ts, float(row["close"]), "TIMEOUT"
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


def run_backtest(symbol: str, m1: pd.DataFrame, m5: pd.DataFrame, config: Config, variant: str, split: str):
    rng = np.random.default_rng(config.seed)
    balance = config.initial_balance
    peak = balance
    max_dd = 0.0
    trades = []
    equity = []
    daily_trades = {}
    daily_pnl = {}
    unavailable_until = pd.Timestamp.min

    for i in range(len(m5) - 1):
        ts = m5.index[i]
        row = m5.iloc[i]
        peak = max(peak, balance)
        max_dd = min(max_dd, (balance - peak) / peak)
        equity.append({"time": ts, "symbol": symbol, "variant": variant, "split": split, "balance": balance, "drawdown_pct": max_dd * 100})
        if ts <= unavailable_until:
            continue
        if config.include_utc_hours is not None and ts.hour not in config.include_utc_hours:
            continue
        if config.exclude_utc_hours and ts.hour in config.exclude_utc_hours:
            continue
        if config.exclude_weekdays and ts.day_name() in config.exclude_weekdays:
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
        if config.entry_delay_minutes > 0:
            entry_time = entry_time + pd.Timedelta(minutes=config.entry_delay_minutes)
            if entry_time not in m1.index:
                continue
            entry_base = float(m1.loc[entry_time, "open"])
        else:
            entry_base = float(next_row["open"])
        spread = max(float(row["atr"]) * config.spread_atr_mult, float(row["close"]) * 0.00001)
        slippage = rng.uniform(config.slippage_min_spread, config.slippage_max_spread) * spread
        if sig["side"] == "BUY":
            entry = entry_base + spread / 2 + slippage
        else:
            entry = entry_base - spread / 2 - slippage
        levels = trade_levels(row, entry, sig["side"], config)
        if levels is None:
            continue
        sl, tp, risk_dist = levels
        risk_pct = config.tier_b_risk if sig["tier"] == "B" else config.tier_a_risk
        risk_base = config.initial_balance if config.fixed_risk_base else balance
        qty = (risk_base * risk_pct) / risk_dist
        if qty <= 0 or not np.isfinite(qty):
            continue
        exit_time, exit_price, result = simulate_exit(m1, entry_time, sig["side"], sl, tp, config.max_hold_minutes)
        unavailable_until = exit_time
        if result == "OPEN" or not np.isfinite(exit_price):
            continue
        gross = (exit_price - entry) * qty
        if sig["side"] == "SELL":
            gross = -gross
        fees = (entry * qty + exit_price * qty) * config.fee_rate
        pnl = gross - fees
        balance += pnl
        daily_trades[day] = daily_trades.get(day, 0) + 1
        daily_pnl[day] = daily_pnl.get(day, 0.0) + pnl
        trades.append(
            {
                "symbol": symbol,
                "variant": variant,
                "split": split,
                "timestamp": ts,
                "entry_time": entry_time,
                "exit_time": exit_time,
                "side": sig["side"],
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "rr": config.rr,
                "tier": sig["tier"],
                "qty": qty,
                "spread": spread,
                "slippage": slippage,
                "fee": fees,
                "pnl": pnl,
                "result": result,
                "balance": balance,
                "m15_adx": row.get("m15_adx"),
                "atr": row.get("atr"),
                "atr_percentile_100": row.get("atr_percentile_100"),
                "volume_ratio": row.get("volume_ratio"),
            }
        )
    return pd.DataFrame(trades), pd.DataFrame(equity)


def metrics(trades: pd.DataFrame, equity: pd.DataFrame, days: int) -> dict:
    if trades.empty:
        return {
            "total_trades": 0,
            "winrate": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "expectancy": 0.0,
            "trades_per_day": 0.0,
            "profit_pct": 0.0,
        }
    wins = trades[trades["pnl"] > 0]
    losses = trades[trades["pnl"] <= 0]
    gross_profit = wins["pnl"].sum()
    gross_loss = -losses["pnl"].sum()
    start = 1000.0
    end = trades["balance"].iloc[-1]
    return {
        "total_trades": len(trades),
        "winrate": len(wins) / len(trades) * 100,
        "profit_factor": gross_profit / gross_loss if gross_loss else 0.0,
        "max_drawdown": float(equity["drawdown_pct"].min()) if not equity.empty else 0.0,
        "expectancy": trades["pnl"].mean(),
        "trades_per_day": len(trades) / max(days, 1),
        "profit_pct": (end - start) / start * 100,
    }


def monthly_performance(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    df = trades.copy()
    df["month"] = pd.to_datetime(df["exit_time"]).dt.to_period("M").astype(str)
    return df.groupby(["symbol", "variant", "split", "month"]).agg(
        trades=("pnl", "count"),
        pnl=("pnl", "sum"),
        wins=("pnl", lambda x: int((x > 0).sum())),
        losses=("pnl", lambda x: int((x <= 0).sum())),
    ).reset_index()


def find_symbol_files(symbol: str) -> Optional[dict]:
    prefix_map = {
        "BTCUSDT": "bitcoin",
        "ETHUSDT": "ethereum",
        "SOLUSDT": "solana",
    }
    prefixes = [symbol.lower(), symbol.replace("USDT", "").lower(), prefix_map.get(symbol, "").lower()]
    suffixes = [
        "2022_2025",
        "2022_2024",
        "2023_2025",
        "365d",
    ]
    for prefix in [p for p in prefixes if p]:
        for suffix in suffixes:
            paths = {
                "m1": DATA_DIR / f"{prefix}_{suffix}_1m.csv",
                "m5": DATA_DIR / f"{prefix}_{suffix}_5m.csv",
                "m15": DATA_DIR / f"{prefix}_{suffix}_15m.csv",
            }
            if all(path.exists() for path in paths.values()):
                return paths

        m1_candidates = sorted(DATA_DIR.glob(f"{prefix}_*_1m.csv"), reverse=True)
        for m1_path in m1_candidates:
            stem = m1_path.name.removesuffix("_1m.csv")
            paths = {
                "m1": m1_path,
                "m5": DATA_DIR / f"{stem}_5m.csv",
                "m15": DATA_DIR / f"{stem}_15m.csv",
            }
            if all(path.exists() for path in paths.values()):
                return paths
    return None


def variant_group(name: str) -> str:
    if name == "base":
        return "Base strategy"
    if name.startswith("volume"):
        return "+Volume filter"
    if name.startswith("volatility"):
        return "+Volatility filter"
    if name.startswith("both"):
        return "+Both filters"
    return "Tier B enhancement"


def variants() -> list[tuple[str, Config]]:
    base = Config()
    out = [("base", base)]
    for mult in [1.2, 1.5, 2.0]:
        out.append((f"volume_{mult}", replace(base, use_volume_filter=True, volume_mult=mult)))
    out.append(("volatility_atr", replace(base, use_atr_expansion=True)))
    out.append(("volatility_bb", replace(base, use_bb_width_expansion=True)))
    out.append(("both_vol1_2_atr", replace(base, use_volume_filter=True, volume_mult=1.2, use_atr_expansion=True)))
    out.append(("both_vol1_5_atr", replace(base, use_volume_filter=True, volume_mult=1.5, use_atr_expansion=True)))
    out.append(("both_vol1_2_bb", replace(base, use_volume_filter=True, volume_mult=1.2, use_bb_width_expansion=True)))
    out.append(("tier_b_bb_volume", replace(base, tier_b_bb_close=True, tier_b_volume_required=True, volume_mult=1.5)))
    return out


def split_data(m1: pd.DataFrame, m5: pd.DataFrame, m15: pd.DataFrame, split_name: str):
    if split_name == "train_2022_2023":
        start, end = "2022-01-01", "2023-12-31 23:59:59"
    elif split_name == "test_2024_2025":
        start, end = "2024-01-01", "2025-12-31 23:59:59"
    else:
        return m1, m5, m15
    return m1.loc[start:end], m5.loc[start:end], m15.loc[start:end]


def write_csv(path: Path, df: pd.DataFrame):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT")
    parser.add_argument("--splits", default="train_2022_2023,test_2024_2025,available")
    args = parser.parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    all_trades = []
    all_equity = []
    warnings = []
    for symbol in [s.strip().upper() for s in args.symbols.split(",") if s.strip()]:
        files = find_symbol_files(symbol)
        if not files:
            warnings.append(f"{symbol}: missing M1/M5/M15 CSV files")
            continue
        m1_raw = load_csv(files["m1"])
        m5_raw = load_csv(files["m5"])
        m15_raw = load_csv(files["m15"])
        for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
            m1_s, m5_s, m15_s = split_data(m1_raw, m5_raw, m15_raw, split)
            if len(m5_s) < 300 or len(m15_s) < 220 or len(m1_s) < 1000:
                warnings.append(f"{symbol} {split}: insufficient data m1={len(m1_s)} m5={len(m5_s)} m15={len(m15_s)}")
                continue
            days = max((m5_s.index[-1].date() - m5_s.index[0].date()).days + 1, 1)
            for name, cfg in variants():
                m5 = prepare(m5_s, m15_s, cfg)
                trades, equity = run_backtest(symbol, m1_s, m5, cfg, name, split)
                row = {"symbol": symbol, "variant": name, "variant_group": variant_group(name), "split": split, "days": days, **metrics(trades, equity, days)}
                row["rejected"] = bool(row["max_drawdown"] < -25 or row["total_trades"] < 50)
                row["reject_reason"] = "; ".join(
                    reason
                    for reason in [
                        "max_drawdown > 25%" if row["max_drawdown"] < -25 else "",
                        "total_trades < 50" if row["total_trades"] < 50 else "",
                    ]
                    if reason
                )
                summary_rows.append(row)
                if not trades.empty:
                    all_trades.append(trades)
                if not equity.empty:
                    all_equity.append(equity)

    summary = pd.DataFrame(summary_rows)
    trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    equity = pd.concat(all_equity, ignore_index=True) if all_equity else pd.DataFrame()
    monthly = monthly_performance(trades)
    write_csv(REPORT_DIR / "summary.csv", summary)
    write_csv(REPORT_DIR / "trades.csv", trades)
    write_csv(REPORT_DIR / "equity_curve.csv", equity)
    write_csv(REPORT_DIR / "monthly_performance.csv", monthly)
    if warnings:
        (REPORT_DIR / "warnings.txt").write_text("\n".join(warnings), encoding="utf-8")

    print("=== Realistic Donchian Backtest ===")
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(" -", warning)
    if summary.empty:
        print("No runnable datasets found.")
        return
    cols = ["symbol", "variant", "split", "total_trades", "winrate", "profit_factor", "max_drawdown", "expectancy", "trades_per_day", "profit_pct", "rejected"]
    print(summary[cols].sort_values(["symbol", "split", "profit_factor"], ascending=[True, True, False]).to_string(index=False))
    print(f"Outputs: {REPORT_DIR}")


if __name__ == "__main__":
    main()
