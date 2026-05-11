"""
forex_bot.py — Donchian Breakout Forex Bot (MetaTrader 5)
Refactored: clean structure, fixed AI-watchlist gate, complete core config.
"""

import json
import os
import sys
import time
import builtins
from datetime import datetime, time as dtime, timezone
from pathlib import Path

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

# ── stdout safety (Windows console) ──────────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── emoji → ASCII fallback (safe on any terminal) ────────────────────────────
_EMOJI_MAP = {
    "🔌": "[CONNECT]", "🔎": "[SCAN]",   "✅": "[OK]",      "⛔": "[BLOCK]",
    "🧠": "[AI]",      "❌": "[ERROR]",  "🧪": "[DRY_RUN]", "📌": "[ORDER]",
    "🚀": "[START]",   "🌙": "[SESSION]","⏳": "[WAIT]",    "👀": "[WATCH]",
    "⏸":  "[SKIP]",   "🛑": "[STOP]",   "🎯": "[SELECT]",  "⚠️": "[WARN]",
}

def _safe(v: object) -> str:
    s = str(v)
    for src, dst in _EMOJI_MAP.items():
        s = s.replace(src, dst)
    return s.encode("ascii", errors="ignore").decode("ascii")

def print(*args, **kwargs):  # noqa: A001
    builtins.print(*[_safe(a) for a in args], **kwargs)

# ── project imports ───────────────────────────────────────────────────────────
BOT_DIR      = Path(__file__).resolve().parent
PROJECT_ROOT = BOT_DIR.parent
sys.path.insert(0, str(BOT_DIR))

from donchian_core import DonchianCoreConfig, latest_signal, _REJECT_STATS
from demo_testcase_logger import log_demo_testcase
from notifier import (
    notify_bot_started, notify_error,
    notify_order_opened, notify_order_result,
    notify_last_error, notify_reconnected,
)

try:
    from dotenv import load_dotenv
    _env = PROJECT_ROOT / ".env"
    load_dotenv(_env if _env.exists() else None)
except ImportError:
    pass

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

class Config:
    # Symbol
    SYMBOL         = os.getenv("FOREX_SYMBOL", "XAUUSD")
    SYMBOL_ALIASES = ["XAUUSD", "GOLD"]
    LOT            = float(os.getenv("FOREX_LOT", "0.01"))

    # Timeframes
    TF_TREND = mt5.TIMEFRAME_M15   # trend direction
    TF_ENTRY = mt5.TIMEFRAME_M5    # signal
    TF_EXEC  = mt5.TIMEFRAME_M1    # execution confirmation
    BARS     = 500

    # Indicators — backtest shows EMA(50/200) > EMA(20/50) for M15 trend quality
    EMA_FAST   = int(os.getenv("FOREX_EMA_FAST", "50"))
    EMA_SLOW   = int(os.getenv("FOREX_EMA_SLOW", "200"))
    EMA_BIG    = 200
    RSI_PERIOD = 14
    MAGIC_NUMBER = int(os.getenv("FOREX_MAGIC_NUMBER", "20260424"))

    # Strategy
    RR                  = float(os.getenv("FOREX_RR", "2.5"))     # raised from 2.0
    TIER_A_RISK_PERCENT = float(os.getenv("FOREX_TIER_A_RISK", "0.25"))
    TIER_B_RISK_PERCENT = float(os.getenv("FOREX_TIER_B_RISK", "1.0"))
    DONCHIAN_N          = int(os.getenv("FOREX_DONCHIAN_N", "20"))
    ADX_MIN             = float(os.getenv("FOREX_ADX_MIN", "18.0"))   # loosened from 20
    ADX_MAX             = float(os.getenv("FOREX_ADX_MAX", "50.0"))   # was missing → blocked strong trends
    ATR_PERCENTILE_MIN  = float(os.getenv("FOREX_ATR_PCT_MIN", "30.0"))  # loosened from 50
    MIN_ATR_PCT         = float(os.getenv("FOREX_MIN_ATR_PCT", "0.0002"))  # fix: was 0.0005 → blocked GBPUSD/USDJPY
    VOLUME_MULT         = float(os.getenv("FOREX_VOLUME_MULT", "0.8"))   # loosened from 1.0
    ALLOWED_SIDE        = os.getenv("FOREX_ALLOWED_SIDE", "BOTH")
    REQUIRE_ATR_EXPANSION = os.getenv("FOREX_REQUIRE_ATR_EXPANSION", "false").lower() == "true"

    # D1 Regime filter — only trade when daily trend is clear (backtest: PF 0.60→0.90)
    USE_D1_REGIME   = os.getenv("FOREX_USE_D1_REGIME", "true").lower() == "true"
    D1_ADX_MIN      = float(os.getenv("FOREX_D1_ADX_MIN", "20.0"))
    D1_BARS         = 260   # ~1 year of daily bars for EMA(200)

    # Pro techniques (Minervini + ICT)
    ADX_BARS_RISING = int(os.getenv("FOREX_ADX_BARS_RISING", "1"))      # loosened from 2
    BREAKEVEN_R     = float(os.getenv("FOREX_BREAKEVEN_R", "0.5"))
    KILL_ZONE_ONLY  = os.getenv("FOREX_KILL_ZONE_ONLY", "false").lower() == "true"  # off by default
    KILL_ZONES_UTC  = [(7, 9), (12, 14)]   # London open, NY open
    CORR_GROUPS     = [
        {"EURUSD", "GBPUSD", "EURGBP"},
        {"USDJPY", "USDCHF", "USDCAD"},
        {"AUDUSD", "NZDUSD", "AUDNZD"},
    ]

    # Risk
    MAX_TRADES_PER_DAY    = int(os.getenv("FOREX_MAX_TRADES_PER_DAY", "5"))   # raised from 3
    MAX_DAILY_LOSS_PCT    = float(os.getenv("FOREX_MAX_DAILY_LOSS_PCT", "3.0"))
    MAX_SPREAD_POINTS     = int(os.getenv("FOREX_MAX_SPREAD_POINTS", "80"))
    # Gold (XAUUSD) spread is quoted in different point units — needs higher limit
    MAX_SPREAD_POINTS_GOLD = int(os.getenv("FOREX_MAX_SPREAD_POINTS_GOLD", "500"))
    MIN_TREND_GAP_PCT     = float(os.getenv("FOREX_MIN_TREND_GAP_PCT", "0.00035"))
    MIN_M5_ATR_POINTS     = int(os.getenv("FOREX_MIN_M5_ATR_POINTS", "120"))
    MAX_M5_ATR_POINTS     = int(os.getenv("FOREX_MAX_M5_ATR_POINTS", "1200"))
    MIN_BODY_RATIO        = float(os.getenv("FOREX_MIN_BODY_RATIO", "0.35"))
    MIN_VOLUME_MULT       = float(os.getenv("FOREX_MIN_VOLUME_MULT", "1.0"))

    # Session — Thai time (UTC+7); bot runs on local clock
    TRADE_START_HOUR  = int(os.getenv("FOREX_TRADE_START_HOUR", "14"))
    TRADE_END_HOUR    = int(os.getenv("FOREX_TRADE_END_HOUR", "23"))
    BLOCK_ENTRY_HOURS = {
        int(h.strip())
        for h in os.getenv("FOREX_BLOCK_ENTRY_HOURS", "").split(",")
        if h.strip()
    }
    EXIT_AFTER_SESSION_END = True

    # MT5 connection
    DRY_RUN              = os.getenv("FOREX_DRY_RUN", "true").lower() == "true"
    MT5_LOGIN            = os.getenv("MT5_LOGIN", "").strip()
    MT5_PASSWORD         = os.getenv("MT5_PASSWORD", "").strip()
    MT5_SERVER           = os.getenv("MT5_SERVER", "").strip()
    MT5_EXPLICIT_LOGIN   = os.getenv("FOREX_MT5_EXPLICIT_LOGIN", "true").lower() == "true"

    # Timing
    CHECK_INTERVAL_SECONDS              = int(os.getenv("FOREX_CHECK_INTERVAL_SECONDS", "60"))
    MT5_RECONNECT_SLEEP_SECONDS         = int(os.getenv("FOREX_MT5_RECONNECT_SLEEP_SECONDS", "10"))
    MT5_DISCONNECT_NOTIFY_COOLDOWN_SECS = int(os.getenv("FOREX_MT5_DISCONNECT_NOTIFY_COOLDOWN_SECONDS", "300"))

    # AI daily scan
    AI_ENABLED         = os.getenv("FOREX_AI_DAILY_SCAN_ENABLED", "true").lower() == "true"
    # FIX: default changed to false — bot no longer hard-blocks when scan hasn't run yet
    REQUIRE_AI_WATCHLIST = os.getenv("FOREX_REQUIRE_AI_WATCHLIST", "false").lower() == "true"
    AI_SCAN_ON_START   = os.getenv("FOREX_AI_SCAN_ON_BOT_START", "true").lower() == "true"
    AI_SCAN_HOUR       = int(os.getenv("FOREX_AI_SCAN_HOUR", str(int(os.getenv("FOREX_TRADE_START_HOUR", "14")))))
    DEEPSEEK_KEY       = os.getenv("DEEPSEEK_API_KEY")
    DEEPSEEK_MODEL     = os.getenv("FOREX_DEEPSEEK_MODEL", "deepseek-reasoner")
    AI_SYMBOL_LIMIT    = int(os.getenv("FOREX_AI_SYMBOL_UNIVERSE_LIMIT", "30"))

    # Backtest shows EURJPY has no edge — removed from priority scan list
    # XAUUSD removed — dedicated gold_bot.py handles Gold with Gold-specific config
    AI_MAJOR_PAIRS = [
        "EURUSD","GBPUSD","USDJPY","USDCHF","USDCAD",
        "AUDUSD","NZDUSD","GBPJPY",
    ]
    AI_ALLOWED_SYMBOLS = [
        "EURUSD","GBPUSD","USDJPY","USDCHF","USDCAD","AUDUSD","NZDUSD",
        "EURJPY","GBPJPY","EURGBP","EURCHF","EURCAD","EURAUD","EURNZD",
        "GBPCHF","GBPCAD","GBPAUD","GBPNZD","AUDJPY","CADJPY","CHFJPY",
        "NZDJPY","AUDCAD","AUDCHF","AUDNZD","CADCHF","NZDCAD","NZDCHF",
    ]

    # Paths
    LOG_DIR               = Path("logs")
    AI_STATE_FILE         = LOG_DIR / "forex_ai_state.json"
    AI_RECOMMENDATION_LOG = LOG_DIR / "forex_ai_recommendations.jsonl"
    KILL_FILE             = LOG_DIR / "STOP"
    STATE_FILE            = LOG_DIR / "forex_state.json"
    MAX_CONSECUTIVE_LOSSES = int(os.getenv("FOREX_MAX_CONSECUTIVE_LOSSES", "3"))

# Global mutable symbol (can be switched by AI watchlist)
_SYMBOL = Config.SYMBOL


def symbol() -> str:
    return _SYMBOL


def set_symbol(name: str) -> None:
    global _SYMBOL
    _SYMBOL = name


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    builtins.print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {_safe(msg)}", flush=True)


def warn(msg: str) -> None:
    builtins.print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] WARN {_safe(msg)}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# MT5 connection
# ─────────────────────────────────────────────────────────────────────────────

def _account_info_str(account=None) -> str:
    account = account or mt5.account_info()
    if account is None:
        return f"No account info | last_error={mt5.last_error()}"
    fields = ("login","server","name","currency","trade_allowed",
              "trade_expert","balance","equity","margin_free")
    return " | ".join(f"{f}={getattr(account, f, '')}" for f in fields)


def _resolve_symbol(preferred: str) -> tuple[str, object]:
    """Find the actual MT5 symbol name for a preferred name (handles aliases)."""
    candidates = [preferred]
    upper = preferred.upper()
    if "XAU" in upper or "GOLD" in upper:
        for alias in Config.SYMBOL_ALIASES:
            if alias not in candidates:
                candidates.append(alias)

    for name in candidates:
        info = mt5.symbol_info(name)
        if info is not None:
            return name, info

    all_syms = mt5.symbols_get()
    if not all_syms:
        raise RuntimeError(f"Cannot load MT5 symbols: {mt5.last_error()}")

    pairs = [(s.name, s.name.upper()) for s in all_syms]
    for alias in candidates:
        alias_up = alias.upper()
        for name, up in pairs:
            if up.startswith(alias_up):
                return name, mt5.symbol_info(name)
    for alias in candidates:
        alias_up = alias.upper()
        for name, up in pairs:
            if alias_up in up:
                return name, mt5.symbol_info(name)

    gold = [n for n, up in pairs if "XAU" in up or "GOLD" in up]
    sample = ", ".join((gold or [n for n, _ in pairs])[:20])
    raise RuntimeError(f"Symbol '{preferred}' not found. Similar: {sample}")


def connect_mt5() -> None:
    global _SYMBOL
    log("🔌 Connecting MT5...")
    try:
        mt5.shutdown()
    except Exception:
        pass
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")

    if Config.MT5_EXPLICIT_LOGIN:
        if Config.MT5_LOGIN and Config.MT5_PASSWORD and Config.MT5_SERVER:
            try:
                login_id = int(Config.MT5_LOGIN)
            except ValueError as exc:
                raise RuntimeError("MT5_LOGIN must be numeric") from exc
            if not mt5.login(login_id, password=Config.MT5_PASSWORD, server=Config.MT5_SERVER):
                raise RuntimeError(f"MT5 login failed: {mt5.last_error()}")
            log(f"✅ MT5 login OK | login={login_id} server={Config.MT5_SERVER}")
        else:
            log("⏳ MT5 explicit login skipped (missing credentials)")

    resolved, info = _resolve_symbol(_SYMBOL)
    if resolved != _SYMBOL:
        log(f"🔎 Symbol resolved: {_SYMBOL} -> {resolved}")
        set_symbol(resolved)

    if not info.visible:
        if not mt5.symbol_select(_SYMBOL, True):
            raise RuntimeError(f"Cannot select {_SYMBOL} in Market Watch: {mt5.last_error()}")

    log(f"✅ MT5 connected | Symbol={_SYMBOL}")
    log(f"🔎 Account: {_account_info_str()}")


def _is_connection_error(error: Exception) -> bool:
    text = str(error)
    return any(k in text for k in ("IPC send failed", "IPC", "-10001",
                                   "No account info", "Cannot load MT5 symbols"))


def mt5_ping() -> None:
    if mt5.terminal_info() is None or mt5.account_info() is None:
        raise RuntimeError(f"MT5 ping failed: {mt5.last_error()}")


def reconnect_mt5(state: dict, error: Exception) -> None:
    now = time.time()
    cooldown_expired = now - float(state.get("last_disconnect_notify") or 0) >= Config.MT5_DISCONNECT_NOTIFY_COOLDOWN_SECS
    if not state.get("mt5_down") or cooldown_expired:
        notify_error("FOREX", f"MT5 disconnected: {error}")
        state["last_disconnect_notify"] = now
    state["mt5_down"] = True
    log(f"🔌 Reconnecting in {Config.MT5_RECONNECT_SLEEP_SECONDS}s...")
    time.sleep(Config.MT5_RECONNECT_SLEEP_SECONDS)
    connect_mt5()
    mt5_ping()
    state["mt5_down"] = False
    notify_reconnected("FOREX", "MT5 connection restored")
    log("✅ MT5 reconnected")


def select_symbol(preferred: str, reason: str = "") -> str:
    resolved, info = _resolve_symbol(preferred)
    if not info.visible:
        if not mt5.symbol_select(resolved, True):
            raise RuntimeError(f"Cannot select {resolved}: {mt5.last_error()}")
    old = symbol()
    set_symbol(resolved)
    suffix = f" | {reason}" if reason else ""
    if old != resolved:
        log(f"🎯 Symbol switched: {old} -> {resolved}{suffix}")
    else:
        log(f"🎯 Symbol: {resolved}{suffix}")
    return resolved


# ─────────────────────────────────────────────────────────────────────────────
# Market data
# ─────────────────────────────────────────────────────────────────────────────

def get_ohlcv(sym: str, timeframe: int, bars: int = 500) -> pd.DataFrame:
    rates = mt5.copy_rates_from_pos(sym, timeframe, 0, bars)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"No data from MT5: {sym}")
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.rename(columns={"tick_volume": "volume"}, inplace=True)
    return df


def _d1_adx(df: pd.DataFrame, period: int = 14) -> float:
    """Wilder's ADX on daily bars. Returns latest ADX value."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    plus_dm  = (high.diff()).where((high.diff() > -low.diff()) & (high.diff() > 0), 0.0)
    minus_dm = (-low.diff()).where((-low.diff() > high.diff()) & (-low.diff() > 0), 0.0)
    alpha = 1 / period
    atr_s    = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_di  = 100 * plus_dm.ewm(alpha=alpha, adjust=False).mean() / atr_s.replace(0, float("nan"))
    minus_di = 100 * minus_dm.ewm(alpha=alpha, adjust=False).mean() / atr_s.replace(0, float("nan"))
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, float("nan"))) * 100
    adx = dx.ewm(alpha=alpha, adjust=False).mean()
    return float(adx.iloc[-1]) if not adx.empty else 0.0


def d1_regime_ok(sym: str, side: str) -> tuple[bool, str]:
    """
    Check D1 regime: ADX must be >= D1_ADX_MIN and D1 trend must align with signal side.
    Returns (ok, reason_string).
    """
    if not Config.USE_D1_REGIME:
        return True, ""
    try:
        df = get_ohlcv(sym, mt5.TIMEFRAME_D1, Config.D1_BARS)
    except RuntimeError:
        return True, ""  # fail-open: don't block if D1 data unavailable

    if len(df) < 60:
        return True, ""

    close = df["close"]
    ema50  = close.ewm(span=50,  adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()
    d1_trend = 1 if ema50.iloc[-1] > ema200.iloc[-1] else -1

    d1_adx = _d1_adx(df)

    if d1_adx < Config.D1_ADX_MIN:
        return False, f"D1 ADX {d1_adx:.1f} < {Config.D1_ADX_MIN} (ranging)"
    if side == "BUY" and d1_trend != 1:
        return False, f"D1 trend bearish (EMA50={ema50.iloc[-1]:.5f} < EMA200={ema200.iloc[-1]:.5f})"
    if side == "SELL" and d1_trend != -1:
        return False, f"D1 trend bullish (EMA50={ema50.iloc[-1]:.5f} > EMA200={ema200.iloc[-1]:.5f})"
    return True, f"D1 ADX={d1_adx:.1f} trend={'UP' if d1_trend == 1 else 'DOWN'}"


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema20"]  = df["close"].ewm(span=Config.EMA_FAST, adjust=False).mean()
    df["ema50"]  = df["close"].ewm(span=Config.EMA_SLOW, adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=Config.EMA_BIG,  adjust=False).mean()

    delta    = df["close"].diff()
    avg_gain = delta.clip(lower=0).rolling(Config.RSI_PERIOD).mean()
    avg_loss = (-delta.clip(upper=0)).rolling(Config.RSI_PERIOD).mean()
    df["rsi"] = 100 - (100 / (1 + avg_gain / avg_loss))

    df["vol_avg"]          = df["volume"].rolling(20).mean()
    df["ema20_50_gap_pct"] = (df["ema20"] - df["ema50"]).abs() / df["close"]
    df["body_ratio"]       = (df["close"] - df["open"]).abs() / (df["high"] - df["low"]).replace(0, np.nan)

    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Filters
# ─────────────────────────────────────────────────────────────────────────────

def pass_session_filter() -> bool:
    now = datetime.now()
    if now.hour in Config.BLOCK_ENTRY_HOURS:
        log(f"⛔ Entry blocked: hour {now.hour} in BLOCK_ENTRY_HOURS")
        return False
    return Config.TRADE_START_HOUR <= now.hour <= Config.TRADE_END_HOUR


def should_exit_after_session() -> bool:
    return Config.EXIT_AFTER_SESSION_END and datetime.now().hour > Config.TRADE_END_HOUR


def get_spread_points() -> float | None:
    tick = mt5.symbol_info_tick(symbol())
    info = mt5.symbol_info(symbol())
    if tick is None or info is None or info.point <= 0:
        return None
    return (tick.ask - tick.bid) / info.point


def pass_kill_zone_filter() -> tuple[bool, str]:
    if not Config.KILL_ZONE_ONLY:
        return True, "kill zone filter off"
    utc_hour = datetime.now(timezone.utc).hour
    for start, end in Config.KILL_ZONES_UTC:
        if start <= utc_hour < end:
            return True, f"in kill zone {start:02d}-{end:02d} UTC"
    zones = ", ".join(f"{s:02d}-{e:02d}" for s, e in Config.KILL_ZONES_UTC)
    return False, f"outside kill zones ({zones} UTC)"


def pass_spread_filter() -> tuple[bool, str]:
    sp = get_spread_points()
    if sp is None:
        return False, "no tick/spread info"
    sym = symbol().upper()
    limit = Config.MAX_SPREAD_POINTS_GOLD if "XAU" in sym or "XAG" in sym else Config.MAX_SPREAD_POINTS
    if sp > limit:
        return False, f"spread {sp:.1f} > {limit} pts"
    return True, f"spread OK {sp:.1f} pts"


def pass_daily_risk_filter() -> tuple[bool, str]:
    account = mt5.account_info()
    if account is None:
        return False, f"no account info: {mt5.last_error()}"
    if float(account.balance) <= 0:
        return False, f"invalid balance {account.balance}"

    today = datetime.now().date()
    deals = mt5.history_deals_get(
        datetime.combine(today, dtime.min),
        datetime.combine(today, dtime.max),
    ) or []

    trades_today = sum(1 for d in deals if d.symbol == symbol() and d.entry == mt5.DEAL_ENTRY_IN)
    profit_today = sum(d.profit for d in deals if d.symbol == symbol())
    max_loss     = account.balance * (Config.MAX_DAILY_LOSS_PCT / 100)

    if trades_today >= Config.MAX_TRADES_PER_DAY:
        return False, f"max trades {trades_today}/{Config.MAX_TRADES_PER_DAY}"
    if profit_today < -max_loss:
        return False, f"daily loss {profit_today:.2f} > limit -{max_loss:.2f}"
    return True, f"risk OK pnl={profit_today:.2f} limit=-{max_loss:.2f}"


# ─────────────────────────────────────────────────────────────────────────────
# Position checks
# ─────────────────────────────────────────────────────────────────────────────

def has_open_position() -> bool:
    pos = mt5.positions_get(symbol=symbol())
    return pos is not None and len(pos) > 0


def has_any_open_position(symbols: list[str]) -> tuple[bool, str]:
    positions = mt5.positions_get() or []
    sym_set = set(symbols)
    for pos in positions:
        if pos.symbol in sym_set:
            return True, pos.symbol
    return False, ""


def has_correlated_position(sym: str) -> tuple[bool, str]:
    positions = mt5.positions_get() or []
    if not positions:
        return False, ""
    open_syms = {p.symbol.upper() for p in positions}
    sym_up = sym.upper()
    for group in Config.CORR_GROUPS:
        if sym_up in group:
            conflict = group & open_syms - {sym_up}
            if conflict:
                return True, f"correlated position open: {', '.join(conflict)}"
    return False, ""


# ─────────────────────────────────────────────────────────────────────────────
# Breakeven stop management
# ─────────────────────────────────────────────────────────────────────────────

def _position_side(pos) -> str:
    return "BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL"


def _move_sl_to_breakeven(pos) -> bool:
    entry = pos.price_open
    side  = _position_side(pos)
    if side == "BUY"  and pos.sl >= entry:
        return True
    if side == "SELL" and 0 < pos.sl <= entry:
        return True
    req = {
        "action":   mt5.TRADE_ACTION_SLTP,
        "symbol":   pos.symbol,
        "position": pos.ticket,
        "sl":       entry,
        "tp":       pos.tp,
    }
    if Config.DRY_RUN:
        log(f"[DRY_BE] {pos.symbol} ticket={pos.ticket} sl→entry {entry}")
        return True
    result = mt5.order_send(req)
    ok = result is not None and result.retcode == mt5.TRADE_RETCODE_DONE
    if ok:
        log(f"[BE] SL moved to entry {entry} on {pos.symbol} ticket={pos.ticket}")
    else:
        warn(f"[BE] Failed: {result}")
    return ok


def check_breakeven_positions(state: dict) -> None:
    if Config.BREAKEVEN_R <= 0:
        return
    for pos in (mt5.positions_get() or []):
        key = f"be_applied_{pos.ticket}"
        if state.get(key):
            continue
        entry = pos.price_open
        side  = _position_side(pos)
        risk  = abs(entry - pos.sl) if pos.sl > 0 else 0
        if risk <= 0:
            continue
        tick = mt5.symbol_info_tick(pos.symbol)
        if tick is None:
            continue
        current  = tick.bid if side == "BUY" else tick.ask
        be_level = (entry + risk * Config.BREAKEVEN_R if side == "BUY"
                    else entry - risk * Config.BREAKEVEN_R)
        triggered = (side == "BUY" and current >= be_level) or (side == "SELL" and current <= be_level)
        if triggered and _move_sl_to_breakeven(pos):
            state[key] = True


# ─────────────────────────────────────────────────────────────────────────────
# Signal generation
# ─────────────────────────────────────────────────────────────────────────────

def _core_config() -> DonchianCoreConfig:
    """
    FIX: previously missing adx_max, session_hours_utc, atr_percentile_min,
    volume_mult, allowed_side — caused signal engine to silently reject most setups.
    Session hours converted Thai → UTC (subtract 7).
    """
    # Thai 14–23 → UTC 7–16
    session_utc = tuple(range(
        Config.TRADE_START_HOUR - 7,
        Config.TRADE_END_HOUR   - 7 + 1,
    ))
    return DonchianCoreConfig(
        allowed_side         = Config.ALLOWED_SIDE,
        ema_fast             = Config.EMA_FAST,
        ema_slow             = Config.EMA_SLOW,
        tier_a_risk          = Config.TIER_A_RISK_PERCENT / 100,
        tier_b_risk          = Config.TIER_B_RISK_PERCENT / 100,
        rr                   = Config.RR,
        donchian_n           = Config.DONCHIAN_N,
        adx_min              = Config.ADX_MIN,
        adx_max              = Config.ADX_MAX,
        session_hours_utc    = session_utc,
        atr_percentile_min   = Config.ATR_PERCENTILE_MIN,
        min_atr_pct          = Config.MIN_ATR_PCT,
        volume_mult          = Config.VOLUME_MULT,
        require_atr_expansion= Config.REQUIRE_ATR_EXPANSION,
        adx_bars_rising      = Config.ADX_BARS_RISING,
        max_trades_per_day      = Config.MAX_TRADES_PER_DAY,
        require_trend_alignment = False,  # EMA50/200 lags too far; Donchian breakout + ADX decide direction
    )


def generate_signal() -> tuple[dict, pd.DataFrame]:
    sym = symbol()
    df_m1  = get_ohlcv(sym, Config.TF_EXEC,  Config.BARS)
    df_m5  = get_ohlcv(sym, Config.TF_ENTRY, Config.BARS)
    df_m15 = get_ohlcv(sym, Config.TF_TREND, Config.BARS)
    signal, reason, candle_time = latest_signal(sym, df_m1, df_m5, df_m15, _core_config())

    def _no_trade(r: str) -> tuple[dict, pd.DataFrame]:
        return {
            "time":   candle_time or datetime.now(),
            "symbol": sym,
            "side":   "NO_TRADE",
            "price":  float(df_m5.iloc[-2]["close"]) if len(df_m5) >= 2 else 0.0,
            "reason": r,
        }, df_m5

    if signal is None:
        return _no_trade(reason)

    regime_ok, regime_reason = d1_regime_ok(sym, signal["side"])
    if not regime_ok:
        log(f"[D1_REGIME] blocked {signal['side']} — {regime_reason}")
        return _no_trade(f"D1 regime: {regime_reason}")

    signal["time"]  = pd.Timestamp(signal["candle_time"])
    signal["price"] = signal["entry"]
    signal["d1_regime"] = regime_reason
    return signal, df_m5


# ─────────────────────────────────────────────────────────────────────────────
# TP / SL / lot sizing
# ─────────────────────────────────────────────────────────────────────────────

def calculate_tp_sl(signal: dict) -> dict | None:
    info = mt5.symbol_info(symbol())
    if info is None:
        raise RuntimeError("No symbol info")
    if signal["side"] == "NO_TRADE":
        return None
    d = info.digits
    entry = float(signal["entry"])
    sl    = float(signal["sl"])
    tp    = float(signal["tp"])
    return {
        "entry":         round(entry, d),
        "sl":            round(sl, d),
        "tp":            round(tp, d),
        "risk_points":   round(abs(entry - sl) / info.point, 1),
        "reward_points": round(abs(tp - entry) / info.point, 1),
        "rr":            Config.RR,
        "tier":          signal.get("tier"),
        "risk_pct":      signal.get("risk_pct"),
    }


def calculate_lot(signal: dict, tp_sl: dict) -> float:
    account = mt5.account_info()
    info    = mt5.symbol_info(symbol())
    if account is None or info is None:
        return Config.LOT

    risk_pct   = float(signal.get("risk_pct", Config.TIER_A_RISK_PERCENT / 100))
    risk_money = account.balance * risk_pct
    stop_dist  = abs(float(tp_sl["entry"]) - float(tp_sl["sl"]))
    if stop_dist <= 0 or info.trade_tick_size <= 0 or info.trade_tick_value <= 0:
        return Config.LOT

    loss_per_lot = (stop_dist / info.trade_tick_size) * info.trade_tick_value
    if loss_per_lot <= 0:
        return Config.LOT

    step    = info.volume_step or 0.01
    min_lot = info.volume_min  or step
    max_lot = info.volume_max  or (risk_money / loss_per_lot)
    lot     = np.floor((risk_money / loss_per_lot) / step) * step
    lot     = min(max(lot, min_lot), max_lot)

    # Margin safety: cap lot if it would consume >90% of free margin
    order_type_mg = mt5.ORDER_TYPE_BUY if signal.get("side") == "BUY" else mt5.ORDER_TYPE_SELL
    margin_needed = mt5.order_calc_margin(order_type_mg, symbol(), lot, float(tp_sl["entry"]))
    if margin_needed is not None and margin_needed > 0:
        free = float(getattr(account, "margin_free", 0))
        if free > 0 and margin_needed > free * 0.9:
            capped = np.floor(free * 0.9 / margin_needed * lot / step) * step
            capped = max(min(capped, max_lot), min_lot)
            warn(f"[MARGIN] needed={margin_needed:.2f} > free*0.9={free*0.9:.2f} — lot {lot:.3f}->{capped:.3f}")
            lot = capped

    digits  = max(0, int(round(-np.log10(step)))) if step < 1 else 0
    return round(float(lot), digits)


# ─────────────────────────────────────────────────────────────────────────────
# Order execution
# ─────────────────────────────────────────────────────────────────────────────

def send_order(signal: dict, tp_sl: dict) -> dict | None:
    tick = mt5.symbol_info_tick(symbol())
    info = mt5.symbol_info(symbol())
    if tick is None or info is None:
        warn("❌ No tick/info — order skipped")
        return None

    side = signal["side"]
    if side == "BUY":
        order_type, price = mt5.ORDER_TYPE_BUY,  tick.ask
    elif side == "SELL":
        order_type, price = mt5.ORDER_TYPE_SELL, tick.bid
    else:
        return None

    # Stops level: verify SL/TP are not too close to current price
    stops_lv = getattr(info, "stops_level", 0)
    if stops_lv > 0:
        min_dist = stops_lv * info.point
        if abs(price - tp_sl["sl"]) < min_dist:
            warn(f"[STOPS] SL too close: dist={abs(price-tp_sl['sl']):.5f} < min={min_dist:.5f} (stops_level={stops_lv})")
            return None
        if abs(tp_sl["tp"] - price) < min_dist:
            warn(f"[STOPS] TP too close: dist={abs(tp_sl['tp']-price):.5f} < min={min_dist:.5f} (stops_level={stops_lv})")
            return None

    lot = calculate_lot(signal, tp_sl)
    request = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       symbol(),
        "volume":       lot,
        "type":         order_type,
        "price":        price,
        "sl":           tp_sl["sl"],
        "tp":           tp_sl["tp"],
        "deviation":    30,
        "magic":        Config.MAGIC_NUMBER,
        "comment":      f"DONCHIAN_T{signal.get('tier','A')}",
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    notify_order_opened(
        "FOREX", symbol(), side, lot, price, tp_sl["sl"], tp_sl["tp"],
        tier=signal.get("tier", ""),
        risk_pct=signal.get("risk_pct", Config.TIER_A_RISK_PERCENT / 100),
        dry_run=Config.DRY_RUN,
    )

    if Config.DRY_RUN:
        log(f"🧪 DRY_RUN order: {request}")
        return request

    result = mt5.order_send(request)
    log(f"📌 Order result: {result}")
    notify_order_result("FOREX", symbol(), side, result, dry_run=False)
    if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
        time.sleep(0.3)
        for p in (mt5.positions_get(symbol=symbol()) or []):
            if p.magic == Config.MAGIC_NUMBER and p.sl == 0:
                warn(f"[VERIFY] Position {p.ticket} has no SL — manual intervention required!")
                notify_error("FOREX", f"Position {p.ticket} on {symbol()} opened without SL!")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Demo testcase logging
# ─────────────────────────────────────────────────────────────────────────────

def _log_testcase(blocked_by: str = "", block_reason: str = "", spread_pts: float | None = None) -> None:
    sym  = symbol()
    info = mt5.symbol_info(sym)
    tick = mt5.symbol_info_tick(sym)
    spread = {}
    if info and tick and info.point > 0:
        spread_val = tick.ask - tick.bid
        mid = (tick.ask + tick.bid) / 2
        spread = {
            "spread":        spread_val,
            "spread_points": spread_pts if spread_pts is not None else spread_val / info.point,
            "spread_pct":    spread_val / mid if mid else 0.0,
        }
    log_demo_testcase(
        "forex_demo", sym,
        get_ohlcv(sym, Config.TF_EXEC,  Config.BARS),
        get_ohlcv(sym, Config.TF_ENTRY, Config.BARS),
        get_ohlcv(sym, Config.TF_TREND, Config.BARS),
        _core_config(),
        external_blocked_by=blocked_by,
        external_block_reason=block_reason,
        spread=spread,
    )


# ─────────────────────────────────────────────────────────────────────────────
# AI daily market scan
# ─────────────────────────────────────────────────────────────────────────────

def _read_ai_state() -> dict:
    try:
        if Config.AI_STATE_FILE.exists():
            return json.loads(Config.AI_STATE_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        pass
    return {}


def _write_ai_state(state: dict) -> None:
    Config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    Config.AI_STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _normalize_symbol(name: str) -> str | None:
    upper = name.upper()
    if "XAU" in upper or "GOLD" in upper:
        return "XAUUSD"
    letters = "".join(c for c in upper if c.isalpha())
    for base in Config.AI_ALLOWED_SYMBOLS:
        if letters.startswith(base):
            return base
    return None


def _ai_candidate_symbols() -> list[str]:
    all_syms = mt5.symbols_get() or []
    available: dict[str, str] = {}
    for item in all_syms:
        base = _normalize_symbol(item.name)
        if base and base not in available:
            available[base] = item.name
    ordered = []
    for s in Config.AI_MAJOR_PAIRS + Config.AI_ALLOWED_SYMBOLS:
        if s in available and s not in ordered:
            ordered.append(s)
    return (ordered or Config.AI_MAJOR_PAIRS)[: Config.AI_SYMBOL_LIMIT]


def _build_ai_prompt(candidates: list[str]) -> str:
    now = datetime.now()
    return f"""You are a professional Forex analyst. Analyze the market and rate pairs for trading suitability.
Time: 2:00 PM Thailand (UTC+7) — London-New York overlap.
Candidate symbols: {", ".join(candidates)}
Return ONLY valid JSON — no markdown, no reasoning chain.
{{
  "timestamp": "{now.strftime('%Y-%m-%d')} 14:00:00",
  "recommendations": [
    {{"symbol": "EURUSD", "rating": 1, "reason": "short reason max 80 chars"}},
    ...
  ]
}}
Rating: 1=best, 5=avoid. Include only rating 1-3, min 3 pairs, max 7 pairs."""


def _build_watchlist(recommendations: list[dict], candidates: list[str] | None = None) -> list[dict]:
    candidates = candidates or _ai_candidate_symbols()
    ranked = sorted(
        [r for r in recommendations if int(r.get("rating", 99)) in {1, 2, 3}],
        key=lambda r: (
            int(r.get("rating", 99)),
            candidates.index(str(r.get("symbol", "")).upper())
            if str(r.get("symbol", "")).upper() in candidates else 999,
        ),
    )
    watchlist, errors = [], []
    for rec in ranked:
        sym = str(rec.get("symbol", "")).strip().upper()
        if not sym:
            continue
        try:
            resolved, info = _resolve_symbol(sym)
            if not info.visible:
                if not mt5.symbol_select(resolved, True):
                    raise RuntimeError(f"Cannot select {resolved}: {mt5.last_error()}")
            watchlist.append({
                "symbol": resolved,
                "rating": int(rec.get("rating", 99)),
                "reason": rec.get("reason", ""),
            })
        except Exception as exc:
            errors.append(f"{sym}: {exc}")
    if watchlist:
        log("🧠 AI watchlist: " + ", ".join(f"{w['symbol']}(r{w['rating']})" for w in watchlist))
    else:
        warn("🧠 No AI symbol available in MT5: " + " | ".join(errors))
    return watchlist


def run_ai_scan(force: bool = False) -> None:
    if not Config.AI_ENABLED:
        return
    now   = datetime.now()
    today = now.strftime("%Y-%m-%d")
    if now.hour < Config.AI_SCAN_HOUR:
        return

    state = _read_ai_state()
    if state.get("last_scan_date") == today and not force:
        log(f"[AI] Using cached scan from {state.get('last_scan_at','?')} | "
            f"symbols={', '.join(state.get('candidate_symbols', []))}")
        # Rebuild watchlist from cached recs if candidate_symbols is empty
        if not state.get("candidate_symbols") and state.get("last_recommendations"):
            wl = _build_watchlist(state["last_recommendations"])
            if wl:
                state["candidate_symbols"]        = [w["symbol"] for w in wl]
                state["candidate_recommendations"] = wl
                state["selected_symbol"]           = wl[0]["symbol"]
                _write_ai_state(state)
        return

    # Clear stale data before new scan
    for key in ("candidate_symbols","candidate_recommendations","selected_symbol","selected_recommendation"):
        state.pop(key, None)

    if not Config.DEEPSEEK_KEY:
        warn("🧠 AI scan skipped: missing DEEPSEEK_API_KEY")
        state.update({"last_scan_date": today, "last_scan_at": now.strftime("%Y-%m-%d %H:%M:%S"),
                      "last_scan_error": "missing DEEPSEEK_API_KEY"})
        _write_ai_state(state)
        return
    if OpenAI is None:
        warn("🧠 AI scan skipped: openai package not installed")
        state.update({"last_scan_date": today, "last_scan_at": now.strftime("%Y-%m-%d %H:%M:%S"),
                      "last_scan_error": "openai not installed"})
        _write_ai_state(state)
        return

    try:
        log("🧠 DeepSeek daily scan starting...")
        candidates = _ai_candidate_symbols()
        log(f"🧠 Candidates: {', '.join(candidates)}")
        client = OpenAI(api_key=Config.DEEPSEEK_KEY, base_url="https://api.deepseek.com/v1")
        resp   = client.chat.completions.create(
            model=Config.DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": _build_ai_prompt(candidates)}],
            temperature=0.2,
            max_tokens=700,
        )
        content = resp.choices[0].message.content.strip()
        for fence in ("```json", "```"):
            if fence in content:
                content = content.split(fence, 1)[1].split("```", 1)[0]
        result = json.loads(content.strip())
        result.setdefault("timestamp", now.strftime("%Y-%m-%d %H:%M:%S"))

        Config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        with Config.AI_RECOMMENDATION_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        recs = result.get("recommendations", [])
        wl   = _build_watchlist(recs, candidates)
        state.update({
            "last_scan_date": today,
            "last_scan_at":   now.strftime("%Y-%m-%d %H:%M:%S"),
            "candidate_universe":     candidates,
            "last_recommendations":   recs,
            "candidate_symbols":      [w["symbol"] for w in wl],
            "candidate_recommendations": wl,
            "selected_symbol":        wl[0]["symbol"] if wl else None,
            "selected_recommendation": wl[0] if wl else None,
        })
        _write_ai_state(state)
        log(f"🧠 Scan done: {json.dumps(result, ensure_ascii=False)}")
    except Exception as exc:
        warn(f"🧠 DeepSeek scan failed: {exc}")


def get_trading_watchlist() -> list[str]:
    """
    FIX: REQUIRE_AI_WATCHLIST now defaults to false.
    Falls back to Config.SYMBOL so the bot never sits idle due to a missed AI scan.
    """
    today = datetime.now().strftime("%Y-%m-%d")
    state = _read_ai_state()

    if Config.AI_ENABLED and Config.REQUIRE_AI_WATCHLIST:
        if state.get("last_scan_date") != today:
            log("🧠 REQUIRE_AI_WATCHLIST=true but no scan today — waiting")
            return []

    symbols = state.get("candidate_symbols") or []
    if not symbols and state.get("selected_symbol"):
        symbols = [state["selected_symbol"]]
    if not symbols:
        symbols = [symbol()]   # fallback: always trade the configured symbol

    seen, unique = set(), []
    for s in symbols:
        if s and s not in seen:
            unique.append(s)
            seen.add(s)
    return unique


# ─────────────────────────────────────────────────────────────────────────────
# Startup notification
# ─────────────────────────────────────────────────────────────────────────────

def _notify_started(phase: str = "READY") -> None:
    extra = (
        f"Status: <code>{phase}</code>\n"
        f"Symbol: <code>{symbol()}</code>\n"
        f"Session: <code>{Config.TRADE_START_HOUR}:00–{Config.TRADE_END_HOUR}:59</code>\n"
        f"AI scan: <code>{Config.AI_ENABLED}</code> | "
        f"Require watchlist: <code>{Config.REQUIRE_AI_WATCHLIST}</code>\n"
        f"RR: <code>1:{Config.RR}</code> | "
        f"ADX: <code>{Config.ADX_MIN}–{Config.ADX_MAX}</code>"
    )
    sent = notify_bot_started(
        "Forex Donchian Bot",
        "DRY_RUN" if Config.DRY_RUN else "LIVE/DEMO",
        extra,
    )
    if sent:
        log(f"✅ Telegram notification sent ({phase})")
    else:
        log(f"⏳ Telegram skipped ({notify_last_error() or 'send failed'})")


# ─────────────────────────────────────────────────────────────────────────────
# Bot state persistence + safety counters
# ─────────────────────────────────────────────────────────────────────────────

def load_forex_state() -> dict:
    defaults: dict = {
        "mt5_down": False,
        "last_disconnect_notify": 0,
        "last_stats_hour": -1,
        "last_stats_date": None,
        "consecutive_losses": 0,
    }
    try:
        if Config.STATE_FILE.exists():
            saved = json.loads(Config.STATE_FILE.read_text(encoding="utf-8"))
            defaults.update(saved)
    except (json.JSONDecodeError, OSError):
        pass
    return defaults


def save_forex_state(state: dict) -> None:
    Config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    saveable = {k: v for k, v in state.items()
                if k not in ("mt5_down", "last_disconnect_notify")}
    try:
        Config.STATE_FILE.write_text(
            json.dumps(saveable, indent=2, default=str), encoding="utf-8"
        )
    except OSError as exc:
        warn(f"[STATE] Cannot save state: {exc}")


def count_consecutive_losses() -> int:
    """Count consecutive losing closed trades from MT5 deal history today."""
    today = datetime.now().date()
    deals = mt5.history_deals_get(
        datetime.combine(today, dtime.min),
        datetime.combine(today, dtime.max),
    ) or []
    closed = [d for d in deals
              if d.entry == mt5.DEAL_ENTRY_OUT
              and d.symbol == symbol()]
    consec = 0
    for d in reversed(closed):
        if d.profit < 0:
            consec += 1
        else:
            break
    return consec


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def run_bot() -> None:
    log(
        f"🚀 Forex bot starting | pid={os.getpid()} | "
        f"session={Config.TRADE_START_HOUR}:00–{Config.TRADE_END_HOUR}:59 | "
        f"interval={Config.CHECK_INTERVAL_SECONDS}s | DRY_RUN={Config.DRY_RUN}"
    )
    _notify_started("BOOTING")
    try:
        connect_mt5()
    except Exception as exc:
        notify_error("FOREX", f"Startup failed: {exc}")
        raise
    _notify_started("READY")

    if Config.AI_SCAN_ON_START:
        run_ai_scan(force=True)

    last_entry_candle: dict[str, object] = {}
    state = load_forex_state()
    state["mt5_down"] = False
    state["last_disconnect_notify"] = 0

    log("🚀 Bot running")

    while True:
        try:
            if Config.KILL_FILE.exists():
                log("[KILL] STOP file detected — exiting cleanly")
                save_forex_state(state)
                break
            mt5_ping()
            if state.get("mt5_down"):
                state["mt5_down"] = False
                notify_reconnected("FOREX", "MT5 restored")

            run_ai_scan()

            # ── Daily stats reset + hourly log ───────────────────────────
            _now = datetime.now()
            today = _now.date()
            if state["last_stats_date"] != today:
                if _REJECT_STATS:
                    log(f"[STATS] Daily summary: {dict(_REJECT_STATS.most_common())}")
                _REJECT_STATS.clear()
                state["last_stats_date"] = today
                save_forex_state(state)
            elif _now.hour != state["last_stats_hour"] and _REJECT_STATS:
                total = _REJECT_STATS.get("_total_attempts", 1)
                top = {
                    k: f"{v}({v/total:.0%})"
                    for k, v in _REJECT_STATS.most_common()
                    if not k.startswith("_")
                }
                log(f"[STATS] Hourly rejections (of {total} attempts): {dict(list(top.items())[:8])}")
                state["last_stats_hour"] = _now.hour

            # ── Session gate ──────────────────────────────────────────────
            if not pass_session_filter():
                if should_exit_after_session():
                    log("🌙 Session ended — bot exiting")
                    break
                log("⏳ Outside session")
                time.sleep(Config.CHECK_INTERVAL_SECONDS)
                continue

            # ── ICT Kill Zone filter ──────────────────────────────────────
            kz_ok, kz_msg = pass_kill_zone_filter()
            if not kz_ok:
                log(f"[KZ] {kz_msg}")
                time.sleep(Config.CHECK_INTERVAL_SECONDS)
                continue

            # ── AI watchlist ──────────────────────────────────────────────
            watchlist = get_trading_watchlist()
            if not watchlist:
                log("⏳ No watchlist symbols — waiting for AI scan")
                time.sleep(Config.CHECK_INTERVAL_SECONDS)
                continue

            log(f"👀 Watchlist: {', '.join(watchlist)}")

            # ── Manage open positions (breakeven) ────────────────────────
            has_pos, pos_sym = has_any_open_position(watchlist)
            if has_pos:
                check_breakeven_positions(state)
                log(f"📌 Open position on {pos_sym} — skip new entries")
                time.sleep(Config.CHECK_INTERVAL_SECONDS)
                continue

            # ── Consecutive loss stop ─────────────────────────────────────
            consec = count_consecutive_losses()
            state["consecutive_losses"] = consec
            if consec >= Config.MAX_CONSECUTIVE_LOSSES:
                log(f"[CONSEC] {consec} consecutive losses >= {Config.MAX_CONSECUTIVE_LOSSES} — new entries paused")
                save_forex_state(state)
                time.sleep(Config.CHECK_INTERVAL_SECONDS)
                continue

            # ── Per-symbol analysis ───────────────────────────────────────
            order_sent = False
            for preferred in watchlist:
                try:
                    select_symbol(preferred, "AI watchlist candidate")
                    sym = symbol()
                    log(f"🔎 Analyzing {sym}")

                    corr_blocked, corr_msg = has_correlated_position(sym)
                    if corr_blocked:
                        _REJECT_STATS["bot_correlated_pair"] += 1
                        log(f"[CORR] {sym} blocked: {corr_msg}")
                        continue

                    spread_ok, spread_msg = pass_spread_filter()
                    if not spread_ok:
                        _REJECT_STATS["bot_spread_too_high"] += 1
                        _log_testcase("SPREAD_TOO_HIGH", spread_msg, get_spread_points())
                        log(f"⛔ {sym} {spread_msg}")
                        continue

                    risk_ok, risk_msg = pass_daily_risk_filter()
                    if not risk_ok:
                        _REJECT_STATS["bot_daily_limit"] += 1
                        _log_testcase("DAILY_TRADE_LIMIT", risk_msg)
                        log(f"🛑 {sym} {risk_msg}")
                        continue

                    if has_open_position():
                        _log_testcase("OPEN_POSITION_EXISTS", f"{sym} has open position")
                        log(f"📌 {sym} position exists — skip")
                        continue

                    _log_testcase()
                    signal, df_m5 = generate_signal()
                    candle_time   = signal["time"]

                    if last_entry_candle.get(sym) == candle_time:
                        log(f"⏸ {sym} same candle {candle_time} — skip")
                        continue

                    log(f"Signal: {signal}")
                    if signal["side"] == "NO_TRADE":
                        log(f"⏸ {sym} NO_TRADE: {signal['reason']}")
                        last_entry_candle[sym] = candle_time
                        continue

                    tp_sl = calculate_tp_sl(signal)
                    log(f"TP/SL: {tp_sl}")
                    send_order(signal, tp_sl)
                    save_forex_state(state)

                    last_entry_candle[sym] = candle_time
                    order_sent = True
                    break

                except Exception as exc:
                    warn(f"❌ {preferred} error: {exc}")
                    if _is_connection_error(exc):
                        reconnect_mt5(state, exc)
                        order_sent = False
                        break
                    notify_error("FOREX", f"{preferred} error: {exc}")

            if not order_sent:
                log("⏳ No order this round")

            time.sleep(Config.CHECK_INTERVAL_SECONDS)

        except KeyboardInterrupt:
            log("🛑 Stopped by user")
            break
        except Exception as exc:
            warn(f"❌ Loop error: {exc}")
            if _is_connection_error(exc):
                try:
                    reconnect_mt5(state, exc)
                except Exception as rc_exc:
                    warn(f"❌ Reconnect failed: {rc_exc}")
                    notify_error("FOREX", f"Reconnect failed: {rc_exc}")
            else:
                notify_error("FOREX", str(exc))
            time.sleep(Config.CHECK_INTERVAL_SECONDS)

    mt5.shutdown()


if __name__ == "__main__":
    run_bot()