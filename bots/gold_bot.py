"""
gold_bot.py — XAUUSDm Dedicated Bot (MetaTrader 5)  [Gold V2 + Indicator V3]

Strategy: session momentum breakout (Gold V2 core)
  - SELL: no additional indicator filter (backtest PF 1.845)
  - BUY:  MACD histogram > 0 on M15 required (backtest PF 1.557, PASS_DRY_RUN_CANDIDATE)
  - Session: UTC 12:00-14:00 (Thai 19:00-21:00, NY early momentum)
  - Signal: Donchian-20 breakout + ADX rising + body quality + ATR percentile
  - SL: max(8-bar swing/entry + ATR×1.0)  |  RR: 1.8  |  Max hold: 720 min
  - D1 regime filter ON (additional live safety layer)
  - AI news gate ON
  - Separate kill file: logs/gold_STOP
  - Separate state file:  logs/gold_state.json
"""

import json
import os
import sys
import time
import builtins
from collections import Counter
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

_EMOJI_MAP = {
    "🔌": "[CONNECT]", "🔎": "[SCAN]",  "✅": "[OK]",     "⛔": "[BLOCK]",
    "❌": "[ERROR]",   "📌": "[ORDER]",  "🚀": "[START]",  "⏳": "[WAIT]",
    "⏸":  "[SKIP]",   "🛑": "[STOP]",   "⚠️": "[WARN]",   "👀": "[WATCH]",
    "🧠": "[AI]",
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
sys.path.insert(0, str(PROJECT_ROOT))   # for strategies.gold_v2_core

_REJECT_STATS: Counter = Counter()

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
    SYMBOL   = os.getenv("GOLD_SYMBOL", "XAUUSDm")
    LOT      = float(os.getenv("GOLD_LOT", "0.01"))

    TF_TREND = mt5.TIMEFRAME_M15
    TF_ENTRY = mt5.TIMEFRAME_M5
    TF_EXEC  = mt5.TIMEFRAME_M1
    BARS     = 500

    # Strategy — Gold V2 + Indicator V3 (SELL PF 1.845 | BUY+MACD PF 1.557)
    ALLOWED_SIDE         = os.getenv("GOLD_ALLOWED_SIDE", "BOTH")    # BOTH: SELL unrestricted, BUY needs MACD
    RR                   = float(os.getenv("GOLD_RR", "1.8"))        # V2 validated RR
    RISK_PCT             = float(os.getenv("GOLD_RISK_PCT", "0.0025"))  # 0.25% per trade
    ADX_MIN              = float(os.getenv("GOLD_ADX_MIN", "18.0"))
    BODY_MULT            = float(os.getenv("GOLD_BODY_MULT", "2.0"))    # body > 2× avg_body20
    MAX_WICK_PCT         = float(os.getenv("GOLD_MAX_WICK_PCT", "0.40"))
    ATR_PERCENTILE_MIN   = float(os.getenv("GOLD_ATR_PCT_MIN", "40.0"))
    MACD_BUY_FILTER      = os.getenv("GOLD_MACD_BUY_FILTER", "true").lower() == "true"  # require MACD hist>0 for BUY
    MOMENTUM_SL_ATR_MULT = float(os.getenv("GOLD_SL_ATR_MULT", "1.0")) # SL = swing + ATR×1.0
    MAX_HOLD_MINUTES     = int(os.getenv("GOLD_MAX_HOLD_MINUTES", "720"))  # 12h timeout exit
    # Session: UTC 12:00-14:00 (Thai 19:00-21:00, NY early momentum)
    SESSION_UTC_START_H  = int(os.getenv("GOLD_SESSION_UTC_START_H", "12"))
    SESSION_UTC_END_H    = int(os.getenv("GOLD_SESSION_UTC_END_H",   "14"))

    # D1 Regime — only trade when daily ADX ≥ 20 and D1 trend aligns
    USE_D1_REGIME = os.getenv("GOLD_USE_D1_REGIME", "true").lower() == "true"
    D1_ADX_MIN    = float(os.getenv("GOLD_D1_ADX_MIN", "20.0"))
    D1_BARS       = 260

    # Session — Gold trades all hours; only block weekend gap (Fri close → Sun open)
    # UTC: market closed Fri 21:00 → Sun 21:00
    BLOCK_WEEKEND_HOURS = os.getenv("GOLD_BLOCK_WEEKEND", "true").lower() == "true"

    # Risk / safety
    MAX_TRADES_PER_DAY     = int(os.getenv("GOLD_MAX_TRADES_PER_DAY", "3"))
    MAX_DAILY_LOSS_PCT     = float(os.getenv("GOLD_MAX_DAILY_LOSS_PCT", "3.0"))
    MAX_SPREAD_POINTS      = int(os.getenv("GOLD_MAX_SPREAD_POINTS", "600"))  # Gold spread wider
    MAX_CONSECUTIVE_LOSSES = int(os.getenv("GOLD_MAX_CONSECUTIVE_LOSSES", "3"))
    MAGIC_NUMBER           = int(os.getenv("GOLD_MAGIC_NUMBER", "20260501"))

    # MT5
    DRY_RUN             = os.getenv("GOLD_DRY_RUN", "false").lower() == "true"
    MT5_LOGIN           = os.getenv("MT5_LOGIN", "").strip()
    MT5_PASSWORD        = os.getenv("MT5_PASSWORD", "").strip()
    MT5_SERVER          = os.getenv("MT5_SERVER", "").strip()
    MT5_EXPLICIT_LOGIN  = os.getenv("GOLD_MT5_EXPLICIT_LOGIN", "true").lower() == "true"

    # AI news gate — checks major gold-moving macro/news risk before entries.
    AI_NEWS_ENABLED       = os.getenv("GOLD_AI_NEWS_ENABLED", "true").lower() == "true"
    AI_NEWS_FAIL_OPEN     = os.getenv("GOLD_AI_NEWS_FAIL_OPEN", "true").lower() == "true"
    AI_NEWS_CACHE_MINUTES = int(os.getenv("GOLD_AI_NEWS_CACHE_MINUTES", "30"))
    AI_NEWS_BLOCK_RISK    = int(os.getenv("GOLD_AI_NEWS_BLOCK_RISK", "70"))
    AI_NEWS_LOOKAHEAD_MIN = int(os.getenv("GOLD_AI_NEWS_LOOKAHEAD_MINUTES", "120"))
    AI_NEWS_BLOCK_MINUTES = int(os.getenv("GOLD_AI_NEWS_BLOCK_MINUTES", "60"))
    DEEPSEEK_KEY          = os.getenv("DEEPSEEK_API_KEY")
    DEEPSEEK_MODEL        = os.getenv("GOLD_DEEPSEEK_MODEL", os.getenv("FOREX_DEEPSEEK_MODEL", "deepseek-reasoner"))

    # Timing
    CHECK_INTERVAL_SECONDS              = int(os.getenv("GOLD_CHECK_INTERVAL_SECONDS", "60"))
    MT5_RECONNECT_SLEEP_SECONDS         = int(os.getenv("GOLD_MT5_RECONNECT_SLEEP_SECONDS", "10"))
    MT5_DISCONNECT_NOTIFY_COOLDOWN_SECS = int(os.getenv("GOLD_MT5_DISCONNECT_NOTIFY_COOLDOWN_SECONDS", "300"))

    # Paths
    LOG_DIR    = Path("logs")
    KILL_FILE  = LOG_DIR / "gold_STOP"
    STATE_FILE = LOG_DIR / "gold_state.json"
    AI_NEWS_STATE_FILE = LOG_DIR / "gold_ai_news_state.json"
    AI_NEWS_LOG        = LOG_DIR / "gold_ai_news.jsonl"


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

def connect_mt5() -> None:
    if not mt5.initialize():
        raise RuntimeError(f"mt5.initialize() failed: {mt5.last_error()}")
    if Config.MT5_EXPLICIT_LOGIN and Config.MT5_LOGIN:
        ok = mt5.login(int(Config.MT5_LOGIN), Config.MT5_PASSWORD, Config.MT5_SERVER)
        if not ok:
            raise RuntimeError(f"mt5.login() failed: {mt5.last_error()}")
    acc = mt5.account_info()
    if acc is None:
        raise RuntimeError("Not logged in to MT5")
    log(f"[CONNECT] account={acc.login}  server={acc.server}  balance={acc.balance:.2f}")


def _is_connection_error(exc: Exception) -> bool:
    return any(k in str(exc).lower() for k in ("terminal", "connection", "timeout", "initialize"))


def mt5_ping() -> None:
    if mt5.account_info() is None:
        raise RuntimeError("MT5 connection lost")


def reconnect_mt5(state: dict, exc: Exception) -> None:
    warn(f"[RECONNECT] {exc}")
    state["mt5_down"] = True
    now = time.time()
    if now - state.get("last_disconnect_notify", 0) > Config.MT5_DISCONNECT_NOTIFY_COOLDOWN_SECS:
        notify_error("GOLD", f"MT5 disconnected: {exc}")
        state["last_disconnect_notify"] = now
    time.sleep(Config.MT5_RECONNECT_SLEEP_SECONDS)
    try:
        mt5.shutdown()
        connect_mt5()
        state["mt5_down"] = False
        notify_reconnected("GOLD", "MT5 restored")
    except Exception as rc_exc:
        warn(f"[RECONNECT] Failed: {rc_exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def get_ohlcv(timeframe: int, bars: int = 500) -> pd.DataFrame:
    rates = mt5.copy_rates_from_pos(Config.SYMBOL, timeframe, 0, bars)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"No data from MT5: {Config.SYMBOL} tf={timeframe}")
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.rename(columns={"tick_volume": "volume"}, inplace=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# D1 Regime filter
# ─────────────────────────────────────────────────────────────────────────────

def _d1_adx(df: pd.DataFrame, period: int = 14) -> float:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    plus_dm  = (high.diff()).where((high.diff() > -low.diff()) & (high.diff() > 0), 0.0)
    minus_dm = (-low.diff()).where((-low.diff() > high.diff()) & (-low.diff() > 0), 0.0)
    alpha = 1 / period
    atr_s    = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_di  = 100 * plus_dm.ewm(alpha=alpha, adjust=False).mean() / atr_s.replace(0, float("nan"))
    minus_di = 100 * minus_dm.ewm(alpha=alpha, adjust=False).mean() / atr_s.replace(0, float("nan"))
    dx  = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, float("nan"))) * 100
    adx = dx.ewm(alpha=alpha, adjust=False).mean()
    return float(adx.iloc[-1]) if not adx.empty else 0.0


def d1_regime_ok(side: str) -> tuple[bool, str]:
    if not Config.USE_D1_REGIME:
        return True, ""
    try:
        df = get_ohlcv(mt5.TIMEFRAME_D1, Config.D1_BARS)
    except RuntimeError:
        return True, ""  # fail-open

    if len(df) < 60:
        return True, ""

    close  = df["close"]
    ema50  = close.ewm(span=50,  adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()
    d1_trend = 1 if ema50.iloc[-1] > ema200.iloc[-1] else -1
    d1_adx   = _d1_adx(df)

    if d1_adx < Config.D1_ADX_MIN:
        return False, f"D1 ADX {d1_adx:.1f} < {Config.D1_ADX_MIN} (ranging)"
    if side == "BUY"  and d1_trend != 1:
        return False, f"D1 trend bearish — Gold below D1 EMA200"
    if side == "SELL" and d1_trend != -1:
        return False, f"D1 trend bullish — Gold above D1 EMA200"
    return True, f"D1 ADX={d1_adx:.1f} trend={'UP' if d1_trend == 1 else 'DOWN'}"


# ─────────────────────────────────────────────────────────────────────────────
# AI news gate
# ─────────────────────────────────────────────────────────────────────────────

def _read_ai_news_state() -> dict:
    try:
        if Config.AI_NEWS_STATE_FILE.exists():
            return json.loads(Config.AI_NEWS_STATE_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        pass
    return {}


def _write_ai_news_state(state: dict) -> None:
    Config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    Config.AI_NEWS_STATE_FILE.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")


def _strip_json_fence(text: str) -> str:
    text = text.strip()
    for fence in ("```json", "```"):
        if fence in text:
            text = text.split(fence, 1)[1].split("```", 1)[0]
    return text.strip()


def _last_closed_snapshot(timeframe: int, bars: int = 120) -> dict:
    df = get_ohlcv(timeframe, bars)
    row = df.iloc[-2] if len(df) >= 2 else df.iloc[-1]
    return {
        "time": str(row.get("time", "")),
        "open": float(row["open"]),
        "high": float(row["high"]),
        "low": float(row["low"]),
        "close": float(row["close"]),
        "volume": float(row.get("volume", 0)),
    }


def _build_ai_news_prompt(signal: dict, market: dict) -> str:
    now_local = datetime.now()
    now_utc = datetime.now(timezone.utc)
    return f"""You are a professional gold (XAUUSD) risk analyst.
Task: decide whether a new gold trade should be blocked because of major market-moving news.

Current time:
- Local/Bangkok: {now_local:%Y-%m-%d %H:%M:%S} UTC+7
- UTC: {now_utc:%Y-%m-%d %H:%M:%S}

Trade setup:
- Symbol: {Config.SYMBOL}
- Side: {signal.get("side")}
- Tier: {signal.get("tier")}
- Entry: {signal.get("entry")}
- SL: {signal.get("sl")}
- TP: {signal.get("tp")}
- D1 regime: {signal.get("d1_regime", "")}

Recent closed candles:
- M5: {market.get("m5")}
- M15: {market.get("m15")}
- D1: {market.get("d1")}

Block only if there is high-impact gold risk now or within the next {Config.AI_NEWS_LOOKAHEAD_MIN} minutes:
- FOMC/Fed rate decision, Fed Chair speech, dot plot, minutes
- US CPI, PCE, NFP, unemployment, jobless claims, GDP, ISM, retail sales
- major USD yield shock, central-bank surprise, war/geopolitical escalation
- unscheduled headline likely to cause gold whipsaw/liquidity sweep

If you cannot verify real-time news/calendar, do not invent headlines. Use known scheduled macro-risk logic and return ALLOW unless risk is clearly high.
Return ONLY valid JSON:
{{
  "action": "ALLOW" or "BLOCK",
  "risk_score": 0-100,
  "block_minutes": integer,
  "event": "short event label or none",
  "reason": "max 120 chars"
}}"""


def _ai_news_fallback(message: str) -> tuple[bool, str]:
    if Config.AI_NEWS_FAIL_OPEN:
        return True, f"AI news unavailable; allowing ({message})"
    return False, f"AI news unavailable; blocking ({message})"


def run_ai_news_scan(signal: dict, force: bool = False) -> dict:
    now = datetime.now()
    state = _read_ai_news_state()
    last_scan_at = state.get("last_scan_at")
    if last_scan_at and not force:
        try:
            age = (now - datetime.strptime(last_scan_at, "%Y-%m-%d %H:%M:%S")).total_seconds()
            if age < Config.AI_NEWS_CACHE_MINUTES * 60:
                return state
        except ValueError:
            pass

    if not Config.AI_NEWS_ENABLED:
        state.update({
            "action": "ALLOW",
            "risk_score": 0,
            "event": "disabled",
            "reason": "AI news gate disabled",
            "last_scan_at": now.strftime("%Y-%m-%d %H:%M:%S"),
        })
        return state
    if not Config.DEEPSEEK_KEY:
        ok, reason = _ai_news_fallback("missing DEEPSEEK_API_KEY")
        action = "ALLOW" if ok else "BLOCK"
        state.update({
            "action": action,
            "risk_score": 0 if ok else 100,
            "event": "api_key_missing",
            "reason": reason,
            "last_scan_at": now.strftime("%Y-%m-%d %H:%M:%S"),
        })
        _write_ai_news_state(state)
        return state
    if OpenAI is None:
        ok, reason = _ai_news_fallback("openai package not installed")
        action = "ALLOW" if ok else "BLOCK"
        state.update({
            "action": action,
            "risk_score": 0 if ok else 100,
            "event": "openai_missing",
            "reason": reason,
            "last_scan_at": now.strftime("%Y-%m-%d %H:%M:%S"),
        })
        _write_ai_news_state(state)
        return state

    try:
        market = {
            "m5": _last_closed_snapshot(Config.TF_ENTRY),
            "m15": _last_closed_snapshot(Config.TF_TREND),
            "d1": _last_closed_snapshot(mt5.TIMEFRAME_D1, Config.D1_BARS),
        }
        client = OpenAI(api_key=Config.DEEPSEEK_KEY, base_url="https://api.deepseek.com/v1")
        resp = client.chat.completions.create(
            model=Config.DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": _build_ai_news_prompt(signal, market)}],
            temperature=0.1,
            max_tokens=260,
        )
        result = json.loads(_strip_json_fence(resp.choices[0].message.content))
        risk_score = int(result.get("risk_score", 0))
        action = str(result.get("action", "ALLOW")).upper()
        if risk_score >= Config.AI_NEWS_BLOCK_RISK:
            action = "BLOCK"
        if action not in {"ALLOW", "BLOCK"}:
            action = "ALLOW"
        state = {
            "last_scan_at": now.strftime("%Y-%m-%d %H:%M:%S"),
            "action": action,
            "risk_score": risk_score,
            "block_minutes": int(result.get("block_minutes") or Config.AI_NEWS_BLOCK_MINUTES),
            "event": str(result.get("event", "none"))[:120],
            "reason": str(result.get("reason", ""))[:240],
            "model": Config.DEEPSEEK_MODEL,
        }
        Config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        with Config.AI_NEWS_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(state, ensure_ascii=False) + "\n")
        _write_ai_news_state(state)
        log(f"🧠 News risk: {state['action']} score={risk_score} event={state['event']} reason={state['reason']}")
        return state
    except Exception as exc:
        ok, reason = _ai_news_fallback(str(exc))
        action = "ALLOW" if ok else "BLOCK"
        state.update({
            "action": action,
            "risk_score": 0 if ok else 100,
            "event": "ai_error",
            "reason": reason,
            "last_scan_at": now.strftime("%Y-%m-%d %H:%M:%S"),
        })
        _write_ai_news_state(state)
        warn(f"🧠 AI news gate failed: {exc}")
        return state


def pass_ai_news_filter(signal: dict) -> tuple[bool, str]:
    state = run_ai_news_scan(signal)
    action = str(state.get("action", "ALLOW")).upper()
    risk_score = int(state.get("risk_score") or 0)
    if action == "BLOCK" or risk_score >= Config.AI_NEWS_BLOCK_RISK:
        block_minutes = int(state.get("block_minutes") or Config.AI_NEWS_BLOCK_MINUTES)
        event = state.get("event", "major news")
        reason = state.get("reason", "")
        return False, f"AI news block {block_minutes}m | score={risk_score} | {event}: {reason}"
    return True, f"AI news OK score={risk_score} | {state.get('reason', '')}"


# ─────────────────────────────────────────────────────────────────────────────
# Session / weekend gate
# ─────────────────────────────────────────────────────────────────────────────

def pass_weekend_gate() -> bool:
    """Block trading during Fri 21:00 UTC – Sun 21:00 UTC (market closed)."""
    if not Config.BLOCK_WEEKEND_HOURS:
        return True
    now = datetime.now(timezone.utc)
    wd  = now.weekday()   # Mon=0 … Sun=6
    h   = now.hour
    # Friday after 21:00 UTC
    if wd == 4 and h >= 21:
        return False
    # All day Saturday
    if wd == 5:
        return False
    # Sunday before 21:00 UTC
    if wd == 6 and h < 21:
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Spread & risk filters
# ─────────────────────────────────────────────────────────────────────────────

def pass_spread_filter() -> tuple[bool, str]:
    info = mt5.symbol_info(Config.SYMBOL)
    tick = mt5.symbol_info_tick(Config.SYMBOL)
    if info is None or tick is None or info.point <= 0:
        return True, ""
    spread_pts = round((tick.ask - tick.bid) / info.point)
    if spread_pts > Config.MAX_SPREAD_POINTS:
        return False, f"spread {spread_pts} pts > max {Config.MAX_SPREAD_POINTS}"
    return True, f"spread={spread_pts}pts"


def pass_daily_risk_filter() -> tuple[bool, str]:
    acc = mt5.account_info()
    if acc is None:
        return True, ""
    today = datetime.now().date()
    deals = mt5.history_deals_get(
        datetime.combine(today, dtime.min),
        datetime.combine(today, dtime.max),
    ) or []
    closed = [d for d in deals if d.entry == mt5.DEAL_ENTRY_OUT
              and d.symbol == Config.SYMBOL and d.magic == Config.MAGIC_NUMBER]
    if len(closed) >= Config.MAX_TRADES_PER_DAY:
        return False, f"max {Config.MAX_TRADES_PER_DAY} trades/day reached"
    day_pnl = sum(d.profit for d in closed)
    limit   = acc.balance * (Config.MAX_DAILY_LOSS_PCT / 100)
    if day_pnl <= -limit:
        return False, f"daily loss {day_pnl:.2f} exceeds -{limit:.2f}"
    return True, ""


def has_open_position() -> bool:
    positions = mt5.positions_get(symbol=Config.SYMBOL) or []
    return any(p.magic == Config.MAGIC_NUMBER for p in positions)


# ─────────────────────────────────────────────────────────────────────────────
# Gold V2 strategy config
# ─────────────────────────────────────────────────────────────────────────────

def _v2_config():
    from strategies.gold_v2_core import GoldV2Config
    side = Config.ALLOWED_SIDE
    return GoldV2Config(
        strategy                    = "momentum",
        variant                     = "live",
        allowed_side                = side if side != "BOTH" else None,
        session_ranges_utc_minutes  = ((Config.SESSION_UTC_START_H * 60,
                                        Config.SESSION_UTC_END_H   * 60),),
        adx_min                     = Config.ADX_MIN,
        momentum_rr                 = Config.RR,
        use_atr_expansion           = False,
        body_mult                   = Config.BODY_MULT,
        max_wick_pct                = Config.MAX_WICK_PCT,
        momentum_sl_atr_mult        = Config.MOMENTUM_SL_ATR_MULT,
        momentum_atr_percentile_min = Config.ATR_PERCENTILE_MIN,
        max_trades_per_day          = Config.MAX_TRADES_PER_DAY,
        max_hold_minutes            = Config.MAX_HOLD_MINUTES,
        risk_pct                    = Config.RISK_PCT,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Signal generation (Gold V2 momentum + Indicator V3 filters)
# ─────────────────────────────────────────────────────────────────────────────

def _m15_macd_hist(df_m15: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> float:
    """Return last *completed* M15 bar MACD histogram value (equivalent to shift(1) in backtest)."""
    close = df_m15["close"]
    if len(close) < slow + signal + 2:
        return 0.0
    macd_line = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    hist = macd_line - macd_line.ewm(span=signal, adjust=False).mean()
    return float(hist.iloc[-2])   # iloc[-1] = forming bar, iloc[-2] = last completed bar


def generate_signal() -> dict:
    from strategies.gold_v2_core import prepare as v2_prepare, momentum_signal, trade_levels

    df_m5  = get_ohlcv(Config.TF_ENTRY, Config.BARS)
    df_m15 = get_ohlcv(Config.TF_TREND, Config.BARS)
    df_m5  = df_m5.set_index("time")
    df_m15 = df_m15.set_index("time")

    _REJECT_STATS["_total_attempts"] += 1
    cfg = _v2_config()

    try:
        prepared = v2_prepare(df_m5, df_m15, cfg)
    except Exception as exc:
        return {"side": "NO_TRADE", "reason": f"prepare_error: {exc}"}

    if len(prepared) < 50:
        return {"side": "NO_TRADE", "reason": "insufficient_bars"}

    row         = prepared.iloc[-2]     # last fully-closed M5 bar
    candle_time = str(prepared.index[-2])

    sig = momentum_signal(row, cfg)
    if sig is None:
        _REJECT_STATS["no_momentum_signal"] += 1
        return {"side": "NO_TRADE", "reason": "no_momentum_signal", "candle_time": candle_time}

    if sig["side"] == "BUY" and Config.MACD_BUY_FILTER:
        macd_val = _m15_macd_hist(df_m15)
        if macd_val <= 0:
            _REJECT_STATS["macd_buy_blocked"] += 1
            return {"side": "NO_TRADE", "reason": f"BUY blocked: MACD hist {macd_val:.4f} <= 0", "candle_time": candle_time}

    tick = mt5.symbol_info_tick(Config.SYMBOL)
    if tick is None:
        return {"side": "NO_TRADE", "reason": "no_tick", "candle_time": candle_time}
    entry = tick.bid if sig["side"] == "SELL" else tick.ask

    levels = trade_levels(row, entry, sig, cfg)
    if levels is None:
        _REJECT_STATS["invalid_levels"] += 1
        return {"side": "NO_TRADE", "reason": "invalid_levels", "candle_time": candle_time}
    sl, tp, risk_dist = levels

    if Config.USE_D1_REGIME:
        regime_ok, regime_reason = d1_regime_ok(sig["side"])
        if not regime_ok:
            _REJECT_STATS["d1_regime_blocked"] += 1
            log(f"[D1] blocked {sig['side']}: {regime_reason}")
            return {"side": "NO_TRADE", "reason": f"D1: {regime_reason}", "candle_time": candle_time}
        regime_note = regime_reason
    else:
        regime_note = ""

    return {
        "side":        sig["side"],
        "entry":       entry,
        "sl":          sl,
        "tp":          tp,
        "risk_dist":   risk_dist,
        "pattern":     sig.get("pattern", ""),
        "candle_time": candle_time,
        "d1_regime":   regime_note,
        "risk_pct":    Config.RISK_PCT,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Position management: timeout exit (matches backtest max_hold_minutes=720)
# ─────────────────────────────────────────────────────────────────────────────

def _check_max_hold() -> None:
    if Config.MAX_HOLD_MINUTES <= 0:
        return
    now = datetime.now(timezone.utc)
    for pos in (mt5.positions_get(symbol=Config.SYMBOL) or []):
        if pos.magic != Config.MAGIC_NUMBER:
            continue
        hold_min = (now - datetime.fromtimestamp(pos.time, tz=timezone.utc)).total_seconds() / 60
        if hold_min < Config.MAX_HOLD_MINUTES:
            continue
        tick = mt5.symbol_info_tick(Config.SYMBOL)
        if tick is None:
            continue
        close_type  = mt5.ORDER_TYPE_BUY if pos.type == mt5.ORDER_TYPE_SELL else mt5.ORDER_TYPE_SELL
        close_price = tick.ask if close_type == mt5.ORDER_TYPE_BUY else tick.bid
        log(f"[TIMEOUT] ticket={pos.ticket} hold={hold_min:.0f}m >= {Config.MAX_HOLD_MINUTES}m — closing at {close_price:.5f}")
        if Config.DRY_RUN:
            log(f"[DRY_RUN] would close timeout {pos.ticket}")
            continue
        req = mt5.order_send({
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       Config.SYMBOL,
            "volume":       pos.volume,
            "type":         close_type,
            "position":     pos.ticket,
            "price":        close_price,
            "deviation":    20,
            "magic":        Config.MAGIC_NUMBER,
            "comment":      "gold_timeout",
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        })
        if req and req.retcode == mt5.TRADE_RETCODE_DONE:
            log(f"[TIMEOUT] Closed ticket={pos.ticket}")
            notify_order_result("GOLD", Config.SYMBOL, "CLOSE_TIMEOUT", req, dry_run=False)


# ─────────────────────────────────────────────────────────────────────────────
# Order execution
# ─────────────────────────────────────────────────────────────────────────────

def _calculate_lot(signal: dict) -> float:
    account = mt5.account_info()
    info    = mt5.symbol_info(Config.SYMBOL)
    if account is None or info is None:
        return Config.LOT

    risk_pct   = float(signal.get("risk_pct", Config.RISK_PCT))
    risk_money = account.balance * risk_pct
    stop_dist  = abs(float(signal["entry"]) - float(signal["sl"]))
    if stop_dist <= 0 or info.trade_tick_size <= 0 or info.trade_tick_value <= 0:
        return Config.LOT

    lot = risk_money / (stop_dist / info.trade_tick_size * info.trade_tick_value)
    step    = info.volume_step
    min_lot = info.volume_min
    max_lot = info.volume_max
    lot = max(min_lot, min(max_lot, np.floor(lot / step) * step))

    # Margin check
    order_type_mg = mt5.ORDER_TYPE_BUY if signal.get("side") == "BUY" else mt5.ORDER_TYPE_SELL
    margin_needed = mt5.order_calc_margin(order_type_mg, Config.SYMBOL, lot, float(signal["entry"]))
    if margin_needed is not None and margin_needed > 0:
        free = float(getattr(account, "margin_free", 0))
        if free > 0 and margin_needed > free * 0.9:
            capped = np.floor(free * 0.9 / margin_needed * lot / step) * step
            capped = max(min_lot, min(max_lot, capped))
            warn(f"[MARGIN] margin {margin_needed:.2f} > free*0.9={free*0.9:.2f} — lot {lot:.3f}->{capped:.3f}")
            lot = capped
    return round(lot, 2)


def send_order(signal: dict) -> None:
    if Config.DRY_RUN:
        log(f"[DRY_RUN] {signal['side']} pattern={signal.get('pattern','')} "
            f"entry={signal['entry']:.5f} sl={signal['sl']:.5f} tp={signal['tp']:.5f}")
        notify_order_opened(
            "GOLD", Config.SYMBOL, signal["side"], 0,
            signal["entry"], signal["sl"], signal["tp"],
            tier="",
            risk_pct=signal.get("risk_pct", Config.RISK_PCT),
            dry_run=True,
        )
        return

    info  = mt5.symbol_info(Config.SYMBOL)
    tick  = mt5.symbol_info_tick(Config.SYMBOL)
    if info is None or tick is None:
        warn("[ORDER] No symbol info")
        return

    side  = signal["side"]
    d     = info.digits
    price = round(tick.ask if side == "BUY" else tick.bid, d)
    sl    = round(float(signal["sl"]), d)
    tp    = round(float(signal["tp"]), d)

    # Stops level check
    stops_lv = getattr(info, "stops_level", 0)
    if stops_lv > 0:
        min_dist = stops_lv * info.point
        if abs(price - sl) < min_dist:
            warn(f"[STOPS] SL too close: dist={abs(price-sl):.5f} < min={min_dist:.5f}")
            return
        if abs(tp - price) < min_dist:
            warn(f"[STOPS] TP too close: dist={abs(tp-price):.5f} < min={min_dist:.5f}")
            return

    lot = _calculate_lot(signal)
    request = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       Config.SYMBOL,
        "volume":       lot,
        "type":         mt5.ORDER_TYPE_BUY if side == "BUY" else mt5.ORDER_TYPE_SELL,
        "price":        price,
        "sl":           sl,
        "tp":           tp,
        "deviation":    10,
        "magic":        Config.MAGIC_NUMBER,
        "comment":      f"gold_v2_{signal.get('pattern', '')}",
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    log(f"[ORDER] result: {result}")
    notify_order_result("GOLD", Config.SYMBOL, side, result, dry_run=False)

    if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
        time.sleep(0.3)
        for p in (mt5.positions_get(symbol=Config.SYMBOL) or []):
            if p.magic == Config.MAGIC_NUMBER and p.sl == 0:
                warn(f"[VERIFY] Position {p.ticket} has no SL — please set manually!")
                notify_error("GOLD", f"Position {p.ticket} on {Config.SYMBOL} opened without SL!")


# ─────────────────────────────────────────────────────────────────────────────
# State persistence
# ─────────────────────────────────────────────────────────────────────────────

def load_state() -> dict:
    defaults = {
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


def save_state(state: dict) -> None:
    Config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    saveable = {k: v for k, v in state.items()
                if k not in ("mt5_down", "last_disconnect_notify")}
    try:
        Config.STATE_FILE.write_text(
            json.dumps(saveable, indent=2, default=str), encoding="utf-8"
        )
    except OSError as exc:
        warn(f"[STATE] Cannot save: {exc}")


def count_consecutive_losses() -> int:
    today = datetime.now().date()
    deals = mt5.history_deals_get(
        datetime.combine(today, dtime.min),
        datetime.combine(today, dtime.max),
    ) or []
    closed = [d for d in deals if d.entry == mt5.DEAL_ENTRY_OUT
              and d.symbol == Config.SYMBOL and d.magic == Config.MAGIC_NUMBER]
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
    macd_note = "MACD_BUY_ON" if Config.MACD_BUY_FILTER else "MACD_BUY_OFF"
    mode = "DRY_RUN" if Config.DRY_RUN else "LIVE"
    log(
        f"[START] Gold V2+IndV3 bot | symbol={Config.SYMBOL} | pid={os.getpid()} | "
        f"interval={Config.CHECK_INTERVAL_SECONDS}s | mode={mode} | DRY_RUN={Config.DRY_RUN} | "
        f"side={Config.ALLOWED_SIDE} | RR={Config.RR} | {macd_note} | "
        f"session=UTC{Config.SESSION_UTC_START_H:02d}:00-{Config.SESSION_UTC_END_H:02d}:00 | "
        f"D1_regime={'ON' if Config.USE_D1_REGIME else 'OFF'}"
    )
    notify_bot_started(
        "Gold V2+IndV3 Momentum Bot",
        "DRY_RUN" if Config.DRY_RUN else "LIVE/DEMO",
        (
            f"Symbol: <code>{Config.SYMBOL}</code>\n"
            f"Side: <code>{Config.ALLOWED_SIDE}</code> | MACD BUY filter: <code>{Config.MACD_BUY_FILTER}</code>\n"
            f"Session: <code>UTC {Config.SESSION_UTC_START_H:02d}:00-{Config.SESSION_UTC_END_H:02d}:00</code>\n"
            f"RR: <code>{Config.RR}</code> | D1 regime: <code>{Config.USE_D1_REGIME}</code>\n"
            f"AI news gate: <code>{Config.AI_NEWS_ENABLED}</code>"
        ),
    )
    try:
        connect_mt5()
    except Exception as exc:
        notify_error("GOLD", f"Startup failed: {exc}")
        raise

    state = load_state()
    state["mt5_down"] = False
    state["last_disconnect_notify"] = 0
    last_entry_candle: str | None = None

    log("🚀 Gold bot running")

    while True:
        try:
            # ── Kill file ─────────────────────────────────────────────────
            if Config.KILL_FILE.exists():
                log("[KILL] gold_STOP detected — exiting cleanly")
                save_state(state)
                break

            mt5_ping()
            if state.get("mt5_down"):
                state["mt5_down"] = False
                notify_reconnected("GOLD", "MT5 restored")

            # ── Hourly stats ──────────────────────────────────────────────
            _now  = datetime.now()
            today = _now.date()
            if state["last_stats_date"] != today:
                if _REJECT_STATS:
                    log(f"[STATS] Daily: {dict(_REJECT_STATS.most_common())}")
                _REJECT_STATS.clear()
                state["last_stats_date"] = today
                save_state(state)
            elif _now.hour != state["last_stats_hour"] and _REJECT_STATS:
                total = _REJECT_STATS.get("_total_attempts", 1)
                top   = {k: f"{v}({v/total:.0%})"
                         for k, v in _REJECT_STATS.most_common()
                         if not k.startswith("_")}
                log(f"[STATS] Hourly ({total} attempts): {dict(list(top.items())[:8])}")
                state["last_stats_hour"] = _now.hour

            # ── Weekend gate ──────────────────────────────────────────────
            if not pass_weekend_gate():
                log("⏳ Weekend — market closed")
                time.sleep(Config.CHECK_INTERVAL_SECONDS)
                continue

            # ── Manage open position (timeout exit) ──────────────────────
            if has_open_position():
                _check_max_hold()
                log("📌 Gold position open — skip new entry")
                time.sleep(Config.CHECK_INTERVAL_SECONDS)
                continue

            # ── Consecutive loss stop ─────────────────────────────────────
            consec = count_consecutive_losses()
            state["consecutive_losses"] = consec
            if consec >= Config.MAX_CONSECUTIVE_LOSSES:
                log(f"[CONSEC] {consec} losses — new entries paused today")
                save_state(state)
                time.sleep(Config.CHECK_INTERVAL_SECONDS)
                continue

            # ── Spread filter ─────────────────────────────────────────────
            spread_ok, spread_msg = pass_spread_filter()
            if not spread_ok:
                _REJECT_STATS["spread_too_high"] += 1
                log(f"⛔ {spread_msg}")
                time.sleep(Config.CHECK_INTERVAL_SECONDS)
                continue

            # ── Daily risk filter ─────────────────────────────────────────
            risk_ok, risk_msg = pass_daily_risk_filter()
            if not risk_ok:
                _REJECT_STATS["daily_limit"] += 1
                log(f"🛑 {risk_msg}")
                time.sleep(Config.CHECK_INTERVAL_SECONDS)
                continue

            # ── Signal ───────────────────────────────────────────────────
            signal = generate_signal()
            candle_time = str(signal.get("candle_time", ""))

            if last_entry_candle == candle_time and candle_time:
                log(f"⏸ Same candle {candle_time} — skip")
                time.sleep(Config.CHECK_INTERVAL_SECONDS)
                continue

            last_entry_candle = candle_time

            if signal["side"] == "NO_TRADE":
                log(f"⏸ NO_TRADE: {signal['reason']}")
                time.sleep(Config.CHECK_INTERVAL_SECONDS)
                continue

            news_ok, news_msg = pass_ai_news_filter(signal)
            if not news_ok:
                _REJECT_STATS["ai_news_blocked"] += 1
                log(f"🧠 {news_msg}")
                save_state(state)
                time.sleep(Config.CHECK_INTERVAL_SECONDS)
                continue

            log(f"👀 Signal: side={signal['side']} pattern={signal.get('pattern','')} "
                f"entry={signal.get('entry')} sl={signal.get('sl')} tp={signal.get('tp')} "
                f"d1={signal.get('d1_regime','')} news={news_msg}")
            send_order(signal)
            save_state(state)
            time.sleep(Config.CHECK_INTERVAL_SECONDS)

        except KeyboardInterrupt:
            log("🛑 Stopped by user")
            save_state(state)
            break
        except Exception as exc:
            warn(f"❌ Loop error: {exc}")
            if _is_connection_error(exc):
                try:
                    reconnect_mt5(state, exc)
                except Exception as rc_exc:
                    warn(f"❌ Reconnect failed: {rc_exc}")
                    notify_error("GOLD", f"Reconnect failed: {rc_exc}")
            else:
                notify_error("GOLD", f"Loop error: {exc}")
            time.sleep(Config.CHECK_INTERVAL_SECONDS)

    mt5.shutdown()
    log("Gold bot stopped.")


if __name__ == "__main__":
    run_bot()
