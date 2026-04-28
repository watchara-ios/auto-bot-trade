import os
import time
import json
import hmac
import hashlib
import logging
import requests
import numpy as np
from datetime import datetime
from urllib.parse import urlencode
from dotenv import load_dotenv
from openai import OpenAI
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

# ===== LOAD ENV =====
load_dotenv()

# ===== LOGGER =====
logging.basicConfig(
    filename='trading_bot.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class Logger:
    @staticmethod
    def _time():
        return datetime.now().strftime("%H:%M:%S")

    @staticmethod
    def info(msg):
        print(f"[{Logger._time()}] 📘 {msg}")
        logging.info(msg)

    @staticmethod
    def success(msg):
        print(f"[{Logger._time()}] ✅ {msg}")
        logging.info(msg)

    @staticmethod
    def error(msg):
        print(f"[{Logger._time()}] ❌ {msg}")
        logging.error(msg)

    @staticmethod
    def warn(msg):
        print(f"[{Logger._time()}] ⚠️ {msg}")
        logging.warning(msg)


# ===== CONFIG =====
class Config:
    # Binance
    API_KEY = os.getenv("BINANCE2_API_KEY")
    SECRET = os.getenv("BINANCE2_SECRET")
    BASE_URL = "https://demo-fapi.binance.com"   # เปลี่ยนเป็น fapi.binance.com สำหรับเทรดจริง
    # ★★★ เปลี่ยนเป็นคู่ที่ทำกำไรได้ (USDCAD, USDCHF) ★★★
    SYMBOLS = ["BTCUSDT","ETHUSDT","SOLUSDT"]

    # DeepSeek
    DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY")
    CHAT_MODEL = "deepseek-chat"
    REASONER_MODEL = "deepseek-reasoner"

    # Runtime parameters
    CHECK_INTERVAL = 30
    ENTRY_COOLDOWN = 60
    LOSS_COOLDOWN = 180

    # Position management
    BASE_POSITION_PERCENT = 3.0
    MAX_POSITIONS = 2
    MAX_TOTAL_EXPOSURE = 20.0
    LEVERAGE = 5

    # ★★★ ขนาด Lot ตามความมั่นใจ ★★★
    BIG_LOT = 0.1          # ไม้ใหญ่ (มั่นใจ ≥70%)
    MEDIUM_LOT = 0.05      # ไม้กลาง (มั่นใจ 50-69%)
    SMALL_LOT = 0.01       # ไม้เล็ก (ต่ำกว่า 50%) – ทดสอบระบบ
    MAX_BIG_TRADES_PER_DAY = 2   # จำกัดไม้ใหญ่ต่อวัน

    # Risk control (ปรับ SL/TP ให้กว้างขึ้นสำหรับไม้ใหญ่ในโค้ด)
    DEFAULT_SL_ATR_MULT = 1.5
    DEFAULT_TP_ATR_MULT = 2.5
    BIG_SL_ATR_MULT = 2.0       # สำหรับไม้ใหญ่ (lot >= 0.1)
    BIG_TP_ATR_MULT = 3.5
    TRAILING_ATR_MULT = 0.8
    MAX_CONSECUTIVE_LOSSES = 4
    DAILY_LOSS_LIMIT_PERCENT = 10.0
    DAILY_PROFIT_TARGET = 8.0

    # AI thresholds
    CHAT_CONFIDENCE_THRESHOLD = 50
    MIN_ATR_PERCENT = 0.12
    REASONER_EXIT_INTERVAL = 3600
    VOLUME_SURGE_MULTIPLIER = 0.8
    MOMENTUM_CONFIRM_BARS = 1
    BREAKOUT_LOOKBACK = 20
    TECHNICAL_FALLBACK_CONFIDENCE = 55


@dataclass
class Position:
    symbol: str
    side: str
    quantity: float
    entry_price: float
    sl_price: float
    tp_price: float
    open_time: float
    trailing_active: bool = False


# ===== BINANCE API (Multi-Symbol) =====
class BinanceAPI:
    _time_offset = 0

    @classmethod
    def sync_time(cls):
        try:
            resp = requests.get(f"{Config.BASE_URL}/fapi/v1/time", timeout=5)
            server_time = resp.json()["serverTime"]
            local_time = int(time.time() * 1000)
            cls._time_offset = server_time - local_time
            Logger.info(f"เวลาซิงค์แล้ว offset: {cls._time_offset}ms")
        except Exception as e:
            Logger.error(f"ซิงค์เวลาล้มเหลว: {e}")

    @staticmethod
    def _sign(params: dict) -> str:
        query = urlencode(params)
        signature = hmac.new(Config.SECRET.encode(), query.encode(), hashlib.sha256).hexdigest()
        return f"{query}&signature={signature}"

    @staticmethod
    def _request(method: str, endpoint: str, params: dict = None, retry: int = 3) -> Any:
        params = params or {}
        params["timestamp"] = int(time.time() * 1000) + BinanceAPI._time_offset
        params["recvWindow"] = 5000
        url = f"{Config.BASE_URL}{endpoint}?{BinanceAPI._sign(params)}"
        headers = {"X-MBX-APIKEY": Config.API_KEY}

        for attempt in range(retry):
            try:
                resp = requests.request(method, url, headers=headers, timeout=10)
                data = resp.json()
                if isinstance(data, dict) and data.get("code") is not None:
                    Logger.error(f"API error: {data}")
                return data
            except Exception as e:
                Logger.error(f"Request error: {e}")
                if attempt < retry - 1:
                    time.sleep(2 ** attempt)
        return {}

    @staticmethod
    def get_balance() -> float:
        res = BinanceAPI._request("GET", "/fapi/v2/account")
        if isinstance(res, dict):
            for asset in res.get("assets", []):
                if asset.get("asset") == "USDT":
                    return float(asset.get("walletBalance", 0))
        return 0.0

    @staticmethod
    def get_price(symbol: str) -> float:
        for _ in range(3):
            try:
                res = requests.get(
                    f"{Config.BASE_URL}/fapi/v1/ticker/price",
                    params={"symbol": symbol},
                    timeout=5
                ).json()
                return float(res.get("price", 0))
            except:
                time.sleep(1)
        return 0.0

    @staticmethod
    def get_klines(symbol: str, interval: str, limit: int = 100) -> List[List]:
        try:
            res = requests.get(
                f"{Config.BASE_URL}/fapi/v1/klines",
                params={"symbol": symbol, "interval": interval, "limit": limit},
                timeout=5
            ).json()
            return res if isinstance(res, list) else []
        except:
            return []

    @staticmethod
    def get_open_position_amt(symbol: str) -> float:
        res = BinanceAPI._request("GET", "/fapi/v2/positionRisk")
        if isinstance(res, list):
            for p in res:
                if p["symbol"] == symbol:
                    return float(p["positionAmt"])
        return 0.0

    @staticmethod
    def place_market_order(symbol: str, side: str, quantity: float) -> dict:
        if quantity <= 0:
            return {}
        return BinanceAPI._request("POST", "/fapi/v1/order", {
            "symbol": symbol,
            "side": side,
            "type": "MARKET",
            "quantity": round(quantity, 3)
        })

    @staticmethod
    def set_leverage(symbol: str, leverage: int) -> bool:
        res = BinanceAPI._request("POST", "/fapi/v1/leverage", {
            "symbol": symbol,
            "leverage": leverage
        })
        return isinstance(res, dict) and res.get("leverage") == leverage


# ===== INDICATORS =====
class Indicators:
    @staticmethod
    def rsi(closes: List[float], period: int = 14) -> float:
        if len(closes) < period + 1:
            return 50.0
        deltas = np.diff(closes[-period-1:])
        gain = np.mean([d for d in deltas if d > 0]) or 0.001
        loss = abs(np.mean([d for d in deltas if d < 0])) or 0.001
        rs = gain / loss
        return round(100 - (100 / (1 + rs)), 2)

    @staticmethod
    def ema(closes: List[float], period: int = 20) -> float:
        if len(closes) < period:
            return closes[-1] if closes else 0.0
        return round(np.mean(closes[-period:]), 2)

    @staticmethod
    def atr(klines: List[List], period: int = 14) -> float:
        if len(klines) < period + 1:
            return 0.0
        highs = [float(k[2]) for k in klines[-period-1:]]
        lows = [float(k[3]) for k in klines[-period-1:]]
        closes = [float(k[4]) for k in klines[-period-1:]]
        tr_values = []
        for i in range(1, len(highs)):
            hl = highs[i] - lows[i]
            hc = abs(highs[i] - closes[i-1])
            lc = abs(lows[i] - closes[i-1])
            tr = max(hl, hc, lc)
            tr_values.append(tr)
        return float(np.mean(tr_values)) if tr_values else 0.0

    @staticmethod
    def ema_slope(closes: List[float], period: int = 20, lookback: int = 3) -> float:
        if len(closes) < period + lookback:
            return 0.0
        ema_vals = []
        for i in range(lookback, 0, -1):
            ema_vals.append(Indicators.ema(closes[:-i] if i > 0 else closes, period))
        return (ema_vals[-1] - ema_vals[0]) / lookback


# ===== MARKET DATA =====
class MarketData:
    @staticmethod
    def fetch(symbol: str) -> Optional[Dict]:
        klines_1m = BinanceAPI.get_klines(symbol, "1m", 100)
        klines_5m = BinanceAPI.get_klines(symbol, "5m", 100)
        klines_15m = BinanceAPI.get_klines(symbol, "15m", 100)
        klines_1h = BinanceAPI.get_klines(symbol, "1h", 100)

        if not klines_1m or not klines_5m or not klines_15m or not klines_1h:
            return None

        closes_1m = [float(k[4]) for k in klines_1m]
        closes_5m = [float(k[4]) for k in klines_5m]
        closes_15m = [float(k[4]) for k in klines_15m]
        closes_1h = [float(k[4]) for k in klines_1h]

        # ★★★ คำนวณ EMA50 สำหรับ Trend Filter ไม้ใหญ่ ★★★
        ema50_15m = Indicators.ema(closes_15m, 50) if len(closes_15m) >= 50 else None
        ema50_1h = Indicators.ema(closes_1h, 50) if len(closes_1h) >= 50 else None

        data = {
            "1m": {
                "price": closes_1m[-1],
                "ema20": Indicators.ema(closes_1m, 20),
                "rsi": Indicators.rsi(closes_1m, 14),
                "atr": Indicators.atr(klines_1m, 14),
                "volume": float(klines_1m[-1][5]),
                "closes": closes_1m,
            },
            "5m": {
                "price": closes_5m[-1],
                "ema20": Indicators.ema(closes_5m, 20),
                "rsi": Indicators.rsi(closes_5m, 14),
                "atr": Indicators.atr(klines_5m, 14),
                "closes": closes_5m,
            },
            "15m": {
                "price": closes_15m[-1],
                "ema20": Indicators.ema(closes_15m, 20),
                "rsi": Indicators.rsi(closes_15m, 14),
                "atr": Indicators.atr(klines_15m, 14),
                "ema_slope": Indicators.ema_slope(closes_15m, 20, 5),
                "ema50": ema50_15m,  # เพิ่ม EMA50
                "closes": closes_15m,
            },
            "1h": {
                "price": closes_1h[-1],
                "ema20": Indicators.ema(closes_1h, 20),
                "rsi": Indicators.rsi(closes_1h, 14),
                "atr": Indicators.atr(klines_1h, 14),
                "ema_slope": Indicators.ema_slope(closes_1h, 20, 5),
                "ema50": ema50_1h,
                "closes": closes_1h,
            },
            "klines_15m_raw": klines_15m,
        }
        return data


# ===== FILTERS =====
class EntryFilters:
    @staticmethod
    def momentum_confirmed(closes_1m: List[float], bars: int = 2) -> bool:
        if len(closes_1m) < bars + 1:
            return False
        recent = closes_1m[-bars-1:]
        return all(recent[i] < recent[i+1] for i in range(bars)) or all(recent[i] > recent[i+1] for i in range(bars))

    @staticmethod
    def volume_surge(klines_1m: List[List], multiplier: float = 1.0) -> bool:
        if len(klines_1m) < 21:
            return False
        volumes = [float(k[5]) for k in klines_1m[-21:]]
        current_vol = volumes[-1]
        avg_vol = np.mean(volumes[:-1])
        return current_vol >= avg_vol * multiplier

    @staticmethod
    def detect_breakout(klines_15m: List[List]) -> tuple:
        lookback = Config.BREAKOUT_LOOKBACK
        if len(klines_15m) < lookback + 1:
            return False, ""
        highs = [float(k[2]) for k in klines_15m[-lookback-1:-1]]
        lows = [float(k[3]) for k in klines_15m[-lookback-1:-1]]
        current = float(klines_15m[-1][4])
        resistance = max(highs)
        support = min(lows)
        if current > resistance:
            return True, "BUY"
        elif current < support:
            return True, "SELL"
        return False, ""

    @staticmethod
    def trend_filter(price: float, ema20: float, ema_slope: float, direction: str) -> bool:
        if direction == "BUY":
            return price > ema20 and ema_slope > 0
        elif direction == "SELL":
            return price < ema20 and ema_slope < 0
        return False

    @staticmethod
    def reclaim_signal(data_5m: dict, data_15m: dict, data_1h: dict) -> tuple:
        price_15m = data_15m["price"]
        ema20_15m = data_15m["ema20"]
        ema50_15m = data_15m.get("ema50")
        slope_15m = data_15m.get("ema_slope", 0.0)
        price_1h = data_1h["price"]
        ema20_1h = data_1h["ema20"]
        slope_1h = data_1h.get("ema_slope", 0.0)
        rsi_5m = data_5m.get("rsi", 50.0)
        rsi_15m = data_15m.get("rsi", 50.0)

        buy_reclaim = (
            price_15m > ema20_15m
            and (ema50_15m is None or price_15m > ema50_15m)
            and price_1h > ema20_1h
            and slope_15m >= 0
            and slope_1h >= 0
            and rsi_5m < 75
            and rsi_15m < 75
        )
        sell_reclaim = (
            price_15m < ema20_15m
            and (ema50_15m is None or price_15m < ema50_15m)
            and price_1h < ema20_1h
            and slope_15m <= 0
            and slope_1h <= 0
            and rsi_5m > 25
            and rsi_15m > 25
        )

        if buy_reclaim:
            return True, "BUY", "15m reclaim aligned with 1h trend"
        if sell_reclaim:
            return True, "SELL", "15m breakdown aligned with 1h trend"
        return False, "", ""

    # ★★★ Filter สำหรับไม้ใหญ่: ราคาต้องอยู่เหนือ/ใต้ EMA50 ★★★
    @staticmethod
    def big_trade_trend_filter(price: float, ema50: float, direction: str) -> bool:
        if ema50 is None:
            return False
        if direction == "BUY":
            return price > ema50
        else:
            return price < ema50


# ===== DEEPSEEK AI =====
deepseek_client = OpenAI(api_key=Config.DEEPSEEK_KEY, base_url="https://api.deepseek.com/v1")

class DeepSeekGatekeeper:
    @staticmethod
    def should_enter(symbol: str, data_1m: dict, data_5m: dict, data_15m: dict) -> tuple:
        Logger.info(f"🤖 Gatekeeper ({symbol}) evaluating market...")
        prompt = f"""You are a strict trading gatekeeper. Analyze the following market data for {symbol} perpetual futures.

1-minute data:
- Price: ${data_1m['price']:.2f}
- EMA20: ${data_1m['ema20']:.2f}
- RSI: {data_1m['rsi']:.2f}
- ATR: {data_1m['atr']:.2f}

5-minute data:
- Price: ${data_5m['price']:.2f}
- EMA20: ${data_5m['ema20']:.2f}
- RSI: {data_5m['rsi']:.2f}

15-minute data:
- Price: ${data_15m['price']:.2f}
- EMA20: ${data_15m['ema20']:.2f}
- RSI: {data_15m['rsi']:.2f}
- EMA Slope: {data_15m.get('ema_slope', 0):.4f}

Based on trend strength, momentum, and volatility, determine if now is a good time to consider a trade.
Respond with a JSON object:
{{"decision": "PROCEED" or "HOLD", "confidence": 0-100, "reason": "brief explanation"}}

Only say PROCEED if there is a clear directional bias and volatility is not too low. Otherwise HOLD.
"""
        try:
            start = time.time()
            resp = deepseek_client.chat.completions.create(
                model=Config.CHAT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=200
            )
            elapsed = (time.time() - start) * 1000
            content = resp.choices[0].message.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            result = json.loads(content.strip())
            decision = result.get("decision", "HOLD")
            confidence = float(result.get("confidence", 0))
            reason = result.get("reason", "")
            Logger.info(f"Gatekeeper ({symbol}): {decision} (conf:{confidence}%) in {elapsed:.0f}ms - {reason}")
            return decision == "PROCEED", confidence
        except Exception as e:
            Logger.error(f"Gatekeeper error for {symbol}: {e}")
            return False, 0.0


class DeepSeekStrategist:
    @staticmethod
    def plan_trade(symbol: str, data_1m: dict, data_5m: dict, data_15m: dict, klines_15m: List) -> Optional[dict]:
        Logger.info(f"🧠 Strategist ({symbol}) planning trade...")
        closes_15m = [float(k[4]) for k in klines_15m[-20:]]
        price_summary = f"Last 20 closes (15m): {', '.join([f'${c:.0f}' for c in closes_15m])}"
        atr_15m = data_15m['atr']

        prompt = f"""You are a professional crypto futures strategist. Based on the provided data, propose a single trade setup for {symbol}.

Market Data:
- 1m: Price ${data_1m['price']:.2f}, EMA20 ${data_1m['ema20']:.2f}, RSI {data_1m['rsi']:.2f}, ATR {data_1m['atr']:.2f}
- 5m: Price ${data_5m['price']:.2f}, EMA20 ${data_5m['ema20']:.2f}, RSI {data_5m['rsi']:.2f}, ATR {data_5m['atr']:.2f}
- 15m: Price ${data_15m['price']:.2f}, EMA20 ${data_15m['ema20']:.2f}, RSI {data_15m['rsi']:.2f}, ATR {data_15m['atr']:.2f}, EMA Slope: {data_15m.get('ema_slope',0):.4f}

{price_summary}

Current ATR (15m) = {atr_15m:.2f}

Please suggest:
- direction: "BUY" or "SELL"
- position_size_percent: what % of current balance to risk (1-5%)
- stop_loss_atr_mult: multiple of ATR for stop loss (suggest 1.0-2.0)
- take_profit_atr_mult: multiple of ATR for take profit (suggest 2.0-3.5)
- confidence: 0-100

Output must be strict JSON:
{{"direction": "BUY/SELL", "position_percent": float, "sl_atr_mult": float, "tp_atr_mult": float, "confidence": int, "reasoning": "short text"}}
"""
        try:
            start = time.time()
            resp = deepseek_client.chat.completions.create(
                model=Config.REASONER_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300
            )
            elapsed = (time.time() - start) * 1000
            content = resp.choices[0].message.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            plan = json.loads(content.strip())
            Logger.success(f"Strategist ({symbol}): {plan['direction']} ({plan['confidence']}%) in {elapsed:.0f}ms")
            return plan
        except Exception as e:
            Logger.error(f"Strategist error for {symbol}: {e}")
            return None

    @staticmethod
    def evaluate_exit(position: Position, current_price: float, market_data: Dict) -> Optional[str]:
        """Return "CLOSE" or "HOLD" based on AI decision"""
        Logger.info(f"🧠 Reasoner evaluating exit for {position.symbol} {position.side}...")
        unrealized_pnl = (current_price - position.entry_price) * position.quantity
        if position.side == "SELL":
            unrealized_pnl = -unrealized_pnl
        pnl_percent = (unrealized_pnl / (position.entry_price * position.quantity)) * 100

        data_15m = market_data.get("15m", {})
        prompt = f"""You are an expert exit strategist. Decide whether to close this open position now.

Position:
- Symbol: {position.symbol}
- Side: {position.side}
- Entry Price: ${position.entry_price:.2f}
- Current Price: ${current_price:.2f}
- Unrealized PnL: ${unrealized_pnl:.2f} ({pnl_percent:.2f}%)
- Stop Loss: ${position.sl_price:.2f}, Take Profit: ${position.tp_price:.2f}

Market (15m):
- Price: ${data_15m.get('price', 0):.2f}
- EMA20: ${data_15m.get('ema20', 0):.2f}
- RSI: {data_15m.get('rsi', 50):.2f}
- ATR: {data_15m.get('atr', 0):.2f}
- EMA Slope: {data_15m.get('ema_slope', 0):.4f}

Analyze if momentum is fading or if risk/reward favors taking profit now. Respond JSON:
{{"action": "CLOSE" or "HOLD", "reason": "short explanation"}}
"""
        try:
            resp = deepseek_client.chat.completions.create(
                model=Config.REASONER_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=150
            )
            content = resp.choices[0].message.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            result = json.loads(content.strip())
            action = result.get("action", "HOLD")
            Logger.info(f"Reasoner Exit for {position.symbol}: {action} - {result.get('reason','')}")
            return action
        except Exception as e:
            Logger.error(f"Reasoner exit error: {e}")
            return None


# ===== TRADING BOT (Multi-Symbol, Power Mode) =====
class TradingBot:
    def __init__(self):
        self.positions: Dict[str, List[Position]] = {sym: [] for sym in Config.SYMBOLS}
        self.balance = 0.0
        self.daily_start_balance = 0.0
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.consecutive_losses = 0
        self.last_trade_time: Dict[str, float] = {sym: 0.0 for sym in Config.SYMBOLS}
        self.last_loss_time: Dict[str, float] = {sym: 0.0 for sym in Config.SYMBOLS}
        self.trading_allowed = True
        self.last_reset_date = datetime.now().date()
        self.last_atr: Dict[str, float] = {sym: 0.0 for sym in Config.SYMBOLS}
        self.last_reasoner_exit_time = 0.0
        self.big_trade_count_today = 0   # ★★★ จำนวนไม้ใหญ่ที่เปิดวันนี้

        for sym in Config.SYMBOLS:
            if Config.LEVERAGE > 1:
                BinanceAPI.set_leverage(sym, Config.LEVERAGE)

    def update_balance(self):
        for attempt in range(5):
            bal = BinanceAPI.get_balance()
            if bal > 0:
                self.balance = bal
                if self.daily_start_balance == 0:
                    self.daily_start_balance = self.balance
                return
            time.sleep(2)
        Logger.error("ไม่สามารถดึงยอดเงินที่ถูกต้องได้หลังจากลอง 5 ครั้ง")

    def daily_reset(self):
        today = datetime.now().date()
        if today != self.last_reset_date:
            Logger.success(f"📅 วันเทรดใหม่ PnL เมื่อวาน: ${self.daily_pnl:.2f}")
            self.daily_start_balance = self.balance
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.consecutive_losses = 0
            self.big_trade_count_today = 0   # ★★★ รีเซ็ตจำนวนไม้ใหญ่รายวัน
            self.trading_allowed = True
            self.last_reset_date = today

    def check_risk_limits(self) -> bool:
        if self.daily_start_balance <= 0:
            return True
        daily_loss = -self.daily_pnl
        daily_loss_percent = (daily_loss / self.daily_start_balance) * 100
        if daily_loss_percent >= Config.DAILY_LOSS_LIMIT_PERCENT:
            Logger.error(f"❌ ขีดจำกัดการขาดทุนรายวันถึงแล้ว: {daily_loss_percent:.2f}%")
            self.trading_allowed = False
            return False
        if self.consecutive_losses >= Config.MAX_CONSECUTIVE_LOSSES:
            Logger.error(f"❌ ขาดทุนติดต่อกันเกิน {Config.MAX_CONSECUTIVE_LOSSES} ครั้ง")
            self.trading_allowed = False
            return False
        total_exposure = 0.0
        for sym, pos_list in self.positions.items():
            for pos in pos_list:
                exposure = (pos.quantity * pos.entry_price) / self.balance * 100
                total_exposure += exposure
        if total_exposure > Config.MAX_TOTAL_EXPOSURE:
            Logger.warn(f"⚠️ Exposure รวม {total_exposure:.2f}% เกิน {Config.MAX_TOTAL_EXPOSURE}% ห้ามเปิดเพิ่ม")
            return False
        return True

    def check_daily_target(self) -> bool:
        if self.daily_start_balance <= 0:
            return False
        profit_percent = self.daily_pnl / self.daily_start_balance * 100
        if profit_percent >= Config.DAILY_PROFIT_TARGET:
            Logger.success(f"🎯 บรรลุเป้ากำไรรายวัน: {profit_percent:.2f}%")
            self.trading_allowed = False
            return True
        return False

    def log_market_snapshot(self, symbol: str, market_data: Dict):
        Logger.info(f"📈 Market snapshot {symbol}")
        for tf in ["1m", "5m", "15m", "1h"]:
            data = market_data.get(tf, {})
            price = data.get("price", 0.0)
            ema20 = data.get("ema20", 0.0)
            ema50 = data.get("ema50")
            rsi = data.get("rsi", 0.0)
            atr = data.get("atr", 0.0)
            slope = data.get("ema_slope")
            atr_pct = (atr / price * 100) if price else 0.0
            trend = "UP" if price > ema20 else "DOWN" if price < ema20 else "FLAT"
            extra = ""
            if slope is not None:
                extra += f" | slope={slope:.4f}"
            if ema50 is not None:
                extra += f" | EMA50={ema50:.2f}"
            Logger.info(
                f"   {tf}: price={price:.2f} EMA20={ema20:.2f} RSI={rsi:.2f} "
                f"ATR={atr:.2f} ({atr_pct:.2f}%) trend={trend}{extra}"
            )

    def log_position_status(self, symbol: str, current_price: float):
        for idx, pos in enumerate(self.positions[symbol]):
            if pos.side == "BUY":
                pnl = (current_price - pos.entry_price) * pos.quantity
                pnl_pct = (current_price - pos.entry_price) / pos.entry_price * 100
                sl_gap = (current_price - pos.sl_price) / current_price * 100
                tp_gap = (pos.tp_price - current_price) / current_price * 100
            else:
                pnl = (pos.entry_price - current_price) * pos.quantity
                pnl_pct = (pos.entry_price - current_price) / pos.entry_price * 100
                sl_gap = (pos.sl_price - current_price) / current_price * 100
                tp_gap = (current_price - pos.tp_price) / current_price * 100
            age_min = (time.time() - pos.open_time) / 60
            Logger.info(
                f"📌 Position {symbol}#{idx} {pos.side} qty={pos.quantity} entry={pos.entry_price:.2f} "
                f"now={current_price:.2f} PnL=${pnl:.2f} ({pnl_pct:.2f}%) "
                f"SL={pos.sl_price:.2f} ({sl_gap:.2f}% away) "
                f"TP={pos.tp_price:.2f} ({tp_gap:.2f}% away) "
                f"age={age_min:.1f}m trailing={pos.trailing_active}"
            )

    # ★★★ คำนวณขนาด Lot ตาม confidence ★★★
    def get_lot_size(self, confidence: float) -> float:
        if confidence >= 70:
            return Config.BIG_LOT
        elif confidence >= 50:
            return Config.MEDIUM_LOT
        else:
            return Config.SMALL_LOT

    def calculate_position_size(self, symbol: str, price: float, confidence: float) -> float:
        lot = self.get_lot_size(confidence)
        # แปลง lot เป็น quantity ตามสัญญาของ Binance (สมมติ 1 lot = 1 unit ของ base currency)
        # ปรับตามความเหมาะสมของโบรกเกอร์
        # สำหรับ Binance Futures USDCAD, USDCHF ขนาด 1 contract = 100,000 base units
        # แต่เราใช้ lot แบบง่าย คูณด้วย 100,000 เพื่อให้ได้ position size
        if symbol in ["USDCAD", "USDCHF"]:
            # 1 lot = 100,000 units
            qty = lot * 100000 / price
        else:
            # fallback
            qty = lot * 100000 / price
        # ปัดเศษให้เหมาะสม
        if qty < 0.001:
            qty = 0.001
        return round(qty, 3)

    def open_position(self, symbol: str, plan: dict, current_price: float, atr: float) -> bool:
        direction = plan["direction"]
        confidence = plan.get("confidence", 50)
        lot = self.get_lot_size(confidence)
        sl_mult = plan.get("sl_atr_mult", Config.DEFAULT_SL_ATR_MULT)
        tp_mult = plan.get("tp_atr_mult", Config.DEFAULT_TP_ATR_MULT)

        # ★★★ ตรวจสอบข้อจำกัดไม้ใหญ่ ★★★
        if lot >= Config.BIG_LOT:
            if self.big_trade_count_today >= Config.MAX_BIG_TRADES_PER_DAY:
                Logger.warn(f"🛑 ครบ {Config.MAX_BIG_TRADES_PER_DAY} ไม้ใหญ่แล้ววันนี้ ไม่เปิดเพิ่ม")
                return False
            # Trend filter พิเศษสำหรับไม้ใหญ่
            ema50 = plan.get("_ema50")  # ถูกส่งมาจาก main loop
            if ema50 and not EntryFilters.big_trade_trend_filter(current_price, ema50, direction):
                Logger.info(f"⛔ ไม้ใหญ่ {symbol} ไม่ผ่าน EMA50 filter (ราคา {current_price} vs EMA50 {ema50})")
                return False

        qty = self.calculate_position_size(symbol, current_price, confidence)
        if qty <= 0:
            return False

        # ★★★ ปรับ SL/TP สำหรับไม้ใหญ่ ★★★
        if lot >= Config.BIG_LOT:
            sl_mult = Config.BIG_SL_ATR_MULT
            tp_mult = Config.BIG_TP_ATR_MULT

        if direction == "BUY":
            sl_price = current_price - atr * sl_mult
            tp_price = current_price + atr * tp_mult
        else:
            sl_price = current_price + atr * sl_mult
            tp_price = current_price - atr * tp_mult

        Logger.info(f"🚀 เปิด {symbol} {direction} {qty} (lot {lot}) @ ${current_price:.2f} conf:{confidence}%")
        Logger.info(f"   SL: ${sl_price:.2f} | TP: ${tp_price:.2f} (ATR={atr:.2f})")

        resp = BinanceAPI.place_market_order(symbol, direction, qty)
        if resp.get("orderId"):
            self.positions[symbol].append(Position(
                symbol=symbol,
                side=direction,
                quantity=qty,
                entry_price=current_price,
                sl_price=sl_price,
                tp_price=tp_price,
                open_time=time.time()
            ))
            self.daily_trades += 1
            if lot >= Config.BIG_LOT:
                self.big_trade_count_today += 1   # ★★★ เพิ่มจำนวนไม้ใหญ่
            self.last_trade_time[symbol] = time.time()
            Logger.success(f"✅ เปิดออเดอร์ {symbol} สำเร็จ")
            return True
        else:
            Logger.error(f"คำสั่ง {symbol} ล้มเหลว")
            return False

    def close_position(self, symbol: str, index: int, current_price: float, reason: str):
        if index >= len(self.positions[symbol]):
            return
        pos = self.positions[symbol][index]
        close_side = "SELL" if pos.side == "BUY" else "BUY"
        resp = BinanceAPI.place_market_order(symbol, close_side, pos.quantity)

        if resp.get("orderId"):
            if pos.side == "BUY":
                pnl = (current_price - pos.entry_price) * pos.quantity
            else:
                pnl = (pos.entry_price - current_price) * pos.quantity
            pnl_pct = pnl / (pos.entry_price * pos.quantity) * 100

            if pnl < 0:
                self.consecutive_losses += 1
                self.last_loss_time[symbol] = time.time()
                Logger.warn(f"⚠️ ขาดทุน {symbol} ติดต่อกัน: {self.consecutive_losses}")
            else:
                self.consecutive_losses = 0

            self.daily_pnl += pnl
            self.positions[symbol].pop(index)
            Logger.success(f"💰 ปิด {symbol}: {reason} | PnL: ${pnl:.2f} ({pnl_pct:.2f}%)")
        else:
            Logger.error(f"ไม่สามารถปิดออเดอร์ {symbol} ได้")

    def check_exits(self, symbol: str, current_price: float, atr: float):
        for i, pos in reversed(list(enumerate(self.positions[symbol]))):
            Logger.info(f"🔎 Exit check {symbol}#{i}: side={pos.side} price={current_price:.2f} SL={pos.sl_price:.2f} TP={pos.tp_price:.2f} ATR={atr:.2f}")
            if pos.side == "BUY":
                if current_price <= pos.sl_price:
                    Logger.warn(f"🚪 {symbol} BUY hit SL: price {current_price:.2f} <= SL {pos.sl_price:.2f}")
                    self.close_position(symbol, i, current_price, "Stop Loss")
                    continue
                if current_price >= pos.tp_price:
                    Logger.success(f"🚪 {symbol} BUY hit TP: price {current_price:.2f} >= TP {pos.tp_price:.2f}")
                    self.close_position(symbol, i, current_price, "Take Profit")
                    continue
                # Trailing stop เมื่อกำไรเกิน 0.5 ATR
                if current_price > pos.entry_price + 0.5 * atr:
                    new_sl = current_price - Config.TRAILING_ATR_MULT * atr
                    if new_sl > pos.sl_price:
                        old_sl = pos.sl_price
                        pos.sl_price = new_sl
                        pos.trailing_active = True
                        Logger.info(f"🔁 {symbol} BUY trailing SL moved {old_sl:.2f} -> {new_sl:.2f}")
                    else:
                        Logger.info(f"⏸️ {symbol} BUY trailing not moved: new SL {new_sl:.2f} <= current SL {pos.sl_price:.2f}")
                else:
                    trigger = pos.entry_price + 0.5 * atr
                    Logger.info(f"⏸️ {symbol} BUY hold: price below trailing trigger {trigger:.2f}")
            else:
                if current_price >= pos.sl_price:
                    Logger.warn(f"🚪 {symbol} SELL hit SL: price {current_price:.2f} >= SL {pos.sl_price:.2f}")
                    self.close_position(symbol, i, current_price, "Stop Loss")
                    continue
                if current_price <= pos.tp_price:
                    Logger.success(f"🚪 {symbol} SELL hit TP: price {current_price:.2f} <= TP {pos.tp_price:.2f}")
                    self.close_position(symbol, i, current_price, "Take Profit")
                    continue
                if current_price < pos.entry_price - 0.5 * atr:
                    new_sl = current_price + Config.TRAILING_ATR_MULT * atr
                    if new_sl < pos.sl_price:
                        old_sl = pos.sl_price
                        pos.sl_price = new_sl
                        pos.trailing_active = True
                        Logger.info(f"🔁 {symbol} SELL trailing SL moved {old_sl:.2f} -> {new_sl:.2f}")
                    else:
                        Logger.info(f"⏸️ {symbol} SELL trailing not moved: new SL {new_sl:.2f} >= current SL {pos.sl_price:.2f}")
                else:
                    trigger = pos.entry_price - 0.5 * atr
                    Logger.info(f"⏸️ {symbol} SELL hold: price above trailing trigger {trigger:.2f}")

    def reasoner_exit_check(self):
        elapsed = time.time() - self.last_reasoner_exit_time
        if elapsed < Config.REASONER_EXIT_INTERVAL:
            remaining = Config.REASONER_EXIT_INTERVAL - elapsed
            total_positions = sum(len(v) for v in self.positions.values())
            if total_positions > 0:
                Logger.info(f"⏸️ Reasoner exit ยังไม่ถึงรอบ เหลือ {remaining/60:.1f} นาที (positions={total_positions})")
            return
        self.last_reasoner_exit_time = time.time()
        Logger.info("🕐 Reasoner กำลังตรวจสอบการปิดออเดอร์ทุกสัญลักษณ์...")

        for symbol in list(self.positions.keys()):
            if not self.positions[symbol]:
                continue
            market_data = MarketData.fetch(symbol)
            if not market_data:
                continue
            current_price = market_data["1m"]["price"]
            for i in reversed(range(len(self.positions[symbol]))):
                pos = self.positions[symbol][i]
                action = DeepSeekStrategist.evaluate_exit(pos, current_price, market_data)
                if action == "CLOSE":
                    self.close_position(symbol, i, current_price, "AI Reasoner Exit")
                elif action == "HOLD":
                    Logger.info(f"⏸️ AI Reasoner ให้ถือ {symbol}#{i} ต่อ")
                else:
                    Logger.warn(f"⚠️ AI Reasoner ไม่มีคำสั่งปิดที่ชัดเจนสำหรับ {symbol}#{i}: {action}")

    def run(self):
        Logger.success("=" * 50)
        Logger.success("🚀 Multi-Symbol Power Trading Bot (ไม้ใหญ่เน้นคุณภาพ)")
        Logger.success("=" * 50)
        Logger.info(f"Symbols: {', '.join(Config.SYMBOLS)}")
        Logger.info(f"Gatekeeper: {Config.CHAT_MODEL} | Strategist: {Config.REASONER_MODEL}")
        Logger.info(f"Leverage: {Config.LEVERAGE}x | Cooldown: {Config.ENTRY_COOLDOWN}s")
        Logger.info(f"ไม้ใหญ่: {Config.BIG_LOT} lot (สูงสุด {Config.MAX_BIG_TRADES_PER_DAY} ครั้ง/วัน)")
        Logger.info("=" * 50)

        self.update_balance()
        if self.balance <= 0:
            Logger.error("ไม่สามารถเริ่มบอทได้เนื่องจากยอดเงินไม่ถูกต้อง")
            return
        self.daily_start_balance = self.balance

        for sym in Config.SYMBOLS:
            existing_amt = BinanceAPI.get_open_position_amt(sym)
            if abs(existing_amt) > 0:
                price = BinanceAPI.get_price(sym)
                self.positions[sym].append(Position(
                    symbol=sym,
                    side="BUY" if existing_amt > 0 else "SELL",
                    quantity=abs(existing_amt),
                    entry_price=price,
                    sl_price=price * 0.99,
                    tp_price=price * 1.02,
                    open_time=time.time()
                ))
                Logger.warn(f"⚠️ พบสถานะ {sym} เปิดค้างอยู่ โหลดเข้าสู่ระบบ")

        while True:
            try:
                self.daily_reset()
                if not self.trading_allowed or not self.check_risk_limits():
                    time.sleep(60)
                    continue
                if self.check_daily_target():
                    time.sleep(60)
                    continue

                self.update_balance()

                self.reasoner_exit_check()

                for symbol in Config.SYMBOLS:
                    Logger.info(f"\n--- กำลังวิเคราะห์ {symbol} ---")
                    market_data = MarketData.fetch(symbol)
                    if not market_data:
                        Logger.warn(f"ไม่สามารถดึงข้อมูล {symbol} ได้")
                        continue

                    current_price = market_data["1m"]["price"]
                    atr_15m = market_data["15m"]["atr"]
                    self.last_atr[symbol] = atr_15m

                    self.log_market_snapshot(symbol, market_data)
                    if self.positions[symbol]:
                        self.log_position_status(symbol, current_price)
                        self.check_exits(symbol, current_price, atr_15m)

                    if atr_15m <= 0 or (atr_15m / current_price * 100) < Config.MIN_ATR_PERCENT:
                        Logger.info(f"⏸️ {symbol} ATR ต่ำเกินไป ({atr_15m/current_price*100:.2f}%) ข้ามเฉพาะการเปิดไม้ใหม่")
                        continue

                    now = time.time()
                    cooldown_active = (now - self.last_trade_time[symbol] < Config.ENTRY_COOLDOWN) or \
                                      (now - self.last_loss_time[symbol] < Config.LOSS_COOLDOWN)
                    if cooldown_active:
                        Logger.info(f"⏸️ {symbol} อยู่ใน cooldown")
                        continue

                    if len(self.positions[symbol]) >= Config.MAX_POSITIONS:
                        Logger.info(f"📊 {symbol} มีตำแหน่งครบ {Config.MAX_POSITIONS} แล้ว")
                        continue

                    reclaim = False
                    breakout, breakout_dir = EntryFilters.detect_breakout(market_data["klines_15m_raw"])
                    if breakout:
                        Logger.success(f"🚨 {symbol} Breakout detected! Direction: {breakout_dir}. Bypassing momentum filter.")
                        direction_guess = breakout_dir
                    else:
                        reclaim, reclaim_dir, reclaim_reason = EntryFilters.reclaim_signal(
                            market_data["5m"], market_data["15m"], market_data["1h"]
                        )
                        if reclaim:
                            Logger.success(f"⚡ {symbol} Reclaim signal: {reclaim_dir} - {reclaim_reason}")
                            direction_guess = reclaim_dir
                        else:
                            reclaim_reason = ""

                        price_15m = market_data["15m"]["price"]
                        ema20_15m = market_data["15m"]["ema20"]
                        slope = market_data["15m"]["ema_slope"]
                        if reclaim:
                            pass
                        elif slope > 0 and price_15m > ema20_15m:
                            direction_guess = "BUY"
                        elif slope < 0 and price_15m < ema20_15m:
                            direction_guess = "SELL"
                        else:
                            Logger.info(f"⏸️ {symbol} แนวโน้มไม่ชัดเจน ข้าม")
                            continue

                        if not EntryFilters.momentum_confirmed(market_data["1m"]["closes"], bars=Config.MOMENTUM_CONFIRM_BARS):
                            Logger.info(f"⏸️ {symbol} โมเมนตัมไม่ยืนยัน ข้าม")
                            continue

                        if not EntryFilters.volume_surge(BinanceAPI.get_klines(symbol, "1m", 30), multiplier=Config.VOLUME_SURGE_MULTIPLIER):
                            Logger.info(f"⏸️ {symbol} ปริมาณต่ำกว่าค่าเฉลี่ย ข้าม")
                            continue

                    should_proceed, conf = DeepSeekGatekeeper.should_enter(
                        symbol,
                        market_data["1m"], market_data["5m"], market_data["15m"]
                    )
                    if not should_proceed or conf < Config.CHAT_CONFIDENCE_THRESHOLD:
                        if breakout or reclaim:
                            Logger.warn(
                                f"⚠️ {symbol} ใช้ technical fallback เพราะมี signal ชัด "
                                f"(breakout={breakout}, reclaim={reclaim}, AI conf={conf})"
                            )
                            plan = {
                                "direction": direction_guess,
                                "position_percent": Config.BASE_POSITION_PERCENT,
                                "sl_atr_mult": Config.DEFAULT_SL_ATR_MULT,
                                "tp_atr_mult": Config.DEFAULT_TP_ATR_MULT,
                                "confidence": Config.TECHNICAL_FALLBACK_CONFIDENCE,
                                "reasoning": "technical fallback"
                            }
                        else:
                            Logger.info(f"🚫 {symbol} Gatekeeper ปฏิเสธ (conf={conf})")
                            continue
                    else:
                        plan = DeepSeekStrategist.plan_trade(
                            symbol,
                            market_data["1m"], market_data["5m"], market_data["15m"],
                            market_data["klines_15m_raw"]
                        )
                        if not plan:
                            continue

                    if plan["direction"] != direction_guess:
                        Logger.info(
                            f"⏸️ {symbol} แผนไม่ตรงกับสัญญาณเทคนิค "
                            f"(plan={plan['direction']} signal={direction_guess}) ข้าม"
                        )
                        continue

                    # Trend filter มาตรฐาน
                    if not EntryFilters.trend_filter(market_data["15m"]["price"],
                                                     market_data["15m"]["ema20"],
                                                     market_data["15m"]["ema_slope"],
                                                     plan["direction"]):
                        Logger.info(f"⏸️ {symbol} เทรนด์ไม่สอดคล้องกับแผน ({plan['direction']}) ข้าม")
                        continue

                    # ส่ง EMA50 ให้ open_position ใช้ตรวจไม้ใหญ่
                    plan["_ema50"] = market_data["15m"].get("ema50")

                    self.open_position(symbol, plan, current_price, atr_15m)

                total_positions = sum(len(v) for v in self.positions.values())
                daily_ret = (self.balance - self.daily_start_balance) / self.daily_start_balance * 100
                Logger.info(f"📊 Balance: ${self.balance:.2f} | Daily Return: {daily_ret:.2f}% | Positions: {total_positions} | Big Trades: {self.big_trade_count_today}")
                time.sleep(Config.CHECK_INTERVAL)

            except KeyboardInterrupt:
                Logger.warn("⏹️ Bot หยุดโดยผู้ใช้")
                break
            except Exception as e:
                Logger.error(f"ข้อผิดพลาดใน loop หลัก: {e}")
                time.sleep(10)


if __name__ == "__main__":
    BinanceAPI.sync_time()
    bot = TradingBot()
    bot.run()
