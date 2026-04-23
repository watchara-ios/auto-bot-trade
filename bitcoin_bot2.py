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
    API_KEY = os.getenv("BINANCE_API_KEY")
    SECRET = os.getenv("BINANCE_SECRET")
    BASE_URL = "https://demo-fapi.binance.com"   # เปลี่ยนเป็น fapi.binance.com สำหรับเทรดจริง
    SYMBOLS = ["BTCUSDT", "ETHUSDT"]

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

    # Risk control
    DEFAULT_SL_ATR_MULT = 1.5
    DEFAULT_TP_ATR_MULT = 2.5
    TRAILING_ATR_MULT = 0.8
    MAX_CONSECUTIVE_LOSSES = 4
    DAILY_LOSS_LIMIT_PERCENT = 10.0
    DAILY_PROFIT_TARGET = 8.0

    # AI thresholds
    CHAT_CONFIDENCE_THRESHOLD = 50
    # เพิ่มเติม: ขีดจำกัด ATR ขั้นต่ำ (% ของราคา)
    MIN_ATR_PERCENT = 0.3


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

    # เพิ่มเติม: EMA slope (ใช้ 3 แท่งล่าสุด)
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

        if not klines_1m or not klines_5m or not klines_15m:
            return None

        closes_1m = [float(k[4]) for k in klines_1m]
        closes_5m = [float(k[4]) for k in klines_5m]
        closes_15m = [float(k[4]) for k in klines_15m]

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
                "ema_slope": Indicators.ema_slope(closes_15m, 20, 5),  # เพิ่ม slope
                "closes": closes_15m,
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
        if all(recent[i] < recent[i+1] for i in range(bars)):
            return True
        if all(recent[i] > recent[i+1] for i in range(bars)):
            return True
        return False

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
        if len(klines_15m) < 20:
            return False, ""
        highs = [float(k[2]) for k in klines_15m[-20:]]
        lows = [float(k[3]) for k in klines_15m[-20:]]
        current = float(klines_15m[-1][4])
        resistance = max(highs)
        support = min(lows)
        if current > resistance:
            return True, "BUY"
        elif current < support:
            return True, "SELL"
        return False, ""

    # เพิ่มเติม: Trend Filter (15m EMA20 slope + ราคาเทียบ EMA)
    @staticmethod
    def trend_filter(price: float, ema20: float, ema_slope: float, direction: str) -> bool:
        if direction == "BUY":
            # ราคาอยู่เหนือ EMA และ slope > 0 (แนวโน้มขึ้น)
            return price > ema20 and ema_slope > 0
        elif direction == "SELL":
            # ราคาอยู่ใต้ EMA และ slope < 0 (แนวโน้มลง)
            return price < ema20 and ema_slope < 0
        return False


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

    # เพิ่มเติม: Reasoner ประเมินว่าควรปิด position หรือไม่
    @staticmethod
    def evaluate_exit(position: Position, current_price: float, market_data: Dict) -> Optional[str]:
        """Return "CLOSE" or "HOLD" based on AI decision"""
        Logger.info(f"🧠 Reasoner evaluating exit for {position.symbol} {position.side}...")
        unrealized_pnl = (current_price - position.entry_price) * position.quantity
        if position.side == "SELL":
            unrealized_pnl = -unrealized_pnl
        pnl_percent = (unrealized_pnl / (position.entry_price * position.quantity)) * 100

        # ดึงข้อมูล market data ที่จำเป็น
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


# ===== TRADING BOT (Multi-Symbol) =====
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
        # เพิ่มเติม: เวลาที่ใช้ Reasoner ตรวจสอบ exit ล่าสุด
        self.last_reasoner_exit_time = 0.0

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

    def calculate_position_size(self, symbol: str, price: float, percent: float) -> float:
        capital_used = self.balance * (percent / 100)
        position_notional = capital_used * Config.LEVERAGE
        qty = position_notional / price
        step = 0.001 if symbol == "BTCUSDT" else 0.01
        return max(step, round(qty, 3 if symbol == "BTCUSDT" else 2))

    def open_position(self, symbol: str, plan: dict, current_price: float, atr: float) -> bool:
        direction = plan["direction"]
        pos_percent = plan["position_percent"]
        sl_mult = plan["sl_atr_mult"]
        tp_mult = plan["tp_atr_mult"]

        qty = self.calculate_position_size(symbol, current_price, pos_percent)
        if qty <= 0:
            return False

        if direction == "BUY":
            sl_price = current_price - atr * sl_mult
            tp_price = current_price + atr * tp_mult
        else:
            sl_price = current_price + atr * sl_mult
            tp_price = current_price - atr * tp_mult

        Logger.info(f"🚀 เปิด {symbol} {direction} {qty} @ ${current_price:.2f}")
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
            if pos.side == "BUY":
                if current_price <= pos.sl_price:
                    self.close_position(symbol, i, current_price, "Stop Loss")
                    continue
                if current_price >= pos.tp_price:
                    self.close_position(symbol, i, current_price, "Take Profit")
                    continue
                # Trailing stop (ใช้เมื่อกำไรเกิน 0.5 ATR)
                if current_price > pos.entry_price + 0.5 * atr:
                    new_sl = current_price - Config.TRAILING_ATR_MULT * atr
                    if new_sl > pos.sl_price:
                        pos.sl_price = new_sl
                        pos.trailing_active = True
            else:
                if current_price >= pos.sl_price:
                    self.close_position(symbol, i, current_price, "Stop Loss")
                    continue
                if current_price <= pos.tp_price:
                    self.close_position(symbol, i, current_price, "Take Profit")
                    continue
                if current_price < pos.entry_price - 0.5 * atr:
                    new_sl = current_price + Config.TRAILING_ATR_MULT * atr
                    if new_sl < pos.sl_price:
                        pos.sl_price = new_sl
                        pos.trailing_active = True

    # เพิ่มเติม: ฟังก์ชันให้ Reasoner ตรวจ exit ทุกชั่วโมง
    def reasoner_exit_check(self):
        """เรียกทุก 1 ชั่วโมง ให้ Reasoner วิเคราะห์ว่าควรปิดออเดอร์หรือไม่"""
        if time.time() - self.last_reasoner_exit_time < 3600:
            return  # ยังไม่ถึงเวลา
        self.last_reasoner_exit_time = time.time()
        Logger.info("🕐 Reasoner กำลังตรวจสอบการปิดออเดอร์ทุกสัญลักษณ์...")

        # ดึงข้อมูลตลาดของทุก symbol ที่มี position
        for symbol in list(self.positions.keys()):
            if not self.positions[symbol]:
                continue
            market_data = MarketData.fetch(symbol)
            if not market_data:
                continue
            current_price = market_data["1m"]["price"]
            # ต้องวนลูปจากหลังไปหน้าเพราะอาจมีการลบระหว่าง iteration
            for i in reversed(range(len(self.positions[symbol]))):
                pos = self.positions[symbol][i]
                action = DeepSeekStrategist.evaluate_exit(pos, current_price, market_data)
                if action == "CLOSE":
                    self.close_position(symbol, i, current_price, "AI Reasoner Exit")

    def run(self):
        Logger.success("=" * 50)
        Logger.success("🚀 Multi-Symbol BTC+ETH Trading Bot (Active Mode with Reasoner Exit)")
        Logger.success("=" * 50)
        Logger.info(f"Symbols: {', '.join(Config.SYMBOLS)}")
        Logger.info(f"Gatekeeper: {Config.CHAT_MODEL} (threshold: {Config.CHAT_CONFIDENCE_THRESHOLD}%)")
        Logger.info(f"Strategist: {Config.REASONER_MODEL}")
        Logger.info(f"Leverage: {Config.LEVERAGE}x | Cooldown: {Config.ENTRY_COOLDOWN}s")
        Logger.info(f"Min ATR%: {Config.MIN_ATR_PERCENT}% | Reasoner Exit Check ทุก 1 ชม.")
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

                # *** เช็ค Reasoner Exit ทุก 1 ชั่วโมง ***
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

                    # เพิ่มเติม: เช็ค ATR ขั้นต่ำ (กันตลาดเงียบ)
                    if atr_15m <= 0 or (atr_15m / current_price * 100) < Config.MIN_ATR_PERCENT:
                        Logger.info(f"⏸️ {symbol} ATR ต่ำเกินไป ({atr_15m/current_price*100:.2f}%) ข้าม")
                        continue

                    # ตรวจสอบการปิด position ตาม SL/TP/Trailing
                    self.check_exits(symbol, current_price, atr_15m)

                    now = time.time()
                    cooldown_active = (now - self.last_trade_time[symbol] < Config.ENTRY_COOLDOWN) or \
                                      (now - self.last_loss_time[symbol] < Config.LOSS_COOLDOWN)
                    if cooldown_active:
                        Logger.info(f"⏸️ {symbol} อยู่ใน cooldown")
                        continue

                    if len(self.positions[symbol]) >= Config.MAX_POSITIONS:
                        Logger.info(f"📊 {symbol} มีตำแหน่งครบ {Config.MAX_POSITIONS} แล้ว")
                        continue

                    # ตัวกรอง
                    breakout, breakout_dir = EntryFilters.detect_breakout(market_data["klines_15m_raw"])
                    if breakout:
                        Logger.success(f"🚨 {symbol} Breakout detected! Direction: {breakout_dir}. Bypassing momentum filter.")
                        direction_guess = breakout_dir
                    else:
                        # ใช้ trend filter เพื่อเดาทิศทางจาก EMA slope (สำหรับ momentum filter)
                        price_15m = market_data["15m"]["price"]
                        ema20_15m = market_data["15m"]["ema20"]
                        slope = market_data["15m"]["ema_slope"]
                        if slope > 0 and price_15m > ema20_15m:
                            direction_guess = "BUY"
                        elif slope < 0 and price_15m < ema20_15m:
                            direction_guess = "SELL"
                        else:
                            # แนวโน้มไม่ชัด
                            Logger.info(f"⏸️ {symbol} แนวโน้มไม่ชัดเจน (slope={slope:.4f}) ข้าม")
                            continue

                        if not EntryFilters.momentum_confirmed(market_data["1m"]["closes"], bars=2):
                            Logger.info(f"⏸️ {symbol} โมเมนตัมไม่ยืนยัน ข้าม")
                            continue

                        if not EntryFilters.volume_surge(BinanceAPI.get_klines(symbol, "1m", 30), multiplier=1.0):
                            Logger.info(f"⏸️ {symbol} ปริมาณต่ำกว่าค่าเฉลี่ย ข้าม")
                            continue

                        # ใช้ trend filter ตรวจสอบอีกครั้งกับทิศทางที่ได้จาก strategist
                        # (เราจะส่งต่อให้ Gatekeeper/Strategist ตัดสินอีกที แต่เพิ่มความแข็งแกร่ง)
                        Logger.info(f"✅ {symbol} ผ่านตัวกรอง เตรียมส่ง AI...")

                    # Gatekeeper
                    should_proceed, conf = DeepSeekGatekeeper.should_enter(
                        symbol,
                        market_data["1m"], market_data["5m"], market_data["15m"]
                    )
                    if not should_proceed or conf < Config.CHAT_CONFIDENCE_THRESHOLD:
                        Logger.info(f"🚫 {symbol} Gatekeeper ปฏิเสธ (conf={conf})")
                        continue

                    # Strategist
                    plan = DeepSeekStrategist.plan_trade(
                        symbol,
                        market_data["1m"], market_data["5m"], market_data["15m"],
                        market_data["klines_15m_raw"]
                    )
                    if not plan:
                        continue

                    # ตรวจสอบ trend filter อีกครั้งด้วยทิศทางจากแผน
                    if not EntryFilters.trend_filter(market_data["15m"]["price"],
                                                     market_data["15m"]["ema20"],
                                                     market_data["15m"]["ema_slope"],
                                                     plan["direction"]):
                        Logger.info(f"⏸️ {symbol} เทรนด์ไม่สอดคล้องกับแผน ({plan['direction']}) ข้าม")
                        continue

                    self.open_position(symbol, plan, current_price, atr_15m)

                # สรุปสถานะ
                total_positions = sum(len(v) for v in self.positions.values())
                daily_ret = (self.balance - self.daily_start_balance) / self.daily_start_balance * 100
                Logger.info(f"📊 Balance: ${self.balance:.2f} | Daily Return: {daily_ret:.2f}% | Total Positions: {total_positions}")
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