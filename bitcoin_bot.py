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
from dataclasses import dataclass, field
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
    BASE_URL = "https://demo-fapi.binance.com"
    # BASE_URL = "https://fapi.binance.com"           # Use fapi.binance.com for live trading, testnet.binancefuture.com for testnet
    SYMBOL = "BTCUSDT"

    # DeepSeek
    DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY")
    CHAT_MODEL = "deepseek-chat"
    REASONER_MODEL = "deepseek-reasoner"

    # Runtime parameters
    CHECK_INTERVAL = 30          # Status check interval (seconds)
    ENTRY_COOLDOWN = 120         # Minimum interval between two entries (seconds)
    LOSS_COOLDOWN = 300          # Cooldown period after a loss (seconds)

    # Position management
    BASE_POSITION_PERCENT = 5.0  # Base position size as % of balance (before leverage)
    MAX_POSITIONS = 2
    MAX_TOTAL_EXPOSURE = 30.0    # Maximum total exposure percentage (after leverage)
    LEVERAGE = 5

    # Risk control
    DEFAULT_SL_ATR_MULT = 1.5    # Stop loss = ATR * multiplier
    DEFAULT_TP_ATR_MULT = 2.5    # Take profit = ATR * multiplier
    TRAILING_ATR_MULT = 0.8      # Trailing stop distance = ATR * multiplier
    MAX_CONSECUTIVE_LOSSES = 4
    DAILY_LOSS_LIMIT_PERCENT = 10.0   # Maximum daily loss percentage (based on starting balance)
    DAILY_PROFIT_TARGET = 8.0         # Daily profit target (stop opening new trades when reached)

    # AI thresholds
    CHAT_CONFIDENCE_THRESHOLD = 70     # Minimum confidence for chat model to allow trade


@dataclass
class Position:
    side: str                  # "BUY" or "SELL"
    quantity: float            # Contract quantity
    entry_price: float
    sl_price: float
    tp_price: float
    open_time: float
    trailing_active: bool = False


# ===== BINANCE API =====
class BinanceAPI:
    @staticmethod
    def _sign(params: dict) -> str:
        query = urlencode(params)
        signature = hmac.new(Config.SECRET.encode(), query.encode(), hashlib.sha256).hexdigest()
        return f"{query}&signature={signature}"

    @staticmethod
    def _request(method: str, endpoint: str, params: dict = None, retry: int = 3) -> Any:
        params = params or {}
        params["timestamp"] = int(time.time() * 1000)
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
    def get_price() -> float:
        for _ in range(3):
            try:
                res = requests.get(
                    f"{Config.BASE_URL}/fapi/v1/ticker/price",
                    params={"symbol": Config.SYMBOL},
                    timeout=5
                ).json()
                return float(res.get("price", 0))
            except:
                time.sleep(1)
        return 0.0

    @staticmethod
    def get_klines(interval: str, limit: int = 100) -> List[List]:
        try:
            res = requests.get(
                f"{Config.BASE_URL}/fapi/v1/klines",
                params={"symbol": Config.SYMBOL, "interval": interval, "limit": limit},
                timeout=5
            ).json()
            return res if isinstance(res, list) else []
        except:
            return []

    @staticmethod
    def get_open_position_amt() -> float:
        res = BinanceAPI._request("GET", "/fapi/v2/positionRisk")
        if isinstance(res, list):
            for p in res:
                if p["symbol"] == Config.SYMBOL:
                    return float(p["positionAmt"])
        return 0.0

    @staticmethod
    def place_market_order(side: str, quantity: float) -> dict:
        if quantity <= 0:
            return {}
        return BinanceAPI._request("POST", "/fapi/v1/order", {
            "symbol": Config.SYMBOL,
            "side": side,
            "type": "MARKET",
            "quantity": round(quantity, 3)      # BTC contract precision is usually 3 decimals
        })

    @staticmethod
    def set_leverage(leverage: int) -> bool:
        res = BinanceAPI._request("POST", "/fapi/v1/leverage", {
            "symbol": Config.SYMBOL,
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


# ===== MARKET DATA =====
class MarketData:
    @staticmethod
    def fetch() -> Optional[Dict]:
        klines_1m = BinanceAPI.get_klines("1m", 100)
        klines_5m = BinanceAPI.get_klines("5m", 100)
        klines_15m = BinanceAPI.get_klines("15m", 100)

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
            },
            "5m": {
                "price": closes_5m[-1],
                "ema20": Indicators.ema(closes_5m, 20),
                "rsi": Indicators.rsi(closes_5m, 14),
                "atr": Indicators.atr(klines_5m, 14),
            },
            "15m": {
                "price": closes_15m[-1],
                "ema20": Indicators.ema(closes_15m, 20),
                "rsi": Indicators.rsi(closes_15m, 14),
                "atr": Indicators.atr(klines_15m, 14),
            },
            "klines_15m_raw": klines_15m,  # Provide richer data to the reasoner
        }
        return data


# ===== FILTERS =====
class EntryFilters:
    @staticmethod
    def momentum_confirmed(closes_1m: List[float], bars: int = 3) -> bool:
        """Consecutive bars moving in the same direction"""
        if len(closes_1m) < bars + 1:
            return False
        recent = closes_1m[-bars-1:]
        if all(recent[i] < recent[i+1] for i in range(bars)):
            return True   # Consecutive up
        if all(recent[i] > recent[i+1] for i in range(bars)):
            return True   # Consecutive down
        return False

    @staticmethod
    def volume_surge(klines_1m: List[List], multiplier: float = 1.2) -> bool:
        """Current volume is higher than average of last 20 periods by multiplier"""
        if len(klines_1m) < 21:
            return False
        volumes = [float(k[5]) for k in klines_1m[-21:]]
        current_vol = volumes[-1]
        avg_vol = np.mean(volumes[:-1])
        return current_vol > avg_vol * multiplier


# ===== DEEPSEEK AI =====
deepseek_client = OpenAI(api_key=Config.DEEPSEEK_KEY, base_url="https://api.deepseek.com/v1")

class DeepSeekGatekeeper:
    """Uses deepseek-chat to decide whether to enter a trade"""
    @staticmethod
    def should_enter(data_1m: dict, data_5m: dict, data_15m: dict) -> tuple[bool, float]:
        Logger.info("🤖 Gatekeeper (chat) evaluating market...")
        prompt = f"""You are a strict trading gatekeeper. Analyze the following market data for BTCUSDT perpetual futures.

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
            # Clean possible markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            result = json.loads(content.strip())
            decision = result.get("decision", "HOLD")
            confidence = float(result.get("confidence", 0))
            reason = result.get("reason", "")
            Logger.info(f"Gatekeeper: {decision} (conf:{confidence}%) in {elapsed:.0f}ms - {reason}")
            return decision == "PROCEED", confidence
        except Exception as e:
            Logger.error(f"Gatekeeper error: {e}")
            return False, 0.0


class DeepSeekStrategist:
    """Uses deepseek-reasoner to determine specific trade parameters"""
    @staticmethod
    def plan_trade(data_1m: dict, data_5m: dict, data_15m: dict, klines_15m: List) -> Optional[dict]:
        Logger.info("🧠 Strategist (reasoner) planning trade...")
        # Prepare recent price summary
        closes_15m = [float(k[4]) for k in klines_15m[-20:]]
        price_summary = f"Last 20 closes (15m): {', '.join([f'${c:.0f}' for c in closes_15m])}"
        atr_15m = data_15m['atr']

        prompt = f"""You are a professional crypto futures strategist. Based on the provided data, propose a single trade setup.

Market Data:
- 1m: Price ${data_1m['price']:.2f}, EMA20 ${data_1m['ema20']:.2f}, RSI {data_1m['rsi']:.2f}, ATR {data_1m['atr']:.2f}
- 5m: Price ${data_5m['price']:.2f}, EMA20 ${data_5m['ema20']:.2f}, RSI {data_5m['rsi']:.2f}, ATR {data_5m['atr']:.2f}
- 15m: Price ${data_15m['price']:.2f}, EMA20 ${data_15m['ema20']:.2f}, RSI {data_15m['rsi']:.2f}, ATR {data_15m['atr']:.2f}

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
            Logger.success(f"Strategist plan: {plan['direction']} ({plan['confidence']}%) in {elapsed:.0f}ms")
            return plan
        except Exception as e:
            Logger.error(f"Strategist error: {e}")
            return None


# ===== TRADING BOT =====
class TradingBot:
    def __init__(self):
        self.positions: List[Position] = []
        self.balance = 0.0
        self.daily_start_balance = 0.0
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.consecutive_losses = 0
        self.last_trade_time = 0.0
        self.last_loss_time = 0.0
        self.trading_allowed = True
        self.last_reset_date = datetime.now().date()
        # Set leverage
        if Config.LEVERAGE > 1:
            BinanceAPI.set_leverage(Config.LEVERAGE)

    def update_balance(self):
        self.balance = BinanceAPI.get_balance()
        if self.daily_start_balance == 0:
            self.daily_start_balance = self.balance
        self.daily_pnl = self.balance - self.daily_start_balance

    def daily_reset(self):
        today = datetime.now().date()
        if today != self.last_reset_date:
            Logger.success(f"📅 New trading day. Yesterday's PnL: ${self.daily_pnl:.2f}")
            self.daily_start_balance = self.balance
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.consecutive_losses = 0
            self.trading_allowed = True
            self.last_reset_date = today

    def check_risk_limits(self) -> bool:
        # Check daily loss limit
        if self.daily_start_balance > 0:
            daily_loss_percent = -self.daily_pnl / self.daily_start_balance * 100
            if daily_loss_percent >= Config.DAILY_LOSS_LIMIT_PERCENT:
                Logger.error(f"❌ Daily loss limit reached: {daily_loss_percent:.2f}%")
                self.trading_allowed = False
                return False
        # Check consecutive losses
        if self.consecutive_losses >= Config.MAX_CONSECUTIVE_LOSSES:
            Logger.error(f"❌ Max consecutive losses ({Config.MAX_CONSECUTIVE_LOSSES}) reached.")
            self.trading_allowed = False
            return False
        return True

    def check_daily_target(self) -> bool:
        if self.daily_start_balance <= 0:
            return False
        profit_percent = self.daily_pnl / self.daily_start_balance * 100
        if profit_percent >= Config.DAILY_PROFIT_TARGET:
            Logger.success(f"🎯 Daily profit target reached: {profit_percent:.2f}%")
            self.trading_allowed = False
            return True
        return False

    def calculate_position_size(self, price: float, percent: float) -> float:
        """Calculate contract quantity (considering leverage)"""
        capital_used = self.balance * (percent / 100)
        position_notional = capital_used * Config.LEVERAGE
        qty = position_notional / price
        # Minimum quantity limit (Binance BTC contract min is 0.001)
        return max(0.001, round(qty, 3))

    def open_position(self, plan: dict, current_price: float, atr: float) -> bool:
        direction = plan["direction"]
        pos_percent = plan["position_percent"]
        sl_mult = plan["sl_atr_mult"]
        tp_mult = plan["tp_atr_mult"]

        qty = self.calculate_position_size(current_price, pos_percent)
        if qty <= 0:
            return False

        if direction == "BUY":
            sl_price = current_price - atr * sl_mult
            tp_price = current_price + atr * tp_mult
        else:
            sl_price = current_price + atr * sl_mult
            tp_price = current_price - atr * tp_mult

        Logger.info(f"🚀 Opening {direction} {qty:.3f} BTC @ ${current_price:.2f}")
        Logger.info(f"   SL: ${sl_price:.2f} | TP: ${tp_price:.2f} (ATR={atr:.2f})")

        resp = BinanceAPI.place_market_order(direction, qty)
        if resp.get("orderId"):
            self.positions.append(Position(
                side=direction,
                quantity=qty,
                entry_price=current_price,
                sl_price=sl_price,
                tp_price=tp_price,
                open_time=time.time()
            ))
            self.daily_trades += 1
            self.last_trade_time = time.time()
            Logger.success("✅ Position opened")
            return True
        else:
            Logger.error("Order failed")
            return False

    def close_position(self, index: int, current_price: float, reason: str):
        if index >= len(self.positions):
            return
        pos = self.positions[index]
        close_side = "SELL" if pos.side == "BUY" else "BUY"
        resp = BinanceAPI.place_market_order(close_side, pos.quantity)

        if resp.get("orderId"):
            if pos.side == "BUY":
                pnl = (current_price - pos.entry_price) * pos.quantity
            else:
                pnl = (pos.entry_price - current_price) * pos.quantity
            pnl_pct = pnl / (pos.entry_price * pos.quantity) * 100

            if pnl < 0:
                self.consecutive_losses += 1
                self.last_loss_time = time.time()
                Logger.warn(f"⚠️ Loss incurred, consecutive losses: {self.consecutive_losses}")
            else:
                self.consecutive_losses = 0

            self.positions.pop(index)
            Logger.success(f"💰 Closed: {reason} | PnL: ${pnl:.2f} ({pnl_pct:.2f}%)")
        else:
            Logger.error("Failed to close position")

    def check_exits(self, current_price: float):
        """Check stop loss, take profit, and trailing stop"""
        for i, pos in enumerate(self.positions):
            # Fixed SL/TP
            if pos.side == "BUY":
                if current_price <= pos.sl_price:
                    self.close_position(i, current_price, "Stop Loss")
                    return
                if current_price >= pos.tp_price:
                    self.close_position(i, current_price, "Take Profit")
                    return
                # Trailing stop: raise stop loss when price rises
                if current_price > pos.entry_price:
                    new_sl = current_price - Config.TRAILING_ATR_MULT * self.last_atr
                    if new_sl > pos.sl_price:
                        pos.sl_price = new_sl
                        pos.trailing_active = True
            else:  # SELL
                if current_price >= pos.sl_price:
                    self.close_position(i, current_price, "Stop Loss")
                    return
                if current_price <= pos.tp_price:
                    self.close_position(i, current_price, "Take Profit")
                    return
                if current_price < pos.entry_price:
                    new_sl = current_price + Config.TRAILING_ATR_MULT * self.last_atr
                    if new_sl < pos.sl_price:
                        pos.sl_price = new_sl
                        pos.trailing_active = True

    def run(self):
        Logger.success("=" * 50)
        Logger.success("🚀 Enhanced BTC Trading Bot (Two-step AI decision)")
        Logger.success("=" * 50)
        Logger.info(f"Gatekeeper: {Config.CHAT_MODEL}")
        Logger.info(f"Strategist: {Config.REASONER_MODEL}")
        Logger.info(f"Leverage: {Config.LEVERAGE}x | Cooldown: {Config.ENTRY_COOLDOWN}s")
        Logger.info("=" * 50)

        self.update_balance()
        self.daily_start_balance = self.balance

        # Check for existing open position
        existing_amt = BinanceAPI.get_open_position_amt()
        if abs(existing_amt) > 0:
            price = BinanceAPI.get_price()
            self.positions.append(Position(
                side="BUY" if existing_amt > 0 else "SELL",
                quantity=abs(existing_amt),
                entry_price=price,
                sl_price=price * 0.99,
                tp_price=price * 1.02,
                open_time=time.time()
            ))
            Logger.warn("⚠️ Detected existing position, loaded")

        self.last_atr = 0.0

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
                market_data = MarketData.fetch()
                if not market_data:
                    time.sleep(Config.CHECK_INTERVAL)
                    continue

                current_price = market_data["1m"]["price"]
                self.last_atr = market_data["15m"]["atr"]
                closes_1m = [float(k[4]) for k in BinanceAPI.get_klines("1m", 20)]

                # ---- Exit checks (always active regardless of cooldown) ----
                self.check_exits(current_price)

                # ---- Entry logic ----
                now = time.time()
                cooldown_active = (now - self.last_trade_time < Config.ENTRY_COOLDOWN) or \
                                  (now - self.last_loss_time < Config.LOSS_COOLDOWN)
                if cooldown_active:
                    time.sleep(Config.CHECK_INTERVAL)
                    continue

                if len(self.positions) >= Config.MAX_POSITIONS:
                    time.sleep(Config.CHECK_INTERVAL)
                    continue

                # 1. Pre-filter with technicals
                if not EntryFilters.momentum_confirmed(closes_1m):
                    Logger.info("⏸️ Momentum not confirmed, skipping")
                    time.sleep(Config.CHECK_INTERVAL)
                    continue
                if not EntryFilters.volume_surge(BinanceAPI.get_klines("1m", 30)):
                    Logger.info("⏸️ Volume insufficient, skipping")
                    time.sleep(Config.CHECK_INTERVAL)
                    continue

                # 2. Gatekeeper (chat) decision
                should_proceed, conf = DeepSeekGatekeeper.should_enter(
                    market_data["1m"], market_data["5m"], market_data["15m"]
                )
                if not should_proceed or conf < Config.CHAT_CONFIDENCE_THRESHOLD:
                    time.sleep(Config.CHECK_INTERVAL)
                    continue

                # 3. Strategist (reasoner) creates plan
                plan = DeepSeekStrategist.plan_trade(
                    market_data["1m"], market_data["5m"], market_data["15m"],
                    market_data["klines_15m_raw"]
                )
                if not plan:
                    time.sleep(Config.CHECK_INTERVAL)
                    continue

                # 4. Execute trade
                self.open_position(plan, current_price, market_data["15m"]["atr"])

                # Print status
                daily_ret = (self.balance - self.daily_start_balance) / self.daily_start_balance * 100
                Logger.info(f"📊 Balance: ${self.balance:.2f} | Daily Return: {daily_ret:.2f}% | Positions: {len(self.positions)}")

                time.sleep(Config.CHECK_INTERVAL)

            except KeyboardInterrupt:
                Logger.warn("⏹️ Bot stopped by user")
                break
            except Exception as e:
                Logger.error(f"Main loop exception: {e}")
                time.sleep(10)


if __name__ == "__main__":
    bot = TradingBot()
    bot.run()