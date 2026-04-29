import os
from datetime import datetime

import requests


class TelegramNotifier:
    def __init__(self):
        self.token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.enabled = bool(self.token and self.chat_id)

    def send(self, message: str) -> bool:
        if not self.enabled:
            return False
        try:
            resp = requests.post(
                f"https://api.telegram.org/bot{self.token}/sendMessage",
                data={
                    "chat_id": self.chat_id,
                    "text": message,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": "true",
                },
                timeout=10,
            )
            return resp.status_code == 200 and resp.json().get("ok", False)
        except Exception:
            return False


notifier = TelegramNotifier()


def notify(message: str) -> bool:
    return notifier.send(message)


def notify_bot_started(name: str, mode: str, extra: str = ""):
    text = (
        f"🤖 <b>{name} started</b>\n"
        f"Mode: <code>{mode}</code>\n"
        f"Time: <code>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</code>"
    )
    if extra:
        text += f"\n{extra}"
    return notify(text)


def notify_order_opened(market: str, symbol: str, side: str, qty, entry, sl, tp, tier="", risk_pct="", dry_run=True):
    action = "BUY/LONG" if side == "BUY" else "SELL/SHORT"
    mode = "DRY_RUN" if dry_run else "LIVE/DEMO"
    return notify(
        f"🚀 <b>{market} ORDER OPENED</b>\n"
        f"Mode: <code>{mode}</code>\n"
        f"Symbol: <code>{symbol}</code>\n"
        f"Side: <b>{action}</b>\n"
        f"Qty/Lot: <code>{qty}</code>\n"
        f"Entry: <code>{_fmt(entry)}</code>\n"
        f"SL: <code>{_fmt(sl)}</code>\n"
        f"TP: <code>{_fmt(tp)}</code>\n"
        f"Tier: <code>{tier}</code>\n"
        f"Risk: <code>{_fmt_pct(risk_pct)}</code>\n"
        f"Time: <code>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</code>"
    )


def notify_order_result(market: str, symbol: str, side: str, result, dry_run=True):
    mode = "DRY_RUN" if dry_run else "LIVE/DEMO"
    return notify(
        f"📌 <b>{market} ORDER RESULT</b>\n"
        f"Mode: <code>{mode}</code>\n"
        f"Symbol: <code>{symbol}</code>\n"
        f"Side: <code>{side}</code>\n"
        f"Result: <code>{_short(result)}</code>"
    )


def notify_order_closed(market: str, symbol: str, side: str, exit_price, pnl="", reason="", dry_run=True):
    mode = "DRY_RUN" if dry_run else "LIVE/DEMO"
    result_icon = "✅" if _is_positive(pnl) else "🛑" if str(pnl) else "📤"
    return notify(
        f"{result_icon} <b>{market} ORDER CLOSED</b>\n"
        f"Mode: <code>{mode}</code>\n"
        f"Symbol: <code>{symbol}</code>\n"
        f"Side: <code>{side}</code>\n"
        f"Exit: <code>{_fmt(exit_price)}</code>\n"
        f"PnL: <code>{_fmt(pnl)}</code>\n"
        f"Reason: <code>{_short(reason, 200)}</code>\n"
        f"Time: <code>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</code>"
    )


def notify_error(market: str, message: str):
    return notify(
        f"⚠️ <b>{market} ERROR</b>\n"
        f"<code>{_short(message, 900)}</code>\n"
        f"Time: <code>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</code>"
    )


def notify_reconnected(market: str, message: str = "Connection restored"):
    return notify(
        f"✅ <b>{market} RECONNECTED</b>\n"
        f"<code>{_short(message, 500)}</code>\n"
        f"Time: <code>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</code>"
    )


def _fmt(value):
    try:
        return f"{float(value):.6f}".rstrip("0").rstrip(".")
    except Exception:
        return str(value)


def _fmt_pct(value):
    try:
        value = float(value)
        if value < 1:
            value *= 100
        return f"{value:.2f}%"
    except Exception:
        return str(value)


def _short(value, limit=700):
    text = str(value)
    return text if len(text) <= limit else text[:limit] + "..."


def _is_positive(value):
    try:
        return float(value) > 0
    except Exception:
        return False
