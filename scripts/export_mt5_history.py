# export_mt5_history.py

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# ===== CONFIG =====
DAYS_BACK = 30
OUTPUT_FILE = Path(__file__).resolve().parent.parent / "data" / "mt5_trade_history.csv"

# ===== CONNECT MT5 =====
if not mt5.initialize():
    print("❌ MT5 initialize failed:", mt5.last_error())
    quit()

print("✅ Connected to MT5")

# ===== ACCOUNT INFO (diagnostic) =====
acc = mt5.account_info()
if acc:
    print(f"   Account : {acc.login}  Server: {acc.server}")
    print(f"   Balance : {acc.balance:.2f}  Equity: {acc.equity:.2f}")
    print(f"   Currency: {acc.currency}")
else:
    print("⚠️  account_info() returned None — not logged in?", mt5.last_error())

# ===== DATE RANGE =====
to_date = datetime.now()
from_date = to_date - timedelta(days=DAYS_BACK)
print(f"   Period  : {from_date.date()} → {to_date.date()} ({DAYS_BACK} days)")

# ===== GET DEAL HISTORY =====
deals = mt5.history_deals_get(from_date, to_date)
print(f"   Raw deals returned: {len(deals) if deals else 0}  last_error={mt5.last_error()}")

if deals is None or len(deals) == 0:
    print("ℹ️  No trade history in this period.")
    print("   Possible reasons:")
    print("   1. Bot has not placed any orders yet (most likely)")
    print("   2. MT5 terminal needs to load history: right-click chart → History → Load")
    print("   3. Account is new / demo with no trades")
    mt5.shutdown()
    quit()

# ===== CONVERT TO DATAFRAME =====
df = pd.DataFrame([d._asdict() for d in deals])

# แปลงเวลา
df["time"] = pd.to_datetime(df["time"], unit="s")

# เอาเฉพาะรายการที่มี symbol
df = df[df["symbol"] != ""]

# เพิ่มประเภท deal
entry_map = {
    0: "IN",
    1: "OUT",
    2: "INOUT",
    3: "OUT_BY"
}
df["entry_type"] = df["entry"].map(entry_map)

type_map = {
    0: "BUY",
    1: "SELL",
    2: "BALANCE",
    3: "CREDIT",
    4: "CHARGE",
    5: "CORRECTION",
    6: "BONUS",
    7: "COMMISSION",
    8: "COMMISSION_DAILY",
    9: "COMMISSION_MONTHLY",
    10: "AGENT_DAILY",
    11: "AGENT_MONTHLY",
    12: "INTEREST",
    13: "BUY_CANCELED",
    14: "SELL_CANCELED"
}
df["deal_type"] = df["type"].map(type_map)

# จัด column ให้อ่านง่าย
cols = [
    "time",
    "ticket",
    "order",
    "position_id",
    "symbol",
    "deal_type",
    "entry_type",
    "volume",
    "price",
    "profit",
    "commission",
    "swap",
    "fee",
    "comment",
    "magic"
]

df = df[[c for c in cols if c in df.columns]]

# ===== SAVE CSV =====
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print(f"✅ Export success: {OUTPUT_FILE}")
print(f"Rows: {len(df)}")
print(df.tail(10))

mt5.shutdown()
