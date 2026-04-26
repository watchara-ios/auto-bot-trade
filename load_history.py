import requests
import pandas as pd
import time

BASE_URL = "https://api.binance.com"

def get_klines(symbol="BTCUSDT", interval="1m", start_time=None, limit=1000):
    url = BASE_URL + "/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    if start_time:
        params["startTime"] = start_time

    response = requests.get(url, params=params)
    return response.json()

def download_all(symbol="BTCUSDT", interval="1h", days=30):
    all_data = []

    # ย้อนหลัง X วัน
    end_time = int(time.time() * 1000)
    start_time = end_time - (days * 24 * 60 * 60 * 1000)

    while True:
        data = get_klines(symbol, interval, start_time)

        if not data:
            break

        all_data.extend(data)
        start_time = data[-1][0] + 1  # candle ถัดไป

        print(f"โหลดแล้ว: {len(all_data)} candles")

        if len(data) < 1000:
            break

        time.sleep(0.2)  # กันโดน rate limit

    # แปลงเป็น DataFrame
    df = pd.DataFrame(all_data, columns=[
        "time","open","high","low","close","volume",
        "close_time","qav","trades","tbbav","tbqav","ignore"
    ])

    df = df[["time","open","high","low","close","volume"]]

    df["time"] = pd.to_datetime(df["time"], unit="ms")
    df = df.astype({
        "open": float,
        "high": float,
        "low": float,
        "close": float,
        "volume": float
    })

    return df


# ===== RUN =====
df = download_all(days=7)  # เปลี่ยนเป็น 365 ได้
df.to_csv("bitcoin_7d_1h.csv", index=False)

print("✅ Saved csv")