"""
Download Binance Futures M15 + M1 OHLCV data for new symbols.
Saves to data/{name}_2022_2025_{tf}.csv  (same format as existing files)

Usage:
  python3 download_symbol_data.py                      # download all NEW_SYMBOLS
  python3 download_symbol_data.py AVAXUSDT LINKUSDT   # specific symbols
"""

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

BASE_URL   = "https://fapi.binance.com"
START_DATE = "2022-01-01"
END_DATE   = "2025-12-31 23:59:59"
TIMEFRAMES = [("15m", "15m"), ("1m", "1m")]

NEW_SYMBOLS = {
    "AVAXUSDT": "avalanche",
    "BNBUSDT":  "bnb",
    "LINKUSDT": "chainlink",
    "DOTUSDT":  "polkadot",
}

SYMBOL_NAME = {          # fallback: use symbol prefix as name
}


def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> list:
    all_data = []
    cur = start_ms
    while cur < end_ms:
        resp = requests.get(
            f"{BASE_URL}/fapi/v1/klines",
            params={"symbol": symbol, "interval": interval,
                    "startTime": cur, "endTime": end_ms, "limit": 1000},
            timeout=20,
        )
        resp.raise_for_status()
        chunk = resp.json()
        if not chunk:
            break
        all_data.extend(chunk)
        cur = int(chunk[-1][6]) + 1   # close_time of last bar + 1ms
        if len(chunk) < 1000:
            break
        time.sleep(0.05)              # ~20 req/s, well within 1200/min limit
    return all_data


def to_df(raw: list) -> pd.DataFrame:
    cols = ["open_time","open","high","low","close","volume",
            "close_time","qav","ntrades","tbbase","tbquote","ignore"]
    df = pd.DataFrame(raw, columns=cols)
    df["time"] = pd.to_datetime(df["open_time"].astype("int64"), unit="ms")
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["time","open","high","low","close","volume"]].dropna()


def download(symbol: str, name: str):
    start_ms = int(datetime.strptime(START_DATE, "%Y-%m-%d")
                   .replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ms   = int(datetime.strptime(END_DATE, "%Y-%m-%d %H:%M:%S")
                   .replace(tzinfo=timezone.utc).timestamp() * 1000)

    for tf_label, tf_api in TIMEFRAMES:
        out = DATA_DIR / f"{name}_2022_2025_{tf_label}.csv"
        if out.exists():
            print(f"  [{symbol} {tf_label}] already exists → skip")
            continue

        print(f"  [{symbol} {tf_label}] downloading …", end=" ", flush=True)
        t0 = time.time()
        raw = fetch_klines(symbol, tf_api, start_ms, end_ms)
        df  = to_df(raw)
        df.to_csv(out, index=False)
        elapsed = time.time() - t0
        print(f"{len(df):,} bars → {out.name}  ({elapsed:.1f}s)")


def main():
    symbols_to_dl = {}
    if len(sys.argv) > 1:
        for sym in sys.argv[1:]:
            sym = sym.upper()
            name = NEW_SYMBOLS.get(sym, sym.lower().replace("usdt", ""))
            symbols_to_dl[sym] = name
    else:
        symbols_to_dl = NEW_SYMBOLS

    print(f"Downloading {len(symbols_to_dl)} symbol(s): {list(symbols_to_dl)}")
    print(f"Period: {START_DATE} → {END_DATE}")
    print(f"Output: {DATA_DIR}")
    print("-" * 60)

    for symbol, name in symbols_to_dl.items():
        print(f"\n[{symbol}]")
        try:
            download(symbol, name)
        except Exception as e:
            print(f"  ERROR: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
