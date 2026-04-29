import argparse
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

SPOT_BASE_URL = "https://api.binance.com"
FUTURES_BASE_URL = "https://fapi.binance.com"

DEFAULT_SYMBOLS = "BTCUSDT"
DEFAULT_INTERVALS = "1m,5m,15m,1h"

SYMBOL_PREFIX = {
    "BTCUSDT": "bitcoin",
    "ETHUSDT": "ethereum",
    "SOLUSDT": "solana",
}


def parse_csv_arg(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def year_bounds_ms(year: int) -> tuple[int, int]:
    start = datetime(year, 1, 1, tzinfo=timezone.utc)
    end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    return int(start.timestamp() * 1000), int(end.timestamp() * 1000) - 1


def symbol_prefix(symbol: str) -> str:
    symbol = symbol.upper()
    return SYMBOL_PREFIX.get(symbol, symbol.replace("USDT", "").lower())


def base_url(market: str) -> str:
    if market == "spot":
        return SPOT_BASE_URL
    if market == "futures":
        return FUTURES_BASE_URL
    raise ValueError(f"Unsupported market: {market}")


def klines_endpoint(market: str) -> str:
    return "/api/v3/klines" if market == "spot" else "/fapi/v1/klines"


def get_klines(
    session: requests.Session,
    market: str,
    symbol: str,
    interval: str,
    start_time: int,
    end_time: int,
    limit: int = 1000,
) -> list:
    url = f"{base_url(market)}{klines_endpoint(market)}"
    params = {
        "symbol": symbol.upper(),
        "interval": interval,
        "startTime": start_time,
        "endTime": end_time,
        "limit": limit,
    }
    response = session.get(url, params=params, timeout=20)
    response.raise_for_status()
    data = response.json()
    if isinstance(data, dict):
        raise RuntimeError(f"Binance error for {symbol} {interval}: {data}")
    return data


def normalize_klines(rows: list) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
    df = pd.DataFrame(
        rows,
        columns=[
            "time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "trades",
            "taker_buy_base_volume",
            "taker_buy_quote_volume",
            "ignore",
        ],
    )
    df = df[["time", "open", "high", "low", "close", "volume"]]
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True).dt.tz_convert(None)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna().drop_duplicates("time").sort_values("time")


def download_range(
    session: requests.Session,
    market: str,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    sleep_seconds: float,
) -> pd.DataFrame:
    all_rows = []
    cursor = start_ms
    request_count = 0

    while cursor <= end_ms:
        rows = get_klines(session, market, symbol, interval, cursor, end_ms)
        request_count += 1
        if not rows:
            break

        all_rows.extend(rows)
        last_open_time = int(rows[-1][0])
        next_cursor = last_open_time + 1
        if next_cursor <= cursor:
            break
        cursor = next_cursor

        print(
            f"  {symbol} {interval}: {len(all_rows):>7} candles "
            f"last={pd.to_datetime(last_open_time, unit='ms')}",
            flush=True,
        )

        if len(rows) < 1000:
            break
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    df = normalize_klines(all_rows)
    print(f"  done {symbol} {interval}: {len(df)} candles, requests={request_count}", flush=True)
    return df


def output_path(symbol: str, year: int, interval: str) -> Path:
    return DATA_DIR / f"{symbol_prefix(symbol)}_{year}_{interval}.csv"


def merged_output_path(symbol: str, years: Iterable[int], interval: str) -> Path:
    years = sorted(years)
    if len(years) == 1:
        suffix = str(years[0])
    else:
        suffix = f"{years[0]}_{years[-1]}"
    return DATA_DIR / f"{symbol_prefix(symbol)}_{suffix}_{interval}.csv"


def write_year_csv(path: Path, df: pd.DataFrame, force: bool):
    if path.exists() and not force:
        print(f"  skip existing: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  saved: {path}")


def write_merged_csv(symbol: str, years: list[int], interval: str, force: bool):
    frames = []
    for year in years:
        path = output_path(symbol, year, interval)
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        return
    merged = pd.concat(frames, ignore_index=True)
    merged["time"] = pd.to_datetime(merged["time"])
    merged = merged.drop_duplicates("time").sort_values("time")
    path = merged_output_path(symbol, years, interval)
    if path.exists() and not force:
        print(f"  skip existing merged: {path}")
        return
    merged.to_csv(path, index=False)
    print(f"  saved merged: {path} ({len(merged)} candles)")


def main():
    parser = argparse.ArgumentParser(
        description="Download Binance OHLCV history for selected years and timeframes."
    )
    parser.add_argument("--symbols", default=DEFAULT_SYMBOLS, help="Comma-separated symbols, e.g. BTCUSDT,ETHUSDT,SOLUSDT")
    parser.add_argument("--years", required=True, help="Comma-separated years, e.g. 2022,2023,2024,2025")
    parser.add_argument("--intervals", default=DEFAULT_INTERVALS, help="Comma-separated intervals, default: 1m,5m,15m,1h")
    parser.add_argument("--market", choices=["spot", "futures"], default="futures")
    parser.add_argument("--sleep", type=float, default=0.15, help="Sleep seconds between Binance requests")
    parser.add_argument("--force", action="store_true", help="Overwrite existing yearly and merged files")
    parser.add_argument("--no-merge", action="store_true", help="Do not create merged multi-year CSV files")
    args = parser.parse_args()

    symbols = [symbol.upper() for symbol in parse_csv_arg(args.symbols)]
    intervals = parse_csv_arg(args.intervals)
    years = sorted({int(year) for year in parse_csv_arg(args.years)})

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    print("=== Binance History Downloader ===")
    print(f"market={args.market}")
    print(f"symbols={','.join(symbols)}")
    print(f"years={','.join(str(year) for year in years)}")
    print(f"intervals={','.join(intervals)}")
    print(f"output={DATA_DIR}")

    for symbol in symbols:
        for interval in intervals:
            for year in years:
                path = output_path(symbol, year, interval)
                if path.exists() and not args.force:
                    print(f"skip existing: {path}")
                    continue

                print(f"\nDownloading {symbol} {interval} {year}...")
                start_ms, end_ms = year_bounds_ms(year)
                df = download_range(
                    session=session,
                    market=args.market,
                    symbol=symbol,
                    interval=interval,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    sleep_seconds=args.sleep,
                )
                write_year_csv(path, df, force=True)

            if not args.no_merge:
                write_merged_csv(symbol, years, interval, force=args.force)

    print("\nSaved csv files successfully.")


if __name__ == "__main__":
    main()
