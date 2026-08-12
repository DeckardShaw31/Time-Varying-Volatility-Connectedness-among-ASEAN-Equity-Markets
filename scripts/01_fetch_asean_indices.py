"""
Stage 1: Fetch ASEAN equity-market index data.

Downloads daily OHLCV data for 6 ASEAN market indices.
  - 5 markets via yfinance
  - Vietnam (VN-Index) via vnstock (yfinance does not cover it)

Output schema: date, country, index_name, ticker, open, high, low, close,
               adjusted_close, volume, currency, source

No merging or forward-filling of missing trading days.
"""

import os
import sys

# Fix Windows encoding BEFORE any imports that may print Unicode
os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import yfinance as yf

import config
from scripts.utils import setup_logger

logger = setup_logger("01_fetch_asean")


def fetch_single_index(country: str, info: dict) -> pd.DataFrame:
    """Download OHLCV for one ASEAN index via yfinance."""
    ticker = info["ticker"]
    logger.info(f"  Downloading {country} ({ticker}) ...")

    tk = yf.Ticker(ticker)
    df = tk.history(start=config.START_DATE, end=config.END_DATE,
                    auto_adjust=False, actions=False)

    if df.empty:
        logger.warning(f"  [!] No data returned for {country} ({ticker})")
        return pd.DataFrame()

    # Standardize column names
    df = df.reset_index()
    col_map = {}
    for c in df.columns:
        cl = c.lower().strip()
        if cl in ("date", "datetime"):
            col_map[c] = "date"
        elif cl == "open":
            col_map[c] = "open"
        elif cl == "high":
            col_map[c] = "high"
        elif cl == "low":
            col_map[c] = "low"
        elif cl == "close":
            col_map[c] = "close"
        elif cl in ("adj close", "adj_close", "adjusted close", "adjclose"):
            col_map[c] = "adjusted_close"
        elif cl == "volume":
            col_map[c] = "volume"
    df = df.rename(columns=col_map)

    # Ensure we have a date column
    if "date" not in df.columns:
        logger.warning(f"  [!] No 'date' column found for {country}")
        return pd.DataFrame()

    # Convert date to date-only (remove timezone/time)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()

    # Add metadata columns
    df["country"]    = country
    df["index_name"] = info["index_name"]
    df["ticker"]     = ticker
    df["currency"]   = info["currency"]
    df["source"]     = "yfinance"

    # If adjusted_close is missing, use close
    if "adjusted_close" not in df.columns:
        df["adjusted_close"] = df["close"]

    # Select and order output columns
    out_cols = ["date", "country", "index_name", "ticker",
                "open", "high", "low", "close", "adjusted_close",
                "volume", "currency", "source"]
    for c in out_cols:
        if c not in df.columns:
            df[c] = None
    df = df[out_cols]

    logger.info(f"  OK {country}: {len(df)} observations, "
                f"{df['date'].min().date()} to {df['date'].max().date()}")
    return df


def fetch_vietnam_index() -> pd.DataFrame:
    """
    Download VN-Index data via vnstock (yfinance does not cover it).
    """
    logger.info("  Downloading Vietnam (VN-Index) via vnstock ...")

    try:
        from vnstock import Vnstock
        stock = Vnstock()
        df = stock.stock(symbol="VNINDEX", source="VCI").quote.history(
            start=config.START_DATE, end=config.END_DATE, interval="1D"
        )

        if df is None or df.empty:
            logger.warning("  [!] vnstock returned empty data for VNINDEX")
            return pd.DataFrame()

        # Standardize columns
        df = df.reset_index(drop=True) if "time" not in df.columns else df
        
        # Find and rename columns
        col_map = {}
        for c in df.columns:
            cl = c.lower().strip()
            if cl in ("time", "date", "datetime", "tradingdate"):
                col_map[c] = "date"
            elif cl == "open":
                col_map[c] = "open"
            elif cl == "high":
                col_map[c] = "high"
            elif cl == "low":
                col_map[c] = "low"
            elif cl == "close":
                col_map[c] = "close"
            elif cl == "volume":
                col_map[c] = "volume"
        df = df.rename(columns=col_map)

        if "date" not in df.columns:
            # Try using the index
            if hasattr(df.index, 'name') and df.index.name and 'time' in df.index.name.lower():
                df = df.reset_index()
                df = df.rename(columns={df.columns[0]: "date"})
            else:
                logger.warning("  [!] Cannot identify date column in vnstock output")
                logger.info(f"  Columns: {list(df.columns)}")
                return pd.DataFrame()

        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
        df["country"] = "Vietnam"
        df["index_name"] = "VN-Index"
        df["ticker"] = "VNINDEX"
        df["currency"] = "VND"
        df["source"] = "vnstock"

        if "adjusted_close" not in df.columns:
            df["adjusted_close"] = df.get("close", None)

        out_cols = ["date", "country", "index_name", "ticker",
                    "open", "high", "low", "close", "adjusted_close",
                    "volume", "currency", "source"]
        for c in out_cols:
            if c not in df.columns:
                df[c] = None
        df = df[out_cols]

        logger.info(f"  OK Vietnam: {len(df)} observations, "
                    f"{df['date'].min().date()} to {df['date'].max().date()}")
        return df

    except ImportError:
        logger.warning("  vnstock not installed. Install with: pip install vnstock")
        return pd.DataFrame()
    except Exception as e:
        logger.warning(f"  vnstock fetch failed: {e}")
        return pd.DataFrame()


def main():
    logger.info("=" * 60)
    logger.info("Stage 1: Fetching ASEAN equity-market index data")
    logger.info("=" * 60)

    frames = []

    # Fetch 5 markets via yfinance (skip Vietnam)
    for country, info in config.ASEAN_MARKETS.items():
        if country == "Vietnam":
            continue
        df = fetch_single_index(country, info)
        if not df.empty:
            frames.append(df)

    # Fetch Vietnam via vnstock
    vn_df = fetch_vietnam_index()
    if not vn_df.empty:
        frames.append(vn_df)

    if not frames:
        logger.error("No data fetched for any market. Aborting.")
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["country", "date"]).reset_index(drop=True)

    # Save to data/raw and deliverables
    raw_path = config.DATA_RAW / "asean_indices_raw.csv"
    del_path = config.DELIVERABLES / "asean_indices_raw.csv"

    combined.to_csv(raw_path, index=False)
    combined.to_csv(del_path, index=False)

    logger.info("-" * 60)
    logger.info(f"Total observations: {len(combined)}")
    logger.info(f"Countries: {combined['country'].nunique()}")
    logger.info(f"Date range: {combined['date'].min().date()} to "
                f"{combined['date'].max().date()}")
    logger.info(f"Saved -> {raw_path}")
    logger.info(f"Saved -> {del_path}")

    # Summary per country
    summary = combined.groupby("country").agg(
        obs=("date", "count"),
        first_date=("date", "min"),
        last_date=("date", "max"),
    )
    logger.info(f"\nPer-country summary:\n{summary.to_string()}")

    return combined


if __name__ == "__main__":
    main()
