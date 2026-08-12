"""
Stage 3: Fetch daily exchange-rate data (LCU per USD).

Downloads daily local-currency-per-US-dollar rates for all 6 ASEAN currencies.
Output schema: date, country, local_currency, local_currency_per_usd
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import yfinance as yf

import config
from scripts.utils import setup_logger

logger = setup_logger("03_fetch_fx")


def fetch_fx_pair(country: str, info: dict) -> pd.DataFrame:
    """Download daily FX rate for one currency pair."""
    pair = info["pair"]
    currency = info["local_currency"]
    logger.info(f"  Downloading {country} ({pair}) ...")

    tk = yf.Ticker(pair)
    df = tk.history(start=config.START_DATE, end=config.END_DATE,
                    auto_adjust=True, actions=False)

    if df.empty:
        logger.warning(f"  [!] No data for {pair}. Trying alternative format ...")
        # Try alternative ticker formats
        alternatives = [
            f"{currency}=X",        # e.g. IDR=X
            f"{currency}USD=X",     # e.g. IDRUSD=X
        ]
        for alt in alternatives:
            tk2 = yf.Ticker(alt)
            df = tk2.history(start=config.START_DATE, end=config.END_DATE,
                             auto_adjust=True, actions=False)
            if not df.empty:
                logger.info(f"    [OK] Found data with {alt}")
                # This might be USD per LCU - need to invert
                # Check: if the values are < 1 for IDR, PHP, VND, it's inverted
                median_val = df["Close"].median()
                if currency in ("IDR", "VND") and median_val < 1:
                    df["Close"] = 1.0 / df["Close"]
                    logger.info(f"    Inverted {alt} to get LCU per USD")
                break

    if df.empty:
        logger.warning(f"  [!] No FX data for {country}")
        return pd.DataFrame()

    df = df.reset_index()
    # Normalize
    rename = {}
    for c in df.columns:
        cl = c.lower().strip()
        if cl in ("date", "datetime"):
            rename[c] = "date"
        elif cl == "close":
            rename[c] = "local_currency_per_usd"
    df = df.rename(columns=rename)

    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df["country"] = country
    df["local_currency"] = currency

    result = df[["date", "country", "local_currency", "local_currency_per_usd"]].copy()
    result = result.dropna(subset=["local_currency_per_usd"])

    logger.info(f"  [OK] {country} ({currency}): {len(result)} observations, "
                f"{result['date'].min().date()} -> {result['date'].max().date()}")
    return result


def main():
    logger.info("=" * 60)
    logger.info("Stage 3: Fetching exchange-rate data (LCU per USD)")
    logger.info("=" * 60)

    frames = []
    for country, info in config.FX_PAIRS.items():
        df = fetch_fx_pair(country, info)
        if not df.empty:
            frames.append(df)

    if not frames:
        logger.error("No FX data fetched. Aborting.")
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["country", "date"]).reset_index(drop=True)

    # Save
    raw_path = config.DATA_RAW / "exchange_rates_raw.csv"
    del_path = config.DELIVERABLES / "exchange_rates_raw.csv"
    combined.to_csv(raw_path, index=False)
    combined.to_csv(del_path, index=False)

    logger.info(f"\n{'-' * 60}")
    logger.info(f"Total observations: {len(combined)}")
    logger.info(f"Saved -> {raw_path}")
    logger.info(f"Saved -> {del_path}")

    return combined


if __name__ == "__main__":
    main()
