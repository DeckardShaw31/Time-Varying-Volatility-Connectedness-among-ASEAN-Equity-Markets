"""
Stage 4: Data cleaning and synchronization.

For every series:
  - Convert dates to common format
  - Sort chronologically
  - Remove duplicate dates
  - Convert price columns to numeric
  - Identify missing, zero, and negative prices
  - Check H_t >= L_t > 0
  - Log observations per market
  - Document first and last valid date
  - Retain raw national trading calendars
  - Create synchronized common-date datasets:
      1. Intersection of trading dates across all six markets
      2. Weekly aggregation (last available observation per week)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import config
from scripts.utils import setup_logger

logger = setup_logger("04_clean")


def clean_asean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw ASEAN indices data."""
    logger.info("Cleaning ASEAN index data ...")

    # 1. Convert dates and enforce sample boundaries
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= config.START_DATE) & (df["date"] <= config.END_DATE)].copy()
    df = df.sort_values(["country", "date"]).reset_index(drop=True)

    # 2. Remove duplicate dates within each country
    n_before = len(df)
    df = df.drop_duplicates(subset=["country", "date"], keep="last")
    n_dupes = n_before - len(df)
    if n_dupes > 0:
        logger.info(f"  Removed {n_dupes} duplicate date rows")

    # 3. Convert price columns to numeric
    price_cols = ["open", "high", "low", "close", "adjusted_close"]
    for col in price_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    # 4. Identify problems
    for country in config.COUNTRY_ORDER:
        sub = df[df["country"] == country]
        n_total = len(sub)

        for col in ["close", "high", "low"]:
            if col not in sub.columns:
                continue
            n_missing = sub[col].isna().sum()
            n_zero = (sub[col] == 0).sum()
            n_neg = (sub[col] < 0).sum()
            if n_missing > 0 or n_zero > 0 or n_neg > 0:
                logger.warning(f"  {country}.{col}: {n_missing} missing, "
                               f"{n_zero} zero, {n_neg} negative (of {n_total})")

    # 5. Check H >= L > 0
    if "high" in df.columns and "low" in df.columns:
        mask_valid = df["high"].notna() & df["low"].notna()
        violations = df[mask_valid & ((df["high"] < df["low"]) | (df["low"] <= 0))]
        if len(violations) > 0:
            logger.warning(f"  {len(violations)} rows violate H >= L > 0:")
            for _, row in violations.head(5).iterrows():
                logger.warning(f"    {row['country']} {row['date'].date()}: "
                               f"H={row['high']}, L={row['low']}")
            # Fix: set low = min(high, low), drop if both <= 0
            bad_hl = mask_valid & (df["high"] < df["low"])
            df.loc[bad_hl, ["high", "low"]] = df.loc[bad_hl, ["low", "high"]].values

            bad_zero = mask_valid & (df["low"] <= 0)
            df = df[~bad_zero]
            logger.info(f"  Fixed high/low swaps; removed {bad_zero.sum()} non-positive rows")

    # 6. Per-market summary
    logger.info("\n  Per-market summary after cleaning:")
    for country in config.COUNTRY_ORDER:
        sub = df[df["country"] == country]
        if len(sub) == 0:
            logger.warning(f"    {country}: NO DATA")
            continue
        logger.info(f"    {country}: {len(sub)} obs, "
                    f"{sub['date'].min().date()} -> {sub['date'].max().date()}")

    return df


def create_intersection_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create synchronized dataset using intersection of trading dates.
    Only dates when ALL 6 markets have data are retained.
    """
    logger.info("Creating intersection-synchronized dataset ...")

    # Pivot to wide format: date × country for close prices
    countries = df["country"].unique()
    date_sets = []
    for country in countries:
        dates = set(df[df["country"] == country]["date"].values)
        date_sets.append(dates)

    common_dates = date_sets[0]
    for ds in date_sets[1:]:
        common_dates = common_dates.intersection(ds)

    common_dates = sorted(common_dates)
    logger.info(f"  Common trading dates: {len(common_dates)} "
                f"(from {pd.Timestamp(common_dates[0]).date()} "
                f"to {pd.Timestamp(common_dates[-1]).date()})")

    # Filter
    synced = df[df["date"].isin(common_dates)].copy()
    synced = synced.sort_values(["date", "country"]).reset_index(drop=True)

    return synced


def create_weekly_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create weekly-aggregated dataset.
    Aggregates daily OHLCV within each ISO week (year_week):
      - High  = max(daily high)
      - Low   = min(daily low)
      - Open  = first(daily open)
      - Close = last(daily close)
      - Volume = sum(daily volume)
      - Date  = last(daily date)
    """
    logger.info("Creating weekly-aggregated dataset ...")

    df = df.copy()
    df["year_week"] = df["date"].dt.isocalendar().year.astype(str) + "-W" + \
                      df["date"].dt.isocalendar().week.astype(str).str.zfill(2)

    agg_dict = {
        "date": "last",
        "close": "last"
    }
    if "open" in df.columns:
        agg_dict["open"] = "first"
    if "high" in df.columns:
        agg_dict["high"] = "max"
    if "low" in df.columns:
        agg_dict["low"] = "min"
    if "adjusted_close" in df.columns:
        agg_dict["adjusted_close"] = "last"
    if "volume" in df.columns:
        agg_dict["volume"] = "sum"

    for c in ["index_name", "ticker", "currency", "source"]:
        if c in df.columns:
            agg_dict[c] = "first"

    # Aggregation by country and year_week
    weekly = df.sort_values("date").groupby(["country", "year_week"]).agg(agg_dict).reset_index()

    # Assign canonical Friday date for exact weekly panel alignment across markets
    yw_str = weekly["year_week"] + "-5"
    weekly["date"] = pd.to_datetime(yw_str, format="%G-W%V-%u")

    # Only keep weeks where all 6 countries have data
    week_counts = weekly.groupby("year_week")["country"].nunique()
    full_weeks = week_counts[week_counts == len(config.COUNTRY_ORDER)].index
    weekly = weekly[weekly["year_week"].isin(full_weeks)].copy()

    weekly = weekly.sort_values(["date", "country"]).reset_index(drop=True)
    logger.info(f"  Weekly observations: {len(weekly)} "
                f"({weekly['year_week'].nunique()} full weeks)")

    return weekly


def clean_global_daily(path: Path) -> pd.DataFrame:
    """Clean the global daily data file."""
    if not path.exists():
        logger.info("  Global daily file not found - skipping.")
        return pd.DataFrame()

    logger.info("Cleaning global daily data ...")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "date"
    df = df.sort_index()

    # Remove duplicate dates and filter date range
    df = df[~df.index.duplicated(keep="last")]
    df = df[(df.index >= config.START_DATE) & (df.index <= config.END_DATE)].copy()

    # Convert to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info(f"  Global daily: {len(df)} dates, columns = {list(df.columns)}")
    for col in df.columns:
        valid = df[col].notna().sum()
        logger.info(f"    {col}: {valid} valid observations")

    return df


def clean_exchange_rates(path: Path) -> pd.DataFrame:
    """Clean the exchange rate data."""
    if not path.exists():
        logger.info("  Exchange rates file not found - skipping.")
        return pd.DataFrame()

    logger.info("Cleaning exchange-rate data ...")
    df = pd.read_csv(path, parse_dates=["date"])
    df = df[(df["date"] >= config.START_DATE) & (df["date"] <= config.END_DATE)].copy()
    df = df.sort_values(["country", "date"]).reset_index(drop=True)
    df = df.drop_duplicates(subset=["country", "date"], keep="last")

    df["local_currency_per_usd"] = pd.to_numeric(
        df["local_currency_per_usd"], errors="coerce")

    # Check for non-positive rates
    bad = df["local_currency_per_usd"] <= 0
    if bad.any():
        logger.warning(f"  Removed {bad.sum()} non-positive FX rates")
        df = df[~bad]

    return df


def main():
    logger.info("=" * 60)
    logger.info("Stage 4: Data cleaning and synchronization")
    logger.info("=" * 60)

    # -- ASEAN indices --
    asean_path = config.DATA_RAW / "asean_indices_raw.csv"
    if not asean_path.exists():
        logger.error(f"ASEAN data not found at {asean_path}. Run Stage 1 first.")
        sys.exit(1)

    asean_raw = pd.read_csv(asean_path)
    asean_clean = clean_asean_data(asean_raw)

    # Save cleaned
    asean_clean.to_csv(config.DATA_CLEANED / "asean_indices_cleaned.csv", index=False)

    # Create synchronized datasets
    asean_intersection = create_intersection_dataset(asean_clean)
    asean_intersection.to_csv(
        config.DATA_CLEANED / "asean_indices_intersection.csv", index=False)

    asean_weekly = create_weekly_dataset(asean_clean)
    asean_weekly.to_csv(config.DATA_CLEANED / "asean_indices_weekly.csv", index=False)

    # -- Global daily --
    global_daily = clean_global_daily(config.DATA_RAW / "global_daily_raw.csv")
    if not global_daily.empty:
        global_daily.to_csv(config.DATA_CLEANED / "global_daily_cleaned.csv")

    # -- Exchange rates --
    fx = clean_exchange_rates(config.DATA_RAW / "exchange_rates_raw.csv")
    if not fx.empty:
        fx.to_csv(config.DATA_CLEANED / "exchange_rates_cleaned.csv", index=False)

    logger.info(f"\n{'-' * 60}")
    logger.info("Stage 4 complete. Cleaned files saved to data/cleaned/")
    logger.info(f"  ASEAN cleaned:      {len(asean_clean)} obs")
    logger.info(f"  ASEAN intersection: {len(asean_intersection)} obs")
    logger.info(f"  ASEAN weekly:       {len(asean_weekly)} obs")

    return asean_clean, asean_intersection, asean_weekly


if __name__ == "__main__":
    main()
