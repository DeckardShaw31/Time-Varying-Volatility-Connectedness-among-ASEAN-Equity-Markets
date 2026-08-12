"""
Stage 5: Calculate returns.

  - Log returns: r = 100 × [ln(P_t) - ln(P_{t-1})]
  - USD returns: r_USD = r_LCU - 100 × Δln(FX)
  - First differences for rates: Δy = 100 × (y_t - y_{t-1})

Applied to:
  - ASEAN indices (local-currency and USD)
  - S&P 500, Brent oil (log returns)
  - Dollar index (log returns)
  - DGS2 (first difference -> basis points)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import config
from scripts.utils import setup_logger, log_returns, first_difference

logger = setup_logger("05_returns")


def compute_asean_returns(asean_df: pd.DataFrame,
                          fx_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Compute local-currency and (optionally) USD returns for ASEAN indices.
    Uses close prices for return calculation.
    """
    logger.info("Computing ASEAN index returns ...")

    results = []

    for country in config.COUNTRY_ORDER:
        sub = asean_df[asean_df["country"] == country].copy()
        sub = sub.sort_values("date").reset_index(drop=True)

        if len(sub) < 2:
            logger.warning(f"  {country}: insufficient data ({len(sub)} obs)")
            continue

        # Log returns in local currency
        sub["return_lcu"] = log_returns(sub["close"])

        # USD returns (if FX data available)
        if fx_df is not None and not fx_df.empty:
            fx_country = fx_df[fx_df["country"] == country].copy()
            if not fx_country.empty:
                fx_country = fx_country.sort_values("date").set_index("date")
                fx_country["dlog_fx"] = 100.0 * np.log(
                    fx_country["local_currency_per_usd"] /
                    fx_country["local_currency_per_usd"].shift(1)
                )

                # Merge FX with index data
                sub = sub.set_index("date")
                sub = sub.join(fx_country[["dlog_fx"]], how="left")
                sub["return_usd"] = sub["return_lcu"] - sub["dlog_fx"]
                sub = sub.reset_index()
            else:
                sub["return_usd"] = np.nan
                sub["dlog_fx"] = np.nan
        else:
            sub["return_usd"] = np.nan
            sub["dlog_fx"] = np.nan

        results.append(sub)

        # Summary
        valid_lcu = sub["return_lcu"].notna().sum()
        valid_usd = sub["return_usd"].notna().sum()
        logger.info(f"  {country}: {valid_lcu} LCU returns, {valid_usd} USD returns")

    if not results:
        return pd.DataFrame()

    combined = pd.concat(results, ignore_index=True)
    return combined


def compute_global_returns(global_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute returns/changes for global variables.
    - Price/index variables: log returns
    - Interest rates (DGS2): first differences
    """
    logger.info("Computing global variable returns ...")

    if global_df.empty:
        return pd.DataFrame()

    result = global_df.copy()

    # Log returns for price/index series
    for col in ["VIX", "SP500", "Brent", "DollarIdx"]:
        if col in result.columns:
            result[f"d_{col}"] = log_returns(result[col])
            valid = result[f"d_{col}"].notna().sum()
            logger.info(f"  Δln({col}): {valid} observations")

    # First difference for interest rates
    for col in ["DGS2"]:
        if col in result.columns:
            result[f"d_{col}"] = first_difference(result[col])
            valid = result[f"d_{col}"].notna().sum()
            logger.info(f"  Δ({col}): {valid} observations (basis points)")

    # Preserve GPR index columns directly
    for col in ["GPR", "GPRD", "GPRD_ACT", "GPRD_THREAT"]:
        if col in global_df.columns and col not in result.columns:
            result[col] = global_df[col]

    return result


def main():
    logger.info("=" * 60)
    logger.info("Stage 5: Calculating returns")
    logger.info("=" * 60)

    # Load cleaned ASEAN data (intersection dataset for synchronized returns)
    asean_path = config.DATA_CLEANED / "asean_indices_cleaned.csv"
    if not asean_path.exists():
        logger.error(f"Cleaned ASEAN data not found. Run Stage 4 first.")
        sys.exit(1)
    asean_df = pd.read_csv(asean_path, parse_dates=["date"])

    # Load FX data if available
    fx_path = config.DATA_CLEANED / "exchange_rates_cleaned.csv"
    fx_df = None
    if fx_path.exists():
        fx_df = pd.read_csv(fx_path, parse_dates=["date"])
        logger.info(f"  Loaded FX data: {len(fx_df)} observations")

    # Compute ASEAN returns
    asean_returns = compute_asean_returns(asean_df, fx_df)
    if not asean_returns.empty:
        out_path = config.DATA_PROC / "asean_returns.csv"
        asean_returns.to_csv(out_path, index=False)
        logger.info(f"  Saved ASEAN returns -> {out_path}")

    # Also compute returns for intersection and weekly datasets
    for label, fname in [("intersection", "asean_indices_intersection.csv"),
                         ("weekly", "asean_indices_weekly.csv")]:
        path = config.DATA_CLEANED / fname
        if path.exists():
            df = pd.read_csv(path, parse_dates=["date"])
            ret = compute_asean_returns(df, fx_df)
            if not ret.empty:
                out = config.DATA_PROC / f"asean_returns_{label}.csv"
                ret.to_csv(out, index=False)
                logger.info(f"  Saved {label} returns -> {out}")

    # Compute global returns
    global_path = config.DATA_CLEANED / "global_daily_cleaned.csv"
    if global_path.exists():
        global_df = pd.read_csv(global_path, index_col=0, parse_dates=True)
        global_returns = compute_global_returns(global_df)
        out_path = config.DATA_PROC / "global_returns.csv"
        global_returns.to_csv(out_path)
        logger.info(f"  Saved global returns -> {out_path}")

    logger.info(f"\n{'-' * 60}")
    logger.info("Stage 5 complete.")


if __name__ == "__main__":
    main()
