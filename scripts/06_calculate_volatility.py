"""
Stage 6: Calculate volatility measures.

Three volatility proxies:
  1. Parkinson range volatility (baseline): v_P = [ln(H/L)]^2 / (4·ln(2))
  2. Squared returns (robustness):          v_SR = r^2
  3. Absolute returns (additional):          v_AR = |r|

Log-transform for VAR estimation: x = ln(v + ε)  where ε = 1e-8
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import config
from scripts.utils import (setup_logger, parkinson_volatility,
                            squared_returns, absolute_returns, log_volatility)

logger = setup_logger("06_volatility")


def compute_volatility(returns_df: pd.DataFrame,
                        label: str = "") -> pd.DataFrame:
    """
    Compute all three volatility measures for ASEAN index returns.
    
    Parameters
    ----------
    returns_df : DataFrame
        Must have columns: date, country, close, high, low, return_lcu
    label : str
        Label for logging
    
    Returns
    -------
    DataFrame with volatility columns added
    """
    logger.info(f"Computing volatility measures ({label}) ...")

    df = returns_df.copy()

    results = []
    for country in config.COUNTRY_ORDER:
        sub = df[df["country"] == country].copy()
        sub = sub.sort_values("date").reset_index(drop=True)

        if len(sub) < 2:
            continue

        # 1. Parkinson range volatility (if H and L available)
        has_hl = ("high" in sub.columns and "low" in sub.columns and
                  sub["high"].notna().any() and sub["low"].notna().any())

        if has_hl:
            sub["vol_parkinson"] = parkinson_volatility(sub["high"], sub["low"])
            # Handle cases where H == L (vol = 0)
            sub.loc[sub["vol_parkinson"] == 0, "vol_parkinson"] = np.nan
            valid_p = sub["vol_parkinson"].notna().sum()
            logger.info(f"  {country} Parkinson: {valid_p} valid observations")
        else:
            sub["vol_parkinson"] = np.nan
            logger.warning(f"  {country}: H/L not available, Parkinson skipped")

        # 2. Squared returns
        if "return_lcu" in sub.columns:
            sub["vol_squared"] = squared_returns(sub["return_lcu"])
        elif "return_usd" in sub.columns:
            sub["vol_squared"] = squared_returns(sub["return_usd"])
        else:
            sub["vol_squared"] = np.nan

        # 3. Absolute returns
        if "return_lcu" in sub.columns:
            sub["vol_absolute"] = absolute_returns(sub["return_lcu"])
        elif "return_usd" in sub.columns:
            sub["vol_absolute"] = absolute_returns(sub["return_usd"])
        else:
            sub["vol_absolute"] = np.nan

        # 4. Log-transforms of all volatility proxies for VAR estimation
        if has_hl:
            sub["log_vol_parkinson"] = log_volatility(
                sub["vol_parkinson"], config.PARKINSON_EPSILON)
        sub["log_vol_squared"] = log_volatility(
            sub["vol_squared"], config.PARKINSON_EPSILON)
        sub["log_vol_absolute"] = log_volatility(
            sub["vol_absolute"], config.PARKINSON_EPSILON)

        results.append(sub)

    if not results:
        return pd.DataFrame()

    combined = pd.concat(results, ignore_index=True)
    return combined


def build_volatility_panel(vol_df: pd.DataFrame,
                            measure: str = "vol_parkinson",
                            sync_type: str = "intersection") -> pd.DataFrame:
    """
    Build a wide-format volatility panel for VAR estimation.
    Columns = countries, index = dates.
    
    Parameters
    ----------
    vol_df : DataFrame
        Long-format with volatility columns
    measure : str
        Which volatility measure to use
    sync_type : str
        For logging
    
    Returns
    -------
    panel : DataFrame (date × country)
    """
    logger.info(f"Building {measure} panel ({sync_type}) ...")

    # Automatically use log-transformed measure for VAR estimation consistency
    log_col = f"log_{measure}" if not measure.startswith("log_") else measure
    if log_col in vol_df.columns:
        use_col = log_col
        logger.info(f"  Using log-transformed column: {use_col}")
    else:
        use_col = measure

    panel = vol_df.pivot_table(
        index="date", columns="country", values=use_col, aggfunc="first"
    )

    # Reorder columns
    panel = panel[[c for c in config.COUNTRY_ORDER if c in panel.columns]]

    # Drop rows with any NaN (complete cases only)
    n_before = len(panel)
    panel = panel.dropna()
    n_after = len(panel)
    logger.info(f"  Panel: {n_after} complete observations "
                f"(dropped {n_before - n_after} incomplete rows)")

    return panel


def main():
    logger.info("=" * 60)
    logger.info("Stage 6: Calculating volatility measures")
    logger.info("=" * 60)

    # Process each dataset variant
    datasets = {
        "full": "asean_returns.csv",
        "intersection": "asean_returns_intersection.csv",
        "weekly": "asean_returns_weekly.csv",
    }

    for label, fname in datasets.items():
        path = config.DATA_PROC / fname
        if not path.exists():
            logger.info(f"  {fname} not found - skipping {label}")
            continue

        df = pd.read_csv(path, parse_dates=["date"])
        vol = compute_volatility(df, label)

        if vol.empty:
            continue

        # Save long-format volatility
        out_long = config.DATA_PROC / f"asean_volatility_{label}.csv"
        vol.to_csv(out_long, index=False)
        logger.info(f"  Saved -> {out_long}")

        # Build wide-format panels for VAR (intersection and weekly only)
        if label in ("intersection", "weekly"):
            for measure in ["vol_parkinson", "vol_squared", "vol_absolute"]:
                panel = build_volatility_panel(vol, measure, label)
                if not panel.empty:
                    panel_path = config.DATA_PROC / f"panel_{measure}_{label}.csv"
                    panel.to_csv(panel_path)
                    logger.info(f"  Saved panel -> {panel_path}")

    logger.info(f"\n{'-' * 60}")
    logger.info("Stage 6 complete.")


if __name__ == "__main__":
    main()
