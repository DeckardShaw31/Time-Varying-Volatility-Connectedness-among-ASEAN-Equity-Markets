"""
Stage 2: Fetch global-shock and control data.

Downloads:
  - Daily: VIX, Brent crude, US 2Y yield, Broad USD index, S&P 500
  - Loads manually-downloaded GPR (daily/monthly) and EPU files
  - Monthly: Global EPU, US EPU, US Monetary Policy Uncertainty

Primary source: yfinance (no API key needed).
Fallback / supplement: FRED via pandas-datareader (if FRED_API_KEY is set).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import yfinance as yf

import config
from scripts.utils import setup_logger

logger = setup_logger("02_fetch_global")


# ----------------------------------------------
# FRED helper (optional)
# ----------------------------------------------

def fetch_fred_series(series_id: str, start: str, end: str) -> pd.Series:
    """Fetch a series from FRED via direct API request or pandas-datareader."""
    api_key = getattr(config, "FRED_API_KEY", "")
    if api_key:
        try:
            import requests
            url = (f"https://api.stlouisfed.org/fred/series/observations?"
                   f"series_id={series_id}&api_key={api_key}&file_type=json"
                   f"&observation_start={start}&observation_end={end}")
            res = requests.get(url, timeout=15).json()
            if "observations" in res:
                df = pd.DataFrame(res["observations"])
                df["date"] = pd.to_datetime(df["date"])
                df["value"] = pd.to_numeric(df["value"], errors="coerce")
                s = df.set_index("date")["value"].dropna()
                logger.info(f"    [OK] FRED API ({series_id}): {len(s)} observations")
                return s
        except Exception as e:
            logger.warning(f"  FRED API direct request failed for {series_id}: {e}")

    try:
        import pandas_datareader.data as web
        df = web.DataReader(series_id, "fred", start, end)
        s = df.iloc[:, 0].dropna()
        logger.info(f"    [OK] FRED datareader ({series_id}): {len(s)} observations")
        return s
    except Exception as e:
        logger.warning(f"  FRED datareader fetch failed for {series_id}: {e}")
        return pd.Series(dtype=float)


def fetch_yfinance_series(ticker: str, col: str = "Close") -> pd.DataFrame:
    """Fetch daily data from yfinance, return date + value."""
    tk = yf.Ticker(ticker)
    df = tk.history(start=config.START_DATE, end=config.END_DATE,
                    auto_adjust=True, actions=False)
    if df.empty:
        return pd.DataFrame()

    df = df.reset_index()
    # Normalize column names
    rename = {}
    for c in df.columns:
        cl = c.lower().strip()
        if cl in ("date", "datetime"):
            rename[c] = "date"
        elif cl == "close":
            rename[c] = "value"
    df = df.rename(columns=rename)

    if "date" not in df.columns or "value" not in df.columns:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    return df[["date", "value"]].dropna()


# ----------------------------------------------
# Fetch daily global variables
# ----------------------------------------------

def fetch_daily_globals() -> pd.DataFrame:
    """Build a combined daily DataFrame of global variables."""
    logger.info("Fetching daily global variables ...")

    all_series = {}

    # 1. VIX
    logger.info("  -> VIX")
    vix = fetch_yfinance_series(config.GLOBAL_DAILY_YF["VIX"])
    if not vix.empty:
        all_series["VIX"] = vix.set_index("date")["value"]
    else:
        # Fallback to FRED
        vix_fred = fetch_fred_series("VIXCLS", config.START_DATE, config.END_DATE)
        if not vix_fred.empty:
            all_series["VIX"] = vix_fred

    # 2. S&P 500
    logger.info("  -> S&P 500")
    sp = fetch_yfinance_series(config.GLOBAL_DAILY_YF["SP500"])
    if not sp.empty:
        all_series["SP500"] = sp.set_index("date")["value"]

    # 3. Brent crude oil
    logger.info("  -> Brent crude oil")
    brent = fetch_yfinance_series(config.GLOBAL_DAILY_YF["Brent"])
    if not brent.empty:
        all_series["Brent"] = brent.set_index("date")["value"]
    else:
        brent_fred = fetch_fred_series("DCOILBRENTEU", config.START_DATE, config.END_DATE)
        if not brent_fred.empty:
            all_series["Brent"] = brent_fred

    # 4. US 2-year Treasury yield (FRED only - no yfinance equivalent)
    logger.info("  -> US 2-year Treasury yield (DGS2)")
    dgs2 = fetch_fred_series("DGS2", config.START_DATE, config.END_DATE)
    if not dgs2.empty:
        all_series["DGS2"] = dgs2
    else:
        # Try yfinance for 2Y yield
        dgs2_yf = fetch_yfinance_series("^IRX")  # 13-week T-bill, not exact
        logger.warning("  DGS2 not available from FRED; consider providing FRED API key.")

    # 5. Broad US dollar index (FRED only)
    logger.info("  -> Broad US Dollar Index")
    dollar = fetch_fred_series("DTWEXBGS", config.START_DATE, config.END_DATE)
    if not dollar.empty:
        all_series["DollarIdx"] = dollar
    else:
        # Try DXY from yfinance as proxy
        dxy = fetch_yfinance_series("DX-Y.NYB")
        if not dxy.empty:
            all_series["DollarIdx"] = dxy.set_index("date")["value"]
            logger.info("  Using DXY (yfinance) as dollar-index proxy.")

    # Combine into single DataFrame
    if not all_series:
        logger.error("No global daily series fetched!")
        return pd.DataFrame()

    combined = pd.DataFrame(all_series)
    combined.index.name = "date"
    combined = combined.sort_index()

    logger.info(f"  [OK] Daily globals: {combined.shape[0]} dates, "
                f"columns = {list(combined.columns)}")
    for col in combined.columns:
        valid = combined[col].notna().sum()
        logger.info(f"    {col}: {valid} non-null observations")

    return combined


# ----------------------------------------------
# Load GPR data (manual downloads)
# ----------------------------------------------

def load_gpr_data() -> dict:
    """Load GPR daily and AI-GPR daily CSVs if available."""
    gpr_data = {}

    for label, path in [("GPR_daily", config.GPR_DAILY_FILE),
                        ("GPR_AI_daily", config.GPR_AI_DAILY_FILE)]:
        if path.exists():
            logger.info(f"  Loading {label} from {path}")
            try:
                df = pd.read_csv(path)
                date_col = None
                for c in df.columns:
                    if c.lower() in ("date", "day", "tradingdate"):
                        date_col = c
                        break
                if date_col is None and len(df.columns) >= 1:
                    date_col = df.columns[0]

                if date_col:
                    # Check if YYYYMMDD integer format
                    if pd.api.types.is_numeric_dtype(df[date_col]):
                        df["date"] = pd.to_datetime(df[date_col].astype(str), format="%Y%m%d", errors="coerce")
                    else:
                        df["date"] = pd.to_datetime(df[date_col], errors="coerce")
                    df = df.dropna(subset=["date"]).sort_values("date")
                    gpr_data[label] = df
                    logger.info(f"    [OK] {len(df)} observations")
            except Exception as e:
                logger.warning(f"    Failed to load {label}: {e}")
        else:
            logger.info(f"  {label} file not found at {path} - skipping.")

    return gpr_data


def load_epu_data() -> dict:
    """Load EPU Excel/CSV files from data/raw/epu/ if available."""
    epu_data = {}
    epu_dir = config.EPU_DIR

    if not epu_dir.exists():
        logger.info("  EPU directory not found - skipping.")
        return epu_data

    for f in epu_dir.iterdir():
        if f.suffix in (".xlsx", ".xls", ".csv"):
            logger.info(f"  Loading EPU file: {f.name}")
            try:
                if f.suffix == ".csv":
                    try:
                        df = pd.read_csv(f, encoding="utf-8")
                    except UnicodeDecodeError:
                        df = pd.read_csv(f, encoding="latin1")
                else:
                    df = pd.read_excel(f, engine="openpyxl")

                # If Year and Month columns exist, construct date
                if "Year" in df.columns and "Month" in df.columns:
                    valid_ym = df["Year"].notna() & df["Month"].notna()
                    df.loc[valid_ym, "date"] = pd.to_datetime(
                        df.loc[valid_ym, "Year"].astype(int).astype(str) + "-" +
                        df.loc[valid_ym, "Month"].astype(int).astype(str).str.zfill(2) + "-01",
                        errors="coerce"
                    )

                epu_data[f.stem] = df
                logger.info(f"    [OK] {len(df)} rows, columns: {list(df.columns)[:5]}")
            except Exception as e:
                logger.warning(f"    Failed: {e}")

    return epu_data


# ----------------------------------------------
# Main
# ----------------------------------------------

def main():
    logger.info("=" * 60)
    logger.info("Stage 2: Fetching global-shock and control data")
    logger.info("=" * 60)

    # 1. Fetch daily globals
    daily = fetch_daily_globals()

    if not daily.empty:
        daily_path = config.DATA_RAW / "global_daily_raw.csv"
        del_path   = config.DELIVERABLES / "global_daily_raw.csv"
        daily.to_csv(daily_path)
        daily.to_csv(del_path)
        logger.info(f"  Saved daily globals -> {daily_path}")

    # 2. Load GPR
    gpr = load_gpr_data()
    # If GPR daily exists, merge into the daily file
    if "GPR_daily" in gpr:
        gpr_df = gpr["GPR_daily"]
        if "date" in gpr_df.columns:
            gpr_df = gpr_df.set_index("date")
            # Identify GPR value columns
            gpr_cols = [c for c in gpr_df.columns if c != "date"]
            if not daily.empty:
                daily = daily.join(gpr_df[gpr_cols], how="outer")
                daily.to_csv(config.DELIVERABLES / "global_daily_raw.csv")

    # 3. Load EPU and GPR Monthly
    epu = load_epu_data()
    monthly_path = config.DELIVERABLES / "global_monthly_raw.csv"

    if epu and "All_Country_Data" in epu:
        epu_df = epu["All_Country_Data"]
        if "date" in epu_df.columns:
            m_df = epu_df[["date"]].copy()
            for col, target in [("GEPU_current", "GEPU"), ("US", "US_EPU"), ("Singapore", "Singapore_EPU")]:
                if col in epu_df.columns:
                    m_df[target] = pd.to_numeric(epu_df[col], errors="coerce")

            # Try to add GPR monthly if available
            gpr_m_path = config.DATA_RAW / "gpr_monthly.csv"
            if gpr_m_path.exists():
                try:
                    gpr_m = pd.read_csv(gpr_m_path)
                    if "month" in gpr_m.columns:
                        gpr_m["date"] = pd.to_datetime(gpr_m["month"], errors="coerce")
                        gpr_m = gpr_m.dropna(subset=["date"])
                        m_df = pd.merge(m_df, gpr_m[["date", "GPR", "GPRT", "GPRA"]], on="date", how="outer")
                except Exception as e:
                    logger.warning(f"Failed to merge GPR monthly: {e}")

            m_df = m_df.sort_values("date").dropna(how="all", subset=[c for c in m_df.columns if c != "date"])
            m_df.to_csv(monthly_path, index=False)
            logger.info(f"  Saved monthly globals -> {monthly_path} ({len(m_df)} rows)")
        else:
            pd.DataFrame(columns=["date", "GEPU", "US_EPU", "GPR"]).to_csv(monthly_path, index=False)
    else:
        pd.DataFrame(columns=["date", "GEPU", "US_EPU", "GPR"]).to_csv(monthly_path, index=False)

    logger.info("\nStage 2 complete.")
    return daily


if __name__ == "__main__":
    main()
