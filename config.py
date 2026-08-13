"""
Central configuration for the ASEAN Volatility Connectedness project.
All parameters, file paths, ticker symbols, and model settings are defined here.
"""

import os
from pathlib import Path

# ----------------------------------------------
# Project paths
# ----------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_RAW     = PROJECT_ROOT / "data" / "raw"
DATA_CLEANED = PROJECT_ROOT / "data" / "cleaned"
DATA_PROC    = PROJECT_ROOT / "data" / "processed"
OUT_TABLES   = PROJECT_ROOT / "outputs" / "tables"
OUT_FIGURES  = PROJECT_ROOT / "outputs" / "figures"
OUT_RESULTS  = PROJECT_ROOT / "outputs" / "results"
DELIVERABLES = PROJECT_ROOT / "deliverables"

# Ensure directories exist
for d in [DATA_RAW, DATA_CLEANED, DATA_PROC, OUT_TABLES, OUT_FIGURES,
          OUT_RESULTS, DELIVERABLES, DATA_RAW / "epu"]:
    d.mkdir(parents=True, exist_ok=True)

# Effective end date (dictated by available Philippines/Thailand market data)
START_DATE = "2010-01-01"
END_DATE   = "2026-07-18"

def find_file(*filenames) -> Path:
    """Helper to locate raw files in PROJECT_ROOT or DATA_RAW."""
    for fn in filenames:
        p1 = PROJECT_ROOT / fn
        if p1.exists():
            return p1
        p2 = DATA_RAW / fn
        if p2.exists():
            return p2
        p3 = DATA_RAW / "epu" / fn
        if p3.exists():
            return p3
    return PROJECT_ROOT / filenames[0]

# ----------------------------------------------
# ASEAN market definitions
# ----------------------------------------------
ASEAN_MARKETS = {
    "Indonesia": {
        "index_name": "Jakarta Composite Index",
        "ticker": "^JKSE",
        "currency": "IDR",
    },
    "Malaysia": {
        "index_name": "FTSE Bursa Malaysia KLCI",
        "ticker": "^KLSE",
        "currency": "MYR",
    },
    "Philippines": {
        "index_name": "PSE Composite Index",
        "ticker": "PSEi.PS",
        "currency": "PHP",
    },
    "Singapore": {
        "index_name": "Straits Times Index",
        "ticker": "^STI",
        "currency": "SGD",
    },
    "Thailand": {
        "index_name": "SET Index",
        "ticker": "^SET.BK",
        "currency": "THB",
    },
    "Vietnam": {
        "index_name": "VN-Index",
        "ticker": "^VNINDEX",
        "currency": "VND",
    },
}

# Ordered list of country names (used as column ordering throughout)
COUNTRY_ORDER = ["Indonesia", "Malaysia", "Philippines", "Singapore",
                 "Thailand", "Vietnam"]

# ----------------------------------------------
# Exchange-rate pairs (yfinance format: XXXUSD=X)
# We download LCU per 1 USD, so we use USDXXX=X
# ----------------------------------------------
FX_PAIRS = {
    "Indonesia": {"pair": "USDIDR=X", "local_currency": "IDR"},
    "Malaysia":  {"pair": "USDMYR=X", "local_currency": "MYR"},
    "Philippines": {"pair": "USDPHP=X", "local_currency": "PHP"},
    "Singapore": {"pair": "USDSGD=X", "local_currency": "SGD"},
    "Thailand":  {"pair": "USDTHB=X", "local_currency": "THB"},
    "Vietnam":   {"pair": "USDVND=X", "local_currency": "VND"},
}

# ----------------------------------------------
# Global daily variables
# ----------------------------------------------
# yfinance tickers used as primary source (no API key needed)
GLOBAL_DAILY_YF = {
    "VIX":        "^VIX",
    "SP500":      "^GSPC",
    "Brent":      "BZ=F",      # Brent crude futures
}

# FRED series IDs (used if pandas-datareader / fredapi available)
FRED_SERIES = {
    "VIX":        "VIXCLS",
    "Brent":      "DCOILBRENTEU",
    "DGS2":       "DGS2",
    "DollarIdx":  "DTWEXBGS",
    "SP500":      "SP500",
}

# FRED API key – set via environment variable
FRED_API_KEY = os.environ.get("FRED_API_KEY", "")

# ----------------------------------------------
# GPR / EPU file paths (manual downloads)
# ----------------------------------------------
GPR_DAILY_FILE    = DATA_RAW / "gpr_daily.csv"
GPR_AI_DAILY_FILE = DATA_RAW / "gpr_ai_daily.csv"
EPU_DIR           = DATA_RAW / "epu"

# ----------------------------------------------
# VAR & connectedness parameters
# ----------------------------------------------
VAR_MAX_LAG        = 10          # Maximum lag for information-criterion search
VAR_IC             = "bic"       # Criterion for lag selection: aic, bic, hqic
FORECAST_HORIZON   = 10          # Generalized FEVD forecast horizon (trading days)
ROLLING_WINDOW     = 250         # Rolling-window size (trading days)

# ----------------------------------------------
# Robustness grids (Section 11)
# ----------------------------------------------
ROBUSTNESS_WINDOWS  = [200, 250, 300]
ROBUSTNESS_HORIZONS = [5, 10, 20]
VOLATILITY_MEASURES = ["parkinson", "squared", "absolute"]
CURRENCY_MODES      = ["lcu", "usd"]
FREQUENCY_MODES     = ["daily_intersection", "weekly"]
GPR_VARIANTS        = ["conventional", "ai"]

# ----------------------------------------------
# Volatility calculation
# ----------------------------------------------
PARKINSON_EPSILON = 1e-8   # Small constant to prevent log(0)

# ----------------------------------------------
# Plotting defaults
# ----------------------------------------------
FIGURE_DPI    = 150
FIGURE_FORMAT = "png"
