# ASEAN Volatility Connectedness & Contagion Research Pipeline (2010–2026)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Framework: Diebold-Yılmaz](https://img.shields.io/badge/Framework-Diebold--Y%C4%B1lmaz%20(2012%2F2014)-green.svg)](https://doi.org/10.1016/j.ijforecast.2011.02.006)

A complete, production-grade Python research framework designed to study **volatility spillovers, directional connectedness, and financial contagion** across six major ASEAN equity markets (**Indonesia, Malaysia, Philippines, Singapore, Thailand, and Vietnam**) over the period **January 2010 – July 2026**.

The methodology strictly implements the **Diebold-Yılmaz (2012, 2014)** VAR-based Generalized Forecast Error Variance Decomposition (GFEVD) connectedness framework, combined with event-window contagion tests (bootstrap confidence intervals) and HAC (Newey-West) global shock regressions.

---

## 📌 Project Purpose

This repository provides an automated, reproducible end-to-end econometric workflow to answer three core empirical questions:
1. **Systemic Spillover Density**: How interconnected are ASEAN equity market volatilities over full-sample and rolling 250-day windows?
2. **Systemic Roles**: Which markets act as net volatility transmitters versus net receivers?
3. **Contagion vs. Interdependence**: Do global economic/geopolitical shocks (e.g., COVID-19, Trade War, Russia-Ukraine War, Global Rate Hikes) trigger statistically significant increases in connectedness (true contagion), and how do global risk proxies (VIX, GPR, EPU, Oil, Dollar, S&P 500) drive total connectedness?

---

## 🧩 Module-by-Module Code Showcase

Below is a detailed technical breakdown of every Python module, helper function, input/output contract, and econometric equation in the codebase.

```
nckh/
├── config.py                       # Central Configuration & Parameters
├── run_all.py                      # Master Execution Pipeline Runner
└── scripts/
    ├── utils.py                    # Shared Econometric & Math Core Engine
    ├── 01_fetch_asean_indices.py   # Stage 1: ASEAN Stock Index Data Ingestion
    ├── 02_fetch_global_data.py     # Stage 2: Global Macro & Shock Data Ingestion
    ├── 03_fetch_exchange_rates.py  # Stage 3: Exchange Rate Data Ingestion
    ├── 04_clean_data.py            # Stage 4: Data Validation & Synchronization
    ├── 05_calculate_returns.py     # Stage 5: Log-Return & Change Calculations
    ├── 06_calculate_volatility.py  # Stage 6: Multi-Proxy Volatility Estimation
    ├── 07_descriptive_stats.py     # Stage 7: Descriptive & Unit Root Diagnostics
    ├── 08_var_model.py             # Stage 8: Full-Sample VAR & GFEVD Decomposition
    ├── 09_connectedness.py         # Stage 9: Rolling-Window Spillover Estimation
    ├── 10_shock_analysis.py        # Stage 10: Event Contagion & HAC Regressions
    └── 11_robustness.py            # Stage 11: Robustness Grid & DCC-GARCH Modeling
```

---

### ⚙️ 1. Configuration & Core Engine

#### `config.py` — *Central Configuration & Master Registry*
- **Role**: Defines all global parameters, date ranges (`2010-01-01` to `2026-07-31`), market tickers, FRED series mappings, directory paths, and model hyperparameters.
- **Key Settings**:
  - `ASEAN_MARKETS`: Maps 6 country indices (`^JKSE` Indonesia, `^KLSE` Malaysia, `PSEi.PS` Philippines, `^STI` Singapore, `^SET.BK` Thailand, `VNINDEX` Vietnam).
  - `FRED_SERIES`: `VIXCLS` (VIX), `DCOILBRENTEU` (Brent Oil), `DGS2` (US 2Y Yield), `DTWEXBGS` (Broad USD Index), `SP500` (S&P 500).
  - `VAR_MAX_LAG = 10`, `VAR_IC = "bic"`, `FORECAST_HORIZON = 10`, `ROLLING_WINDOW = 250`.
  - `FRED_API_KEY`: Configured for automated FRED data retrieval.

#### `scripts/utils.py` — *Shared Econometric & Mathematical Engine*
- **Role**: Contains the core mathematical algorithms for log-returns, Parkinson volatility, companion matrices, MA representation, Generalized FEVD, and Diebold-Yılmaz metrics.
- **Key Functions**:
  - `log_returns(prices)`: Computes $r_t = 100 \cdot [\ln(P_t) - \ln(P_{t-1})]$.
  - `parkinson_volatility(high, low)`: Calculates Parkinson (1980) range volatility:
    $$v_P = \frac{1}{4 \ln(2)} \left[ \ln\left(\frac{H_t}{L_t}\right) \right]^2$$
  - `var_companion_matrix(coefs)` & `ma_coefficients(coefs, horizon)`: Computes the VAR companion matrix $C$ and MA($\infty$) impulse-response matrices $\Psi_0, \Psi_1, \dots, \Psi_{H-1}$.
  - `generalized_fevd(coefs, sigma, horizon)`: Implements Pesaran-Shin (1998) Generalized Forecast Error Variance Decomposition:
    $$\theta_{ij}(H) = \frac{\sigma_{jj}^{-1} \sum_{h=0}^{H-1} (\mathbf{e}_i' \Psi_h \Sigma \mathbf{e}_j)^2}{\sum_{h=0}^{H-1} (\mathbf{e}_i' \Psi_h \Sigma \Psi_h' \mathbf{e}_i)}$$
  - `normalize_fevd(theta)`: Row-normalizes the GFEVD matrix $\tilde{\theta}_{ij} = \frac{\theta_{ij}}{\sum_{j=1}^K \theta_{ij}}$ so each row sums to 1 (100%).
  - `connectedness_measures(theta_norm, labels)`: Extracts Diebold-Yılmaz spillover metrics:
    - **Total Connectedness Index ($\text{TCI}$)**:
      $$\text{TCI} = \frac{100}{K} \sum_{\substack{i,j=1 \\ i \neq j}}^K \tilde{\theta}_{ij}$$
    - **Directional FROM**: $\text{FROM}_i = \sum_{j \neq i} \tilde{\theta}_{ij}$
    - **Directional TO**: $\text{TO}_i = \sum_{j \neq i} \tilde{\theta}_{ji}$
    - **Net Connectedness**: $\text{Net}_i = \text{TO}_i - \text{FROM}_i$
    - **Net Pairwise**: $C_{ij} = \tilde{\theta}_{ji} - \tilde{\theta}_{ij}$
  - `rolling_connectedness(data, window, horizon, max_lag, ic)`: Executes rolling-window estimation with automatic stability checks (`result.is_stable()`) and lag-fallback logic.
  - `setup_logger(name)`: Configures UTF-8 console logging with Windows encoding fixes.

---

### 📥 2. Data Ingestion Modules (Stages 1–3)

#### `scripts/01_fetch_asean_indices.py` — *Stage 1: ASEAN Stock Data Ingestion*
- **Role**: Ingests daily OHLCV data for all 6 ASEAN equity indices from 2010 to 2026.
- **Implementation**:
  - Downloads Indonesia (`^JKSE`), Malaysia (`^KLSE`), Philippines (`PSEi.PS`), Singapore (`^STI`), and Thailand (`^SET.BK`) via `yfinance`.
  - Ingests Vietnam (`VN-Index`) via `vnstock` (since `yfinance` lacks VN-Index coverage).
  - Enforces schema: `date, country, index_name, ticker, open, high, low, close, adjusted_close, volume, currency, source`.
  - **No forward-filling** of missing trading days (preserves raw national calendars).
- **Outputs**: `data/raw/asean_indices_raw.csv` and `deliverables/asean_indices_raw.csv`.

#### `scripts/02_fetch_global_data.py` — *Stage 2: Global Macro & Risk Data Ingestion*
- **Role**: Ingests global risk proxies, commodity prices, interest rates, and policy uncertainty metrics.
- **Implementation**:
  - Fetches daily `VIX`, `S&P 500`, `Brent Crude Oil`, `DGS2` (US 2Y Treasury yield), and `DollarIdx` (Broad USD Index) using `yfinance` and the FRED API.
  - Loads manually provided Caldara-Iacoviello Geopolitical Risk CSVs (`gpr_daily.csv`, `gpr_monthly.csv`).
  - Loads Economic Policy Uncertainty CSVs (`All_Country_Data.csv`) with `latin1` fallback handling.
- **Outputs**: `deliverables/global_daily_raw.csv` (4,326 daily obs) and `deliverables/global_monthly_raw.csv` (499 monthly obs).

#### `scripts/03_fetch_exchange_rates.py` — *Stage 3: Exchange Rate Ingestion*
- **Role**: Ingests daily local-currency-per-USD exchange rates for currency robustness checks.
- **Implementation**: Downloads `USDIDR=X`, `USDMYR=X`, `USDPHP=X`, `USDSGD=X`, `USDTHB=X`, and `USDVND=X` via `yfinance`, automatically handling quote inversions for pairs listed as USD/LCU.
- **Outputs**: `deliverables/exchange_rates_raw.csv`.

---

### 🧹 3. Data Processing & Diagnostic Modules (Stages 4–7)

#### `scripts/04_clean_data.py` — *Stage 4: Validation & Synchronization*
- **Role**: Cleans raw data and builds two common-date synchronized datasets.
- **Implementation**:
  - Normalizes dates, removes duplicates, validates numeric types, and verifies $H_t \ge L_t > 0$.
  - **Intersection Dataset**: Keeps only trading days common to **all 6 ASEAN markets** (3,343 observations).
  - **Weekly Dataset**: Takes the last available observation per ISO week to eliminate nonsynchronous trading friction (774 weekly observations).
- **Outputs**: `data/cleaned/asean_indices_intersection.csv` and `data/cleaned/asean_indices_weekly.csv`.

#### `scripts/05_calculate_returns.py` — *Stage 5: Log-Return Calculations*
- **Role**: Computes continuously compounded log-returns for stocks/commodities and basis-point changes for interest rates.
- **Equations**:
  - Local Currency Returns: $r_{i,t}^{LCU} = 100 \cdot [\ln(P_{i,t}) - \ln(P_{i,t-1})]$
  - USD Returns: $r_{i,t}^{USD} = r_{i,t}^{LCU} - 100 \cdot \Delta \ln(FX_{i,t})$
  - Interest Rate Changes: $\Delta y_t = 100 \cdot (y_t - y_{t-1})$ (basis points)
- **Outputs**: `data/processed/asean_returns_intersection.csv` and `data/processed/global_returns.csv`.

#### `scripts/06_calculate_volatility.py` — *Stage 6: Multi-Proxy Volatility Estimation*
- **Role**: Computes three distinct volatility proxies for baseline analysis and robustness testing.
- **Proxies Computed**:
  1. **Parkinson Range Volatility (Baseline)**: $v_{i,t}^P = \frac{[\ln(H_{i,t} / L_{i,t})]^2}{4 \ln(2)}$
  2. **Squared Returns (Robustness)**: $v_{i,t}^{SR} = (r_{i,t})^2$
  3. **Absolute Returns (Additional)**: $v_{i,t}^{AR} = |r_{i,t}|$
  4. **Log Transformation**: $x_{i,t} = \ln(v_{i,t}^P + \varepsilon)$ with $\varepsilon = 10^{-8}$ to prevent $\log(0)$ in VAR estimation.
- **Outputs**: Wide-format volatility panels `data/processed/panel_vol_parkinson_intersection.csv` and `panel_vol_squared_intersection.csv`.

#### `scripts/07_descriptive_stats.py` — *Stage 7: Statistics & Unit Root Diagnostics*
- **Role**: Computes descriptive statistics and econometric tests for returns and volatility series.
- **Tests Implemented**:
  - Summary stats: Mean, Median, Std Dev, Min, Max, Skewness, Excess Kurtosis.
  - **Jarque-Bera Normality Test**: Evaluates $H_0: \text{Normality}$.
  - **Augmented Dickey-Fuller (ADF) Test**: Evaluates $H_0: \text{Unit Root (Non-stationary)}$.
  - **KPSS Test**: Evaluates $H_0: \text{Level Stationarity}$.
  - Correlation heatmaps & Autocorrelation Function (ACF) plots.
- **Outputs**: `outputs/tables/table_returns_stats_intersection.csv`, `table_vol_parkinson_stats_intersection.csv`, and figure heatmaps under `outputs/figures/`.

---

### 📉 4. Econometric Modeling & Contagion Modules (Stages 8–11)

#### `scripts/08_var_model.py` — *Stage 8: Full-Sample VAR & GFEVD*
- **Role**: Estimates the 6-variable vector autoregression $x_t = \sum_{k=1}^p \Phi_k x_{t-k} + \varepsilon_t$ and full-sample GFEVD.
- **Implementation**:
  - Selects optimal lag order using AIC, BIC, and HQIC criteria (max lag 10; BIC selects lag 3 for Parkinson, lag 6 for Squared).
  - Verifies stability: `result.is_stable()` checks that all VAR polynomial roots lie outside the unit circle ($|z| > 1$).
  - Computes residual Durbin-Watson and Portmanteau autocorrelation tests.
  - Generates the full-sample $6 \times 6$ GFEVD matrix and formats the baseline connectedness table.
- **Outputs**: `outputs/tables/connectedness_table_vol_parkinson_intersection.csv` and `outputs/figures/gfevd_heatmap_vol_parkinson_intersection.png`.

#### `scripts/09_connectedness.py` — *Stage 9: Rolling-Window Connectedness*
- **Role**: Computes time-varying spillover dynamics over 250-day rolling windows (step = 1 day).
- **Implementation**:
  - Re-estimates the VAR model for each of ~3,100 rolling windows.
  - Calculates rolling $\text{TCI}$, $\text{FROM}_i$, $\text{TO}_i$, $\text{Net}_i$, and net pairwise matrices.
  - Plotting routines generate time-series figures for rolling $\text{TCI}$ and net directional positions.
- **Outputs**: `outputs/results/rolling_connectedness_vol_parkinson_intersection.csv` (3,094 windows) and plots `tci_rolling_*.png`, `net_connectedness_*.png`.

#### `scripts/10_shock_analysis.py` — *Stage 10: Event Contagion & HAC Regressions*
- **Role**: Tests for financial contagion during 8 major historical shocks and estimates global driver regressions.
- **Implementation**:
  - **Event-Window Analysis**: Compares mean $\text{TCI}$ during shock windows vs. pre-shock tranquil benchmark windows.
  - **Bootstrap Confidence Intervals**: 10,000 iterations to evaluate $H_0: \mu_{\text{shock}} - \mu_{\text{tranquil}} = 0$. Significant positive shifts ($\Delta \text{TCI} > 0, p < 0.05$) confirm **contagion** rather than simple interdependence.
  - **HAC (Newey-West) Regression**: Estimates OLS regression of $\text{TCI}$ on global variables with heteroskedasticity and autocorrelation-consistent standard errors:
    $$\text{TCI}_t = \alpha + \beta_1 \text{GPR}_t + \beta_2 \text{VIX}_t + \beta_3 \Delta \text{Oil}_t + \beta_4 \Delta \text{DGS2}_t + \beta_5 \Delta \text{Dollar}_t + \beta_6 \Delta \text{S\&P500}_t + \varepsilon_t$$
- **Outputs**: `deliverables/event_windows.csv`, `outputs/tables/shock_analysis_results.csv`, and `outputs/tables/hac_regression_coefficients.csv`.

#### `scripts/11_robustness.py` — *Stage 11: Robustness Matrix & DCC-GARCH*
- **Role**: Conducts sensitivity analysis across alternative model specifications and estimates multivariate GARCH.
- **Implementation**:
  - **Robustness Grid**: Re-estimates rolling connectedness across window sizes $W \in \{200, 250, 300\}$, forecast horizons $H \in \{5, 10, 20\}$, and volatility proxies (Parkinson vs. Squared).
  - **DCC-GARCH**: Fits univariate GARCH(1,1) models to each market and computes time-varying conditional correlations using exponential smoothing ($\lambda = 0.94$).
- **Outputs**: `outputs/tables/robustness_summary.csv` (18 combinations), `outputs/figures/robustness_comparison.png`, and `outputs/results/dcc_garch_correlations_vol_parkinson_intersection.csv`.

---

### 🕹️ 5. Pipeline Orchestration

#### `run_all.py` — *Master Pipeline Script*
- **Role**: Provides a centralized CLI entry point to run the entire pipeline or specific sub-stages.
- **CLI Options**:
  ```bash
  python run_all.py           # Executes Stages 1 -> 11 sequentially
  python run_all.py 8 9       # Runs Stage 8 and Stage 9
  python run_all.py --from 4  # Runs Stage 4 onward
  ```

---

## 📊 Key Empirical Findings

### 1. Connectedness & Systemic Roles (2010–2026)

- **Baseline Total Connectedness Index ($\text{TCI}$)**:
  - **Parkinson Range Volatility (Baseline)**: Full-sample $\text{TCI} = \mathbf{17.15\%}$. Rolling 250-day $\text{TCI}$ averages $\mathbf{20.04\%}$ (range: 3.21% – 56.03%).
  - **Squared Returns (Proxy)**: Full-sample $\text{TCI} = \mathbf{53.90\%}$. Rolling 250-day $\text{TCI}$ averages $\mathbf{39.79\%}$ (range: 4.85% – 82.58%).
- **Market Classifications**:
  - **Net Transmitters**: **Thailand** ($\text{Net} = +33.50$) and **Singapore** ($\text{Net} = +5.65$).
  - **Net Receivers**: **Indonesia** ($\text{Net} = -15.31$) and **Philippines** ($\text{Net} = -15.02$).
  - **Isolated Market**: **Vietnam** ($\text{FROM} = 6.89\%, \text{TO} = 5.94\%$), displaying low direct spillover sensitivity and acting as a regional diversification hedge.

### 2. Event-Window Contagion Analysis

Contagion requires a **statistically significant increase in connectedness** ($\Delta \text{TCI} > 0$), not merely high isolated volatility:

| Shock Event | Shock Window | Tranquil Mean $\text{TCI}$ | Shock Mean $\text{TCI}$ | $\Delta \text{TCI}$ | Bootstrap $p$-value | Empirical Finding |
|---|:---:|:---:|:---:|:---:|:---:|---|
| **COVID-19 Pandemic** | 2020-01-30 to 2020-06-30 | 21.77% | **41.75%** | **+19.98%** | $p < 0.0001$ | **Severe Contagion** |
| **US-China Trade War** | 2018-03-22 to 2018-12-31 | 11.73% | **23.86%** | **+12.13%** | $p < 0.0001$ | **High Contagion** |
| **China Stock Crash** | 2015-06-12 to 2016-02-29 | 10.29% | **18.82%** | **+8.53%** | $p < 0.0001$ | **Contagion** |
| **European Debt Crisis** | 2011-07-01 to 2012-06-30 | 21.10% | **28.73%** | **+7.63%** | $p < 0.0001$ | **Contagion** |
| **Taper Tantrum** | 2013-05-22 to 2013-09-30 | 13.51% | **20.75%** | **+7.24%** | $p < 0.0001$ | **Contagion** |
| **Monetary Tightening** | 2022-03-16 to 2022-12-31 | 13.33% | **16.86%** | **+3.53%** | $p < 0.0001$ | **Contagion** |
| **Russia-Ukraine War** | 2022-02-24 to 2022-06-30 | 13.21% | **13.80%** | **+0.58%** | $p = 0.0081$ | **Minor Contagion** |
| **US Banking Crisis** | 2023-03-10 to 2023-05-31 | 19.03% | **16.87%** | **-2.16%** | $p < 0.0001$ | **No Contagion** |

### 3. Drivers of Connectedness (HAC Newey-West Regression)

$$\text{TCI}_t = 7.64 + 0.684 \cdot \text{VIX}_t + 0.154 \cdot \Delta\text{Oil}_t + 0.027 \cdot \Delta\text{DGS2}_t - 1.346 \cdot \Delta\text{Dollar}_t + 0.767 \cdot \Delta\text{S\&P500}_t$$

- **Global Equity Volatility ($\text{VIX}$)**: $\beta = +0.684$ ($t = +6.53, p < 0.0001$) — The dominant driver of connectedness.
- **External Market Returns ($\Delta\text{S\&P 500}$)**: $\beta = +0.767$ ($t = +4.13, p < 0.0001$).
- **Dollar Index Changes ($\Delta\text{Dollar}$)**: $\beta = -1.346$ ($t = -2.04, p = 0.041$).

---

## 🚀 Quick Start Guide

### 1. Prerequisites & Installation

Clone the repository and install required dependencies:

```bash
git clone https://github.com/your-username/asean-connectedness.git
cd asean-connectedness
pip install -r requirements.txt
```

### 2. Configure API Keys & External Data (Optional)

- **FRED API Key**: Set your key in `config.py` or export it as an environment variable:
  ```powershell
  $env:FRED_API_KEY="your_api_key_here"
  ```
- **Manual Data Files**: Place downloaded Caldara-Iacoviello GPR (`gpr_daily.csv`, `gpr_monthly.csv`) and EPU files (`All_Country_Data.csv`) in `data/raw/` and `data/raw/epu/`.

---

## 💻 Running the Pipeline

### Execute the Full Pipeline End-to-End

To execute all 11 stages sequentially:

```bash
python run_all.py
```

### Execute Specific Stages

You can run individual stages or stage ranges by passing command-line arguments:

```bash
# Run Stage 1 (Fetch ASEAN Data)
python run_all.py 1

# Run Stage 8 and Stage 9 (VAR Model & Rolling Connectedness)
python run_all.py 8 9

# Run Stage 10 and Stage 11 (Shock Analysis & Robustness Grid)
python run_all.py 10 11

# Run from Stage 4 (Data Cleaning) onward
python run_all.py --from 4
```

---

## 📈 Figures & Visualizations Generated

The pipeline automatically saves publication-ready figures to `outputs/figures/`:

- `tci_rolling_vol_parkinson_intersection.png`: 250-day rolling Total Connectedness Index ($\text{TCI}$) time-series.
- `net_connectedness_vol_parkinson_intersection.png`: Net directional position for all 6 ASEAN countries over time.
- `from_connectedness_vol_parkinson_intersection.png` & `to_connectedness_vol_parkinson_intersection.png`: Directional spillovers.
- `gfevd_heatmap_vol_parkinson_intersection.png`: Full-sample GFEVD spillover intensity matrix.
- `corr_vol_parkinson_intersection.png`: Volatility correlation heatmap.
- `robustness_comparison.png`: TCI comparison across window sizes ($W=200, 250, 300$) and horizons ($H=5, 10, 20$).

---

## 📜 Citation & References

If you use this pipeline or code in your academic research, please cite the underlying methodology:

- **Diebold, F. X., & Yılmaz, K. (2012)**. Better measures of econometric connectedness and propagation, with application to global equity markets. *The Economic Journal*, 122(559), 401-421.
- **Diebold, F. X., & Yılmaz, K. (2014)**. On the network topology of variance decompositions: Measuring connectedness of financial firms. *Journal of Econometrics*, 182(1), 119-134.
- **Pesaran, H. H., & Shin, Y. (1998)**. Generalized impulse response analysis in linear multivariate models. *Economics Letters*, 58(1), 17-29.
- **Parkinson, M. (1980)**. The extreme value method for estimating the variance of the rate of return. *Journal of Business*, 53(1), 61-65.
