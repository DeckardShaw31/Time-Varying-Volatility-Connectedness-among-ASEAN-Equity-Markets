# ASEAN Volatility Connectedness & Shock Analysis Pipeline (2010–2026)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Framework: Diebold-Yılmaz](https://img.shields.io/badge/Framework-Diebold--Y%C4%B1lmaz%20(2012%2F2014)-green.svg)](https://doi.org/10.1016/j.ijforecast.2011.02.006)

A complete, production-grade Python research framework designed to study **volatility spillovers, directional connectedness, and financial shock transmission** across six major ASEAN equity markets (**Indonesia, Malaysia, Philippines, Singapore, Thailand, and Vietnam**) over the period **January 2010 – July 2026**.

The methodology strictly implements the **Diebold-Yılmaz (2012, 2014)** VAR-based Generalized Forecast Error Variance Decomposition (GFEVD) connectedness framework, combined with event-window shock tests (Moving-Block Bootstrap confidence intervals) and HAC (Newey-West) global shock regressions.

---

## 📌 Project Purpose

This repository provides an automated, reproducible end-to-end econometric workflow to answer three core empirical questions:
1. **Systemic Spillover Density**: How interconnected are ASEAN equity market volatilities over full-sample and rolling 250-day windows?
2. **Systemic Roles**: Which markets act as net volatility transmitters versus net receivers?
3. **Shock-Associated Shifts vs. Interdependence**: Do global economic/geopolitical shocks (e.g., COVID-19, Trade War, Russia-Ukraine War, Global Rate Hikes) trigger statistically significant increases in connectedness, and how do global risk proxies (VIX, GPR, EPU, Oil, Dollar, S&P 500) drive total connectedness?

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
    ├── 10_shock_analysis.py        # Stage 10: Event Analysis & HAC Regressions
    └── 11_robustness.py            # Stage 11: Robustness Grid & GARCH-Filtered EWMA
```

---

### ⚙️ 1. Configuration & Core Engine

#### `config.py` — *Central Configuration & Master Registry*
- **Role**: Defines all global parameters, date ranges (`2010-01-01` to `2026-07-18`), market tickers, FRED series mappings, directory paths, and model hyperparameters.
- **Key Settings**:
  - `ASEAN_MARKETS`: Maps 6 country indices (`^JKSE` Indonesia, `^KLSE` Malaysia, `PSEi.PS` Philippines, `^STI` Singapore, `^SET.BK` Thailand, `VNINDEX` Vietnam).
  - `FRED_SERIES`: `VIXCLS` (VIX), `DCOILBRENTEU` (Brent Oil), `DGS2` (US 2Y Yield), `DTWEXBGS` (Broad USD Index), `SP500` (S&P 500).
  - `VAR_MAX_LAG = 10`, `VAR_IC = "bic"`, `FORECAST_HORIZON = 10`, `ROLLING_WINDOW = 250`.
  - `FRED_API_KEY`: Configured via environment variable (`FRED_API_KEY`).

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
- **Role**: Ingests daily OHLCV data for all 6 ASEAN equity indices from 2010 to July 17, 2026.
- **Implementation**:
  - Downloads Indonesia (`^JKSE`), Malaysia (`^KLSE`), Philippines (`PSEi.PS`), Singapore (`^STI`), and Thailand (`^SET.BK`) via `yfinance`.
  - Ingests Vietnam (`VN-Index`) via `vnstock`.
  - Enforces schema: `date, country, index_name, ticker, open, high, low, close, adjusted_close, volume, currency, source`.
- **Outputs**: `data/raw/asean_indices_raw.csv` and `deliverables/asean_indices_raw.csv`.

#### `scripts/02_fetch_global_data.py` — *Stage 2: Global Macro & Risk Data Ingestion*
- **Role**: Ingests global risk proxies, commodity prices, interest rates, and policy uncertainty metrics.
- **Implementation**:
  - Fetches daily `VIX`, `S&P 500`, `Brent Crude Oil`, `DGS2` (US 2Y Treasury yield), and `DollarIdx` (Broad USD Index) using `yfinance` and the FRED API.
  - Loads Caldara-Iacoviello Geopolitical Risk CSVs (`gpr_daily.csv`).
- **Outputs**: `deliverables/global_daily_raw.csv` and `deliverables/global_monthly_raw.csv`.

#### `scripts/03_fetch_exchange_rates.py` — *Stage 3: Exchange Rate Ingestion*
- **Role**: Ingests daily local-currency-per-USD exchange rates for currency robustness checks.
- **Outputs**: `deliverables/exchange_rates_raw.csv`.

---

### 🧹 3. Data Processing & Diagnostic Modules (Stages 4–7)

#### `scripts/04_clean_data.py` — *Stage 4: Validation & Synchronization*
- **Role**: Cleans raw data and builds common-date synchronized datasets.
- **Implementation**:
  - **Intersection Dataset**: Keeps trading days common to **all 6 ASEAN markets** (3,366 observations; sample end date **July 17, 2026**).
  - **Weekly Dataset**: Aggregates daily OHLCV within each ISO week (`year_week`) and assigns a **canonical Friday date** (`pd.to_datetime(year_week + '-5', format='%G-W%V-%u')`).
- **Outputs**: `data/cleaned/asean_indices_intersection.csv` and `data/cleaned/asean_indices_weekly.csv`.

#### `scripts/05_calculate_returns.py` — *Stage 5: Log-Return Calculations*
- **Role**: Computes continuously compounded log-returns for stocks/commodities and changes for interest rates.
- **Outputs**: `data/processed/asean_returns_intersection.csv` and `data/processed/global_returns.csv`.

#### `scripts/06_calculate_volatility.py` — *Stage 6: Multi-Proxy Volatility Estimation*
- **Role**: Computes three volatility proxies across Local Currency (`lcu`) and USD (`usd`) returns.
- **Proxies Computed**:
  1. **Parkinson Range Volatility (Baseline)**: $v_{i,t}^P = \frac{[\ln(H_{i,t} / L_{i,t})]^2}{4 \ln(2)}$
  2. **Squared Returns**: $v_{i,t}^{SR} = (r_{i,t})^2$
  3. **Absolute Returns**: $v_{i,t}^{AR} = |r_{i,t}|$
  4. **Consistent Log Transformation**: $x_{i,t} = \ln(v_{i,t} + \varepsilon)$ with $\varepsilon = 10^{-8}$ applied **consistently across all proxies**.
- **Outputs**: Wide-format volatility panels `panel_vol_parkinson_intersection.csv`, `panel_vol_squared_intersection.csv`, `panel_vol_squared_usd_intersection.csv`, `panel_vol_absolute_intersection.csv`, etc.

#### `scripts/07_descriptive_stats.py` — *Stage 7: Statistics & Unit Root Diagnostics*
- **Role**: Computes descriptive statistics, Jarque-Bera normality tests, ADF, and KPSS stationarity tests.
- **Outputs**: `outputs/tables/table_returns_stats_intersection.csv`, `table_vol_parkinson_stats_intersection.csv`, etc.

---

### 📉 4. Econometric Modeling & Contagion Modules (Stages 8–11)

#### `scripts/08_var_model.py` — *Stage 8: Full-Sample VAR & GFEVD*
- **Role**: Estimates the 6-variable vector autoregression $x_t = \sum_{k=1}^p \Phi_k x_{t-k} + \varepsilon_t$ and full-sample GFEVD.
- **Lag Selection Note**: BIC selects raw lag 0 for Log Squared Returns; an **enforced minimum lag 1** is used for VAR model stability.
- **Outputs**: `outputs/tables/connectedness_table_vol_parkinson_intersection.csv` and `outputs/figures/gfevd_heatmap_vol_parkinson_intersection.png`.

#### `scripts/09_connectedness.py` — *Stage 9: Rolling-Window Connectedness*
- **Role**: Computes time-varying spillover dynamics over 250-day rolling windows.
- **Outputs**: `outputs/results/rolling_connectedness_vol_parkinson_intersection.csv` and time-series plots under `outputs/figures/`.

#### `scripts/10_shock_analysis.py` — *Stage 10: Event Analysis & HAC Regressions*
- **Role**: Evaluates shock-associated shifts in TCI and estimates global driver regressions.
- **Implementation**:
  - **Moving-Block Bootstrap (MBB)**: Evaluates feasible block sizes ($B \le \frac{1}{2} \min(n_{\text{shock}}, n_{\text{tranquil}})$; $B10$ and $B20$). Infeasible block sizes are skipped. Zero bootstrap probabilities are reported as $p < 0.0002$.
  - **Month-End HAC Regression**: Resamples daily global level series to month-end levels ($\text{Brent}_m, \text{DollarIdx}_m, \text{SP500}_m, \text{DGS2}_m$) before computing monthly log/first differences, and takes monthly averages for $\text{VIX}_m$ and $\text{GPR}_m$. Fits OLS with Newey-West HAC standard errors ($L = 12$ months) to accommodate rolling window persistence.
- **Outputs**: `outputs/tables/shock_analysis_results.csv` and `outputs/tables/hac_regression_coefficients.csv`.

#### `scripts/11_robustness.py` — *Stage 11: Robustness Matrix & GARCH-Filtered EWMA*
- **Role**: Conducts sensitivity analysis across 90 specifications and estimates GARCH-filtered EWMA conditional market correlations.
- **Implementation**:
  - **90-Specification Grid**: Evaluates 10 panel variants (Parkinson, Squared LCU/USD, Absolute LCU/USD $\times$ Daily/Weekly), using econometrically equivalent weekly parameters ($W_{\text{weekly}} \in \{40, 50, 60\}$ weeks, $H_{\text{weekly}} \in \{1, 2, 4\}$ weeks).
  - **Directional Spillover Metrics**: Exports mean net directional connectedness for all 6 countries and `share_Vietnam_net_transmitter`.
  - **Fixed VAR Lags**: Evaluates fixed lag orders $p \in \{1, 2, 3, 4, 7\}$ covering BIC ($p=1, 3$), HQIC ($p=4$), and AIC ($p=7$) lag selections.
  - **GARCH-Filtered EWMA**: Fits univariate GARCH(1,1) models to daily stock return series (%) and computes time-varying conditional market correlations ($\lambda = 0.94$).
- **Outputs**: `outputs/tables/robustness_summary.csv`, `outputs/tables/robustness_alternative_lags.csv`, and `outputs/results/garch_ewma_correlations_*.csv`.

#### `scripts/12_portfolio_diversification.py` — *Stage 12: Portfolio Diversification across Connectedness Regimes*
- **Role**: Directly evaluates out-of-sample portfolio risk and diversification benefits across Low-TCI ($\le Q_{25}$), Moderate-TCI, and High-TCI ($\ge Q_{75}$) regimes.
- **Implementation**: Simulates rolling 250-day Equal-Weighted (1/N) and Global Minimum Variance (GMV) allocations, computing realized volatility, Choueifaty & Coignard (2008) Diversification Ratios ($DR$), 95% Expected Shortfall ($\text{ES}_{95}$), and net performance after 10 bps transaction costs.
- **Outputs**: `outputs/tables/portfolio_diversification_results.csv`, `outputs/tables/portfolio_regime_comparison.csv`, and `outputs/figures/portfolio_diversification_regimes.png`.

#### `scripts/generate_manuscript_results.py` — *Stage 13: Single-Source-of-Truth Dataset Generator*
- **Role**: Compiles every single empirical number, test statistic, diagnostic, regression estimate, robustness cell, and portfolio metric into a single authoritative master dataset.
- **Outputs**: `deliverables/manuscript_results.json` and `deliverables/manuscript_results.csv` (as well as mirrored in `outputs/`).

---

## 📊 Key Empirical Findings

### 1. Connectedness & Systemic Roles (2010–2026)

- **Total Connectedness Index ($\text{TCI}$)** (Sample: Jan 4, 2010 – Jun 29, 2026, $N=3,330$):
  - **Log Parkinson Range Volatility (Baseline)**: Full-sample $\text{TCI} = \mathbf{17.22\%}$. Rolling 250-day $\text{TCI}$ averages $\mathbf{20.08\%}$ (range: 9.88% – 48.84%).
  - **Log Squared Returns**: Full-sample $\text{TCI} = \mathbf{9.27\%}$. Rolling 250-day mean $\text{TCI} = \mathbf{11.89\%}$.
- **Market Roles**:
  - **Net Transmitters**: **Thailand** ($\text{Net} = +3.21\%$) and **Indonesia** ($\text{Net} = +2.80\%$).
  - **Net Receivers**: **Singapore** ($\text{Net} = -2.91\%$), **Philippines** ($\text{Net} = -1.61\%$), and **Vietnam** ($\text{Net} = -0.96\%$).
  - **Sensitivity Note**: Vietnam's net spillover position is sensitive to the choice of volatility proxy ($\text{Net} = -0.72\%$ with Parkinson vs. $\text{Net} = +0.64\%$ with squared returns).

### 2. Event-Window Analysis (Moving-Block Bootstrap & Multiple Testing)

| Shock Event | Shock Window | Tranquil Mean $\text{TCI}$ | Shock Mean $\text{TCI}$ | $\Delta \text{TCI}$ | MBB $B=20$ 95% CI | Raw $p$-value | Holm $p$-value | BH $p$-value | Shift Status |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **COVID-19 Pandemic** | 2020-01-30 to 2020-06-30 | 21.77% | **41.75%** | **+19.98%** | $[7.56, 31.12]$ | $p = 0.0014$ | $p = 0.0042$ | $p = 0.0018$ | **Shock Increase** |
| **US-China Trade War** | 2018-03-22 to 2018-12-31 | 11.73% | **23.86%** | **+12.13%** | $[11.08, 15.46]$ | $p < 0.0002$ | $p = 0.0008$ | $p < 0.0002$ | **Shock Increase** |
| **China Stock Crash** | 2015-06-12 to 2016-02-29 | 10.29% | **18.82%** | **+8.53%** | $[6.04, 11.60]$ | $p < 0.0002$ | $p = 0.0008$ | $p < 0.0002$ | **Shock Increase** |
| **European Debt Crisis** | 2011-07-01 to 2012-06-30 | 21.10% | **28.73%** | **+7.63%** | $[5.53, 10.09]$ | $p < 0.0002$ | $p = 0.0008$ | $p < 0.0002$ | **Shock Increase** |
| **Taper Tantrum** | 2013-05-22 to 2013-09-30 | 13.51% | **20.75%** | **+7.24%** | $[5.80, 10.04]$ | $p < 0.0002$ | $p = 0.0008$ | $p < 0.0002$ | **Shock Increase** |
| **Monetary Tightening** | 2022-06-01 to 2022-12-31 | 13.21% | **17.90%** | **+4.69%** | $[2.57, 7.04]$ | $p < 0.0002$ | $p = 0.0008$ | $p < 0.0002$ | **Shock Increase** |
| **Russia-Ukraine War** | 2022-02-24 to 2022-05-31 | 13.21% | **13.82%** | **+0.61%** | $[-0.92, 2.08]$ | $p = 0.2370$ | $p = 0.2370$ | $p = 0.2370$ | No shift |
| **US Banking Crisis** | 2023-03-10 to 2023-05-31 | 19.03% | **16.87%** | **-2.16%** | $[-3.75, -0.72]$ | $p = 0.0016$ | $p = 0.0042$ | $p = 0.0018$ | No shift |

*Note: All six positive shock shifts remain statistically significant at 5% after adjusting for multiple testing across all eight events via Holm-Bonferroni (FWER) and Benjamini-Hochberg (FDR).*

### 3. Drivers of Connectedness (HAC Newey-West Multi-Specification Models)

$$\text{TCI}_m = 16.36 + 0.962 \cdot \text{VIX}_m - 0.130 \cdot \text{GPR}_m + 0.121 \cdot \Delta\text{Oil}_m + 2.222 \cdot \Delta\text{DGS2\_pp}_m - 0.370 \cdot \Delta\text{Dollar}_m + 0.245 \cdot \Delta\text{S\&P500}_m$$

- **Global Financial Volatility ($\text{VIX}_m$)**: $\beta = \mathbf{+0.9616}$ ($z = +3.718, p < 0.001$, Standardized $\beta^* = \mathbf{+0.5567}$) — Dominant positive systemic spillover driver.
- **Geopolitical Risk ($\text{GPR}_m$)**: $\beta = \mathbf{-0.1300}$ ($z = -4.381, p < 0.001$, Standardized $\beta^* = \mathbf{-0.4672}$) — Statistically significant negative coefficient (flight-to-safety / international decoupling).
- **Oil Price Changes ($\Delta\text{Oil}_m$)**: $\beta = \mathbf{+0.1213}$ ($z = +2.185, p = 0.0289$, Standardized $\beta^* = \mathbf{+0.1248}$) — Statistically significant positive commodity driver.
- **GPR Diagnostic Suite**:
  - **No Multicollinearity**: Regressor $\text{VIF} \in [1.089, 1.659]$ and $\text{Corr}(\text{VIX}, \text{GPR}) = 0.0372$.
  - **Isolated Models**: $\text{VIX-only}$ ($R^2 = 0.2834, p < 0.001$) and $\text{GPR-only}$ ($R^2 = 0.1933, p < 0.001$).
  - **Orthogonalized GPR**: $\beta_{\text{GPR}^{\perp}} = -0.1300$ ($p < 0.001$).
  - **Differenced & AR(1) Models**: $\beta_{\Delta\text{VIX}} = +0.3291$ ($p = 0.010$) and AR(1) autoregressive $\rho = 0.8825$ ($p < 0.001, R^2 = 0.9294$).

### 4. Portfolio Diversification across Connectedness Regimes

| Connectedness Regime | Mean $\text{TCI}$ | EW Ann. Vol. | GMV Ann. Vol. | GMV Risk Reduction | EW Diversification Ratio ($DR$) | GMV Diversification Ratio ($DR$) | GMV Expected Shortfall ($\text{ES}_{95}$) | Net Sharpe Ratio |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Full Sample** | 19.45% | 11.62% | 10.60% | 8.78% | 1.481 | 1.366 | -1.63% | 0.249 |
| **Low TCI ($\le Q_{25}$)** | 9.28% | **9.66%** | **9.06%** | 6.28% | **1.593** | **1.460** | **-1.37%** | **0.841** |
| **Moderate TCI ($Q_{25}$–$Q_{75}$)** | 17.45% | 10.89% | 10.11% | 7.16% | 1.482 | 1.364 | -1.55% | -0.467 |
| **High TCI ($\ge Q_{75}$)** | 33.59% | **14.45%** | **12.73%** | **11.92%** | **1.368** | **1.276** | **-1.99%** | **0.972** |
| **Crisis Peak ($\ge Q_{90}$)** | 45.79% | **17.32%** | **15.86%** | 8.43% | **1.265** | **1.229** | **-2.38%** | **1.461** |

- **Moving-Block Bootstrap Inference ($B=20$ days, 2,000 replications)**:
  - $\Delta \sigma_{\text{EW}}$: Observed $\mathbf{+4.79\%}$ (Bootstrap mean $+4.93\%$, $95\%\text{ CI } [0.85, 10.07], p = 0.009$)
  - $\Delta \sigma_{\text{GMV}}$: Observed $\mathbf{+3.67\%}$ (Bootstrap mean $+3.81\%$, $95\%\text{ CI } [0.65, 7.79], p = 0.018$)
  - $\Delta DR$: Observed $\mathbf{-0.184}$ (Bootstrap mean $-0.189$, $95\%\text{ CI } [-0.23, -0.14], p < 0.001$)
  - $\Delta \text{ES}_{95}$: Observed $\mathbf{-0.62\%}$ (Bootstrap mean $-0.63\%$, $95\%\text{ CI } [-1.49, 0.07], p = 0.091$)

---

## 🚀 Quick Start Guide

### 1. Prerequisites & Installation

```bash
git clone https://github.com/DeckardShaw31/Time-Varying-Volatility-Connectedness-among-ASEAN-Equity-Markets.git
cd Time-Varying-Volatility-Connectedness-among-ASEAN-Equity-Markets
pip install -r requirements.txt
```

### 2. Running the Pipeline

```bash
python run_all.py
```

All empirical tables, diagnostics, figures, and deliverables (`deliverables/manuscript_results.json` and `deliverables/manuscript_results.csv`) will be generated automatically.

---

## 📜 Citation & References

- **Diebold, F. X., & Yılmaz, K. (2012)**. Better measures of econometric connectedness and propagation, with application to global equity markets. *The Economic Journal*, 122(559), 401-421.
- **Diebold, F. X., & Yılmaz, K. (2014)**. On the network topology of variance decompositions: Measuring connectedness of financial firms. *Journal of Econometrics*, 182(1), 119-134.
- **Choueifaty, Y., & Coignard, Y. (2008)**. Toward maximum diversification. *The Journal of Portfolio Management*, 35(1), 40-51.
- **Corsi, F. (2009)**. A simple approximate long-memory model of realized volatility. *Journal of Financial Econometrics*, 7(2), 174-196.
- **Antonakakis, N., Chatziantoniou, I., & Gabauer, D. (2020)**. Refined measures of dynamic connectedness based on time-varying parameter vector autoregressions. *Journal of Risk and Financial Management*, 13(4), 84.
- **Bessler, W., Opfer, H., & Wolff, D. (2017)**. Multi-asset portfolio optimization and out-of-sample performance: An evaluation of Black–Litterman, minimum-variance, and equal-weighted approaches. *The European Journal of Finance*, 23(1), 1-30.
- **Pesaran, H. H., & Shin, Y. (1998)**. Generalized impulse response analysis in linear multivariate models. *Economics Letters*, 58(1), 17-29.
- **Parkinson, M. (1980)**. The extreme value method for estimating the variance of the rate of return. *Journal of Business*, 53(1), 61-65.


