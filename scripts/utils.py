"""
Shared utility functions for the ASEAN Volatility Connectedness project.

Provides reusable routines for:
  - Log-return and first-difference calculations
  - Parkinson range volatility
  - Generalized Forecast Error Variance Decomposition (GFEVD)
  - Diebold-Yılmaz connectedness measures
  - Plotting helpers
  - Logging setup
"""

import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import linalg

# Suppress statsmodels frequency warnings in rolling loops
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="statsmodels")
try:
    from statsmodels.tools.sm_exceptions import ValueWarning
    warnings.filterwarnings("ignore", category=ValueWarning)
except ImportError:
    pass

# ----------------------------------------------
# Logging
# ----------------------------------------------

def setup_logger(name: str, level=logging.INFO) -> logging.Logger:
    """Create a console logger with a standard format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        if hasattr(sys.stdout, "reconfigure"):
            try:
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
                              datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ----------------------------------------------
# Return calculations
# ----------------------------------------------

def log_returns(prices: pd.Series) -> pd.Series:
    """
    Continuously compounded returns (× 100).
    r_t = 100 * [ln(P_t) - ln(P_{t-1})]
    """
    return 100.0 * np.log(prices / prices.shift(1))


def first_difference(series: pd.Series) -> pd.Series:
    """
    First difference scaled by 100 (for interest rates in percentage points).
    Δy_t = 100 * (y_t - y_{t-1})
    """
    return 100.0 * series.diff()


# ----------------------------------------------
# Volatility calculations
# ----------------------------------------------

def parkinson_volatility(high: pd.Series, low: pd.Series) -> pd.Series:
    """
    Parkinson (1980) range-based volatility estimator:
      v_P = [ln(H/L)]^2 / (4 * ln(2))
    """
    log_hl = np.log(high / low)
    return (log_hl ** 2) / (4.0 * np.log(2.0))


def squared_returns(returns: pd.Series) -> pd.Series:
    """Squared-return volatility proxy: v_SR = r^2."""
    return returns ** 2


def absolute_returns(returns: pd.Series) -> pd.Series:
    """Absolute-return volatility proxy: v_AR = |r|."""
    return returns.abs()


def log_volatility(vol: pd.Series, epsilon: float = 1e-8) -> pd.Series:
    """Log-transform of volatility for VAR estimation: x = ln(v + ε)."""
    return np.log(vol + epsilon)


# ----------------------------------------------
# VAR companion form & MA coefficients
# ----------------------------------------------

def var_companion_matrix(coefs: np.ndarray) -> np.ndarray:
    """
    Build the companion matrix from VAR coefficient matrices.
    
    Parameters
    ----------
    coefs : ndarray of shape (p, K, K)
        Φ_1, ..., Φ_p  where K = number of variables, p = lag order.

    Returns
    -------
    C : ndarray of shape (K*p, K*p)
    """
    p, K, _ = coefs.shape
    Kp = K * p
    C = np.zeros((Kp, Kp))
    # First K rows: [Φ_1, Φ_2, ..., Φ_p]
    for i in range(p):
        C[:K, i*K:(i+1)*K] = coefs[i]
    # Identity blocks on the sub-diagonal
    if p > 1:
        C[K:, :K*(p-1)] = np.eye(K*(p-1))
    return C


def ma_coefficients(coefs: np.ndarray, horizon: int) -> list:
    """
    Compute MA(∞) coefficient matrices Ψ_0, Ψ_1, ..., Ψ_{H-1}
    from VAR coefficient matrices using the companion form.
    
    Parameters
    ----------
    coefs : ndarray of shape (p, K, K)
    horizon : int
    
    Returns
    -------
    psi : list of ndarray, each of shape (K, K)
    """
    p, K, _ = coefs.shape
    C = var_companion_matrix(coefs)
    psi = []
    J = np.zeros((K, K * p))
    J[:K, :K] = np.eye(K)
    
    C_power = np.eye(K * p)
    for h in range(horizon):
        psi_h = J @ C_power @ J.T
        psi.append(psi_h)
        C_power = C_power @ C
    return psi


# ----------------------------------------------
# Generalized FEVD (Pesaran–Shin, 1998)
# ----------------------------------------------

def generalized_fevd(coefs: np.ndarray, sigma: np.ndarray,
                     horizon: int) -> np.ndarray:
    """
    Compute the Generalized Forecast Error Variance Decomposition.
    
    Parameters
    ----------
    coefs : ndarray of shape (p, K, K)
        VAR lag coefficient matrices.
    sigma : ndarray of shape (K, K)
        Residual covariance matrix.
    horizon : int
        Forecast horizon H.
    
    Returns
    -------
    theta : ndarray of shape (K, K)
        Un-normalized GFEVD matrix.  θ_{ij} = share of variable i's
        H-step forecast-error variance attributable to shocks in variable j.
    """
    K = sigma.shape[0]
    psi = ma_coefficients(coefs, horizon)
    sigma_diag = np.diag(sigma)

    theta = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            numerator = 0.0
            denominator = 0.0
            for h in range(horizon):
                # e_j' Σ e_i  (the (j,i) element of Σ, but we use e_i' Σ e_j)
                numerator += (psi[h][i, :] @ sigma[:, j]) ** 2
                denominator += psi[h][i, :] @ sigma @ psi[h][i, :]
            theta[i, j] = (numerator / sigma_diag[j]) / denominator

    return theta


def normalize_fevd(theta: np.ndarray) -> np.ndarray:
    """Row-normalize the GFEVD matrix so each row sums to 1 (or 100%)."""
    row_sums = theta.sum(axis=1, keepdims=True)
    return theta / row_sums


# ----------------------------------------------
# Connectedness measures (Diebold-Yılmaz)
# ----------------------------------------------

def connectedness_measures(theta_norm: np.ndarray,
                           labels: list = None) -> dict:
    """
    Compute all connectedness measures from a row-normalized GFEVD matrix.
    
    Parameters
    ----------
    theta_norm : ndarray of shape (K, K)
        Row-normalized GFEVD (each row sums to 1).
    labels : list of str, optional
        Names of the K variables.
    
    Returns
    -------
    dict with keys:
        'tci'      : float   - Total Connectedness Index (%)
        'from_i'   : array   - Directional FROM others (%)
        'to_i'     : array   - Directional TO others (%)
        'net_i'    : array   - Net directional (TO - FROM) (%)
        'pairwise' : ndarray - Pairwise net connectedness (K×K)
        'table'    : DataFrame - Full connectedness table
    """
    K = theta_norm.shape[0]
    theta_pct = theta_norm * 100.0

    # Own shares on diagonal
    own = np.diag(theta_pct)

    # FROM_i = sum of off-diagonal elements in row i
    from_i = theta_pct.sum(axis=1) - own   # shape (K,)

    # TO_i = sum of off-diagonal elements in column i
    to_i = theta_pct.sum(axis=0) - own      # shape (K,)

    # Net_i = TO_i - FROM_i
    net_i = to_i - from_i

    # TCI = average of FROM (or TO), both sum to the same total
    tci = from_i.sum() / K

    # Pairwise net connectedness: C_{ij} = θ_ji - θ_ij  (net from i to j)
    pairwise = theta_pct.T - theta_pct

    # Build summary table
    if labels is None:
        labels = [f"Var{i+1}" for i in range(K)]

    table = pd.DataFrame(theta_pct, index=labels, columns=labels)
    table["FROM"] = from_i
    to_row = list(to_i) + [from_i.sum()]
    table.loc["TO"] = to_row
    net_row = list(net_i) + [np.nan]
    table.loc["Net"] = net_row

    return {
        "tci": tci,
        "from_i": from_i,
        "to_i": to_i,
        "net_i": net_i,
        "pairwise": pairwise,
        "table": table,
    }


# ----------------------------------------------
# Rolling-window connectedness
# ----------------------------------------------

def rolling_connectedness(data: pd.DataFrame, window: int,
                          horizon: int, max_lag: int = 10,
                          ic: str = "bic",
                          logger=None) -> pd.DataFrame:
    """
    Compute rolling-window TCI and directional connectedness.
    
    Parameters
    ----------
    data : DataFrame
        Columns = volatility series for each market, index = DatetimeIndex.
    window : int
        Rolling-window size.
    horizon : int
        GFEVD forecast horizon.
    max_lag : int
        Maximum VAR lag for IC selection.
    ic : str
        Information criterion: 'aic', 'bic', or 'hqic'.
    logger : Logger, optional
    
    Returns
    -------
    results : DataFrame
        Columns: date, TCI, FROM_<country>, TO_<country>, Net_<country>, lag_selected
    """
    from statsmodels.tsa.api import VAR

    labels = list(data.columns)
    K = len(labels)
    n = len(data)
    records = []

    for end in range(window, n + 1):
        start = end - window
        subset = data.iloc[start:end]
        date = data.index[end - 1]

        try:
            model = VAR(subset)
            lag_result = model.select_order(maxlags=min(max_lag, window // K - 2))
            selected_lag = getattr(lag_result, ic)
            if selected_lag is None or selected_lag == 0:
                selected_lag = 1

            # Fit model with selected lag
            result = model.fit(selected_lag)

            # Check stability (statsmodels result.is_stable() returns True if stable)
            if not result.is_stable():
                # Try fallback to smaller lag order if unstable
                stable_found = False
                for lag_fb in range(selected_lag - 1, 0, -1):
                    res_fb = model.fit(lag_fb)
                    if res_fb.is_stable():
                        result = res_fb
                        selected_lag = lag_fb
                        stable_found = True
                        break
                if not stable_found:
                    if logger:
                        logger.warning(f"Unstable VAR at {date}, skipping.")
                    continue

            # Extract coefficient matrices: shape (p, K, K)
            coefs = np.array(result.coefs)  # (p, K, K)
            sigma = np.array(result.sigma_u)  # (K, K)

            # GFEVD
            theta = generalized_fevd(coefs, sigma, horizon)
            theta_norm = normalize_fevd(theta)

            # Connectedness
            cm = connectedness_measures(theta_norm, labels)

            record = {"date": date, "TCI": cm["tci"], "lag_selected": selected_lag}
            for i, lbl in enumerate(labels):
                record[f"FROM_{lbl}"] = cm["from_i"][i]
                record[f"TO_{lbl}"] = cm["to_i"][i]
                record[f"Net_{lbl}"] = cm["net_i"][i]
            records.append(record)

        except Exception as e:
            if logger:
                logger.warning(f"Window ending {date}: {e}")
            continue

        if logger and (end - window) % 500 == 0:
            logger.info(f"  Rolling window {end - window + 1}/{n - window + 1} - date {date}")

    return pd.DataFrame(records)


# ----------------------------------------------
# Plotting helpers
# ----------------------------------------------

def setup_plot_style():
    """Apply a clean publication-quality matplotlib style."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.figsize": (12, 6),
        "figure.dpi": 150,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
    })


def save_figure(fig, name: str, out_dir: Path = None):
    """Save a matplotlib figure to the outputs/figures directory."""
    if out_dir is None:
        # Import here to avoid circular dependency
        import config
        out_dir = config.OUT_FIGURES
    filepath = out_dir / f"{name}.png"
    fig.savefig(filepath, bbox_inches="tight", dpi=150)
    print(f"  Saved figure -> {filepath}")
