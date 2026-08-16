"""
Stage 8: VAR model estimation and Generalized FEVD.

  - Construct 6-variable volatility vector x_t = (v_Indonesia, ..., v_Vietnam)'
  - Select lag order via AIC, BIC, HQIC (max 10)
  - Check VAR stability (eigenvalues < 1)
  - Residual autocorrelation diagnostics
  - Compute MA(∞) representation
  - Calculate generalized forecast-error variance decompositions
  - Normalize every variance-decomposition row

Baseline: BIC lag selection, 10-day forecast horizon
"""

import sys
import warnings
from pathlib import Path

# Suppress statsmodels warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="statsmodels")
try:
    from statsmodels.tools.sm_exceptions import ValueWarning
    warnings.filterwarnings("ignore", category=ValueWarning)
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from scripts.utils import (setup_logger, generalized_fevd, normalize_fevd,
                            connectedness_measures, setup_plot_style, save_figure)

logger = setup_logger("08_var")


def select_and_report_lag(model, max_lag: int) -> dict:
    """
    Select lag order and report all information criteria.
    
    Returns dict with aic, bic, hqic lag selections.
    """
    logger.info(f"  Selecting lag order (max = {max_lag}) ...")

    K = model.neqs
    effective_max = min(max_lag, len(model.endog) // K - 2)
    if effective_max < 1:
        effective_max = 1

    lag_order = model.select_order(maxlags=effective_max)

    logger.info(f"    AIC  -> lag {lag_order.aic}")
    logger.info(f"    BIC  -> lag {lag_order.bic}")
    logger.info(f"    HQIC -> lag {lag_order.hqic}")

    return {
        "aic": lag_order.aic,
        "bic": lag_order.bic,
        "hqic": lag_order.hqic,
    }


def check_stability(result) -> bool:
    """Check VAR stability: statsmodels result.is_stable() returns True if stable."""
    is_stable = result.is_stable()
    min_root = np.min(np.abs(result.roots)) if len(result.roots) > 0 else 0.0

    logger.info(f"  Stability check: min |poly root| = {min_root:.6f}")
    if is_stable:
        logger.info(f"    [OK] VAR is stable")
    else:
        logger.warning(f"    [X] VAR is UNSTABLE")

    return is_stable


def residual_diagnostics(result, labels: list):
    """Run and report residual autocorrelation tests."""
    logger.info("  Residual diagnostics ...")

    # Durbin-Watson
    try:
        from statsmodels.stats.stattools import durbin_watson
        dw = durbin_watson(result.resid)
        for i, lbl in enumerate(labels):
            logger.info(f"    Durbin-Watson ({lbl}): {dw[i]:.4f}")
    except Exception as e:
        logger.warning(f"    Durbin-Watson failed: {e}")

    # Portmanteau test for residual autocorrelation
    try:
        test = result.test_whiteness(nlags=10, signif=0.05)
        logger.info(f"    Portmanteau test: stat = {test.test_statistic:.2f}, "
                    f"p-value = {test.pvalue:.4f}")
        if test.pvalue < 0.05:
            logger.warning("    [!] Residuals show significant autocorrelation")
        else:
            logger.info("    [OK] No significant residual autocorrelation at 5%")
    except Exception as e:
        logger.warning(f"    Portmanteau test failed: {e}")


def get_var_diagnostics(result, lag_orders: dict, labels: list, nlags: int = 10) -> pd.DataFrame:
    """
    Extract comprehensive VAR diagnostics into a structured DataFrame.
    """
    from statsmodels.stats.stattools import durbin_watson
    from scipy import linalg
    from scripts.utils import var_companion_matrix

    is_stable = result.is_stable()
    min_root = float(np.min(np.abs(result.roots))) if len(result.roots) > 0 else np.nan
    
    coefs = np.array(result.coefs)
    comp_mat = var_companion_matrix(coefs)
    eigvals = linalg.eigvals(comp_mat)
    max_eig = float(np.max(np.abs(eigvals))) if len(eigvals) > 0 else np.nan

    # Durbin-Watson per market
    dw_vals = durbin_watson(result.resid)
    dw_dict = {f"DW_{lbl}": round(float(dw_vals[i]), 4) for i, lbl in enumerate(labels)}

    # Portmanteau whiteness test
    try:
        pw_test = result.test_whiteness(nlags=nlags, signif=0.05)
        portmanteau_stat = float(pw_test.test_statistic)
        portmanteau_pval = float(pw_test.pvalue)
        portmanteau_df = int(pw_test.df) if hasattr(pw_test, "df") else int(nlags * len(labels)**2 - result.k_ar * len(labels)**2)
    except Exception as e:
        logger.warning(f"Portmanteau calculation note: {e}")
        portmanteau_stat = np.nan
        portmanteau_pval = np.nan
        portmanteau_df = np.nan

    diag_data = {
        "n_obs": int(result.nobs),
        "k_vars": int(result.neqs),
        "selected_lag": int(result.k_ar),
        "lag_aic": int(lag_orders.get("aic", np.nan)),
        "lag_bic": int(lag_orders.get("bic", np.nan)),
        "lag_hqic": int(lag_orders.get("hqic", np.nan)),
        "is_stable": bool(is_stable),
        "min_poly_root": round(min_root, 6),
        "max_eigenvalue": round(max_eig, 6),
        "portmanteau_stat": round(portmanteau_stat, 4),
        "portmanteau_df": portmanteau_df,
        "portmanteau_pvalue": round(portmanteau_pval, 6),
        **dw_dict
    }

    return pd.DataFrame([diag_data])


def estimate_full_sample_var(panel: pd.DataFrame, ic: str = "bic",
                              max_lag: int = 10,
                              horizon: int = 10) -> dict:
    """
    Estimate a full-sample VAR and compute the GFEVD connectedness table.
    
    Returns
    -------
    dict with keys: result, lag_orders, is_stable, theta, theta_norm,
                    connectedness, table, diagnostics
    """
    labels = list(panel.columns)
    logger.info(f"Full-sample VAR estimation on {labels}")
    logger.info(f"  Observations: {len(panel)}")

    model = VAR(panel)

    # Lag selection
    lag_orders = select_and_report_lag(model, max_lag)
    raw_selected = lag_orders.get(ic, 1)
    if raw_selected == 0:
        selected = 1
        logger.info(f"  {ic.upper()} selected raw lag 0; enforced minimum lag {selected} used for VAR estimation.")
    else:
        selected = raw_selected
        logger.info(f"  Using {ic.upper()} -> lag {selected}")

    # Estimate
    result = model.fit(selected)
    logger.info(f"  VAR({selected}) estimated with {result.nobs} observations")

    # Stability
    is_stable = check_stability(result)

    # Residual diagnostics
    residual_diagnostics(result, labels)

    # Full diagnostics dataframe
    diagnostics_df = get_var_diagnostics(result, lag_orders, labels, nlags=10)

    # GFEVD
    logger.info(f"  Computing GFEVD (horizon = {horizon}) ...")
    coefs = np.array(result.coefs)
    sigma = np.array(result.sigma_u)

    theta = generalized_fevd(coefs, sigma, horizon)
    theta_norm = normalize_fevd(theta)

    # Connectedness measures
    cm = connectedness_measures(theta_norm, labels)

    logger.info(f"\n  Full-sample connectedness table:")
    logger.info(f"\n{cm['table'].round(2).to_string()}")
    logger.info(f"\n  Total Connectedness Index (TCI): {cm['tci']:.2f}%")

    return {
        "result": result,
        "lag_orders": lag_orders,
        "is_stable": is_stable,
        "diagnostics": diagnostics_df,
        "theta": theta,
        "theta_norm": theta_norm,
        "connectedness": cm,
        "table": cm["table"],
    }


def plot_fevd_heatmap(theta_norm: np.ndarray, labels: list, title: str,
                       filename: str):
    """Plot the GFEVD matrix as a heatmap."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(9, 7))

    theta_pct = theta_norm * 100
    im = ax.imshow(theta_pct, cmap="YlOrRd", aspect="equal")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    # Annotate
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = theta_pct[i, j]
            color = "white" if val > 50 else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                    color=color, fontsize=9)

    ax.set_title(title)
    ax.set_xlabel("Shock source")
    ax.set_ylabel("Affected market")
    plt.colorbar(im, ax=ax, label="% of FEVD")
    plt.tight_layout()
    save_figure(fig, filename)
    plt.close(fig)


def main():
    logger.info("=" * 60)
    logger.info("Stage 8: VAR model estimation and GFEVD")
    logger.info("=" * 60)

    # Process baseline configuration
    baseline_panels = [
        ("vol_parkinson", "intersection", "panel_vol_parkinson_intersection.csv"),
        ("vol_squared", "intersection", "panel_vol_squared_intersection.csv"),
    ]

    for measure, sync, fname in baseline_panels:
        panel_path = config.DATA_PROC / fname
        if not panel_path.exists():
            logger.info(f"  {fname} not found - skipping")
            continue

        panel = pd.read_csv(panel_path, index_col=0, parse_dates=True)
        label = f"{measure}_{sync}"

        logger.info(f"\n{'-' * 40}")
        logger.info(f"Processing: {label}")

        results = estimate_full_sample_var(
            panel,
            ic=config.VAR_IC,
            max_lag=config.VAR_MAX_LAG,
            horizon=config.FORECAST_HORIZON,
        )

        # Save connectedness table
        table_path = config.OUT_TABLES / f"connectedness_table_{label}.csv"
        results["table"].to_csv(table_path)
        deliv_table_path = config.DELIVERABLES / f"connectedness_table_{label}.csv"
        results["table"].to_csv(deliv_table_path)
        logger.info(f"  Saved -> {table_path} and {deliv_table_path}")

        # Save VAR diagnostics
        diag_path = config.OUT_TABLES / f"var_diagnostics_{label}.csv"
        results["diagnostics"].to_csv(diag_path, index=False)
        deliv_diag_path = config.DELIVERABLES / f"var_diagnostics_{label}.csv"
        results["diagnostics"].to_csv(deliv_diag_path, index=False)
        logger.info(f"  Saved diagnostics -> {diag_path} and {deliv_diag_path}")

        # Save GFEVD heatmap
        if measure == "vol_parkinson" and sync == "intersection":
            heatmap_title = "Full-Sample Volatility Variance Decomposition"
        else:
            heatmap_title = f"Full-Sample Volatility Variance Decomposition ({measure})"

        plot_fevd_heatmap(
            results["theta_norm"],
            list(panel.columns),
            heatmap_title,
            f"gfevd_heatmap_{label}"
        )

        # Save lag order comparison
        lag_path = config.OUT_TABLES / f"lag_selection_{label}.csv"
        pd.DataFrame([results["lag_orders"]]).to_csv(lag_path, index=False)

    logger.info(f"\n{'-' * 60}")
    logger.info("Stage 8 complete.")


if __name__ == "__main__":
    main()
