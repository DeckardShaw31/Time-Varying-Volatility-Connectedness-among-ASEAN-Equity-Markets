"""
Stage 7: Descriptive statistics and plots.

Computes for returns and volatility:
  - Number of observations, mean, median, std, min, max
  - Skewness, excess kurtosis
  - Jarque-Bera normality test
  - Augmented Dickey-Fuller test
  - Phillips-Perron (via KPSS as proxy) test
  - Correlation matrices
  - Autocorrelation function plots
  - Volatility and return time-series plots

Produces manuscript Tables 3-4 and Figure 1.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import config
from scripts.utils import setup_logger, setup_plot_style, save_figure

logger = setup_logger("07_descriptive")


def compute_summary_statistics(panel: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Compute comprehensive summary statistics for each column of a panel.
    
    Returns a DataFrame with rows = statistics, columns = markets/variables.
    """
    logger.info(f"Computing summary statistics ({label}) ...")

    records = {}
    for col in panel.columns:
        s = panel[col].dropna()

        # Basic stats
        n = len(s)
        mean = s.mean()
        median = s.median()
        std = s.std()
        minimum = s.min()
        maximum = s.max()
        skew = s.skew()
        kurt = s.kurtosis()  # excess kurtosis

        # Jarque-Bera test
        jb_stat, jb_pval = stats.jarque_bera(s)

        # ADF test (with constant)
        try:
            adf_stat, adf_pval, _, _, _, _ = adfuller(s, autolag="AIC")
        except Exception:
            adf_stat, adf_pval = np.nan, np.nan

        # KPSS test (level stationarity)
        try:
            kpss_stat, kpss_pval, _, _ = kpss(s, regression="c", nlags="auto")
        except Exception:
            kpss_stat, kpss_pval = np.nan, np.nan

        records[col] = {
            "N": n,
            "Mean": round(mean, 4),
            "Median": round(median, 4),
            "Std Dev": round(std, 4),
            "Min": round(minimum, 4),
            "Max": round(maximum, 4),
            "Skewness": round(skew, 4),
            "Excess Kurtosis": round(kurt, 4),
            "JB Stat": round(jb_stat, 2),
            "JB p-value": round(jb_pval, 4),
            "ADF Stat": round(adf_stat, 4) if not np.isnan(adf_stat) else np.nan,
            "ADF p-value": round(adf_pval, 4) if not np.isnan(adf_pval) else np.nan,
            "KPSS Stat": round(kpss_stat, 4) if not np.isnan(kpss_stat) else np.nan,
            "KPSS p-value": round(kpss_pval, 4) if not np.isnan(kpss_pval) else np.nan,
        }

    result = pd.DataFrame(records)
    logger.info(f"  [OK] {len(result.columns)} variables analyzed")
    return result


def compute_correlation_matrix(panel: pd.DataFrame, label: str) -> pd.DataFrame:
    """Compute pairwise correlation matrix."""
    logger.info(f"Computing correlation matrix ({label}) ...")
    corr = panel.corr()
    return corr


def plot_return_series(returns_df: pd.DataFrame, label: str):
    """Plot time-series of returns for all ASEAN markets."""
    setup_plot_style()
    n_countries = len(config.COUNTRY_ORDER)
    fig, axes = plt.subplots(n_countries, 1, figsize=(14, 3 * n_countries),
                              sharex=True)

    colors = plt.cm.Set2(np.linspace(0, 1, n_countries))

    for i, country in enumerate(config.COUNTRY_ORDER):
        sub = returns_df[returns_df["country"] == country].sort_values("date")
        if sub.empty:
            continue
        ax = axes[i] if n_countries > 1 else axes
        ax.plot(sub["date"], sub["return_lcu"], color=colors[i],
                linewidth=0.5, alpha=0.8)
        ax.set_ylabel(country)
        ax.axhline(y=0, color="gray", linewidth=0.5)

    axes[0].set_title(f"Daily Returns - ASEAN Markets ({label})")
    plt.xlabel("Date")
    plt.tight_layout()
    save_figure(fig, f"returns_timeseries_{label}")
    plt.close(fig)


def plot_volatility_series(vol_df: pd.DataFrame, measure: str, label: str):
    """Plot time-series of volatility for all ASEAN markets."""
    setup_plot_style()
    n_countries = len(config.COUNTRY_ORDER)
    fig, axes = plt.subplots(n_countries, 1, figsize=(14, 3 * n_countries),
                              sharex=True)

    colors = plt.cm.Set1(np.linspace(0, 1, n_countries))

    for i, country in enumerate(config.COUNTRY_ORDER):
        sub = vol_df[vol_df["country"] == country].sort_values("date")
        if sub.empty or measure not in sub.columns:
            continue
        ax = axes[i] if n_countries > 1 else axes
        ax.plot(sub["date"], sub[measure], color=colors[i],
                linewidth=0.5, alpha=0.8)
        ax.set_ylabel(country)

    measure_name = measure.replace("_", " ").title()
    axes[0].set_title(f"{measure_name} - ASEAN Markets ({label})")
    plt.xlabel("Date")
    plt.tight_layout()
    save_figure(fig, f"volatility_{measure}_{label}")
    plt.close(fig)


def plot_correlation_heatmap(corr: pd.DataFrame, title: str, filename: str):
    """Plot a correlation heatmap."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".3f", cmap="RdBu_r",
                center=0, square=True, linewidths=0.5, ax=ax,
                vmin=-1, vmax=1)
    ax.set_title(title)
    plt.tight_layout()
    save_figure(fig, filename)
    plt.close(fig)


def plot_acf(panel: pd.DataFrame, label: str, nlags: int = 30):
    """Plot autocorrelation functions for each series."""
    from statsmodels.graphics.tsaplots import plot_acf as sm_plot_acf

    setup_plot_style()
    n_cols = len(panel.columns)
    fig, axes = plt.subplots(n_cols, 1, figsize=(12, 3 * n_cols))

    for i, col in enumerate(panel.columns):
        ax = axes[i] if n_cols > 1 else axes
        s = panel[col].dropna()
        if len(s) > nlags + 1:
            sm_plot_acf(s, lags=nlags, ax=ax, title=f"ACF - {col}")
        else:
            ax.set_title(f"ACF - {col} (insufficient data)")

    plt.tight_layout()
    save_figure(fig, f"acf_{label}")
    plt.close(fig)


def main():
    logger.info("=" * 60)
    logger.info("Stage 7: Descriptive statistics and plots")
    logger.info("=" * 60)

    # -- Returns descriptive stats --
    for label, fname in [("intersection", "asean_returns_intersection.csv"),
                         ("weekly", "asean_returns_weekly.csv")]:
        path = config.DATA_PROC / fname
        if not path.exists():
            logger.info(f"  {fname} not found - skipping")
            continue

        df = pd.read_csv(path, parse_dates=["date"])

        # Build returns panel
        ret_panel = df.pivot_table(
            index="date", columns="country", values="return_lcu", aggfunc="first"
        )
        ret_panel = ret_panel[[c for c in config.COUNTRY_ORDER if c in ret_panel.columns]]
        ret_panel = ret_panel.dropna()

        # Summary statistics (Table 3)
        stats_table = compute_summary_statistics(ret_panel, f"returns_{label}")
        stats_path = config.OUT_TABLES / f"table_returns_stats_{label}.csv"
        stats_table.to_csv(stats_path)
        logger.info(f"  Saved -> {stats_path}")
        logger.info(f"\n{stats_table.to_string()}\n")

        # Correlation matrix
        corr = compute_correlation_matrix(ret_panel, f"returns_{label}")
        corr_path = config.OUT_TABLES / f"table_returns_corr_{label}.csv"
        corr.to_csv(corr_path)
        plot_correlation_heatmap(corr, f"Return Correlations ({label})",
                                 f"corr_returns_{label}")

        # Return time-series plot (Figure 1)
        plot_return_series(df, label)

        # ACF
        plot_acf(ret_panel, f"returns_{label}")

    # -- Volatility descriptive stats --
    for label in ["intersection", "weekly"]:
        vol_path = config.DATA_PROC / f"asean_volatility_{label}.csv"
        if not vol_path.exists():
            continue

        vol_df = pd.read_csv(vol_path, parse_dates=["date"])

        for measure in ["vol_parkinson", "vol_squared", "vol_absolute"]:
            if measure not in vol_df.columns:
                continue

            # Build panel
            vol_panel = vol_df.pivot_table(
                index="date", columns="country", values=measure, aggfunc="first"
            )
            vol_panel = vol_panel[[c for c in config.COUNTRY_ORDER
                                   if c in vol_panel.columns]]
            vol_panel = vol_panel.dropna()

            if vol_panel.empty:
                continue

            # Summary statistics (Table 4)
            stats_table = compute_summary_statistics(
                vol_panel, f"{measure}_{label}")
            stats_path = config.OUT_TABLES / f"table_{measure}_stats_{label}.csv"
            stats_table.to_csv(stats_path)
            logger.info(f"  Saved -> {stats_path}")

            # Correlation
            corr = compute_correlation_matrix(vol_panel, f"{measure}_{label}")
            corr_path = config.OUT_TABLES / f"table_{measure}_corr_{label}.csv"
            corr.to_csv(corr_path)
            plot_correlation_heatmap(
                corr, f"{measure} Correlations ({label})",
                f"corr_{measure}_{label}")

            # Volatility plot
            plot_volatility_series(vol_df, measure, label)

    logger.info(f"\n{'-' * 60}")
    logger.info("Stage 7 complete. Tables and figures saved.")


if __name__ == "__main__":
    main()
