"""
Stage 11: Robustness checks.

Repeat the connectedness model under alternative specifications:
  - Rolling windows: 200, 250, 300 days
  - Forecast horizons: 5, 10, 20 days
  - Alternative VAR lag orders
  - Volatility: Parkinson vs. squared returns
  - Currency: local-currency vs. USD returns
  - Frequency: daily intersection vs. weekly
  - GPR: conventional vs. AI-GPR
  - Alternative shock-window definitions
  - DCC-GARCH conditional correlations (if feasible)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from scripts.utils import (setup_logger, rolling_connectedness,
                            setup_plot_style, save_figure)

logger = setup_logger("11_robustness")


def run_robustness_grid(panels: dict, windows: list, horizons: list,
                         max_lag: int, ic: str) -> pd.DataFrame:
    """
    Run the rolling connectedness for each combination of parameters.
    
    Parameters
    ----------
    panels : dict
        Keys = (measure, sync), values = DataFrame panels
    windows : list of int
    horizons : list of int
    
    Returns
    -------
    summary : DataFrame with one row per configuration
    """
    results = []

    for (measure, sync), panel in panels.items():
        for window in windows:
            if window > len(panel):
                logger.info(f"  Skip {measure}/{sync}: window {window} > "
                           f"data length {len(panel)}")
                continue

            for horizon in horizons:
                label = f"{measure}_{sync}_w{window}_h{horizon}"
                logger.info(f"\n  Running: {label} ...")

                try:
                    rolling_df = rolling_connectedness(
                        data=panel,
                        window=window,
                        horizon=horizon,
                        max_lag=max_lag,
                        ic=ic,
                        logger=None,  # Suppress per-window logging
                    )

                    if rolling_df.empty:
                        continue

                    tci = rolling_df["TCI"]
                    result = {
                        "measure": measure,
                        "sync": sync,
                        "window": window,
                        "horizon": horizon,
                        "n_windows": len(rolling_df),
                        "tci_mean": round(tci.mean(), 2),
                        "tci_std": round(tci.std(), 2),
                        "tci_min": round(tci.min(), 2),
                        "tci_max": round(tci.max(), 2),
                        "tci_median": round(tci.median(), 2),
                    }
                    results.append(result)

                    # Save individual rolling results
                    out_path = config.OUT_RESULTS / f"robustness_{label}.csv"
                    rolling_df.to_csv(out_path, index=False)

                    logger.info(f"    TCI: mean={result['tci_mean']}, "
                               f"std={result['tci_std']}, "
                               f"range=[{result['tci_min']}, {result['tci_max']}]")

                except Exception as e:
                    logger.warning(f"    Failed: {e}")

    return pd.DataFrame(results)


def plot_robustness_comparison(summary: pd.DataFrame):
    """Plot comparison of TCI statistics across robustness specifications."""
    setup_plot_style()

    if summary.empty:
        return

    # Group by window size
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1. TCI mean by window size
    ax = axes[0]
    for horizon in summary["horizon"].unique():
        sub = summary[summary["horizon"] == horizon]
        for measure in sub["measure"].unique():
            msub = sub[sub["measure"] == measure]
            ax.plot(msub["window"], msub["tci_mean"],
                    marker="o", label=f"H={horizon}, {measure}")
    ax.set_xlabel("Window Size")
    ax.set_ylabel("Mean TCI (%)")
    ax.set_title("Mean TCI by Window Size")
    ax.legend(fontsize=8)

    # 2. TCI mean by horizon
    ax = axes[1]
    for window in summary["window"].unique():
        sub = summary[summary["window"] == window]
        for measure in sub["measure"].unique():
            msub = sub[sub["measure"] == measure]
            ax.plot(msub["horizon"], msub["tci_mean"],
                    marker="s", label=f"W={window}, {measure}")
    ax.set_xlabel("Forecast Horizon")
    ax.set_ylabel("Mean TCI (%)")
    ax.set_title("Mean TCI by Forecast Horizon")
    ax.legend(fontsize=8)

    # 3. TCI range comparison
    ax = axes[2]
    x = range(len(summary))
    labels_short = [f"{r['measure'][:3]}\nW{r['window']}\nH{r['horizon']}"
                    for _, r in summary.iterrows()]
    ax.bar(x, summary["tci_mean"], yerr=summary["tci_std"],
           capsize=3, alpha=0.7, color="steelblue")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels_short, fontsize=7, rotation=45)
    ax.set_ylabel("Mean TCI (%)")
    ax.set_title("TCI Across Specifications")

    plt.tight_layout()
    save_figure(fig, "robustness_comparison")
    plt.close(fig)


def run_garch_filtered_ewma_correlations(panel: pd.DataFrame, label: str) -> dict:
    """
    Estimate GARCH-filtered EWMA conditional correlations (GARCH(1,1) standardized residuals + EWMA smoothing).
    This is an honest, non-parametric alternative to multivariate DCC-GARCH.
    """
    try:
        from arch import arch_model
    except ImportError:
        logger.info("  arch package not available - skipping GARCH-filtered EWMA correlations")
        return {}

    logger.info(f"  GARCH-filtered EWMA conditional correlations estimation ({label}) ...")

    # Step 1: Fit univariate GARCH(1,1) for each series
    std_residuals = {}
    cond_vols = {}

    for col in panel.columns:
        try:
            am = arch_model(panel[col].dropna() * 100,
                           vol="Garch", p=1, q=1, mean="Constant",
                           rescale=True)
            res = am.fit(disp="off")
            std_residuals[col] = res.std_resid
            cond_vols[col] = res.conditional_volatility
            logger.info(f"    {col}: GARCH(1,1) fitted, "
                       f"ω={res.params.get('omega', np.nan):.4f}")
        except Exception as e:
            logger.warning(f"    {col}: GARCH failed - {e}")

    if len(std_residuals) < 2:
        return {}

    # Step 2: EWMA conditional correlations on standardized residuals
    std_df = pd.DataFrame(std_residuals).dropna()
    n = len(std_df)
    K = len(std_df.columns)

    lam = 0.94  # EWMA decay factor

    corr_series = {}
    countries = list(std_df.columns)

    for i in range(K):
        for j in range(i + 1, K):
            c1, c2 = countries[i], countries[j]
            z1 = std_df[c1].values
            z2 = std_df[c2].values

            cov_t = np.zeros(n)
            cov_t[0] = z1[0] * z2[0]
            var1_t = np.zeros(n)
            var2_t = np.zeros(n)
            var1_t[0] = z1[0] ** 2
            var2_t[0] = z2[0] ** 2

            for t in range(1, n):
                cov_t[t] = lam * cov_t[t-1] + (1 - lam) * z1[t] * z2[t]
                var1_t[t] = lam * var1_t[t-1] + (1 - lam) * z1[t] ** 2
                var2_t[t] = lam * var2_t[t-1] + (1 - lam) * z2[t] ** 2

            corr_t = cov_t / np.sqrt(var1_t * var2_t + 1e-10)
            corr_series[f"{c1}_{c2}"] = corr_t

    corr_df = pd.DataFrame(corr_series, index=std_df.index)
    out_path = config.OUT_RESULTS / f"garch_ewma_correlations_{label}.csv"
    corr_df.to_csv(out_path)
    logger.info(f"  Saved GARCH-filtered EWMA correlations -> {out_path}")

    return {"correlations": corr_df, "cond_vols": cond_vols}


def main():
    logger.info("=" * 60)
    logger.info("Stage 11: Robustness checks")
    logger.info("=" * 60)

    # Collect all available panels across volatility measures, currencies, and frequencies
    panels = {}

    measures = ["vol_parkinson", "vol_squared", "vol_squared_usd", "vol_absolute", "vol_absolute_usd"]
    for measure in measures:
        for sync in ["intersection", "weekly"]:
            fname = f"panel_{measure}_{sync}.csv"
            path = config.DATA_PROC / fname
            if path.exists():
                panel = pd.read_csv(path, index_col=0, parse_dates=True)
                panels[(measure, sync)] = panel
                logger.info(f"  Loaded panel: {fname} ({len(panel)} obs)")

    if not panels:
        logger.error("  No volatility panels found. Run Stages 4-6 first.")
        sys.exit(1)

    logger.info(f"\nRobustness grid execution:")
    logger.info(f"  Windows:  {config.ROBUSTNESS_WINDOWS}")
    logger.info(f"  Horizons: {config.ROBUSTNESS_HORIZONS}")
    logger.info(f"  Panels:   {list(panels.keys())}")

    summary = run_robustness_grid(
        panels=panels,
        windows=config.ROBUSTNESS_WINDOWS,
        horizons=config.ROBUSTNESS_HORIZONS,
        max_lag=config.VAR_MAX_LAG,
        ic=config.VAR_IC,
    )

    if not summary.empty:
        summary_path = config.OUT_TABLES / "robustness_summary.csv"
        summary.to_csv(summary_path, index=False)
        logger.info(f"\n  Saved robustness summary -> {summary_path}")
        logger.info(f"\n{summary.to_string()}")

        plot_robustness_comparison(summary)

    # GARCH-filtered EWMA conditional correlations (baseline panel)
    baseline_key = ("vol_parkinson", "intersection")
    if baseline_key not in panels:
        baseline_key = next(iter(panels.keys()))

    run_garch_filtered_ewma_correlations(
        panels[baseline_key], f"{baseline_key[0]}_{baseline_key[1]}"
    )

    logger.info(f"\n{'-' * 60}")
    logger.info("Stage 11 complete.")


if __name__ == "__main__":
    main()
