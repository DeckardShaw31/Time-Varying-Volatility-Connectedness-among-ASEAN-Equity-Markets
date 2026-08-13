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
    Uses econometrically equivalent parameters for weekly panels:
      - Daily: W in {200, 250, 300} trading days, H in {5, 10, 20} days
      - Weekly: W in {40, 50, 60} weeks, H in {1, 2, 4} weeks
    """
    results = []

    for (measure, sync), panel in panels.items():
        # Set equivalent time horizons based on frequency
        if sync == "weekly":
            effective_windows = [40, 50, 60]
            effective_horizons = [1, 2, 4]
        else:
            effective_windows = windows
            effective_horizons = horizons

        for window in effective_windows:
            if window > len(panel):
                logger.info(f"  Skip {measure}/{sync}: window {window} > "
                           f"data length {len(panel)}")
                continue

            for horizon in effective_horizons:
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
                    # Add net directional connectedness averages for each country
                    for country in config.COUNTRY_ORDER:
                        net_col = f"Net_{country}"
                        if net_col in rolling_df.columns:
                            result[f"mean_net_{country}"] = round(rolling_df[net_col].mean(), 2)

                    # Add percentage of windows where Vietnam is a net transmitter (Net_Vietnam > 0)
                    if "Net_Vietnam" in rolling_df.columns:
                        vn_trans_pct = (rolling_df["Net_Vietnam"] > 0).mean() * 100.0
                        result["share_Vietnam_net_transmitter"] = round(vn_trans_pct, 2)

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
    """Plot comparison of TCI statistics across robustness specifications cleanly without text/legend overlap."""
    setup_plot_style()

    if summary.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    measures = ["vol_parkinson", "vol_squared", "vol_squared_usd", "vol_absolute", "vol_absolute_usd"]
    m_labels = ["Parkinson", "Squared", "Squared USD", "Absolute", "Absolute USD"]

    # Filter to available measures
    avail_measures = [m for m in measures if m in summary["measure"].unique()]
    avail_labels = [m_labels[measures.index(m)] for m in avail_measures]

    # 1. Bar plot by measure & frequency
    ax1 = axes[0]
    x = np.arange(len(avail_measures))
    width = 0.35

    d_means = [summary[(summary["measure"] == m) & (summary["sync"] == "intersection")]["tci_mean"].mean() for m in avail_measures]
    d_stds = [summary[(summary["measure"] == m) & (summary["sync"] == "intersection")]["tci_mean"].std() for m in avail_measures]
    w_means = [summary[(summary["measure"] == m) & (summary["sync"] == "weekly")]["tci_mean"].mean() for m in avail_measures]
    w_stds = [summary[(summary["measure"] == m) & (summary["sync"] == "weekly")]["tci_mean"].std() for m in avail_measures]

    ax1.bar(x - width / 2, d_means, width, yerr=d_stds, label="Daily", capsize=3, color="#1f77b4", alpha=0.85)
    ax1.bar(x + width / 2, w_means, width, yerr=w_stds, label="Weekly", capsize=3, color="#ff7f0e", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(avail_labels, rotation=25, ha="right")
    ax1.set_ylabel("Mean TCI (%)")
    ax1.set_title("Mean TCI by Proxy & Frequency")
    ax1.legend(loc="upper right")

    # 2. Window size sensitivity (Daily)
    ax2 = axes[1]
    d_sub = summary[summary["sync"] == "intersection"]
    for m, m_lbl in zip(avail_measures, avail_labels):
        w_sub = d_sub[d_sub["measure"] == m].groupby("window")["tci_mean"].mean()
        if not w_sub.empty:
            ax2.plot(w_sub.index, w_sub.values, marker="o", label=m_lbl)
    ax2.set_xlabel("Window Size (Trading Days)")
    ax2.set_ylabel("Mean TCI (%)")
    ax2.set_title("Window Size Sensitivity (Daily)")
    ax2.set_xticks([200, 250, 300])
    ax2.legend(fontsize=8, loc="best")

    # 3. Forecast Horizon sensitivity (Daily)
    ax3 = axes[2]
    for m, m_lbl in zip(avail_measures, avail_labels):
        h_sub = d_sub[d_sub["measure"] == m].groupby("horizon")["tci_mean"].mean()
        if not h_sub.empty:
            ax3.plot(h_sub.index, h_sub.values, marker="s", label=m_lbl)
    ax3.set_xlabel("Forecast Horizon (Days)")
    ax3.set_ylabel("Mean TCI (%)")
    ax3.set_title("Forecast Horizon Sensitivity (Daily)")
    ax3.set_xticks([5, 10, 20])
    ax3.legend(fontsize=8, loc="best")

    plt.tight_layout()
    save_figure(fig, "robustness_comparison")
    plt.close(fig)


def run_garch_filtered_ewma_correlations(panel: pd.DataFrame, label: str) -> dict:
    """
    Estimate GARCH-filtered EWMA conditional market correlations (GARCH(1,1) standardized residuals + EWMA smoothing).
    Fits univariate GARCH(1,1) to return series to extract standardized innovations and compute conditional return correlations.
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

    # Alternative fixed VAR lag order checks (p = 1, 2, 3) on baseline panel
    baseline_key = ("vol_parkinson", "intersection")
    if baseline_key not in panels:
        baseline_key = next(iter(panels.keys()))

    b_panel = panels[baseline_key]
    logger.info(f"\n  Running alternative fixed VAR lag orders (p=1, 2, 3) on {baseline_key} ...")
    lag_results = []
    for p in [1, 2, 3]:
        try:
            r_df = rolling_connectedness(
                data=b_panel, window=250, horizon=10,
                fixed_lag=p, logger=None
            )
            if not r_df.empty:
                lag_results.append({
                    "lag_order": p,
                    "tci_mean": round(r_df["TCI"].mean(), 2),
                    "tci_std": round(r_df["TCI"].std(), 2),
                    "tci_min": round(r_df["TCI"].min(), 2),
                    "tci_max": round(r_df["TCI"].max(), 2),
                    "tci_median": round(r_df["TCI"].median(), 2),
                })
        except Exception as e:
            logger.warning(f"    Lag p={p} failed: {e}")

    if lag_results:
        lag_df = pd.DataFrame(lag_results)
        lag_path = config.OUT_TABLES / "robustness_alternative_lags.csv"
        lag_df.to_csv(lag_path, index=False)
        logger.info(f"  Saved alternative VAR lags -> {lag_path}")

    # GARCH-filtered EWMA conditional correlations (fitted on daily return series in %)
    ret_path = config.DATA_PROC / "asean_returns_intersection.csv"
    if ret_path.exists():
        ret_df = pd.read_csv(ret_path, parse_dates=["date"])
        ret_panel = ret_df.pivot_table(index="date", columns="country", values="return_lcu").dropna()
        ret_panel = ret_panel[[c for c in config.COUNTRY_ORDER if c in ret_panel.columns]]
        run_garch_filtered_ewma_correlations(ret_panel, "returns_intersection")
    else:
        run_garch_filtered_ewma_correlations(panels[baseline_key], f"{baseline_key[0]}_{baseline_key[1]}")

    logger.info(f"\n{'-' * 60}")
    logger.info("Stage 11 complete.")


if __name__ == "__main__":
    main()
