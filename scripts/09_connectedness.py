"""
Stage 9: Rolling-window connectedness measures.

From the normalized GFEVD, calculate over every 250-day rolling window:
  - Total Connectedness Index (TCI)
  - Directional connectedness received "FROM" others
  - Directional connectedness transmitted "TO" others
  - Net directional connectedness
  - Pairwise directional and net pairwise connectedness

Baseline: 250-day rolling window, 10-day forecast horizon, BIC lag selection.
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

logger = setup_logger("09_connectedness")


def plot_tci(rolling_df: pd.DataFrame, title: str, filename: str):
    """Plot the Total Connectedness Index over time."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(rolling_df["date"], rolling_df["TCI"], color="#1f77b4",
            linewidth=1.0)
    ax.fill_between(rolling_df["date"], rolling_df["TCI"],
                    alpha=0.15, color="#1f77b4")
    ax.set_ylabel("TCI (%)")
    ax.set_title(title)
    ax.set_ylim(bottom=0)

    # Add mean line
    mean_tci = rolling_df["TCI"].mean()
    ax.axhline(y=mean_tci, color="red", linestyle="--", linewidth=0.8,
               label=f"Mean = {mean_tci:.1f}%")
    ax.legend()

    plt.tight_layout()
    save_figure(fig, filename)
    plt.close(fig)


def plot_net_connectedness(rolling_df: pd.DataFrame, title: str,
                            filename: str):
    """Plot net directional connectedness for each market in a 3x2 grid layout."""
    setup_plot_style()
    net_cols = [c for c in rolling_df.columns if c.startswith("Net_")]
    countries = [c.replace("Net_", "") for c in net_cols]

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    axes_flat = axes.flatten()

    colors = plt.cm.tab10(np.linspace(0, 1, len(countries)))
    dates = rolling_df["date"]

    for i, (col, country) in enumerate(zip(net_cols, countries)):
        ax = axes_flat[i]
        values = rolling_df[col]

        ax.fill_between(dates, values, where=values >= 0,
                        color=colors[i], alpha=0.4, label="Net transmitter")
        ax.fill_between(dates, values, where=values < 0,
                        color="gray", alpha=0.3, label="Net receiver")
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.set_title(country, fontsize=12)
        ax.set_ylabel("Net (%)")

    fig.suptitle(title, fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, filename)
    plt.close(fig)


def plot_from_to(rolling_df: pd.DataFrame, direction: str,
                  title: str, filename: str):
    """Plot FROM or TO directional connectedness."""
    setup_plot_style()
    prefix = f"{direction}_"
    cols = [c for c in rolling_df.columns if c.startswith(prefix)]
    countries = [c.replace(prefix, "") for c in cols]

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(countries)))

    for i, (col, country) in enumerate(zip(cols, countries)):
        ax.plot(rolling_df["date"], rolling_df[col],
                label=country, color=colors[i], linewidth=0.8)

    ax.set_ylabel(f"{direction} (%)")
    ax.set_title(title)
    ax.legend(loc="upper right", ncol=3)
    plt.tight_layout()
    save_figure(fig, filename)
    plt.close(fig)


def main():
    logger.info("=" * 60)
    logger.info("Stage 9: Rolling-window connectedness")
    logger.info("=" * 60)

    # Process baseline configuration
    configs = [
        ("vol_parkinson", "intersection", "panel_vol_parkinson_intersection.csv"),
    ]

    # Also process squared returns if available
    sq_path = config.DATA_PROC / "panel_vol_squared_intersection.csv"
    if sq_path.exists():
        configs.append(
            ("vol_squared", "intersection", "panel_vol_squared_intersection.csv"))

    for measure, sync, fname in configs:
        panel_path = config.DATA_PROC / fname
        if not panel_path.exists():
            logger.info(f"  {fname} not found - skipping")
            continue

        panel = pd.read_csv(panel_path, index_col=0, parse_dates=True)
        label = f"{measure}_{sync}"

        logger.info(f"\n{'-' * 40}")
        logger.info(f"Rolling connectedness: {label}")
        logger.info(f"  Window: {config.ROLLING_WINDOW}, "
                    f"Horizon: {config.FORECAST_HORIZON}")

        # Compute rolling connectedness
        rolling_df = rolling_connectedness(
            data=panel,
            window=config.ROLLING_WINDOW,
            horizon=config.FORECAST_HORIZON,
            max_lag=config.VAR_MAX_LAG,
            ic=config.VAR_IC,
            logger=logger,
        )

        if rolling_df.empty:
            logger.warning("  No rolling results - check data.")
            continue

        rolling_df["date"] = pd.to_datetime(rolling_df["date"])

        # Save results
        out_path = config.OUT_RESULTS / f"rolling_connectedness_{label}.csv"
        rolling_df.to_csv(out_path, index=False)
        logger.info(f"  Saved -> {out_path}")

        # Summary
        logger.info(f"  TCI range: {rolling_df['TCI'].min():.2f}% - "
                    f"{rolling_df['TCI'].max():.2f}%")
        logger.info(f"  TCI mean:  {rolling_df['TCI'].mean():.2f}%")

        # Titles matching publication requirements
        if measure == "vol_parkinson" and sync == "intersection":
            tci_title = "Time-Varying Total Volatility Connectedness"
            net_title = "Net Directional Volatility Connectedness by Market"
        else:
            tci_title = f"Time-Varying Total Volatility Connectedness ({measure})"
            net_title = f"Net Directional Volatility Connectedness ({measure})"

        # Plots
        plot_tci(rolling_df, tci_title, f"tci_rolling_{label}")

        plot_net_connectedness(rolling_df, net_title, f"net_connectedness_{label}")

        plot_from_to(rolling_df, "FROM",
                     f"Directional FROM Others - {measure} ({sync})",
                     f"from_connectedness_{label}")

        plot_from_to(rolling_df, "TO",
                     f"Directional TO Others - {measure} ({sync})",
                     f"to_connectedness_{label}")

    logger.info(f"\n{'-' * 60}")
    logger.info("Stage 9 complete.")


if __name__ == "__main__":
    main()
