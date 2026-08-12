"""
Stage 10: Shock and contagion analysis.

For each shock event:
  - Define event windows (shock period vs. tranquil benchmark)
  - Calculate mean TCI during shock and tranquil periods
  - Compute difference in means and medians
  - Bootstrap confidence intervals
  - HAC-standard-error regression

Regression: TCI_t = α + β₁·GPR_t + β₂·VIX_t + β₃·ΔOil_t + β₄·ΔDGS2_t + β₅·ΔDollar_t + ε_t

Evidence of contagion requires a significant increase in connectedness.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from scripts.utils import setup_logger, setup_plot_style, save_figure

logger = setup_logger("10_shock")


# ----------------------------------------------
# Default event windows (from literature)
# ----------------------------------------------

DEFAULT_EVENTS = [
    {
        "event_name": "European Debt Crisis",
        "event_date": "2011-07-01",
        "shock_start": "2011-07-01",
        "shock_end": "2012-06-30",
        "tranquil_start": "2011-03-30",
        "tranquil_end": "2011-06-30",
        "source_for_dates": "Literature (Diebold-Yilmaz 2014); Note: 49 rolling TCI obs pre-shock",
    },
    {
        "event_name": "Taper Tantrum",
        "event_date": "2013-05-22",
        "shock_start": "2013-05-22",
        "shock_end": "2013-09-30",
        "tranquil_start": "2013-01-02",
        "tranquil_end": "2013-05-21",
        "source_for_dates": "Bernanke testimony, May 2013",
    },
    {
        "event_name": "China Stock Market Crash",
        "event_date": "2015-06-12",
        "shock_start": "2015-06-12",
        "shock_end": "2016-02-29",
        "tranquil_start": "2014-12-01",
        "tranquil_end": "2015-06-11",
        "source_for_dates": "SSE Composite crash",
    },
    {
        "event_name": "US-China Trade War Escalation",
        "event_date": "2018-03-22",
        "shock_start": "2018-03-22",
        "shock_end": "2018-12-31",
        "tranquil_start": "2017-09-01",
        "tranquil_end": "2018-03-21",
        "source_for_dates": "Trump tariff announcement",
    },
    {
        "event_name": "COVID-19 Pandemic",
        "event_date": "2020-01-30",
        "shock_start": "2020-01-30",
        "shock_end": "2020-06-30",
        "tranquil_start": "2019-07-01",
        "tranquil_end": "2020-01-29",
        "source_for_dates": "WHO PHEIC declaration",
    },
    {
        "event_name": "Russia-Ukraine War",
        "event_date": "2022-02-24",
        "shock_start": "2022-02-24",
        "shock_end": "2022-05-31",
        "tranquil_start": "2021-09-01",
        "tranquil_end": "2022-02-23",
        "source_for_dates": "Russian invasion of Ukraine (non-overlapping window)",
    },
    {
        "event_name": "Global Monetary Tightening 2022",
        "event_date": "2022-06-01",
        "shock_start": "2022-06-01",
        "shock_end": "2022-12-31",
        "tranquil_start": "2021-09-01",
        "tranquil_end": "2022-02-23",
        "source_for_dates": "Fed 75bps rate hike acceleration (non-overlapping window)",
    },
    {
        "event_name": "US Banking Crisis 2023",
        "event_date": "2023-03-10",
        "shock_start": "2023-03-10",
        "shock_end": "2023-05-31",
        "tranquil_start": "2022-12-01",
        "tranquil_end": "2023-03-09",
        "source_for_dates": "SVB collapse",
    },
]


def create_event_windows() -> pd.DataFrame:
    """Create the event_windows.csv deliverable."""
    events = pd.DataFrame(DEFAULT_EVENTS)
    for col in ["event_date", "shock_start", "shock_end",
                "tranquil_start", "tranquil_end"]:
        events[col] = pd.to_datetime(events[col])
    return events


def moving_block_bootstrap_mean_diff(shock_vals: np.ndarray, tranquil_vals: np.ndarray,
                                     block_size: int = 20, n_boot: int = 5000, ci: float = 0.95) -> dict:
    """
    Moving-block bootstrap confidence interval for difference in means (mean_shock - mean_tranquil).
    Accurately preserves time-series dependence within event windows.
    """
    observed_diff = shock_vals.mean() - tranquil_vals.mean()
    n_shock = len(shock_vals)
    n_tranquil = len(tranquil_vals)

    def draw_mbb_sample(arr: np.ndarray, target_len: int, b_size: int, rng: np.random.Generator) -> np.ndarray:
        n = len(arr)
        if n <= b_size:
            return rng.choice(arr, size=target_len, replace=True)
        n_blocks_needed = int(np.ceil(target_len / b_size))
        starts = rng.integers(0, n - b_size + 1, size=n_blocks_needed)
        sample_blocks = [arr[s:s + b_size] for s in starts]
        return np.concatenate(sample_blocks)[:target_len]

    boot_diffs = np.empty(n_boot)
    rng = np.random.default_rng(42)

    for b in range(n_boot):
        b_shock = draw_mbb_sample(shock_vals, n_shock, block_size, rng)
        b_tranquil = draw_mbb_sample(tranquil_vals, n_tranquil, block_size, rng)
        boot_diffs[b] = b_shock.mean() - b_tranquil.mean()

    alpha = (1 - ci) / 2
    ci_lower = np.percentile(boot_diffs, alpha * 100)
    ci_upper = np.percentile(boot_diffs, (1 - alpha) * 100)
    p_value = np.mean(boot_diffs <= 0) if observed_diff > 0 else np.mean(boot_diffs >= 0)

    return {
        "observed_diff": observed_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_value,
        "significant": (ci_lower > 0) if observed_diff > 0 else (ci_upper < 0),
    }


def event_analysis(tci_series: pd.Series, events_df: pd.DataFrame,
                    net_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    For each event, compare TCI during shock vs. tranquil periods using Moving-Block Bootstrap.
    """
    logger.info("Event-window analysis with Moving-Block Bootstrap ...")

    results = []
    for _, event in events_df.iterrows():
        name = event["event_name"]

        shock_mask = (tci_series.index >= event["shock_start"]) & \
                     (tci_series.index <= event["shock_end"])
        tranquil_mask = (tci_series.index >= event["tranquil_start"]) & \
                        (tci_series.index <= event["tranquil_end"])

        shock_vals = tci_series[shock_mask].dropna().values
        tranquil_vals = tci_series[tranquil_mask].dropna().values

        if len(shock_vals) < 5 or len(tranquil_vals) < 5:
            logger.warning(f"  {name}: insufficient data "
                          f"(shock={len(shock_vals)}, tranquil={len(tranquil_vals)})")
            continue

        shock_mean = shock_vals.mean()
        tranquil_mean = tranquil_vals.mean()
        diff_mean = shock_mean - tranquil_mean
        diff_median = np.median(shock_vals) - np.median(tranquil_vals)

        t_stat, t_pval = stats.ttest_ind(shock_vals, tranquil_vals)
        boot = moving_block_bootstrap_mean_diff(shock_vals, tranquil_vals)

        result = {
            "event_name": name,
            "shock_n": len(shock_vals),
            "tranquil_n": len(tranquil_vals),
            "shock_mean_tci": round(shock_mean, 2),
            "tranquil_mean_tci": round(tranquil_mean, 2),
            "diff_mean": round(diff_mean, 2),
            "diff_median": round(diff_median, 2),
            "t_stat": round(t_stat, 3),
            "t_pval": round(t_pval, 4),
            "boot_ci_lower": round(boot["ci_lower"], 2),
            "boot_ci_upper": round(boot["ci_upper"], 2),
            "boot_pval": round(boot["p_value"], 4),
            "contagion": "Yes" if (diff_mean > 0 and boot["significant"]) else "No",
        }

        if net_df is not None:
            for country in config.COUNTRY_ORDER:
                col = f"Net_{country}"
                if col in net_df.columns:
                    net_shock = net_df.loc[shock_mask, col].mean()
                    net_tranquil = net_df.loc[tranquil_mask, col].mean()
                    result[f"net_change_{country}"] = round(
                        net_shock - net_tranquil, 2)

        results.append(result)
        sign = "↑ CONTAGION" if result["contagion"] == "Yes" else "-"
        logger.info(f"  {name}: Δ TCI = {diff_mean:+.2f}% "
                    f"CI=[{boot['ci_lower']:.2f}, {boot['ci_upper']:.2f}] {sign}")

    return pd.DataFrame(results)


def hac_regression(tci_series: pd.Series, global_df: pd.DataFrame) -> dict:
    """
    OLS regression of TCI on global factors with HAC standard errors.
    Subsamples month-end observations to eliminate 249-day rolling-window dependence.
    Equation: TCI_t = α + β₁·GPR_t + β₂·VIX_t + β₃·ΔOil_t + β₄·ΔDGS2_t + β₅·ΔDollar_t + ε_t
    """
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools import add_constant

    logger.info("HAC regression (month-end subsampled): TCI ~ global factors ...")

    merged = pd.DataFrame({"TCI": tci_series})

    # Add global factors
    for src_col, label in [("VIX", "VIX"), ("d_Brent", "ΔOil"), ("d_DGS2", "ΔDGS2"),
                           ("d_DollarIdx", "ΔDollar"), ("d_SP500", "ΔSP500")]:
        if src_col in global_df.columns:
            merged[label] = global_df[src_col]

    # Search for GPR in global_df or raw files
    gpr_found = False
    for col in ["GPRD", "GPR", "GPRD_ACT", "GPRD_THREAT", "gpr"]:
        if col in global_df.columns:
            merged["GPR"] = global_df[col]
            gpr_found = True
            break

    if not gpr_found:
        gpr_path = config.find_file("gpr_daily.csv", "data_gpr_daily(till_aug_10).csv", "global_daily_raw.csv")
        if gpr_path.exists():
            try:
                gpr_raw = pd.read_csv(gpr_path)
                if "DAY" in gpr_raw.columns and "GPRD" in gpr_raw.columns:
                    gpr_raw["date"] = pd.to_datetime(gpr_raw["DAY"].astype(str), format="%Y%m%d", errors="coerce")
                    gpr_raw = gpr_raw.set_index("date")
                    merged["GPR"] = gpr_raw["GPRD"]
            except Exception:
                pass

    merged = merged.dropna()

    if len(merged) < 30:
        logger.warning(f"  Insufficient observations ({len(merged)}) for regression")
        return {}

    # Subsample to month-end observations to eliminate rolling-window 249-day overlap
    monthly_merged = merged.resample("ME").last().dropna()
    logger.info(f"  Month-end subsampled observations: {len(monthly_merged)} months")

    y = monthly_merged["TCI"]
    x_cols = [c for c in monthly_merged.columns if c != "TCI"]

    if not x_cols:
        logger.warning("  No explanatory variables available for regression")
        return {}

    X = add_constant(monthly_merged[x_cols])

    # OLS with HAC (Newey-West) standard errors using 12-month maxlags
    nw_lags = 12
    model = OLS(y, X)
    result = model.fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags})

    logger.info(f"\n{result.summary()}")

    coef_table = pd.DataFrame({
        "coefficient": result.params,
        "std_error": result.bse,
        "t_stat": result.tvalues,
        "p_value": result.pvalues,
    })

    out_table = config.OUT_TABLES / "hac_regression_coefficients.csv"
    coef_table.to_csv(out_table)
    logger.info(f"  Saved HAC regression table -> {out_table}")

    return {
        "result": result,
        "coefficients": coef_table,
        "r_squared": result.rsquared,
        "adj_r_squared": result.rsquared_adj,
        "nobs": result.nobs,
        "nw_lags": nw_lags,
    }


def main():
    logger.info("=" * 60)
    logger.info("Stage 10: Shock and contagion analysis")
    logger.info("=" * 60)

    # 1. Create event windows
    events = create_event_windows()
    event_path = config.DELIVERABLES / "event_windows.csv"
    events.to_csv(event_path, index=False)
    logger.info(f"  Saved event windows -> {event_path}")
    logger.info(f"  {len(events)} events defined")

    # 2. Load rolling connectedness
    rolling_path = config.OUT_RESULTS / "rolling_connectedness_vol_parkinson_intersection.csv"
    if not rolling_path.exists():
        # Try squared returns
        rolling_path = config.OUT_RESULTS / "rolling_connectedness_vol_squared_intersection.csv"

    if not rolling_path.exists():
        logger.error("  Rolling connectedness not found. Run Stage 9 first.")
        sys.exit(1)

    rolling_df = pd.read_csv(rolling_path, parse_dates=["date"])
    rolling_df = rolling_df.set_index("date")

    tci_series = rolling_df["TCI"]

    # 3. Event-window analysis
    event_results = event_analysis(tci_series, events, rolling_df)
    if not event_results.empty:
        out_path = config.OUT_TABLES / "shock_analysis_results.csv"
        event_results.to_csv(out_path, index=False)
        logger.info(f"  Saved shock analysis -> {out_path}")

        # Summary
        n_contagion = (event_results["contagion"] == "Yes").sum()
        logger.info(f"\n  Contagion detected in {n_contagion}/{len(event_results)} events")

    # 4. HAC regression
    global_path = config.DATA_PROC / "global_returns.csv"
    if global_path.exists():
        global_df = pd.read_csv(global_path, index_col=0, parse_dates=True)
        reg_results = hac_regression(tci_series, global_df)

        if reg_results:
            coef_path = config.OUT_TABLES / "hac_regression_coefficients.csv"
            reg_results["coefficients"].to_csv(coef_path)
            logger.info(f"  Saved regression coefficients -> {coef_path}")
    else:
        logger.info("  Global returns not found - skipping HAC regression. "
                    "Run Stage 2 first.")

    logger.info(f"\n{'-' * 60}")
    logger.info("Stage 10 complete.")


if __name__ == "__main__":
    main()
