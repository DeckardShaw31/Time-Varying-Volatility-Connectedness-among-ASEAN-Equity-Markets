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
                                     block_sizes: list = [10, 20], n_boot: int = 5000, ci: float = 0.95) -> dict:
    """
    Moving-block bootstrap confidence intervals for difference in means (mean_shock - mean_tranquil).
    Evaluates feasible block sizes (B=10, B=20) where B <= min(n_shock, n_tranquil)/2.
    Infeasible block sizes are skipped to avoid silent fallback to i.i.d. sampling.
    """
    observed_diff = shock_vals.mean() - tranquil_vals.mean()
    n_shock = len(shock_vals)
    n_tranquil = len(tranquil_vals)
    min_len = min(n_shock, n_tranquil)

    def draw_mbb_sample(arr: np.ndarray, target_len: int, b_size: int, rng: np.random.Generator) -> np.ndarray:
        n = len(arr)
        n_blocks_needed = int(np.ceil(target_len / b_size))
        starts = rng.integers(0, n - b_size + 1, size=n_blocks_needed)
        sample_blocks = [arr[s:s + b_size] for s in starts]
        return np.concatenate(sample_blocks)[:target_len]

    results_by_block = {}
    rng = np.random.default_rng(42)

    for b_size in block_sizes:
        if b_size > min_len // 2:
            results_by_block[f"B{b_size}"] = {
                "ci_lower": np.nan,
                "ci_upper": np.nan,
                "p_value": np.nan,
                "significant": False,
                "feasible": False,
            }
            continue

        boot_diffs = np.empty(n_boot)
        for b in range(n_boot):
            b_shock = draw_mbb_sample(shock_vals, n_shock, b_size, rng)
            b_tranquil = draw_mbb_sample(tranquil_vals, n_tranquil, b_size, rng)
            boot_diffs[b] = b_shock.mean() - b_tranquil.mean()

        alpha = (1 - ci) / 2
        ci_lower = np.percentile(boot_diffs, alpha * 100)
        ci_upper = np.percentile(boot_diffs, (1 - alpha) * 100)
        p_val = np.mean(boot_diffs <= 0) if observed_diff > 0 else np.mean(boot_diffs >= 0)
        results_by_block[f"B{b_size}"] = {
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "p_value": p_val,
            "significant": (ci_lower > 0) if observed_diff > 0 else (ci_upper < 0),
            "feasible": True,
        }

    b10_res = results_by_block.get("B10", {})
    b20_res = results_by_block.get("B20", {})

    # Significance across all feasible blocks
    feasible_sigs = [v["significant"] for v in results_by_block.values() if v.get("feasible", False)]
    is_sig = all(feasible_sigs) if feasible_sigs else False

    p_val_report = b20_res.get("p_value", b10_res.get("p_value", np.nan))

    return {
        "observed_diff": observed_diff,
        "ci_lower_b10": b10_res.get("ci_lower", np.nan),
        "ci_upper_b10": b10_res.get("ci_upper", np.nan),
        "ci_lower_b20": b20_res.get("ci_lower", np.nan),
        "ci_upper_b20": b20_res.get("ci_upper", np.nan),
        "p_value": p_val_report,
        "significant": is_sig,
    }


def event_analysis(tci_series: pd.Series, events_df: pd.DataFrame,
                    net_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    For each event, evaluate shock-associated shifts in TCI using Moving-Block Bootstrap.
    """
    logger.info("Event-window analysis with Feasible Moving-Block Bootstrap (B10, B20) ...")

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

        boot = moving_block_bootstrap_mean_diff(shock_vals, tranquil_vals)

        ci_b10_str = f"[{boot['ci_lower_b10']:.2f}, {boot['ci_upper_b10']:.2f}]" if not np.isnan(boot['ci_lower_b10']) else "N/A"
        ci_b20_str = f"[{boot['ci_lower_b20']:.2f}, {boot['ci_upper_b20']:.2f}]" if not np.isnan(boot['ci_lower_b20']) else "Infeasible"

        pval_val = boot["p_value"]
        if np.isnan(pval_val):
            pval_str = "N/A"
        elif pval_val < 0.0002:
            pval_str = "< 0.0002"
        else:
            pval_str = f"{pval_val:.4f}"

        result = {
            "event_name": name,
            "shock_n": len(shock_vals),
            "tranquil_n": len(tranquil_vals),
            "shock_mean_tci": round(shock_mean, 2),
            "tranquil_mean_tci": round(tranquil_mean, 2),
            "diff_mean": round(diff_mean, 2),
            "diff_median": round(diff_median, 2),
            "mbb_ci_b10": ci_b10_str,
            "mbb_ci_b20": ci_b20_str,
            "boot_pval": pval_str,
            "shock_associated_increase": "Yes" if (diff_mean > 0 and boot["significant"]) else "No",
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
        sign = "↑ SHOCK INCREASE" if result["shock_associated_increase"] == "Yes" else "-"
        logger.info(f"  {name}: Δ TCI = {diff_mean:+.2f}% "
                    f"B10 CI={result['mbb_ci_b10']} B20 CI={result['mbb_ci_b20']} p={pval_str} {sign}")

    return pd.DataFrame(results)


def hac_regression(tci_series: pd.Series, global_df: pd.DataFrame) -> dict:
    """
    OLS regression of TCI on global factors with HAC standard errors.
    Constructs true monthly level differences for price/yield variables,
    and monthly averages for VIX and GPR.
    Equation: TCI_m = α + β₁·GPR_m + β₂·VIX_m + β₃·ΔOil_m + β₄·ΔDGS2_pp_m + β₅·ΔDollar_m + β₆·ΔSP500_m + ε_m
    Note: HAC (12 lags) accommodates (rather than eliminates) 250-day window overlap.
    """
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools import add_constant

    logger.info("HAC regression (monthly level differences): TCI ~ global factors ...")

    # 1. Month-end TCI level
    m_tci = tci_series.resample("ME").last().dropna()
    m_df = pd.DataFrame({"TCI": m_tci})

    # 2. Daily global levels to monthly levels / means
    g_levels = global_df.copy()
    if not isinstance(g_levels.index, pd.DatetimeIndex):
        g_levels.index = pd.to_datetime(g_levels.index)

    # Convert all numeric columns safely
    for col in g_levels.columns:
        g_levels[col] = pd.to_numeric(g_levels[col], errors="coerce")

    # Monthly means for VIX
    if "VIX" in g_levels.columns:
        m_df["VIX"] = g_levels["VIX"].resample("ME").mean()

    # Monthly means for GPR
    for c in ["GPRD", "GPR", "GPRD_ACT", "GPRD_THREAT"]:
        if c in g_levels.columns:
            m_df["GPR"] = g_levels[c].resample("ME").mean()
            break

    if "GPR" not in m_df.columns:
        gpr_path = config.find_file("gpr_daily.csv", "data_gpr_daily(till_aug_10).csv")
        if gpr_path.exists():
            try:
                gpr_raw = pd.read_csv(gpr_path)
                if "DAY" in gpr_raw.columns and "GPRD" in gpr_raw.columns:
                    gpr_raw["date"] = pd.to_datetime(gpr_raw["DAY"].astype(str), format="%Y%m%d", errors="coerce")
                    gpr_raw = gpr_raw.set_index("date")
                    gpr_raw["GPRD"] = pd.to_numeric(gpr_raw["GPRD"], errors="coerce")
                    m_df["GPR"] = gpr_raw["GPRD"].resample("ME").mean()
            except Exception:
                pass

    # Monthly level log differences for Brent oil, DollarIdx, SP500
    if "Brent" in g_levels.columns:
        m_brent = g_levels["Brent"].resample("ME").last()
        m_df["ΔOil"] = 100 * (np.log(m_brent) - np.log(m_brent.shift(1)))

    if "DollarIdx" in g_levels.columns:
        m_dollar = g_levels["DollarIdx"].resample("ME").last()
        m_df["ΔDollar"] = 100 * (np.log(m_dollar) - np.log(m_dollar.shift(1)))

    if "SP500" in g_levels.columns:
        m_sp = g_levels["SP500"].resample("ME").last()
        m_df["ΔSP500"] = 100 * (np.log(m_sp) - np.log(m_sp.shift(1)))

    # Monthly level first difference for interest rates (DGS2 in percentage points)
    if "DGS2" in g_levels.columns:
        m_dgs2 = g_levels["DGS2"].resample("ME").last()
        m_df["ΔDGS2_pp"] = m_dgs2 - m_dgs2.shift(1)

    m_df = m_df.dropna()
    logger.info(f"  Monthly regression observations: {len(m_df)} months")

    if len(m_df) < 20:
        logger.warning("  Insufficient observations for regression")
        return {}

    y = m_df["TCI"]
    x_cols = [c for c in m_df.columns if c != "TCI"]

    if not x_cols:
        logger.warning("  No explanatory variables available for regression")
        return {}

    X = add_constant(m_df[x_cols])

    # OLS with HAC (Newey-West) standard errors using 12-month maxlags
    # Accommodates persistence and overlap in rolling TCI estimates
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

    # Add model summary metrics for Table 7 manuscript integration
    stats_rows = pd.DataFrame({
        "coefficient": [result.nobs, result.rsquared, result.rsquared_adj, nw_lags, result.fvalue, result.f_pvalue],
        "std_error": [np.nan] * 6,
        "t_stat": [np.nan] * 6,
        "p_value": [np.nan] * 6,
    }, index=["N_obs", "R_squared", "Adj_R_squared", "HAC_maxlags", "F_stat", "Prob_F"])

    full_reg_table = pd.concat([coef_table, stats_rows])

    out_table = config.OUT_TABLES / "hac_regression_coefficients.csv"
    full_reg_table.to_csv(out_table)
    deliv_table = config.DELIVERABLES / "hac_regression_coefficients.csv"
    full_reg_table.to_csv(deliv_table)
    logger.info(f"  Saved HAC regression table -> {out_table} and {deliv_table}")
    logger.info(f"  Model fit: N={int(result.nobs)}, R²={result.rsquared:.4f}, Adj R²={result.rsquared_adj:.4f}, HAC Lags={nw_lags}")

    return {
        "result": result,
        "coefficients": full_reg_table,
        "r_squared": result.rsquared,
        "adj_r_squared": result.rsquared_adj,
        "nobs": result.nobs,
        "nw_lags": nw_lags,
    }


def create_alternative_event_windows() -> pd.DataFrame:
    """
    Create alternative event windows with extended shock endpoints (+20 trading days)
    to test sensitivity of shock-associated TCI shifts to window definitions.
    """
    alt_events = []
    for ev in DEFAULT_EVENTS:
        ev_copy = dict(ev)
        s_end = pd.to_datetime(ev["shock_end"]) + pd.Timedelta(days=30)
        ev_copy["shock_end"] = s_end.strftime("%Y-%m-%d")
        alt_events.append(ev_copy)

    events = pd.DataFrame(alt_events)
    for col in ["event_date", "shock_start", "shock_end",
                "tranquil_start", "tranquil_end"]:
        events[col] = pd.to_datetime(events[col])
    return events


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
        rolling_path = config.OUT_RESULTS / "rolling_connectedness_vol_squared_intersection.csv"

    if not rolling_path.exists():
        logger.error("  Rolling connectedness not found. Run Stage 9 first.")
        sys.exit(1)

    rolling_df = pd.read_csv(rolling_path, parse_dates=["date"])
    rolling_df = rolling_df.set_index("date")

    tci_series = rolling_df["TCI"]

    # 3. Baseline Event-window analysis
    event_results = event_analysis(tci_series, events, rolling_df)
    if not event_results.empty:
        out_path = config.OUT_TABLES / "shock_analysis_results.csv"
        event_results.to_csv(out_path, index=False)
        logger.info(f"  Saved shock analysis -> {out_path}")

        n_increases = (event_results["shock_associated_increase"] == "Yes").sum()
        logger.info(f"\n  Significant TCI increases detected in {n_increases}/{len(event_results)} events")

    # 3b. Alternative Event-window Sensitivity Analysis (+20 trading days extended endpoints)
    alt_events = create_alternative_event_windows()
    alt_results = event_analysis(tci_series, alt_events, rolling_df)
    if not alt_results.empty:
        sens_path = config.OUT_TABLES / "shock_analysis_sensitivity.csv"
        alt_results.to_csv(sens_path, index=False)
        logger.info(f"  Saved alternative window sensitivity analysis -> {sens_path}")

    # 4. HAC regression (reading raw daily levels for true monthly level differencing)
    global_path = config.DATA_RAW / "global_daily_raw.csv"
    if not global_path.exists():
        global_path = config.find_file("global_daily_raw.csv", "global_returns.csv")

    if global_path.exists():
        global_df = pd.read_csv(global_path, index_col=0, parse_dates=True)
        reg_results = hac_regression(tci_series, global_df)

        if reg_results:
            coef_path = config.OUT_TABLES / "hac_regression_coefficients.csv"
            reg_results["coefficients"].to_csv(coef_path)
            deliv_path = config.DELIVERABLES / "hac_regression_coefficients.csv"
            reg_results["coefficients"].to_csv(deliv_path)
            logger.info(f"  Saved regression coefficients -> {coef_path} and {deliv_path}")
    else:
        logger.info("  Global raw data not found - skipping HAC regression. "
                    "Run Stage 2 first.")

    logger.info(f"\n{'-' * 60}")
    logger.info("Stage 10 complete.")


if __name__ == "__main__":
    main()
