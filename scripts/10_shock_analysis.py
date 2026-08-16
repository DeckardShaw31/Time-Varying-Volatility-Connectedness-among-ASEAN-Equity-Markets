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
    For each event, evaluate shock-associated shifts in TCI using Moving-Block Bootstrap,
    and apply Holm and Benjamini-Hochberg adjustments across the 8 events.
    """
    from statsmodels.stats.multitest import multipletests

    logger.info("Event-window analysis with Feasible Moving-Block Bootstrap (B10, B20) ...")

    results = []
    raw_pvals = []
    observed_diffs = []

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
        effective_pval = max(pval_val, 0.0001) if not np.isnan(pval_val) else 1.0
        raw_pvals.append(effective_pval)
        observed_diffs.append(diff_mean)

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
            "boot_pval_raw": round(pval_val, 6) if not np.isnan(pval_val) else np.nan,
            "boot_pval": pval_str,
            "raw_significant": bool(diff_mean > 0 and boot["significant"]),
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

    if not results:
        return pd.DataFrame()

    # Apply multiple-testing corrections across the events
    _, holm_pvals, _, _ = multipletests(raw_pvals, alpha=0.05, method="holm")
    _, bh_pvals, _, _ = multipletests(raw_pvals, alpha=0.05, method="fdr_bh")

    for i, res in enumerate(results):
        hp = holm_pvals[i]
        bp = bh_pvals[i]
        diff_m = observed_diffs[i]

        res["boot_pval_holm"] = round(hp, 6)
        res["boot_pval_bh"] = round(bp, 6)
        res["boot_pval_holm_str"] = "< 0.0002" if hp < 0.0002 else f"{hp:.4f}"
        res["boot_pval_bh_str"] = "< 0.0002" if bp < 0.0002 else f"{bp:.4f}"

        res["sig_holm_5pct"] = "Yes" if (diff_m > 0 and hp < 0.05 and res["raw_significant"]) else "No"
        res["sig_bh_5pct"] = "Yes" if (diff_m > 0 and bp < 0.05 and res["raw_significant"]) else "No"
        res["shock_associated_increase"] = "Yes" if res["raw_significant"] else "No"

        sign = "↑ SHOCK INCREASE" if res["shock_associated_increase"] == "Yes" else "-"
        logger.info(f"  {res['event_name']}: Δ TCI = {res['diff_mean']:+.2f}% "
                    f"B20 CI={res['mbb_ci_b20']} p_raw={res['boot_pval']} "
                    f"p_Holm={res['boot_pval_holm_str']} p_BH={res['boot_pval_bh_str']} {sign}")

    return pd.DataFrame(results)


def hac_regression(tci_series: pd.Series, global_df: pd.DataFrame) -> dict:
    """
    OLS regressions of TCI on global factors with Newey-West HAC standard errors (L=12 months).
    Estimates:
      1. Baseline contemporaneous model
      2. Standardized coefficients (Beta weights)
      3. Lagged uncertainty model (lagged VIX and GPR)
      4. Log-GPR model: ln(GPR)
      5. Differenced GPR model: ΔGPR
      6. Threat vs. Act GPR decomposition
    """
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools import add_constant

    logger.info("HAC regression suite (monthly level differences & robustness): TCI ~ global factors ...")

    # 1. Month-end TCI level
    m_tci = tci_series.resample("ME").last().dropna()
    m_df = pd.DataFrame({"TCI": m_tci})

    # 2. Daily global levels to monthly levels / means
    g_levels = global_df.copy()
    if not isinstance(g_levels.index, pd.DatetimeIndex):
        g_levels.index = pd.to_datetime(g_levels.index)

    for col in g_levels.columns:
        g_levels[col] = pd.to_numeric(g_levels[col], errors="coerce")

    # Monthly means for VIX
    if "VIX" in g_levels.columns:
        m_df["VIX"] = g_levels["VIX"].resample("ME").mean()

    # Monthly means for GPR series
    gpr_cols_found = [c for c in ["GPRD", "GPR", "GPRD_ACT", "GPRD_THREAT"] if c in g_levels.columns]
    for c in gpr_cols_found:
        m_df[c] = g_levels[c].resample("ME").mean()

    # Fallback to load direct GPR daily if needed
    if "GPRD" not in m_df.columns and "GPR" not in m_df.columns:
        gpr_path = config.find_file("gpr_daily.csv", "data_gpr_daily(till_aug_10).csv")
        if gpr_path.exists():
            try:
                gpr_raw = pd.read_csv(gpr_path)
                if "DAY" in gpr_raw.columns:
                    gpr_raw["date"] = pd.to_datetime(gpr_raw["DAY"].astype(str), format="%Y%m%d", errors="coerce")
                    gpr_raw = gpr_raw.set_index("date")
                    for gc in ["GPRD", "GPRD_ACT", "GPRD_THREAT"]:
                        if gc in gpr_raw.columns:
                            gpr_raw[gc] = pd.to_numeric(gpr_raw[gc], errors="coerce")
                            m_df[gc] = gpr_raw[gc].resample("ME").mean()
            except Exception as e:
                logger.warning(f"GPR file loading note: {e}")

    # Standardize primary GPR column name
    if "GPR" not in m_df.columns and "GPRD" in m_df.columns:
        m_df["GPR"] = m_df["GPRD"]

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

    # Transformations for robustness
    if "GPR" in m_df.columns:
        m_df["ln_GPR"] = np.log(m_df["GPR"].replace(0, np.nan))
        m_df["ΔGPR"] = m_df["GPR"] - m_df["GPR"].shift(1)
        m_df["lag_GPR"] = m_df["GPR"].shift(1)

    if "VIX" in m_df.columns:
        m_df["lag_VIX"] = m_df["VIX"].shift(1)

    if "GPRD_THREAT" in m_df.columns:
        m_df["GPR_THREAT"] = m_df["GPRD_THREAT"]
    if "GPRD_ACT" in m_df.columns:
        m_df["GPR_ACT"] = m_df["GPRD_ACT"]

    base_cols = ["TCI", "VIX", "GPR", "ΔOil", "ΔDGS2_pp", "ΔDollar", "ΔSP500"]
    available_base = [c for c in base_cols if c in m_df.columns]
    m_clean = m_df.dropna(subset=available_base)

    logger.info(f"  Monthly regression complete cases: {len(m_clean)} months")
    if len(m_clean) < 20:
        logger.warning("  Insufficient observations for regression")
        return {}

    nw_lags = 12

    # Helper function to fit HAC OLS and format results
    def fit_model(y_series, x_df, model_name=""):
        X_mat = add_constant(x_df)
        res = OLS(y_series, X_mat).fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags})
        return res

    # 1. Baseline Model
    x_base_cols = [c for c in ["VIX", "GPR", "ΔOil", "ΔDGS2_pp", "ΔDollar", "ΔSP500"] if c in m_clean.columns]
    res_base = fit_model(m_clean["TCI"], m_clean[x_base_cols], "Baseline")

    coef_table = pd.DataFrame({
        "coefficient": res_base.params,
        "std_error": res_base.bse,
        "t_stat": res_base.tvalues,
        "p_value": res_base.pvalues,
    })
    stats_rows = pd.DataFrame({
        "coefficient": [res_base.nobs, res_base.rsquared, res_base.rsquared_adj, nw_lags, res_base.fvalue, res_base.f_pvalue],
        "std_error": [np.nan] * 6,
        "t_stat": [np.nan] * 6,
        "p_value": [np.nan] * 6,
    }, index=["N_obs", "R_squared", "Adj_R_squared", "HAC_maxlags", "F_stat", "Prob_F"])
    full_base_table = pd.concat([coef_table, stats_rows])

    # 2. Standardized Model
    std_df = (m_clean[available_base] - m_clean[available_base].mean()) / m_clean[available_base].std()
    res_std = fit_model(std_df["TCI"], std_df[x_base_cols], "Standardized")

    # 3. Lagged Uncertainty Model
    lag_cols = [c for c in ["lag_VIX", "lag_GPR", "ΔOil", "ΔDGS2_pp", "ΔDollar", "ΔSP500"] if c in m_clean.columns]
    m_lag = m_clean.dropna(subset=lag_cols)
    res_lag = fit_model(m_lag["TCI"], m_lag[lag_cols], "Lagged")

    # 4. Log GPR Model
    log_gpr_cols = [c for c in ["VIX", "ln_GPR", "ΔOil", "ΔDGS2_pp", "ΔDollar", "ΔSP500"] if c in m_clean.columns]
    m_log = m_clean.dropna(subset=log_gpr_cols)
    res_log = fit_model(m_log["TCI"], m_log[log_gpr_cols], "Log_GPR")

    # 5. Differenced GPR Model
    diff_gpr_cols = [c for c in ["VIX", "ΔGPR", "ΔOil", "ΔDGS2_pp", "ΔDollar", "ΔSP500"] if c in m_clean.columns]
    m_diff = m_clean.dropna(subset=diff_gpr_cols)
    res_diff = fit_model(m_diff["TCI"], m_diff[diff_gpr_cols], "Diff_GPR")

    # 6. GPR Threat vs Act Decomposition
    threat_act_cols = [c for c in ["VIX", "GPR_THREAT", "GPR_ACT", "ΔOil", "ΔDGS2_pp", "ΔDollar", "ΔSP500"] if c in m_clean.columns]
    res_ta = fit_model(m_clean["TCI"], m_clean[threat_act_cols], "Threat_Act") if len(threat_act_cols) >= 3 else None

    # Construct unified multi-specification comparison table
    models_dict = {
        "(1) Baseline": res_base,
        "(2) Standardized": res_std,
        "(3) Lagged Uncertainty": res_lag,
        "(4) Log GPR": res_log,
        "(5) Diff GPR": res_diff,
    }
    if res_ta is not None:
        models_dict["(6) Threat/Act"] = res_ta

    all_regressors = ["const", "VIX", "GPR", "lag_VIX", "lag_GPR", "ln_GPR", "ΔGPR",
                      "GPR_THREAT", "GPR_ACT", "ΔOil", "ΔDGS2_pp", "ΔDollar", "ΔSP500"]

    robust_rows = []
    for var in all_regressors:
        row_coef = {"Variable": var}
        row_se = {"Variable": f"{var}_se"}
        for m_name, res in models_dict.items():
            if var in res.params.index:
                c_val = res.params[var]
                se_val = res.bse[var]
                p_val = res.pvalues[var]
                stars = "***" if p_val < 0.01 else ("**" if p_val < 0.05 else ("*" if p_val < 0.1 else ""))
                row_coef[m_name] = f"{c_val:.4f}{stars}"
                row_se[m_name] = f"({se_val:.4f})"
            else:
                row_coef[m_name] = ""
                row_se[m_name] = ""
        robust_rows.append(row_coef)
        robust_rows.append(row_se)

    # Add summary statistics rows
    stats_map = {
        "Observations (N)": lambda r: str(int(r.nobs)),
        "R-squared": lambda r: f"{r.rsquared:.4f}",
        "Adj. R-squared": lambda r: f"{r.rsquared_adj:.4f}",
        "HAC Maxlags": lambda r: str(nw_lags),
        "F-statistic": lambda r: f"{r.fvalue:.2f}" if not np.isnan(r.fvalue) else "N/A",
        "Prob > F": lambda r: f"{r.f_pvalue:.4f}" if not np.isnan(r.f_pvalue) else "N/A",
    }
    for stat_name, func in stats_map.items():
        s_row = {"Variable": stat_name}
        for m_name, res in models_dict.items():
            s_row[m_name] = func(res)
        robust_rows.append(s_row)

    robust_table = pd.DataFrame(robust_rows)

    fit_stats = pd.DataFrame([{
        "nobs": int(res_base.nobs),
        "r_squared": round(res_base.rsquared, 4),
        "adj_r_squared": round(res_base.rsquared_adj, 4),
        "hac_lags": nw_lags,
        "frequency": "monthly",
    }])

    fit_path = config.OUT_TABLES / "hac_regression_fit.csv"
    fit_stats.to_csv(fit_path, index=False)
    deliv_fit_path = config.DELIVERABLES / "hac_regression_fit.csv"
    fit_stats.to_csv(deliv_fit_path, index=False)

    robust_path = config.OUT_TABLES / "hac_regression_robustness.csv"
    robust_table.to_csv(robust_path, index=False)
    deliv_robust_path = config.DELIVERABLES / "hac_regression_robustness.csv"
    robust_table.to_csv(deliv_robust_path, index=False)
    logger.info(f"  Saved HAC multi-model robustness table -> {robust_path} and {deliv_robust_path}")

    return {
        "result": res_base,
        "coefficients": full_base_table,
        "robustness_table": robust_table,
        "fit_stats": fit_stats,
        "r_squared": res_base.rsquared,
        "adj_r_squared": res_base.rsquared_adj,
        "nobs": res_base.nobs,
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
        deliv_event_path = config.DELIVERABLES / "shock_analysis_results.csv"
        event_results.to_csv(deliv_event_path, index=False)
        logger.info(f"  Saved shock analysis -> {out_path} and {deliv_event_path}")

        n_increases = (event_results["shock_associated_increase"] == "Yes").sum()
        logger.info(f"\n  Significant TCI increases detected in {n_increases}/{len(event_results)} events")

    # 3b. Alternative Event-window Sensitivity Analysis (+20 trading days extended endpoints)
    alt_events = create_alternative_event_windows()
    alt_results = event_analysis(tci_series, alt_events, rolling_df)
    if not alt_results.empty:
        sens_path = config.OUT_TABLES / "shock_analysis_sensitivity.csv"
        alt_results.to_csv(sens_path, index=False)
        deliv_sens_path = config.DELIVERABLES / "shock_analysis_sensitivity.csv"
        alt_results.to_csv(deliv_sens_path, index=False)
        logger.info(f"  Saved alternative window sensitivity analysis -> {sens_path} and {deliv_sens_path}")

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
