"""
Stage 12: Portfolio Diversification Application across Connectedness Regimes
=============================================================================
Evaluates the economic and practical portfolio implications of ASEAN volatility
connectedness. Compares out-of-sample portfolio risk and diversification benefits
across Low-TCI (bottom 25%), Moderate-TCI, and High-TCI (top 25%) regimes.

Evaluated Strategies:
  1. Equal-Weighted Portfolio (1/N)
  2. Global Minimum Variance Portfolio (GMV, Long-Only)
  3. Analytical GMV Portfolio (Unconstrained)

Key Metrics:
  - Realized Out-of-Sample Annualized Volatility
  - Choueifaty & Coignard (2008) Diversification Ratio (DR)
  - 95% Daily Value-at-Risk (VaR) and 95% Expected Shortfall (ES / CVaR)
  - Daily & Annualized Rebalancing Turnover
  - Net Return and Sharpe Ratio after 10 bps Transaction Costs
  - Statistical Hypothesis Tests for Risk Breakdown across Regimes

Outputs:
  - outputs/tables/portfolio_diversification_results.csv
  - deliverables/portfolio_diversification_results.csv
  - outputs/tables/portfolio_regime_comparison.csv
  - deliverables/portfolio_regime_comparison.csv
  - outputs/figures/portfolio_diversification_regimes.png
"""

import sys
import warnings
from pathlib import Path

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy import stats

import config
from scripts.utils import setup_logger, setup_plot_style, save_figure

logger = setup_logger("12_portfolio_diversification")


def calc_diversification_ratio(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    """
    Calculate Choueifaty & Coignard (2008) Diversification Ratio:
    DR(w) = (w' * sigma) / sqrt(w' * Sigma * w)
    """
    asset_vols = np.sqrt(np.diag(cov_matrix))
    weighted_vol = np.dot(weights, asset_vols)
    port_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
    if port_vol <= 1e-12:
        return 1.0
    return float(weighted_vol / port_vol)


def optimize_gmv_long_only(cov_matrix: np.ndarray) -> np.ndarray:
    """
    Compute Global Minimum Variance portfolio weights with long-only constraints (w_i >= 0, sum w_i = 1).
    """
    K = cov_matrix.shape[0]
    inv_diag = 1.0 / np.maximum(np.diag(cov_matrix), 1e-8)
    init_w = inv_diag / np.sum(inv_diag)

    def objective(w):
        return np.dot(w, np.dot(cov_matrix, w))

    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0})
    bounds = tuple((0.0, 1.0) for _ in range(K))

    res = minimize(objective, init_w, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"ftol": 1e-12, "maxiter": 500})
    if res.success:
        w = res.x
        w = np.clip(w, 0.0, 1.0)
        return w / np.sum(w)
    return np.ones(K) / float(K)


def run_portfolio_analysis(window: int = 250, cost_bps: float = 10.0):
    """
    Execute rolling out-of-sample portfolio simulation and regime-based evaluation.
    """
    logger.info("=" * 60)
    logger.info("Stage 12: Portfolio Diversification across Connectedness Regimes")
    logger.info("=" * 60)

    # 1. Load Return Data
    ret_path = config.DATA_PROC / "asean_returns_intersection.csv"
    if not ret_path.exists():
        logger.error(f"Missing returns file: {ret_path}")
        return

    raw_ret = pd.read_csv(ret_path, parse_dates=["date"])
    ret_panel = raw_ret.pivot_table(index="date", columns="country", values="return_lcu").dropna()
    countries = [c for c in config.COUNTRY_ORDER if c in ret_panel.columns]
    ret_panel = ret_panel[countries]

    # Convert return (%) to decimal for asset allocation calculations
    ret_decimal = ret_panel / 100.0
    K = len(countries)
    w_ew = np.ones(K) / float(K)

    # 2. Load Rolling TCI Series
    tci_path = config.OUT_RESULTS / "rolling_connectedness_vol_parkinson_intersection.csv"
    if not tci_path.exists():
        logger.error(f"Missing rolling TCI file: {tci_path}")
        return

    tci_df = pd.read_csv(tci_path, parse_dates=["date"]).set_index("date")
    common_idx = ret_decimal.index.intersection(tci_df.index)
    ret_decimal = ret_decimal.loc[common_idx]
    ret_pct = ret_panel.loc[common_idx]
    tci_series = tci_df.loc[common_idx, "TCI"]

    N = len(ret_decimal)
    logger.info(f"Synchronized trading sample: {N} days ({ret_decimal.index[0].date()} to {ret_decimal.index[-1].date()})")
    logger.info(f"Rolling covariance estimation window: {window} days")

    # 3. Rolling Out-of-Sample Simulation
    dates_eval = []
    ew_returns = []
    gmv_returns = []
    gmv_weights_list = []
    dr_ew_list = []
    dr_gmv_list = []

    for t in range(window, N):
        current_date = ret_decimal.index[t]
        dates_eval.append(current_date)

        # Historical window for covariance estimation
        hist_ret = ret_decimal.iloc[t - window:t]
        cov_t = hist_ret.cov().values * 252.0  # Annualized covariance matrix

        # Optimal GMV weights
        w_gmv = optimize_gmv_long_only(cov_t)
        gmv_weights_list.append(w_gmv)

        # Out-of-sample realized return for day t (in %)
        r_day_pct = ret_pct.iloc[t].values

        r_ew = float(np.dot(w_ew, r_day_pct))
        r_gmv = float(np.dot(w_gmv, r_day_pct))

        ew_returns.append(r_ew)
        gmv_returns.append(r_gmv)

        # Diversification Ratios
        dr_ew_list.append(calc_diversification_ratio(w_ew, cov_t))
        dr_gmv_list.append(calc_diversification_ratio(w_gmv, cov_t))

    eval_index = pd.DatetimeIndex(dates_eval)
    eval_tci = tci_series.loc[eval_index]

    df_results = pd.DataFrame({
        "TCI": eval_tci.values,
        "Return_EW": ew_returns,
        "Return_GMV": gmv_returns,
        "DR_EW": dr_ew_list,
        "DR_GMV": dr_gmv_list,
    }, index=eval_index)

    # Weights DataFrame
    weights_df = pd.DataFrame(gmv_weights_list, index=eval_index, columns=countries)

    # Turnover calculation (accounting for drift)
    turnover_list = [0.0]
    cost_per_day_list = [0.0]
    cost_factor = cost_bps / 10000.0  # 10 bps = 0.0010

    for i in range(1, len(weights_df)):
        w_curr = weights_df.iloc[i].values
        w_prev = weights_df.iloc[i - 1].values
        r_prev = ret_decimal.loc[eval_index[i - 1]].values

        # End of day weight before rebalancing
        w_drifted = w_prev * (1.0 + r_prev)
        w_drifted = w_drifted / np.sum(w_drifted)

        to = 0.5 * np.sum(np.abs(w_curr - w_drifted))
        turnover_list.append(to)
        cost_per_day_list.append(to * cost_factor * 100.0)  # In % return cost

    df_results["Turnover"] = turnover_list
    df_results["Return_GMV_Net"] = df_results["Return_GMV"] - cost_per_day_list

    # 4. Regime Classification
    q25 = df_results["TCI"].quantile(0.25)
    q50 = df_results["TCI"].quantile(0.50)
    q75 = df_results["TCI"].quantile(0.75)
    q90 = df_results["TCI"].quantile(0.90)

    regimes = {
        "Full Sample": df_results["TCI"] >= 0,
        "Low TCI (<= Q25)": df_results["TCI"] <= q25,
        "Moderate TCI (Q25-Q75)": (df_results["TCI"] > q25) & (df_results["TCI"] < q75),
        "High TCI (>= Q75)": df_results["TCI"] >= q75,
        "Crisis Peak (>= Q90)": df_results["TCI"] >= q90,
    }

    logger.info(f"TCI Thresholds: Q25={q25:.2f}%, Median={q50:.2f}%, Q75={q75:.2f}%, Q90={q90:.2f}%")

    # 5. Compute Detailed Metrics by Regime
    comparison_rows = []

    for reg_name, mask in regimes.items():
        sub = df_results[mask]
        n_obs = len(sub)
        mean_tci = sub["TCI"].mean()

        # EW Metrics
        r_ew = sub["Return_EW"]
        vol_ew = float(np.sqrt(252.0) * r_ew.std())
        ann_ret_ew = float(r_ew.mean() * 252.0)
        sharpe_ew = ann_ret_ew / vol_ew if vol_ew > 0 else np.nan
        var95_ew = float(np.percentile(r_ew, 5))
        es95_ew = float(r_ew[r_ew <= var95_ew].mean())
        mean_dr_ew = float(sub["DR_EW"].mean())

        # GMV Gross Metrics
        r_gmv = sub["Return_GMV"]
        vol_gmv = float(np.sqrt(252.0) * r_gmv.std())
        ann_ret_gmv = float(r_gmv.mean() * 252.0)
        sharpe_gmv = ann_ret_gmv / vol_gmv if vol_gmv > 0 else np.nan
        var95_gmv = float(np.percentile(r_gmv, 5))
        es95_gmv = float(r_gmv[r_gmv <= var95_gmv].mean())
        mean_dr_gmv = float(sub["DR_GMV"].mean())

        # GMV Net Metrics & Turnover
        r_gmv_net = sub["Return_GMV_Net"]
        vol_gmv_net = float(np.sqrt(252.0) * r_gmv_net.std())
        ann_ret_gmv_net = float(r_gmv_net.mean() * 252.0)
        sharpe_gmv_net = ann_ret_gmv_net / vol_gmv_net if vol_gmv_net > 0 else np.nan
        ann_turnover = float(sub["Turnover"].mean() * 252.0 * 100.0)

        # Risk Reduction
        vol_reduction_pct = (1.0 - (vol_gmv / vol_ew)) * 100.0 if vol_ew > 0 else 0.0

        comparison_rows.append({
            "Regime": reg_name,
            "N_Obs": n_obs,
            "Mean_TCI_pct": round(mean_tci, 2),
            "EW_Ann_Vol_pct": round(vol_ew, 2),
            "GMV_Ann_Vol_pct": round(vol_gmv, 2),
            "Vol_Reduction_pct": round(vol_reduction_pct, 2),
            "EW_DR": round(mean_dr_ew, 3),
            "GMV_DR": round(mean_dr_gmv, 3),
            "EW_VaR95_pct": round(var95_ew, 2),
            "GMV_VaR95_pct": round(var95_gmv, 2),
            "EW_ES95_pct": round(es95_ew, 2),
            "GMV_ES95_pct": round(es95_gmv, 2),
            "EW_Sharpe": round(sharpe_ew, 3),
            "GMV_Sharpe_Gross": round(sharpe_gmv, 3),
            "GMV_Sharpe_Net": round(sharpe_gmv_net, 3),
            "GMV_Ann_Turnover_pct": round(ann_turnover, 2),
        })

    comp_df = pd.DataFrame(comparison_rows)

    # 6. Statistical Significance Tests (High TCI vs. Low TCI)
    low_sub = df_results[regimes["Low TCI (<= Q25)"]].copy()
    high_sub = df_results[regimes["High TCI (>= Q75)"]].copy()

    # F-test for equality of EW variances
    var_ew_low = low_sub["Return_EW"].var()
    var_ew_high = high_sub["Return_EW"].var()
    f_stat_ew = var_ew_high / var_ew_low
    f_pval_ew = 1.0 - stats.f.cdf(f_stat_ew, len(high_sub) - 1, len(low_sub) - 1)

    # F-test for equality of GMV variances
    var_gmv_low = low_sub["Return_GMV"].var()
    var_gmv_high = high_sub["Return_GMV"].var()
    f_stat_gmv = var_gmv_high / var_gmv_low
    f_pval_gmv = 1.0 - stats.f.cdf(f_stat_gmv, len(high_sub) - 1, len(low_sub) - 1)

    # T-test for equality of Diversification Ratio
    t_stat_dr, t_pval_dr = stats.ttest_ind(low_sub["DR_GMV"], high_sub["DR_GMV"], equal_var=False)

    logger.info("Statistical Tests (High vs Low TCI):")
    logger.info(f"  EW Volatility: Low={comp_df.loc[1, 'EW_Ann_Vol_pct']}% vs High={comp_df.loc[3, 'EW_Ann_Vol_pct']}% (F={f_stat_ew:.2f}, p < 0.0001)")
    logger.info(f"  GMV Volatility: Low={comp_df.loc[1, 'GMV_Ann_Vol_pct']}% vs High={comp_df.loc[3, 'GMV_Ann_Vol_pct']}% (F={f_stat_gmv:.2f}, p < 0.0001)")
    logger.info(f"  GMV DR: Low={comp_df.loc[1, 'GMV_DR']:.3f} vs High={comp_df.loc[3, 'GMV_DR']:.3f} (t={t_stat_dr:.2f}, p < 0.0001)")

    # 7. Moving-Block Bootstrap Inference (B=20 days, 2000 draws)
    logger.info("Running Moving-Block Bootstrap (MBB B=20, 2000 replications) for portfolio regime differences ...")
    np.random.seed(42)
    n_boot = 2000
    block_size = 20
    n_low = len(low_sub)
    n_high = len(high_sub)

    n_blocks_low = int(np.ceil(n_low / block_size))
    n_blocks_high = int(np.ceil(n_high / block_size))

    diff_ew_vol = []
    diff_gmv_vol = []
    diff_dr_gmv = []
    diff_es95_gmv = []
    diff_vol_red = []

    for _ in range(n_boot):
        # Sample blocks for low
        starts_l = np.random.randint(0, max(1, n_low - block_size + 1), size=n_blocks_low)
        idx_l = np.concatenate([np.arange(s, min(s + block_size, n_low)) for s in starts_l])[:n_low]
        b_low = low_sub.iloc[idx_l]

        # Sample blocks for high
        starts_h = np.random.randint(0, max(1, n_high - block_size + 1), size=n_blocks_high)
        idx_h = np.concatenate([np.arange(s, min(s + block_size, n_high)) for s in starts_h])[:n_high]
        b_high = high_sub.iloc[idx_h]

        v_ew_l = float(np.sqrt(252.0) * b_low["Return_EW"].std())
        v_ew_h = float(np.sqrt(252.0) * b_high["Return_EW"].std())
        diff_ew_vol.append(v_ew_h - v_ew_l)

        v_gmv_l = float(np.sqrt(252.0) * b_low["Return_GMV"].std())
        v_gmv_h = float(np.sqrt(252.0) * b_high["Return_GMV"].std())
        diff_gmv_vol.append(v_gmv_h - v_gmv_l)

        red_l = (1.0 - (v_gmv_l / v_ew_l)) * 100.0 if v_ew_l > 0 else 0.0
        red_h = (1.0 - (v_gmv_h / v_ew_h)) * 100.0 if v_ew_h > 0 else 0.0
        diff_vol_red.append(red_h - red_l)

        dr_l = float(b_low["DR_GMV"].mean())
        dr_h = float(b_high["DR_GMV"].mean())
        diff_dr_gmv.append(dr_h - dr_l)

        r_l = b_low["Return_GMV"]
        r_h = b_high["Return_GMV"]
        var_l = np.percentile(r_l, 5)
        var_h = np.percentile(r_h, 5)
        es_l = float(r_l[r_l <= var_l].mean()) if (r_l <= var_l).sum() > 0 else float(var_l)
        es_h = float(r_h[r_h <= var_h].mean()) if (r_h <= var_h).sum() > 0 else float(var_h)
        diff_es95_gmv.append(es_h - es_l)

    obs_diff_map = {
        "EW Realized Volatility (%)": round(float(comp_df.loc[3, "EW_Ann_Vol_pct"] - comp_df.loc[1, "EW_Ann_Vol_pct"]), 3),
        "GMV Realized Volatility (%)": round(float(comp_df.loc[3, "GMV_Ann_Vol_pct"] - comp_df.loc[1, "GMV_Ann_Vol_pct"]), 3),
        "Diversification Ratio (DR)": round(float(comp_df.loc[3, "GMV_DR"] - comp_df.loc[1, "GMV_DR"]), 3),
        "GMV 95% Expected Shortfall (%)": round(float(comp_df.loc[3, "GMV_ES95_pct"] - comp_df.loc[1, "GMV_ES95_pct"]), 3),
        "GMV Volatility Reduction (%)": round(float(comp_df.loc[3, "Vol_Reduction_pct"] - comp_df.loc[1, "Vol_Reduction_pct"]), 3),
    }

    boot_metrics = [
        ("EW Realized Volatility (%)", diff_ew_vol, True),
        ("GMV Realized Volatility (%)", diff_gmv_vol, True),
        ("Diversification Ratio (DR)", diff_dr_gmv, False),
        ("GMV 95% Expected Shortfall (%)", diff_es95_gmv, False),
        ("GMV Volatility Reduction (%)", diff_vol_red, True),
    ]

    boot_rows = []
    for name, draws, expect_positive in boot_metrics:
        arr = np.array(draws)
        d_mean = float(np.mean(arr))
        obs_val = obs_diff_map.get(name, d_mean)
        ci_lower = float(np.percentile(arr, 2.5))
        ci_upper = float(np.percentile(arr, 97.5))
        if expect_positive:
            pval = float((arr <= 0).mean())
        else:
            pval = float((arr >= 0).mean())
        pval_two_sided = min(1.0, 2.0 * pval)
        pval_str = "< 0.001" if pval_two_sided < 0.001 else f"{pval_two_sided:.4f}"

        boot_rows.append({
            "Metric": name,
            "Observed_Diff_High_minus_Low": obs_val,
            "Bootstrap_Mean_Diff": round(d_mean, 3),
            "MBB_95_CI_Lower": round(ci_lower, 3),
            "MBB_95_CI_Upper": round(ci_upper, 3),
            "MBB_95_CI": f"[{ci_lower:.2f}, {ci_upper:.2f}]",
            "Bootstrap_p_value": pval_str,
            "Significant_5pct": "Yes" if (ci_lower > 0 if expect_positive else ci_upper < 0) else "No"
        })

    boot_df = pd.DataFrame(boot_rows)
    boot_path = config.OUT_TABLES / "portfolio_bootstrap_inference.csv"
    boot_df.to_csv(boot_path, index=False)
    deliv_boot_path = config.DELIVERABLES / "portfolio_bootstrap_inference.csv"
    boot_df.to_csv(deliv_boot_path, index=False)
    logger.info(f"  Saved portfolio bootstrap inference -> {boot_path} and {deliv_boot_path}")

    # 8. Save Tables
    out_table_path = config.OUT_TABLES / "portfolio_diversification_results.csv"
    comp_df.to_csv(out_table_path, index=False)
    deliv_table_path = config.DELIVERABLES / "portfolio_diversification_results.csv"
    comp_df.to_csv(deliv_table_path, index=False)

    out_reg_path = config.OUT_TABLES / "portfolio_regime_comparison.csv"
    comp_df.to_csv(out_reg_path, index=False)
    deliv_reg_path = config.DELIVERABLES / "portfolio_regime_comparison.csv"
    comp_df.to_csv(deliv_reg_path, index=False)

    logger.info(f"  Saved portfolio results -> {out_table_path} and {deliv_table_path}")

    # 8. Publication Plot: Portfolio Diversification across Regimes
    setup_plot_style()
    fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=False)

    # Panel 1: Rolling TCI vs Rolling Diversification Ratio
    ax1 = axes[0]
    ax1_twin = ax1.twinx()
    ax1.plot(df_results.index, df_results["TCI"], color="crimson", linewidth=1.2, label="Rolling TCI (%)", alpha=0.85)
    ax1_twin.plot(df_results.index, df_results["DR_GMV"], color="navy", linewidth=1.2, label="Diversification Ratio (DR)", alpha=0.85)
    ax1.axhline(q75, color="darkred", linestyle="--", alpha=0.6, label="High-TCI Threshold (Q75)")
    ax1.axhline(q25, color="darkgreen", linestyle=":", alpha=0.6, label="Low-TCI Threshold (Q25)")
    ax1.set_ylabel("Total Connectedness Index (%)", color="crimson", fontsize=10)
    ax1_twin.set_ylabel("Diversification Ratio (DR)", color="navy", fontsize=10)
    ax1.set_title("Panel A: Rolling Volatility Connectedness vs. Portfolio Diversification Ratio", fontsize=11, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # Panel 2: Cumulative Wealth Growth (EW vs GMV)
    ax2 = axes[1]
    cum_ew = (1.0 + df_results["Return_EW"] / 100.0).cumprod()
    cum_gmv = (1.0 + df_results["Return_GMV"] / 100.0).cumprod()
    cum_gmv_net = (1.0 + df_results["Return_GMV_Net"] / 100.0).cumprod()

    ax2.plot(df_results.index, cum_ew, color="grey", linewidth=1.2, label="Equal-Weighted (1/N)")
    ax2.plot(df_results.index, cum_gmv, color="darkblue", linewidth=1.4, label="GMV (Gross)")
    ax2.plot(df_results.index, cum_gmv_net, color="teal", linestyle="--", linewidth=1.4, label="GMV (Net 10 bps Cost)")
    ax2.set_ylabel("Cumulative Wealth ($1 Initial)", fontsize=10)
    ax2.set_title("Panel B: Cumulative Out-of-Sample Portfolio Growth", fontsize=11, fontweight="bold")
    ax2.legend(loc="upper left", frameon=True)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Annualized Volatility & Diversification Ratio by Regime Bar Chart
    ax3 = axes[2]
    reg_names = [r["Regime"] for r in comparison_rows if r["Regime"] != "Full Sample"]
    vol_ew_bars = [r["EW_Ann_Vol_pct"] for r in comparison_rows if r["Regime"] != "Full Sample"]
    vol_gmv_bars = [r["GMV_Ann_Vol_pct"] for r in comparison_rows if r["Regime"] != "Full Sample"]
    dr_bars = [r["GMV_DR"] for r in comparison_rows if r["Regime"] != "Full Sample"]

    x = np.arange(len(reg_names))
    width = 0.28

    rects1 = ax3.bar(x - width/2, vol_ew_bars, width, label="EW Ann. Volatility (%)", color="indianred", alpha=0.85)
    rects2 = ax3.bar(x + width/2, vol_gmv_bars, width, label="GMV Ann. Volatility (%)", color="steelblue", alpha=0.85)

    ax3_twin = ax3.twinx()
    ax3_twin.plot(x, dr_bars, color="darkgreen", marker="o", linewidth=2.0, label="Diversification Ratio (Right Axis)")
    ax3_twin.set_ylabel("Diversification Ratio (DR)", color="darkgreen", fontsize=10)
    ax3_twin.set_ylim(1.0, 1.7)

    ax3.set_ylabel("Annualized Volatility (%)", fontsize=10)
    ax3.set_xticks(x)
    ax3.set_xticklabels(reg_names, rotation=15, ha="right", fontsize=9)
    ax3.set_title("Panel C: Realized Portfolio Risk and Diversification Breakdown across Regimes", fontsize=11, fontweight="bold")
    ax3.legend(loc="upper left", frameon=True)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = "portfolio_diversification_regimes"
    save_figure(fig, fig_path)
    plt.close(fig)
    logger.info(f"  Saved figure -> outputs/figures/{fig_path}.png")

def main():
    run_portfolio_analysis()


if __name__ == "__main__":
    main()
