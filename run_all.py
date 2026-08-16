"""
Master script: run the entire ASEAN Volatility Connectedness pipeline.

Usage:
    python run_all.py              # Run all stages
    python run_all.py 1 2 3        # Run specific stages
    python run_all.py --from 4     # Run from stage 4 onward
"""

import os
import sys

# Fix Windows encoding BEFORE any imports
os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import time
import warnings
import importlib
from pathlib import Path

# Suppress statsmodels date index warnings globally
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="statsmodels")
try:
    from statsmodels.tools.sm_exceptions import ValueWarning
    warnings.filterwarnings("ignore", category=ValueWarning)
except ImportError:
    pass


# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.utils import setup_logger

logger = setup_logger("run_all")

# Stage registry
STAGES = {
    1:  ("Fetch ASEAN indices",       "scripts.01_fetch_asean_indices"),
    2:  ("Fetch global data",         "scripts.02_fetch_global_data"),
    3:  ("Fetch exchange rates",      "scripts.03_fetch_exchange_rates"),
    4:  ("Clean data",                "scripts.04_clean_data"),
    5:  ("Calculate returns",         "scripts.05_calculate_returns"),
    6:  ("Calculate volatility",      "scripts.06_calculate_volatility"),
    7:  ("Descriptive statistics",    "scripts.07_descriptive_stats"),
    8:  ("VAR model & GFEVD",        "scripts.08_var_model"),
    9:  ("Rolling connectedness",     "scripts.09_connectedness"),
    10: ("Shock & contagion analysis","scripts.10_shock_analysis"),
    11: ("Robustness checks",         "scripts.11_robustness"),
    12: ("Generate manuscript results","scripts.generate_manuscript_results"),
}


def run_stage(stage_num: int):
    """Run a single stage by number."""
    if stage_num not in STAGES:
        logger.error(f"Unknown stage: {stage_num}")
        return False

    name, module_name = STAGES[stage_num]
    logger.info(f"\n{'=' * 60}")
    logger.info(f"STAGE {stage_num}: {name}")
    logger.info(f"{'=' * 60}")

    t0 = time.time()
    try:
        module = importlib.import_module(module_name)
        module.main()
        elapsed = time.time() - t0
        logger.info(f"[OK] Stage {stage_num} completed in {elapsed:.1f}s")
        return True
    except Exception as e:
        elapsed = time.time() - t0
        logger.error(f"[X] Stage {stage_num} failed after {elapsed:.1f}s: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    logger.info("=" * 60)
    logger.info("ASEAN Volatility Connectedness - Full Pipeline")
    logger.info("=" * 60)

    # Parse arguments
    args = sys.argv[1:]

    if "--from" in args:
        idx = args.index("--from")
        start_stage = int(args[idx + 1])
        stages_to_run = [s for s in sorted(STAGES.keys()) if s >= start_stage]
    elif args:
        stages_to_run = [int(a) for a in args]
    else:
        stages_to_run = sorted(STAGES.keys())

    logger.info(f"Stages to run: {stages_to_run}")

    t_total = time.time()
    results = {}

    for stage in stages_to_run:
        success = run_stage(stage)
        results[stage] = success
        if not success:
            logger.warning(f"Stage {stage} failed. Continuing with next stage ...")

    elapsed_total = time.time() - t_total

    # Summary
    logger.info(f"\n{'=' * 60}")
    logger.info(f"PIPELINE SUMMARY (total: {elapsed_total:.1f}s)")
    logger.info(f"{'=' * 60}")
    for stage, success in results.items():
        status = "[OK]" if success else "[X]"
        name = STAGES[stage][0]
        logger.info(f"  {status} Stage {stage:2d}: {name}")


if __name__ == "__main__":
    main()
