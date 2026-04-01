"""
main.py

Entry point for the complete analytical pipeline.

Usage:
    python main.py                           # Full pipeline (all steps)
    python main.py --skip-conflicts          # Skip building conflict pairs (fast)
    python main.py --skip-mip               # Skip MIP model (run heuristic only)
    python main.py --skip-heuristic         # Skip heuristic (run MIP only)
    python main.py --mip-events 200         # Limit MIP to 200 displaced events
    python main.py --mip-time 120           # MIP solver time limit (seconds)
    python main.py --new-mip                # Also run mip_new.py (full objective)
    python main.py --new-heuristic          # Also run heuristic_new.py (full score)
    python main.py --only-new-models        # Skip originals, run only new models
    python main.py --replace-with-new       # Run new models and use them for viz/summary

Pipeline Steps:
    ─────────────────────────────────────────────────────────────────────────
    Step 1  DATA PREPROCESSING
            Load all Excel files, parse timeslots, tag displaced events,
            build conflict pairs (student → event co-occurrence graph).
            → Outputs: events.csv, student_events.csv, conflict_pairs.csv

    Step 2  BASELINE ANALYSIS
            Compute current utilisation, clash rates, lunch break feasibility
            for the unmodified timetable and both proposed scenarios.
            Answers: Q1, Q2, Q3, Q4, Q5 (descriptive, no optimisation).
            → Outputs: baseline_*.csv, utilisation_comparison.csv, ...

    Step 3  MIP MODEL (FICO Xpress)
            For each scenario (S1: 9am-5pm, S2: No Fri PM):
              - Build and solve a Binary Integer Programme
              - Minimise weighted student clashes for displaced WholeClass events
              - Extract assignments and post-solve metrics
            → Outputs: mip_assignment_*.csv, mip_summary_*.csv

    Step 4  HEURISTIC MODEL
            For each scenario:
              - Greedy most-constrained-first assignment
              - Local search (swap neighbourhood) improvement
              - Compute final clash scores and lunch feasibility
            → Outputs: heuristic_assignment_*.csv, heuristic_summary_*.csv,
                        heuristic_iteration_log_*.csv

    Step 5  VISUALISATION
            Generate all figures (heatmaps, bar charts, convergence curves).
            → Outputs: outputs/run_YYYYMMDD_HHMMSS/figures/*.png

    Step 6  FINAL REPORT SUMMARY
            Print consolidated summary table to console and save CSV.
            → Outputs: final_summary.csv
    ─────────────────────────────────────────────────────────────────────────

Research Questions Addressed:
    Q1: Can core teaching be reduced to Mon-Fri 9am-5pm?
        → See displaced_summary, utilisation_comparison, mip_summary_S1

    Q2: Can Friday teaching be eliminated after 12pm?
        → See displaced_summary, mip_summary_S2, heuristic_summary_S2

    Q3: Can core lectures be delivered without clashes?
        → See clash_analysis, mip_summary (n_clashes after rescheduling)

    Q4: Can majority of students have 12-2pm lunch break?
        → See lunch_break_analysis, heuristic_summary (Lunch_Free_Pct)

    Q5: How do changes affect timeslot/room utilisation?
        → See utilisation_comparison, timeslot_load_before_after.png
"""

import time
import datetime
import traceback
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Local modules
from data_preprocessing import run_preprocessing
from baseline_analysis  import run_baseline_analysis
from visualization      import run_all_visualisations
from utils import (
    parse_args,
    setup_logging,
    teardown_logging,
    build_model_comparison,
    print_final_summary,
)

OUT_DIR     = Path(__file__).resolve().parent / "outputs"
CLEANED_DIR = OUT_DIR / "cleaned_data"
OUT_DIR.mkdir(exist_ok=True)
CLEANED_DIR.mkdir(exist_ok=True)



# Main Pipeline
def main():
    args    = parse_args()
    t_start = time.time()

    # Create per-run output directory
    ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUT_DIR / f"run_{ts}"
    run_dir.mkdir(exist_ok=True)
    (run_dir / "figures").mkdir(exist_ok=True)

    # Start logging
    setup_logging(log_dir=run_dir)
    print(f"[main] Run outputs → {run_dir.relative_to(OUT_DIR.parent)}")
    print("=" * 70)
    print("UNIVERSITY TEACHING TIME OPTIMIZATION — TIME Scenario")
    print("=" * 70)

    # ── Step 1: Data Preprocessing ───────────────────────────────────────────
    print("\n>>> STEP 1: DATA PREPROCESSING")
    conflict_path  = CLEANED_DIR / "conflict_pairs.csv"
    build_conflicts = not (args.skip_conflicts and conflict_path.exists())
    if not build_conflicts:
        print("  [main] Loading existing conflict_pairs.csv from cleaned_data/")
    data = run_preprocessing(build_conflicts=build_conflicts, out_dir=CLEANED_DIR)

    # ── Step 2: Baseline Analysis ────────────────────────────────────────────
    print("\n>>> STEP 2: BASELINE ANALYSIS")
    baseline_results = run_baseline_analysis(data, out_dir=run_dir)

    # Determine whether new models replace originals in downstream steps
    use_new = args.replace_with_new or args.only_new_models

    # ── Step 3: MIP Model ────────────────────────────────────────────────────
    mip_results = {}
    run_orig_mip = not args.skip_mip and not use_new
    if run_orig_mip:
        mip_results = _run_mip(args, data, run_dir)
    else:
        print("\n>>> STEP 3: MIP (original) skipped")

    # ── Step 3b: MIP NEW (full objective) ────────────────────────────────────
    mip_new_results = {}
    if args.new_mip or use_new:
        mip_new_results = _run_new_mip(args, data, run_dir)
        if use_new:
            mip_results = mip_new_results   # replace: new results flow into steps 5/6

    # ── Step 4: Heuristic Model ───────────────────────────────────────────────
    heuristic_results = {}
    run_orig_heur = not args.skip_heuristic and not use_new
    if run_orig_heur:
        heuristic_results = _run_heuristic(args, data, run_dir)
    else:
        print("\n>>> STEP 4: Heuristic (original) skipped")

    # ── Step 4b: Heuristic NEW (full clash score) ─────────────────────────────
    heuristic_new_results = {}
    if args.new_heuristic or use_new:
        heuristic_new_results = _run_new_heuristic(args, data, run_dir)
        if use_new:
            heuristic_results = heuristic_new_results  # replace: flows into steps 5/6

    # ── Step 5: Visualisation ─────────────────────────────────────────────────
    if not args.skip_viz:
        print("\n>>> STEP 5: VISUALISATION")
        try:
            run_all_visualisations(data, baseline_results, mip_results,
                                   heuristic_results, out_dir=run_dir)
        except Exception as e:
            print(f"  [ERROR] Visualisation failed: {e}")
            traceback.print_exc()
    else:
        print("\n>>> STEP 5: Visualisation skipped (--skip-viz)")

    # ── Step 6: Final Summary ─────────────────────────────────────────────────
    print("\n>>> STEP 6: FINAL SUMMARY")
    print_final_summary(baseline_results, mip_results, heuristic_results,
                        run_dir=run_dir, out_dir=OUT_DIR)

    # ── Step 6b: Model Comparison ─────────────────────────────────────────────
    if mip_results or heuristic_results:
        print("\n>>> STEP 6b: MODEL COMPARISON (MIP vs Heuristic)")
        build_model_comparison(mip_results, heuristic_results,
                               run_dir=run_dir, out_dir=OUT_DIR)

    elapsed = time.time() - t_start
    print(f"\nTotal elapsed time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    teardown_logging()



# Private step helpers
def _run_mip(args, data: dict, run_dir: Path) -> dict:
    """Run two-phase MIP optimisation (WholeClass + SubGroup) and return results."""
    mip_results = {}
    print("\n>>> STEP 3: MIP OPTIMISATION (two-phase: WholeClass + SubGroup)")
    try:
        from mip_model import run_all_mip_scenarios, run_mip_scenario
        if args.scenario == "both":
            mip_results = run_all_mip_scenarios(
                data, max_events=args.mip_events, time_limit=args.mip_time,
                out_dir=run_dir)
        else:
            mip_results[args.scenario] = run_mip_scenario(
                scenario=args.scenario,
                events=data["events"],
                conflict_pairs=data.get("conflict_pairs"),
                max_events=args.mip_events,
                time_limit=args.mip_time,
                out_dir=run_dir,
            )
    except ImportError as e:
        print(f"  [WARNING] Could not import mip_model: {e}")
        print("  [WARNING] Make sure FICO Xpress is installed.")
    except Exception as e:
        print(f"  [ERROR] MIP model failed: {e}")
        traceback.print_exc()
    return mip_results


def _run_new_mip(args, data: dict, run_dir: Path) -> dict:
    """Run full-objective MIP (mip_new.py: disp–disp + disp–fixed in objective)."""
    new_mip_results = {}
    print("\n>>> STEP 3b: MIP NEW OPTIMISATION (full objective: disp–disp + disp–fixed)")
    try:
        from mip_new import run_all_mip_scenarios_new, run_mip_scenario_new
        if args.scenario == "both":
            new_mip_results = run_all_mip_scenarios_new(
                data, max_events=args.mip_events, time_limit=args.mip_time,
                out_dir=run_dir)
        else:
            new_mip_results[args.scenario] = run_mip_scenario_new(
                scenario       = args.scenario,
                events         = data["events"],
                conflict_pairs = data.get("conflict_pairs"),
                student_events = data["student_events"],
                max_events     = args.mip_events,
                time_limit     = args.mip_time,
                out_dir        = run_dir,
            )
    except ImportError as e:
        print(f"  [WARNING] Could not import mip_new: {e}")
    except Exception as e:
        print(f"  [ERROR] MIP new model failed: {e}")
        traceback.print_exc()
    return new_mip_results


def _run_new_heuristic(args, data: dict, run_dir: Path) -> dict:
    """Run full-objective heuristic (heuristic_new.py: full clash score)."""
    new_heur_results = {}
    print("\n>>> STEP 4b: HEURISTIC NEW (full clash score: disp–disp + disp–fixed)")
    try:
        from heuristic_new import run_all_heuristic_scenarios_new, run_heuristic_scenario_new
        if args.scenario == "both":
            new_heur_results = run_all_heuristic_scenarios_new(
                data,
                max_events = args.heur_events,
                run_ls     = not args.no_ls,
                ls_iter    = args.heur_iter,
                out_dir    = run_dir,
            )
        else:
            new_heur_results[args.scenario] = run_heuristic_scenario_new(
                scenario         = args.scenario,
                events           = data["events"],
                conflict_pairs   = data.get("conflict_pairs"),
                student_events   = data["student_events"],
                max_events       = args.heur_events,
                run_local_search = not args.no_ls,
                ls_max_iter      = args.heur_iter,
                out_dir          = run_dir,
            )
    except Exception as e:
        print(f"  [ERROR] Heuristic new model failed: {e}")
        traceback.print_exc()
    return new_heur_results


def _run_heuristic(args, data: dict, run_dir: Path) -> dict:
    """Run heuristic optimisation and return results."""
    heuristic_results = {}
    print("\n>>> STEP 4: HEURISTIC OPTIMISATION")
    try:
        from heuristic_model import run_all_heuristic_scenarios, run_heuristic_scenario
        if args.scenario == "both":
            heuristic_results = run_all_heuristic_scenarios(
                data,
                max_events=args.heur_events,
                run_ls=not args.no_ls,
                ls_iter=args.heur_iter,
                out_dir=run_dir,
            )
        else:
            heuristic_results[args.scenario] = run_heuristic_scenario(
                scenario=args.scenario,
                events=data["events"],
                conflict_pairs=data.get("conflict_pairs"),
                student_events=data["student_events"],
                max_events=args.heur_events,
                run_local_search=not args.no_ls,
                ls_max_iter=args.heur_iter,
                out_dir=run_dir,
            )
    except Exception as e:
        print(f"  [ERROR] Heuristic model failed: {e}")
        traceback.print_exc()

    return heuristic_results


if __name__ == "__main__":
    main()
