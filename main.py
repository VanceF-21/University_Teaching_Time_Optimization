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

import argparse
import sys
import time
import datetime
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ── Local modules ─────────────────────────────────────────────────────────────
from data_preprocessing import run_preprocessing
from baseline_analysis  import run_baseline_analysis
from visualization      import run_all_visualisations

OUT_DIR      = Path(__file__).resolve().parent / "outputs"
CLEANED_DIR  = OUT_DIR / "cleaned_data"          # preprocessed data (static)
OUT_DIR.mkdir(exist_ok=True)
CLEANED_DIR.mkdir(exist_ok=True)


# ============================================================
# Log Tee  —  mirror all stdout/stderr to a .txt file
# ============================================================
class _Tee:
    """Write every write() call to both the original stream and a log file."""

    def __init__(self, original_stream, log_file):
        self._original = original_stream
        self._log      = log_file

    def write(self, data):
        self._original.write(data)
        self._log.write(data)

    def flush(self):
        self._original.flush()
        self._log.flush()

    def isatty(self):
        return False


def setup_logging(log_dir: Path = None) -> Path:
    """
    Redirect sys.stdout and sys.stderr so that all print() output and
    tracebacks are saved to <log_dir>/pipeline_log_<YYYYMMDD_HHMMSS>.txt
    in addition to appearing in the terminal as normal.

    Parameters
    ----------
    log_dir : Path, optional
        Directory where the log file is saved. Defaults to OUT_DIR.

    Returns the Path of the log file that was created.
    Call teardown_logging() at the end of the run to close the file.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir  = log_dir or OUT_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path  = save_dir / f"pipeline_log_{timestamp}.txt"
    # Expose timestamp so main() can reference it
    setup_logging._timestamp = timestamp
    log_file  = open(log_path, "w", encoding="utf-8", buffering=1)

    # Write a header so the file is self-contained
    header = (
        f"{'=' * 70}\n"
        f"University Teaching Time Optimization — Pipeline Log\n"
        f"Run started : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"{'=' * 70}\n\n"
    )
    log_file.write(header)

    # Monkey-patch stdout and stderr
    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)

    # Store the open file handle on the module so teardown can close it
    setup_logging._log_file = log_file
    setup_logging._log_path = log_path

    print(f"[log] Pipeline log → {log_path}")
    return log_path


def teardown_logging():
    """
    Restore sys.stdout / sys.stderr and close the log file.
    Safe to call even if setup_logging() was never called.
    """
    log_file = getattr(setup_logging, "_log_file", None)
    log_path = getattr(setup_logging, "_log_path", None)

    if log_file is None:
        return

    # Append a footer
    footer = (
        f"\n{'=' * 70}\n"
        f"Run finished: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"{'=' * 70}\n"
    )
    try:
        log_file.write(footer)
    except Exception:
        pass

    # Restore original streams before closing
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

    try:
        log_file.close()
    except Exception:
        pass

    if log_path:
        print(f"[log] Log saved → {log_path}")


# ============================================================
# Argument Parser
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="University Teaching Time Optimization — TIME Scenario",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--skip-conflicts",  action="store_true",
                        help="Load conflict_pairs.csv if it exists instead of rebuilding")
    parser.add_argument("--skip-mip",        action="store_true",
                        help="Skip the MIP optimisation step")
    parser.add_argument("--skip-heuristic",  action="store_true",
                        help="Skip the heuristic optimisation step")
    parser.add_argument("--skip-viz",        action="store_true",
                        help="Skip visualisation step")
    parser.add_argument("--mip-events",  type=int, default=300,
                        help="Max displaced WholeClass events in MIP (default: 300)")
    parser.add_argument("--mip-time",    type=int, default=180,
                        help="MIP solver wall-clock time limit in seconds (default: 180)")
    parser.add_argument("--heur-events", type=int, default=2000,
                        help="Max displaced events in heuristic (default: 2000)")
    parser.add_argument("--heur-iter",   type=int, default=200,
                        help="Local search max iterations (default: 200)")
    parser.add_argument("--no-ls",       action="store_true",
                        help="Heuristic: greedy only (no local search)")
    parser.add_argument("--scenario",    default="both",
                        choices=["S1_9am5pm","S2_NoFriPM","both"],
                        help="Which scenario(s) to optimise (default: both)")
    return parser.parse_args()


# ============================================================
# Final Summary
# ============================================================
def print_final_summary(baseline_results: dict,
                          mip_results: dict,
                          heuristic_results: dict,
                          run_dir: Path = None):
    """
    Print a comprehensive summary table answering all research questions.
    """
    print("\n" + "=" * 70)
    print("FINAL RESULTS SUMMARY")
    print("=" * 70)

    rows = []

    # ── Q1 & Q2: Displaced events + NO_SLOT feasibility ─────────────────────
    disp = baseline_results.get("displaced_summary")
    noslot = baseline_results.get("noslot_feasibility")
    if disp is not None:
        for sc in ["S1_9am5pm", "S2_NoFriPM"]:
            row = disp[disp["Scenario"] == sc]
            if len(row):
                r = row.iloc[0]
                print(f"\n[Q{'1' if sc=='S1_9am5pm' else '2'}] Scenario {sc}:")
                print(f"  Displaced events: {r['Displaced_Events']:,} / {r['Total_Events']:,} "
                      f"({r['Displaced_Pct']:.1f}%)")
                print(f"  WholeClass displaced: {r['Displaced_WholeClass']:,}")
                print(f"  Unique modules affected: {r['Displaced_Unique_Modules']:,}")
                # NEW: NO_SLOT events (cannot fit any allowed timeslot)
                if noslot is not None:
                    ns_row = noslot[noslot["Scenario"] == sc]
                    if len(ns_row):
                        ns = ns_row.iloc[0]
                        print(f"  NO_SLOT (too long for window): {int(ns['N_NoSlot']):,} "
                              f"({ns['NoSlot_Pct']:.1f}%)  — "
                              f"WholeClass: {int(ns['N_NoSlot_WholeClass']):,}")
                rows.append({
                    "RQ": f"Q{'1' if sc=='S1_9am5pm' else '2'}",
                    "Scenario":             sc,
                    "Metric":              "Displaced_Events_Pct",
                    "Value":               r["Displaced_Pct"],
                })

    # ── Q3: Clashes ──────────────────────────────────────────────────────────
    clash = baseline_results.get("clashes")
    if clash is not None:
        print("\n[Q3] Scheduling clashes in existing timetable (before optimisation):")
        for _, row in clash.iterrows():
            print(f"  {row['Scenario']}: {row['total_clash_pairs']:,} clash pairs, "
                  f"{row['student_clash_pct']:.1f}% students affected")

    print("\n[Q3] Clashes after optimisation (rescheduling):")
    for sc in ["S1_9am5pm", "S2_NoFriPM"]:
        if mip_results and sc in mip_results:
            mr = mip_results[sc]
            print(f"  MIP {sc}: {mr.get('n_clashes', 'N/A')} clash pairs remaining "
                  f"(status: {mr.get('status','N/A')})")
        if heuristic_results and sc in heuristic_results:
            hr = heuristic_results[sc]
            sm = hr.get("summary", {})
            print(f"  Heuristic {sc}: greedy={sm.get('Greedy_Clash_Score','N/A')}, "
                  f"local_search={sm.get('LocalSearch_Clash_Score','N/A')}")
            # NEW: WholeClass clash-free breakdown
            wc_cf  = sm.get('Greedy_ClashFree_WholeClass', 'N/A')
            wc_pct = sm.get('Greedy_ClashFree_WholeClass_Pct', 'N/A')
            print(f"    → WholeClass clash-free (greedy): {wc_cf} ({wc_pct}%)")

    # ── Q4: Lunch breaks ─────────────────────────────────────────────────────
    lunch = baseline_results.get("lunch_break")
    if lunch is not None:
        print("\n[Q4] Lunch break (12-2pm) feasibility:")
        for _, row in lunch.iterrows():
            print(f"  {row['Scenario']}: {row['Lunch_Free_Pct']:.1f}% students free for lunch")

    # ── Q5: Utilisation ──────────────────────────────────────────────────────
    util = baseline_results.get("utilisation_comparison")
    if util is not None:
        print("\n[Q5] Room/timeslot utilisation comparison:")
        for _, row in util.iterrows():
            print(f"  {row['Scenario']}: {row['In_Window_Events']:,} events in window, "
                  f"room utilisation = {row['Room_Utilisation_Pct']:.1f}%")
    # NEW: hourly load peak per scenario
    hourly = baseline_results.get("hourly_load")
    if hourly is not None:
        print("\n[Q5] Peak teaching slot per scenario (events in window):")
        for sc in ["S0_Baseline", "S1_9am5pm", "S2_NoFriPM"]:
            sc_df = hourly[hourly["Scenario"] == sc]
            if len(sc_df):
                peak = sc_df.loc[sc_df["Num_Events"].idxmax()]
                print(f"  {sc}: peak at {peak['Day']} {int(peak['Start_Hour'])}:00 "
                      f"({int(peak['Num_Events'])} events, "
                      f"{int(peak['Num_WholeClass'])} WholeClass)")
    # NEW: heuristic day distribution (Q5 after rescheduling)
    if heuristic_results:
        print("\n[Q5] Rescheduled event distribution by day (heuristic):")
        for sc in ["S1_9am5pm", "S2_NoFriPM"]:
            if sc in heuristic_results:
                sm = heuristic_results[sc].get("summary", {})
                day_dist = {d: sm.get(f"Rescheduled_{d}", 0)
                            for d in ["Monday","Tuesday","Wednesday","Thursday","Friday"]}
                print(f"  {sc}: {day_dist}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    if rows:
        dest = (run_dir or OUT_DIR) / "final_summary.csv"
        pd.DataFrame(rows).to_csv(dest, index=False)
        print(f"\n[main] Final summary saved to {dest.relative_to(OUT_DIR.parent)}")

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)


# ============================================================
# Main Pipeline
# ============================================================
def main():
    args = parse_args()
    t_start = time.time()

    # ── Create per-run output directory (timestamp determined here) ──────────
    ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUT_DIR / f"run_{ts}"
    run_dir.mkdir(exist_ok=True)
    (run_dir / "figures").mkdir(exist_ok=True)

    # ── Start logging (mirrors all output to run_dir/pipeline_log_*.txt) ─────
    setup_logging(log_dir=run_dir)
    print(f"[main] Run outputs → {run_dir.relative_to(OUT_DIR.parent)}")

    print("=" * 70)
    print("UNIVERSITY TEACHING TIME OPTIMIZATION — TIME Scenario")
    print("=" * 70)

    # ── Step 1: Data Preprocessing ───────────────────────────────────────────
    print("\n>>> STEP 1: DATA PREPROCESSING")
    build_conflicts = not args.skip_conflicts
    # If conflict_pairs.csv already exists in cleaned_data, reuse it
    conflict_path = CLEANED_DIR / "conflict_pairs.csv"
    if args.skip_conflicts and conflict_path.exists():
        print("  [main] Loading existing conflict_pairs.csv from cleaned_data/")
        build_conflicts = False

    data = run_preprocessing(build_conflicts=build_conflicts, out_dir=CLEANED_DIR)

    # ── Step 2: Baseline Analysis ────────────────────────────────────────────
    print("\n>>> STEP 2: BASELINE ANALYSIS")
    baseline_results = run_baseline_analysis(data, out_dir=run_dir)

    # ── Step 3: MIP Model ────────────────────────────────────────────────────
    mip_results = {}
    if not args.skip_mip:
        print("\n>>> STEP 3: MIP OPTIMISATION (Xpress)")
        try:
            from mip_model import run_all_mip_scenarios, run_mip_scenario
            scenarios_to_run = (
                ["S1_9am5pm", "S2_NoFriPM"] if args.scenario == "both"
                else [args.scenario]
            )
            if args.scenario == "both":
                mip_results = run_all_mip_scenarios(
                    data,
                    max_events=args.mip_events,
                    time_limit=args.mip_time,
                    out_dir=run_dir,
                )
            else:
                res = run_mip_scenario(
                    scenario=args.scenario,
                    events=data["events"],
                    conflict_pairs=data.get("conflict_pairs"),
                    max_events=args.mip_events,
                    time_limit=args.mip_time,
                    out_dir=run_dir,
                )
                mip_results[args.scenario] = res

        except ImportError as e:
            print(f"  [WARNING] Could not import mip_model: {e}")
            print("  [WARNING] Make sure FICO Xpress is installed.")
        except Exception as e:
            print(f"  [ERROR] MIP model failed: {e}")
            import traceback; traceback.print_exc()
    else:
        print("\n>>> STEP 3: MIP skipped (--skip-mip)")

    # ── Step 4: Heuristic Model ───────────────────────────────────────────────
    heuristic_results = {}
    if not args.skip_heuristic:
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
                res = run_heuristic_scenario(
                    scenario=args.scenario,
                    events=data["events"],
                    conflict_pairs=data.get("conflict_pairs"),
                    student_events=data["student_events"],
                    max_events=args.heur_events,
                    run_local_search=not args.no_ls,
                    ls_max_iter=args.heur_iter,
                    out_dir=run_dir,
                )
                heuristic_results[args.scenario] = res

        except Exception as e:
            print(f"  [ERROR] Heuristic model failed: {e}")
            import traceback; traceback.print_exc()
    else:
        print("\n>>> STEP 4: Heuristic skipped (--skip-heuristic)")

    # ── Step 5: Visualisation ─────────────────────────────────────────────────
    if not args.skip_viz:
        print("\n>>> STEP 5: VISUALISATION")
        try:
            run_all_visualisations(data, baseline_results, mip_results, heuristic_results,
                                   out_dir=run_dir)
        except Exception as e:
            print(f"  [ERROR] Visualisation failed: {e}")
            import traceback; traceback.print_exc()
    else:
        print("\n>>> STEP 5: Visualisation skipped (--skip-viz)")

    # ── Step 6: Final Summary ─────────────────────────────────────────────────
    print("\n>>> STEP 6: FINAL SUMMARY")
    print_final_summary(baseline_results, mip_results, heuristic_results, run_dir=run_dir)

    elapsed = time.time() - t_start
    print(f"\nTotal elapsed time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # ── Close log file and restore stdout/stderr ──────────────────────────────
    teardown_logging()


if __name__ == "__main__":
    main()
