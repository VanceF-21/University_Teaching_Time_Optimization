"""
utils.py

Shared utility functions for the University Teaching Time Optimization pipeline.

Includes:
  - _Tee                  : stdout/stderr tee to log file
  - setup_logging()       : redirect output to timestamped log file
  - teardown_logging()    : restore streams and close log file
  - parse_args()          : CLI argument parser
  - build_model_comparison() : MIP vs Heuristic side-by-side comparison table
  - print_final_summary() : consolidated Q1–Q5 results summary
"""

import sys
import datetime
import argparse
import pandas as pd
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Logging helpers
# ─────────────────────────────────────────────────────────────────────────────

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


def setup_logging(log_dir: Path) -> Path:
    """
    Redirect sys.stdout and sys.stderr so that all print() output and
    tracebacks are saved to <log_dir>/pipeline_log_<YYYYMMDD_HHMMSS>.txt
    in addition to appearing in the terminal as normal.

    Parameters
    ----------
    log_dir : Path
        Directory where the log file is saved.

    Returns the Path of the log file that was created.
    Call teardown_logging() at the end of the run to close the file.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"pipeline_log_{timestamp}.txt"

    # Expose timestamp so callers can reference it
    setup_logging._timestamp = timestamp

    log_file = open(log_path, "w", encoding="utf-8", buffering=1)

    header = (
        f"{'=' * 70}\n"
        f"University Teaching Time Optimization — Pipeline Log\n"
        f"Run started : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"{'=' * 70}\n\n"
    )
    log_file.write(header)

    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)

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

    footer = (
        f"\n{'=' * 70}\n"
        f"Run finished: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"{'=' * 70}\n"
    )
    try:
        log_file.write(footer)
    except Exception:
        pass

    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

    try:
        log_file.close()
    except Exception:
        pass

    if log_path:
        print(f"[log] Log saved → {log_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    """Parse CLI arguments for the pipeline entry point."""
    parser = argparse.ArgumentParser(
        description="University Teaching Time Optimization — TIME Scenario",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
                        choices=["S1_9am5pm", "S2_NoFriPM", "both"],
                        help="Which scenario(s) to optimise (default: both)")
    parser.add_argument("--mip-wholeclass-only", action="store_true",
                        help="Revert MIP to original mode: WholeClass displaced events only "
                             "(faster, ~1–5 min). Default is full two-phase MIP "
                             "(WholeClass + SubGroup, ~10–30 min per scenario).")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Model comparison table
# ─────────────────────────────────────────────────────────────────────────────

def build_model_comparison(mip_results: dict,
                            heuristic_results: dict,
                            run_dir: Path,
                            out_dir: Path) -> pd.DataFrame:
    """
    Build a side-by-side comparison table of MIP vs Heuristic metrics
    for each scenario and save to model_comparison.csv.

    Returns a long-format DataFrame: Scenario, Metric, MIP, Heuristic, Note.
    """
    save_dir = run_dir or out_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    scenarios = ["S1_9am5pm", "S2_NoFriPM"]
    rows = []

    def _v(val, fmt=None):
        """Format a value, returning 'N/A' for None."""
        if val is None:
            return "N/A"
        if fmt == "int":
            return int(val)
        if fmt == "pct":
            return f"{float(val):.1f}%"
        if fmt == "float":
            return round(float(val), 1)
        return val

    for sc in scenarios:
        mr = (mip_results or {}).get(sc, {})
        hr = (heuristic_results or {}).get(sc, {})
        ms = mr.get("summary", {}) if mr else {}
        hs = hr.get("summary", {}) if hr else {}

        def add(metric, mip_val, heur_val, note=""):
            rows.append({
                "Scenario":  sc,
                "Metric":    metric,
                "MIP":       mip_val,
                "Heuristic": heur_val,
                "Note":      note,
            })

        # Detect whether this is a full two-phase MIP result
        is_full_mip = "N_Displaced_Total" in ms

        if is_full_mip:
            mip_scope_label    = "WholeClass + SubGroup (two-phase)"
            mip_events_val     = _v(ms.get("N_Displaced_Total"), "int")
            mip_reschedule_val = _v(ms.get("N_Rescheduled_Total"), "int")
            mip_clash_before   = _v(
                (ms.get("N_Clash_Pairs_Before_P1") or 0) +
                (ms.get("N_Clash_Pairs_Before_P2") or 0), "int"
            )
            mip_clash_after    = _v(
                (ms.get("N_Clash_Pairs_After_P1") or 0) +
                (ms.get("N_Clash_Pairs_After_P2") or 0), "int"
            )
            mip_objective      = _v(ms.get("Objective_Total"), "float")
            mip_solve_time     = (
                f"{ms.get('Phase1_Solve_Time_s','?')}s + "
                f"{ms.get('Phase2_Solve_Time_s','?')}s"
            )
            mip_x_vars = (
                f"{_v(ms.get('N_X_Variables_Phase1'),'int')} + "
                f"{_v(ms.get('N_X_Variables_Phase2'),'int')}"
            )
            mip_z_vars = (
                f"{_v(ms.get('N_Z_Variables_Phase1'),'int')} + "
                f"{_v(ms.get('N_Z_Variables_Phase2'),'int')}"
            )
        else:
            mip_scope_label    = "WholeClass only (legacy)"
            mip_events_val     = _v(ms.get("N_Displaced_Events_MIP"), "int")
            mip_reschedule_val = _v(ms.get("N_Rescheduled_Events"), "int")
            mip_clash_before   = _v(ms.get("N_Clash_Pairs_Before"), "int")
            mip_clash_after    = _v(ms.get("N_Clash_Pairs_After_MIP"), "int")
            mip_objective      = _v(ms.get("Objective_Value"), "float")
            mip_solve_time     = _v(ms.get("Solve_Time_s"), "float")
            mip_x_vars         = _v(ms.get("N_X_Variables"), "int")
            mip_z_vars         = _v(ms.get("N_Z_Variables"), "int")

        # Scope
        add("Event_Scope",
            mip_scope_label,
            "All displaced events",
            "MIP mode: two-phase covers all; use --mip-wholeclass-only for legacy mode")
        add("Events_Processed",
            mip_events_val,
            _v(hs.get("N_Displaced_Processed"), "int"),
            "Number of displaced events actually passed to each model")
        add("Events_Rescheduled",
            mip_reschedule_val,
            _v(hs.get("N_Rescheduled"), "int"),
            "Events that received a new timeslot in the solution")
        add("N_NoSlot",
            "N/A",
            _v(hs.get("N_NoSlot"), "int"),
            "Events with duration too long for any allowed slot (heuristic only)")
        add("N_NoSlot_WholeClass",
            "N/A",
            _v(hs.get("N_NoSlot_WholeClass"), "int"),
            "WholeClass subset of NO_SLOT events")

        # Clash quality
        add("Clash_Pairs_Before_Optimisation",
            mip_clash_before,
            "N/A",
            "disp–disp conflict pairs entering MIP; heuristic does not pre-count this")
        add("Clash_Score_After",
            mip_objective,
            _v(hs.get("LocalSearch_Clash_Score"), "float"),
            "Weighted clash score (Σ shared_students for clashing disp–disp pairs); "
            "same metric — directly comparable")
        add("Clash_Pairs_After_Optimisation",
            mip_clash_after,
            "N/A",
            "Pairs where z[e1,e2]=1 after MIP solve; "
            "heuristic reports weighted score, not pair count")

        # Heuristic-only improvement
        add("Greedy_Clash_Score",
            "N/A",
            _v(hs.get("Greedy_Clash_Score"), "float"),
            "Clash score after greedy phase, before local search (heuristic only)")
        add("LocalSearch_Improvement_Pct",
            "N/A",
            _v(hs.get("Improvement_Pct"), "float"),
            "(Greedy − LocalSearch) / Greedy × 100%; measures local search gain")
        add("Greedy_ClashFree_WholeClass_Pct",
            "N/A",
            _v(hs.get("Greedy_ClashFree_WholeClass_Pct"), "float"),
            "% of WholeClass events placed clash-free by greedy phase")

        # Feasibility / solver quality
        add("Solve_Status",
            _v(ms.get("Solve_Status")),
            "FEASIBLE" if hs else "N/A",
            "OPTIMAL/FEASIBLE/INFEASIBLE for MIP; heuristic always produces a solution")
        add("Solve_Time_s",
            mip_solve_time,
            "N/A",
            "Wall-clock solver time (MIP only; two-phase shows Phase1s + Phase2s)")
        add("MIP_X_Variables",
            mip_x_vars,
            "N/A",
            "Binary assignment variables (two-phase: Phase1 + Phase2 counts)")
        add("MIP_Z_Variables",
            mip_z_vars,
            "N/A",
            "Binary clash indicator variables (two-phase: Phase1 + Phase2 counts)")

        # Lunch feasibility
        add("Lunch_Free_Pct",
            _v(ms.get("Lunch_Free_Pct"), "float"),
            _v(hs.get("Lunch_Free_Pct"), "float"),
            "% students with free 12-14 lunch slot after rescheduling")
        add("Slot_Balance_CV",
            "N/A",
            _v(hs.get("Slot_Balance_CV"), "float"),
            "Coefficient of variation of events-per-slot (heuristic only); "
            "lower = more balanced distribution")

    df = pd.DataFrame(rows)

    # Console print
    print("\n" + "=" * 90)
    print("MODEL COMPARISON: MIP vs Heuristic")
    print("=" * 90)
    col_w = [42, 30, 22]
    for sc in scenarios:
        sc_df = df[df["Scenario"] == sc][["Metric", "MIP", "Heuristic"]]
        print(f"\n── Scenario: {sc} ──")
        header = f"{'Metric':<{col_w[0]}}  {'MIP':>{col_w[1]}}  {'Heuristic':>{col_w[2]}}"
        print(header)
        print("-" * (sum(col_w) + 4))
        for _, r in sc_df.iterrows():
            print(f"{r['Metric']:<{col_w[0]}}  {str(r['MIP']):>{col_w[1]}}  "
                  f"{str(r['Heuristic']):>{col_w[2]}}")
    print("=" * 90)
    print("Note: Clash_Score_After uses the same weighted disp–disp metric for both models.")
    print("      Default MIP = two-phase (WholeClass + SubGroup); "
          "use --mip-wholeclass-only for legacy.")

    # Save CSV
    out_path = save_dir / "model_comparison.csv"
    df.to_csv(out_path, index=False)
    try:
        display_path = out_path.relative_to(out_dir.parent)
    except ValueError:
        display_path = out_path
    print(f"\n[comparison] Saved → {display_path}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Final summary
# ─────────────────────────────────────────────────────────────────────────────

def print_final_summary(baseline_results: dict,
                         mip_results: dict,
                         heuristic_results: dict,
                         run_dir: Path,
                         out_dir: Path):
    """
    Print a comprehensive summary table answering all research questions (Q1–Q5)
    and save a condensed CSV to run_dir/final_summary.csv.
    """
    print("\n" + "=" * 70)
    print("FINAL RESULTS SUMMARY")
    print("=" * 70)

    rows = []

    # Q1 & Q2: Displaced events + NO_SLOT feasibility
    disp   = baseline_results.get("displaced_summary")
    noslot = baseline_results.get("noslot_feasibility")
    if disp is not None:
        for sc in ["S1_9am5pm", "S2_NoFriPM"]:
            row = disp[disp["Scenario"] == sc]
            if len(row):
                r = row.iloc[0]
                q_label = "Q1" if sc == "S1_9am5pm" else "Q2"
                print(f"\n[{q_label}] Scenario {sc}:")
                print(f"  Displaced events: {r['Displaced_Events']:,} / {r['Total_Events']:,} "
                      f"({r['Displaced_Pct']:.1f}%)")
                print(f"  WholeClass displaced: {r['Displaced_WholeClass']:,}")
                print(f"  Unique modules affected: {r['Displaced_Unique_Modules']:,}")
                if noslot is not None:
                    ns_row = noslot[noslot["Scenario"] == sc]
                    if len(ns_row):
                        ns = ns_row.iloc[0]
                        print(f"  NO_SLOT (too long for window): {int(ns['N_NoSlot']):,} "
                              f"({ns['NoSlot_Pct']:.1f}%)  — "
                              f"WholeClass: {int(ns['N_NoSlot_WholeClass']):,}")
                rows.append({
                    "RQ":      q_label,
                    "Scenario": sc,
                    "Metric":  "Displaced_Events_Pct",
                    "Value":   r["Displaced_Pct"],
                })

    # Q3: Clashes
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
                  f"(status: {mr.get('status', 'N/A')})")
        if heuristic_results and sc in heuristic_results:
            hr = heuristic_results[sc]
            sm = hr.get("summary", {})
            print(f"  Heuristic {sc}: greedy={sm.get('Greedy_Clash_Score','N/A')}, "
                  f"local_search={sm.get('LocalSearch_Clash_Score','N/A')}")
            wc_cf  = sm.get("Greedy_ClashFree_WholeClass", "N/A")
            wc_pct = sm.get("Greedy_ClashFree_WholeClass_Pct", "N/A")
            print(f"    → WholeClass clash-free (greedy): {wc_cf} ({wc_pct}%)")

    # Q4: Lunch breaks
    lunch = baseline_results.get("lunch_break")
    if lunch is not None:
        print("\n[Q4] Lunch break (12-2pm) feasibility:")
        for _, row in lunch.iterrows():
            print(f"  {row['Scenario']}: {row['Lunch_Free_Pct']:.1f}% students free for lunch")

    # Q5: Utilisation
    util = baseline_results.get("utilisation_comparison")
    if util is not None:
        print("\n[Q5] Room/timeslot utilisation comparison:")
        for _, row in util.iterrows():
            print(f"  {row['Scenario']}: {row['In_Window_Events']:,} events in window, "
                  f"room utilisation = {row['Room_Utilisation_Pct']:.1f}%")

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

    if heuristic_results:
        print("\n[Q5] Rescheduled event distribution by day (heuristic):")
        for sc in ["S1_9am5pm", "S2_NoFriPM"]:
            if sc in heuristic_results:
                sm = heuristic_results[sc].get("summary", {})
                day_dist = {d: sm.get(f"Rescheduled_{d}", 0)
                            for d in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]}
                print(f"  {sc}: {day_dist}")

    # Save CSV
    if rows:
        dest = (run_dir or out_dir) / "final_summary.csv"
        pd.DataFrame(rows).to_csv(dest, index=False)
        print(f"\n[main] Final summary saved to {dest.relative_to(out_dir.parent)}")

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
