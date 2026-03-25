"""
mip_full.py

Two-phase MIP model that processes ALL displaced events (WholeClass + SubGroup),
matching the heuristic model's scope.

Strategy
--------
Phase 1 — WholeClass events
    Identical to the original mip_model.run_mip_scenario().
    Produces new timeslots for all displaced WholeClass lectures.

Phase 2 — SubGroup events
    Uses Phase 1 assignments as additional fixed events.
    Any WholeClass event whose Phase 1 slot is known is locked in at its
    NEW slot; any that failed to get assigned is locked at its ORIGINAL slot.
    SubGroup displaced events are then optimised against this combined
    fixed background.

Combined output
    assignment_df containing all rescheduled events (both phases).
    Summary contains aggregate statistics across both phases.

This keeps each individual MIP tractable (~same size as the original single-phase
model) while covering the full displaced-event set.

Outputs saved to <out_dir>/
    mip_full_assignment_<scenario>.csv   — all rescheduled events
    mip_full_summary_<scenario>.csv      — aggregate summary
    mip_phase1_assignment_<scenario>.csv — phase 1 detail
    mip_phase2_assignment_<scenario>.csv — phase 2 detail
"""

import xpress as xp
import pandas as pd
import numpy as np
from pathlib import Path
import time
import warnings
warnings.filterwarnings("ignore")

# Re-use pure helpers and constants from the original MIP module
from mip_model import (
    get_event_allowed_slots,
    events_overlap,
    compute_mip_metrics,
    OUT_DIR,
    DAY_ORDER,
)

# Default limits  (per-phase)
MAX_EVENTS_PHASE    = 500     # max displaced events per phase
SOLVER_TLIMIT_PHASE = 300     # seconds per phase
MIP_GAP             = 0.02    # 2% optimality gap



# Core single-phase solver  (shared by both phases)
def _run_single_mip_phase(
    phase_name: str,
    scenario: str,
    disp_events: pd.DataFrame,
    fixed_events: pd.DataFrame,
    conflict_pairs: pd.DataFrame,
    extra_fixed_time: dict,         # Event_ID → (day, start_hour, duration_min)
    max_events: int,
    time_limit: int,
    mip_gap: float,
    verbose: bool,
    save_dir: Path,
) -> dict:
    """
    Run a single MIP phase for the given displaced / fixed event sets.

    Parameters
    ----------
    phase_name      : label used in print messages ("Phase1_WholeClass", etc.)
    scenario        : "S1_9am5pm" or "S2_NoFriPM"
    disp_events     : displaced events to optimise in this phase
    fixed_events    : in-window events that cannot be moved
    conflict_pairs  : DataFrame[Event_A, Event_B, Shared_Students]
    extra_fixed_time: additional events locked at specific slots (Phase 1 results)
                      used only when building conflict blocking for Phase 2
    max_events      : cap on problem size (top-N by Event_Size)
    time_limit      : Xpress wall-clock limit in seconds
    mip_gap         : MIP relative optimality gap tolerance
    verbose         : whether Xpress prints solver log
    save_dir        : directory for CSVs

    Returns
    -------
    dict with keys: status, objective, n_displaced, n_clashes,
                    assignment_df, summary
    """
    print(f"\n{'─'*60}")
    print(f"  [{phase_name}]  scenario={scenario}")
    print(f"{'─'*60}")

    # Limit problem size
    if len(disp_events) > max_events:
        print(f"  Limiting to {max_events} events (by Event_Size)")
        disp_events = disp_events.nlargest(max_events, "Event_Size")

    # Deduplicate by Event_ID — duplicate rows cause identical xp.var names
    # which Xpress rejects with "Duplicate column names are not allowed".
    n_before = len(disp_events)
    disp_events = disp_events.drop_duplicates(subset=["Event_ID"])
    if len(disp_events) < n_before:
        print(f"  [INFO] Removed {n_before - len(disp_events)} duplicate Event_ID rows")

    disp_ids  = set(disp_events["Event_ID"].astype(str))
    fixed_ids = set(fixed_events["Event_ID"].astype(str))

    print(f"  Displaced events  : {len(disp_ids):,}")
    print(f"  Fixed events      : {len(fixed_ids):,}")
    print(f"  Extra-fixed (locked Phase 1): {len(extra_fixed_time):,}")

    # Allowed timeslot sets
    slot_map = {}
    for _, row in disp_events.iterrows():
        eid = str(row["Event_ID"])
        slots = get_event_allowed_slots(row["Duration_min"], scenario)
        slot_map[eid] = slots

    # Conflict pairs
    if conflict_pairs is not None and len(conflict_pairs) > 0:
        cp = conflict_pairs.copy()
        cp["Event_A"] = cp["Event_A"].astype(str)
        cp["Event_B"] = cp["Event_B"].astype(str)

        # All events that should block slots: fixed_events + extra_fixed
        all_fixed_ids = fixed_ids | set(extra_fixed_time.keys())

        df_mask = (
            (cp["Event_A"].isin(disp_ids) & cp["Event_B"].isin(all_fixed_ids)) |
            (cp["Event_B"].isin(disp_ids) & cp["Event_A"].isin(all_fixed_ids))
        )
        df_conflicts = cp[df_mask].copy()

        dd_mask = cp["Event_A"].isin(disp_ids) & cp["Event_B"].isin(disp_ids)
        dd_conflicts = cp[dd_mask].copy()
    else:
        print("  [WARNING] No conflict_pairs — skipping blocking constraints.")
        df_conflicts = pd.DataFrame(columns=["Event_A", "Event_B", "Shared_Students"])
        dd_conflicts = pd.DataFrame(columns=["Event_A", "Event_B", "Shared_Students"])

    print(f"  Disp–Fixed conflict pairs : {len(df_conflicts):,}")
    print(f"  Disp–Disp  conflict pairs : {len(dd_conflicts):,}")

    # Fixed event time lookup
    # Includes both original fixed_events AND extra_fixed (Phase 1 results)
    fixed_time: dict = {}
    for _, row in fixed_events.iterrows():
        eid = str(row["Event_ID"])
        if pd.notna(row.get("Day")) and pd.notna(row.get("Start_Hour")):
            fixed_time[eid] = (row["Day"], float(row["Start_Hour"]), float(row["Duration_min"]))

    for eid, slot_info in extra_fixed_time.items():
        fixed_time[str(eid)] = slot_info   # (day, start_hour, duration_min)

    # Build disp→fixed conflict lookup
    df_dict: dict = {}
    all_fixed_ids_str = set(fixed_time.keys())
    for _, row in df_conflicts.iterrows():
        ea, eb = str(row["Event_A"]), str(row["Event_B"])
        if ea in disp_ids and eb in all_fixed_ids_str:
            df_dict.setdefault(ea, set()).add(eb)
        elif eb in disp_ids and ea in all_fixed_ids_str:
            df_dict.setdefault(eb, set()).add(ea)

    # Variables — pre-filter viable slots
    x = {}
    event_slot_list: dict = {}
    all_x_vars = []
    n_skipped_no_window   = 0
    n_skipped_all_blocked = 0

    eid_to_idx = {eid: i for i, eid in enumerate(sorted(disp_ids))}

    for _, row in disp_events.iterrows():
        eid = str(row["Event_ID"])
        dur = float(row["Duration_min"])

        if eid not in slot_map or len(slot_map[eid]) == 0:
            n_skipped_no_window += 1
            continue

        fixed_set = df_dict.get(eid, set())
        viable_slots = []
        for slot in slot_map[eid]:
            d_day, d_start = slot
            blocked = any(
                fid in fixed_time and
                events_overlap(d_day, d_start, dur, *fixed_time[fid])
                for fid in fixed_set
            )
            if not blocked:
                viable_slots.append(slot)

        if len(viable_slots) == 0:
            n_skipped_all_blocked += 1
            continue

        e_idx = eid_to_idx[eid]
        event_slot_list[eid] = []
        for t_idx, slot in enumerate(viable_slots):
            var = xp.var(vartype=xp.binary, name=f"x_{e_idx}_{t_idx}")
            x[(eid, t_idx)] = var
            event_slot_list[eid].append((var, slot))
            all_x_vars.append(var)

    print(f"  Skipped (no window)    : {n_skipped_no_window:,}")
    print(f"  Skipped (all blocked)  : {n_skipped_all_blocked:,}")
    print(f"  Events entering MIP    : {len(event_slot_list):,}")

    # Clash indicator variables z
    z: dict = {}
    all_z_vars = []
    dd_list = []
    z_idx = 0

    for _, row in dd_conflicts.iterrows():
        e1, e2 = str(row["Event_A"]), str(row["Event_B"])
        if e1 not in event_slot_list or e2 not in event_slot_list:
            continue
        shared = int(row["Shared_Students"])
        key = (e1, e2)
        zvar = xp.var(vartype=xp.binary, name=f"z_{z_idx}")
        z[key] = zvar
        all_z_vars.append(zvar)
        dd_list.append((e1, e2, shared))
        z_idx += 1

    # Build Xpress problem
    prob = xp.problem(name=f"TT_{phase_name}_{scenario}")
    prob.controls.outputlog  = 1 if verbose else 0
    prob.controls.maxtime    = -time_limit
    prob.controls.miprelstop = mip_gap
    prob.controls.threads    = 4

    prob.addVariable(all_x_vars + all_z_vars)
    print(f"  Variables: {len(all_x_vars):,} x-vars, {len(all_z_vars):,} z-vars")

    # Constraints
    # (1) Assignment
    n_assignment = 0
    for eid, slot_vars in event_slot_list.items():
        prob.addConstraint(xp.Sum(v for v, _ in slot_vars) == 1)
        n_assignment += 1

    # (2) Clash detection (full overlap check)
    n_clash_det = 0
    disp_dur_map = {str(row["Event_ID"]): float(row["Duration_min"])
                    for _, row in disp_events.iterrows()}

    for (e1, e2, _) in dd_list:
        if e1 not in event_slot_list or e2 not in event_slot_list:
            continue
        key = (e1, e2)
        if key not in z:
            continue
        dur1 = disp_dur_map.get(e1, 60.0)
        dur2 = disp_dur_map.get(e2, 60.0)
        slots_e1 = {slot: var for var, slot in event_slot_list[e1]}
        slots_e2 = {slot: var for var, slot in event_slot_list[e2]}
        for (d1, sh1), var1 in slots_e1.items():
            end1 = sh1 + dur1 / 60.0
            for (d2, sh2), var2 in slots_e2.items():
                if d1 != d2:
                    continue
                end2 = sh2 + dur2 / 60.0
                if sh1 < end2 and sh2 < end1:
                    prob.addConstraint(var1 + var2 <= 1 + z[key])
                    n_clash_det += 1

    print(f"  Assignment constraints : {n_assignment:,}")
    print(f"  Clash-det  constraints : {n_clash_det:,}")

    # Objective
    obj_terms = [shared * z[(e1, e2)] for (e1, e2, shared) in dd_list
                 if (e1, e2) in z]
    if obj_terms:
        prob.setObjective(xp.Sum(obj_terms), sense=xp.minimize)
    else:
        print("  [INFO] No clash variables — feasibility model only.")
        prob.setObjective(xp.Sum(0), sense=xp.minimize)

    # Solve
    print(f"\n  Solving (time limit: {time_limit}s, gap: {mip_gap*100:.0f}%) ...")
    t0 = time.time()
    prob.solve()
    solve_time = time.time() - t0

    # Extract solution
    status_int = prob.getProbStatus()
    _OPT  = getattr(xp, "mip_optimal",     6)
    _FEAS = getattr(xp, "mip_solution",     5)
    _STATUS_NAMES = {
        getattr(xp, "mip_not_loaded",     0): "NOT_LOADED",
        getattr(xp, "mip_lp_not_optimal", 1): "LP_NOT_OPTIMAL",
        getattr(xp, "mip_lp_optimal",     2): "LP_OPTIMAL_NO_INT",
        getattr(xp, "mip_no_sol_found",   3): "INFEASIBLE",
        _FEAS: "FEASIBLE",
        _OPT:  "OPTIMAL",
    }
    status_name = _STATUS_NAMES.get(status_int, f"UNKNOWN({status_int})")
    print(f"  Status: {status_name}  |  Time: {solve_time:.1f}s")

    assignment_rows = []
    n_clashes_found = 0
    obj_val = None

    if status_int in (_FEAS, _OPT):
        obj_val = prob.getObjVal()
        print(f"  Objective (weighted clashes): {obj_val:.0f}")

        x_sol = {(eid, t_idx): round(prob.getSolution(var))
                 for (eid, t_idx), var in x.items()}

        for eid, slot_vars in event_slot_list.items():
            for t_idx, (var, (day, sh)) in enumerate(slot_vars):
                if x_sol.get((eid, t_idx), 0) == 1:
                    orig = disp_events[disp_events["Event_ID"].astype(str) == eid]
                    if len(orig) == 0:
                        continue
                    orig = orig.iloc[0]
                    assignment_rows.append({
                        "Event_ID":       eid,
                        "Module_Code":    orig.get("Module_Code", ""),
                        "Module_Name":    orig.get("Module_Name", ""),
                        "Event_Type":     orig.get("Event_Type", ""),
                        "Duration_min":   orig["Duration_min"],
                        "Event_Size":     orig.get("Event_Size", 0),
                        "WholeClass":     orig.get("WholeClass", False),
                        "Original_Day":   orig.get("Day", ""),
                        "Original_Start": orig.get("Start_Hour", ""),
                        "New_Day":        day,
                        "New_Start_Hour": sh,
                        "Scenario":       scenario,
                        "MIP_Phase":      phase_name,
                    })

        z_sol = {key: round(prob.getSolution(var)) for key, var in z.items()}
        n_clashes_found = sum(z_sol.values())
        print(f"  Clash pairs after solve: {n_clashes_found:,}")
    else:
        print(f"  [WARNING] No feasible solution found in {time_limit}s.")

    assignment_df = pd.DataFrame(assignment_rows)
    summary = {
        "Phase":               phase_name,
        "Scenario":            scenario,
        "Solve_Status":        status_name,
        "Solve_Time_s":        round(solve_time, 1),
        "N_Displaced_In":      len(event_slot_list),
        "N_X_Variables":       len(all_x_vars),
        "N_Z_Variables":       len(all_z_vars),
        "N_Clash_Pairs_Before": len(dd_list),
        "N_Clash_Pairs_After": n_clashes_found,
        "Objective_Value":     obj_val,
        "N_Rescheduled":       len(assignment_rows),
    }

    return {
        "status":        status_name,
        "objective":     obj_val,
        "n_displaced":   len(event_slot_list),
        "n_clashes":     n_clashes_found,
        "assignment_df": assignment_df,
        "summary":       summary,
    }



# Two-phase orchestrator for one scenario
def run_mip_full_scenario(
    scenario: str,
    events: pd.DataFrame,
    conflict_pairs: pd.DataFrame,
    max_events: int  = MAX_EVENTS_PHASE,
    time_limit: int  = SOLVER_TLIMIT_PHASE,
    mip_gap: float   = MIP_GAP,
    verbose: bool    = True,
    out_dir: Path    = None,
) -> dict:
    """
    Run the two-phase full MIP for one scenario.

    Phase 1 optimises WholeClass displaced events.
    Phase 2 optimises SubGroup displaced events, locking Phase 1 results.

    Parameters
    ----------
    scenario       : "S1_9am5pm" or "S2_NoFriPM"
    events         : full events DataFrame
    conflict_pairs : DataFrame[Event_A, Event_B, Shared_Students]
    max_events     : per-phase cap on displaced events (top-N by Event_Size)
    time_limit     : per-phase Xpress solver time limit in seconds
    mip_gap        : MIP relative optimality gap tolerance
    verbose        : print Xpress solver log
    out_dir        : save directory for CSV outputs

    Returns
    -------
    dict with keys:
        status              — combined status string
        objective           — sum of Phase 1 + Phase 2 objectives
        n_displaced         — total displaced events handled across both phases
        n_clashes           — total clash pairs across both phases
        assignment_df       — combined assignment DataFrame
        summary             — merged summary dict
        phase1_result       — raw Phase 1 result dict
        phase2_result       — raw Phase 2 result dict
    """
    save_dir = out_dir or OUT_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    displaced_col = f"Displaced_{scenario.split('_')[0]}"
    disp_mask     = events[displaced_col]

    print(f"\n{'='*60}")
    print(f"MIP FULL MODEL (two-phase): {scenario}")
    print(f"{'='*60}")

    total_disp = disp_mask.sum()
    total_wc   = (disp_mask & (events["WholeClass"] == True)).sum()
    total_sg   = (disp_mask & (events["WholeClass"] == False)).sum()
    print(f"  Total displaced events   : {total_disp:,}")
    print(f"    WholeClass (Phase 1)   : {total_wc:,}")
    print(f"    SubGroup   (Phase 2)   : {total_sg:,}")

    fixed_base = events[~disp_mask].copy()

    # Phase 1: WholeClass
    wc_disp = events[disp_mask & (events["WholeClass"] == True)].copy()

    p1 = _run_single_mip_phase(
        phase_name      = "Phase1_WholeClass",
        scenario        = scenario,
        disp_events     = wc_disp,
        fixed_events    = fixed_base,
        conflict_pairs  = conflict_pairs,
        extra_fixed_time= {},           # no extra fixed for Phase 1
        max_events      = max_events,
        time_limit      = time_limit,
        mip_gap         = mip_gap,
        verbose         = verbose,
        save_dir        = save_dir,
    )

    if len(p1["assignment_df"]) > 0:
        p1["assignment_df"].to_csv(
            save_dir / f"mip_phase1_assignment_{scenario}.csv", index=False
        )

    # Build extra_fixed_time from Phase 1 results
    # For every WholeClass displaced event:
    #   - if assigned by Phase 1 → lock at NEW slot
    #   - if NOT assigned (no viable slot) → lock at ORIGINAL slot
    #     (they remain in the timetable at their current time)
    extra_fixed_time: dict = {}
    p1_assigned_ids: set = set()

    if len(p1["assignment_df"]) > 0:
        for _, row in p1["assignment_df"].iterrows():
            eid = str(row["Event_ID"])
            extra_fixed_time[eid] = (
                row["New_Day"],
                float(row["New_Start_Hour"]),
                float(row["Duration_min"]),
            )
            p1_assigned_ids.add(eid)

    # WholeClass displaced events not assigned → keep at original slot
    for _, row in wc_disp.iterrows():
        eid = str(row["Event_ID"])
        if eid not in p1_assigned_ids:
            if pd.notna(row.get("Day")) and pd.notna(row.get("Start_Hour")):
                extra_fixed_time[eid] = (
                    row["Day"],
                    float(row["Start_Hour"]),
                    float(row["Duration_min"]),
                )

    print(f"\n  Phase 1 complete — {len(p1_assigned_ids):,} WholeClass events assigned.")
    print(f"  Locking {len(extra_fixed_time):,} WholeClass events as fixed for Phase 2.")

    # Phase 2: SubGroup
    sg_disp = events[disp_mask & (events["WholeClass"] == False)].copy()

    p2 = _run_single_mip_phase(
        phase_name      = "Phase2_SubGroup",
        scenario        = scenario,
        disp_events     = sg_disp,
        fixed_events    = fixed_base,
        conflict_pairs  = conflict_pairs,
        extra_fixed_time= extra_fixed_time,
        max_events      = max_events,
        time_limit      = time_limit,
        mip_gap         = mip_gap,
        verbose         = verbose,
        save_dir        = save_dir,
    )

    if len(p2["assignment_df"]) > 0:
        p2["assignment_df"].to_csv(
            save_dir / f"mip_phase2_assignment_{scenario}.csv", index=False
        )

    # Merge results
    combined_df = pd.concat(
        [p1["assignment_df"], p2["assignment_df"]],
        ignore_index=True,
    )
    if len(combined_df) > 0:
        combined_df.to_csv(
            save_dir / f"mip_full_assignment_{scenario}.csv", index=False
        )

    total_obj    = (p1["objective"] or 0) + (p2["objective"] or 0)
    total_clashes = p1["n_clashes"] + p2["n_clashes"]

    # Combined status: OPTIMAL only if both phases optimal
    def _status_rank(s):
        return {"OPTIMAL": 2, "FEASIBLE": 1}.get(s, 0)
    combined_status = (
        p1["status"] if _status_rank(p1["status"]) <= _status_rank(p2["status"])
        else p2["status"]
    )

    summary = {
        "Scenario":              scenario,
        "Solve_Status":          combined_status,
        "Phase1_Status":         p1["status"],
        "Phase2_Status":         p2["status"],
        "Phase1_Solve_Time_s":   p1["summary"]["Solve_Time_s"],
        "Phase2_Solve_Time_s":   p2["summary"]["Solve_Time_s"],
        "N_Displaced_WholeClass": p1["summary"]["N_Displaced_In"],
        "N_Displaced_SubGroup":   p2["summary"]["N_Displaced_In"],
        "N_Displaced_Total":      p1["summary"]["N_Displaced_In"] + p2["summary"]["N_Displaced_In"],
        "N_Rescheduled_WholeClass": p1["summary"]["N_Rescheduled"],
        "N_Rescheduled_SubGroup":   p2["summary"]["N_Rescheduled"],
        "N_Rescheduled_Total":      p1["summary"]["N_Rescheduled"] + p2["summary"]["N_Rescheduled"],
        "N_X_Variables_Phase1":  p1["summary"]["N_X_Variables"],
        "N_X_Variables_Phase2":  p2["summary"]["N_X_Variables"],
        "N_Z_Variables_Phase1":  p1["summary"]["N_Z_Variables"],
        "N_Z_Variables_Phase2":  p2["summary"]["N_Z_Variables"],
        "N_Clash_Pairs_Before_P1": p1["summary"]["N_Clash_Pairs_Before"],
        "N_Clash_Pairs_Before_P2": p2["summary"]["N_Clash_Pairs_Before"],
        "N_Clash_Pairs_After_P1":  p1["summary"]["N_Clash_Pairs_After"],
        "N_Clash_Pairs_After_P2":  p2["summary"]["N_Clash_Pairs_After"],
        "Objective_Phase1":      p1["objective"],
        "Objective_Phase2":      p2["objective"],
        "Objective_Total":       total_obj,
    }

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(save_dir / f"mip_full_summary_{scenario}.csv", index=False)

    print(f"\n  ── Full MIP {scenario} complete ──")
    print(f"  WholeClass rescheduled : {p1['summary']['N_Rescheduled']:,}")
    print(f"  SubGroup   rescheduled : {p2['summary']['N_Rescheduled']:,}")
    print(f"  Combined objective     : {total_obj:.0f}")
    print(f"  Combined clash pairs   : {total_clashes:,}")
    print(f"  Saved → mip_full_assignment_{scenario}.csv")

    return {
        "status":        combined_status,
        "objective":     total_obj,
        "n_displaced":   p1["n_displaced"] + p2["n_displaced"],
        "n_clashes":     total_clashes,
        "assignment_df": combined_df,
        "summary":       summary,
        "phase1_result": p1,
        "phase2_result": p2,
    }



# Run both scenarios
def run_all_mip_full_scenarios(
    data: dict,
    max_events: int  = MAX_EVENTS_PHASE,
    time_limit: int  = SOLVER_TLIMIT_PHASE,
    mip_gap: float   = MIP_GAP,
    out_dir: Path    = None,
) -> dict:
    """
    Run two-phase full MIP for both S1_9am5pm and S2_NoFriPM.

    Parameters
    ----------
    data       : dict from data_preprocessing.run_preprocessing()
    max_events : per-phase cap (top-N displaced events by student count)
    time_limit : per-phase Xpress wall-clock limit in seconds
    mip_gap    : MIP relative optimality gap tolerance
    out_dir    : save directory (defaults to OUT_DIR)

    Returns
    -------
    dict: scenario → result dict from run_mip_full_scenario()
    """
    events         = data["events"]
    student_events = data["student_events"]
    conflict_pairs = data.get("conflict_pairs")

    results = {}
    for scenario in ["S1_9am5pm", "S2_NoFriPM"]:
        res = run_mip_full_scenario(
            scenario       = scenario,
            events         = events,
            conflict_pairs = conflict_pairs,
            max_events     = max_events,
            time_limit     = time_limit,
            mip_gap        = mip_gap,
            out_dir        = out_dir,
        )
        results[scenario] = res

        # Post-optimisation metrics (lunch feasibility, etc.)
        if res["assignment_df"] is not None and len(res["assignment_df"]) > 0:
            metrics = compute_mip_metrics(
                res["assignment_df"], events, student_events, scenario,
                out_dir=out_dir,
            )
            lunch_pct = metrics.get("Lunch_Free_Pct_After_Reschedule", 0)
            print(f"  Post-MIP Lunch Free: {lunch_pct:.1f}%")
            # Store for comparison table
            res["summary"]["Lunch_Free_Pct"] = lunch_pct

    return results



# Standalone entry point
if __name__ == "__main__":
    import argparse
    from data_preprocessing import run_preprocessing

    parser = argparse.ArgumentParser(
        description="Run two-phase full MIP (WholeClass + SubGroup) for TIME scenarios"
    )
    parser.add_argument("--scenario",   default="both",
                        choices=["S1_9am5pm", "S2_NoFriPM", "both"])
    parser.add_argument("--max-events", type=int, default=MAX_EVENTS_PHASE)
    parser.add_argument("--time-limit", type=int, default=SOLVER_TLIMIT_PHASE)
    parser.add_argument("--no-conflicts", action="store_true")
    args = parser.parse_args()

    print("Loading data...")
    data = run_preprocessing(
        build_conflicts=not args.no_conflicts,
        out_dir=OUT_DIR / "cleaned_data",
    )

    scenarios = (["S1_9am5pm", "S2_NoFriPM"] if args.scenario == "both"
                 else [args.scenario])
    for sc in scenarios:
        run_mip_full_scenario(
            scenario       = sc,
            events         = data["events"],
            conflict_pairs = data.get("conflict_pairs"),
            max_events     = args.max_events,
            time_limit     = args.time_limit,
        )
