"""
mip_model.py

Purpose:
    Mixed-Integer Programming (MIP) model using the FICO Xpress Python interface.
    The model tries to RESCHEDULE displaced events (those outside the proposed
    time windows) into allowed timeslots, minimising student-level clashes.

Model Description:
    ─────────────────────────────────────────────────────────────────────────────
    SETS & PARAMETERS
    ─────────────────────────────────────────────────────────────────────────────
    E_disp    : set of displaced events (WholeClass lectures, for tractability)
    E_fixed   : set of in-window events (timeslots unchanged)
    T_e       : allowed timeslots for displaced event e (depends on duration &
                scenario constraints)
    C         : set of conflict pairs (e1, e2) sharing at least one student
                restricted to pairs where at least one event is in E_disp

    ─────────────────────────────────────────────────────────────────────────────
    DECISION VARIABLES
    ─────────────────────────────────────────────────────────────────────────────
    x[e, t]   ∈ {0,1}   event e ∈ E_disp assigned to timeslot t ∈ T_e
    z[e1,e2]  ∈ {0,1}   clash indicator: pair (e1,e2) ∈ C clashes after
                         rescheduling (1 = clash occurs, 0 = no clash)
    lunch[s]  ∈ {0,1}   student s has a free lunch slot (12-14) after
                         rescheduling  [computed post-optimisation as soft metric]

    ─────────────────────────────────────────────────────────────────────────────
    CONSTRAINTS
    ─────────────────────────────────────────────────────────────────────────────
    (1) Assignment:  Σ_{t ∈ T_e} x[e,t] = 1    ∀ e ∈ E_disp
        Each displaced event must be assigned to exactly one allowed timeslot.

    (2) Conflict with fixed events (hard):
        x[e, t_fixed] = 0
        if e ∈ E_disp conflicts with e_fixed ∈ E_fixed and
        t_fixed ∈ T_e (the fixed event's current slot is in the allowed set).
        Prevents placing a displaced event at a slot already occupied by a
        conflicting fixed event.

    (3) Clash detection for displaced–displaced pairs:
        x[e1, t] + x[e2, t] ≤ 1 + z[e1,e2]   ∀ (e1,e2) ∈ C_dd, ∀ t ∈ T_e1 ∩ T_e2
        If both events land on the same slot, z=1 (clash is registered).

    ─────────────────────────────────────────────────────────────────────────────
    OBJECTIVE
    ─────────────────────────────────────────────────────────────────────────────
    Minimise: Σ_{(e1,e2) ∈ C_dd} w[e1,e2] · z[e1,e2]

    where w[e1,e2] = number of students shared by e1 and e2 (impact-weighted).

    A lower objective means fewer (and less impactful) student clashes.

    ─────────────────────────────────────────────────────────────────────────────
    SCOPE LIMITATION (for tractability)
    ─────────────────────────────────────────────────────────────────────────────
    With ~32k events and ~930k student-event records, solving the full MIP is
    computationally prohibitive. We therefore:
      • Focus only on WholeClass lecture events (most severe clash type)
      • Limit the problem to the first MAX_EVENTS displaced events if needed
      • Use pre-built conflict_pairs (from data_preprocessing) for conflict data
      • Set a solver time limit (default: 300 seconds)

Outputs:
    outputs/mip_results_{scenario}.csv  → assignment of displaced events
    outputs/mip_summary_{scenario}.csv  → solve status, clash counts, metrics
"""

import xpress as xp
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import time
warnings.filterwarnings("ignore")

OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

DAY_ORDER  = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
DAY_INDEX  = {d: i for i, d in enumerate(DAY_ORDER)}


# Constants
MAX_EVENTS    = 500    # Max displaced events to include in MIP (for tractability)
SOLVER_TLIMIT = 300    # Seconds
MIP_GAP       = 0.02   # 2% optimality gap acceptable

# Teaching hour slots available (whole hours)
ALL_HOURS = list(range(9, 18))   # 9, 10, ..., 17


# Helper: Build allowed timeslots for an event
def get_event_allowed_slots(duration_min: float,
                             scenario: str,
                             current_slot: tuple = None) -> list:
    """
    Return list of (day, start_hour) tuples that are ALLOWED under the scenario
    for an event of given duration.

    Rules:
      - Event must start at a whole hour between 9 and 17
      - Event must END by 17:00 (S1) or 18:00 (S2/baseline)
      - For S2: no events on Friday starting at 12 or later
    """
    end_limit = 17.0 if scenario == "S1_9am5pm" else 18.0
    allowed = []
    for day in DAY_ORDER:
        for sh in ALL_HOURS:
            end_h = sh + duration_min / 60.0
            if end_h > end_limit:
                continue
            if scenario == "S2_NoFriPM" and day == "Friday" and sh + duration_min / 60.0 > 12.0:
                continue
            allowed.append((day, float(sh)))
    return allowed



# Helper: Check if two events overlap given their timeslots
def events_overlap(day1, start1, dur1, day2, start2, dur2) -> bool:
    """Return True if events at (day1, start1, dur1) and (day2, start2, dur2) overlap."""
    if day1 != day2:
        return False
    end1 = start1 + dur1 / 60.0
    end2 = start2 + dur2 / 60.0
    return start1 < end2 and start2 < end1



# Main MIP Model
def run_mip_scenario(scenario: str,
                     events: pd.DataFrame,
                     conflict_pairs: pd.DataFrame,
                     max_events: int = MAX_EVENTS,
                     time_limit: int = SOLVER_TLIMIT,
                     mip_gap: float = MIP_GAP,
                     verbose: bool = True,
                     out_dir: Path = None) -> dict:
    """
    Run the MIP model for a given scenario.

    Parameters
    ----------
    scenario       : "S1_9am5pm" or "S2_NoFriPM"
    events         : full events DataFrame (from data_preprocessing)
    conflict_pairs : DataFrame with Event_A, Event_B, Shared_Students
    max_events     : maximum number of displaced events to include
    time_limit     : solver time limit in seconds
    mip_gap        : MIP relative optimality gap
    verbose        : print solver progress

    Returns
    -------
    dict with keys:
        status, objective, n_displaced, n_clashes, assignment_df, summary
    """
    save_dir = out_dir or OUT_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    displaced_col = f"Displaced_{scenario.split('_')[0]}"  # e.g. "Displaced_S1"
    print(f"\n{'='*60}")
    print(f"MIP MODEL: {scenario}")
    print(f"{'='*60}")

    # ── Step 1: Identify displaced and fixed events ──────────────────────────
    disp_mask   = events[displaced_col]
    disp_events = events[disp_mask & (events["WholeClass"] == True)].copy()
    fixed_events= events[~disp_mask].copy()

    print(f"  Total displaced WholeClass events: {len(disp_events):,}")
    print(f"  Fixed (in-window) events:          {len(fixed_events):,}")

    # Limit problem size for tractability
    if len(disp_events) > max_events:
        print(f"  [MIP] Limiting to {max_events} displaced events (most impacted by student count)")
        # Prioritise events with most students affected
        disp_events = disp_events.nlargest(max_events, "Event_Size")

    disp_ids  = set(disp_events["Event_ID"])
    fixed_ids = set(fixed_events["Event_ID"])

    # ── Step 2: Build allowed timeslot sets ──────────────────────────────────
    # Map Event_ID → list of allowed (day, start_hour) tuples
    slot_map = {}
    for _, row in disp_events.iterrows():
        eid  = row["Event_ID"]
        dur  = row["Duration_min"]
        slots = get_event_allowed_slots(dur, scenario)
        slot_map[eid] = slots

    # ── Step 3: Filter conflict pairs to relevant ones ───────────────────────
    # We need:
    #   (a) disp–fixed conflicts: to block certain slots from displaced events
    #   (b) disp–disp conflicts: to set up clash variables

    if conflict_pairs is not None and len(conflict_pairs) > 0:
        cp = conflict_pairs.copy()
        # Ensure string dtype
        cp["Event_A"] = cp["Event_A"].astype(str)
        cp["Event_B"] = cp["Event_B"].astype(str)

        # disp–fixed conflicts (a)
        df_mask = (
            (cp["Event_A"].isin(disp_ids) & cp["Event_B"].isin(fixed_ids)) |
            (cp["Event_B"].isin(disp_ids) & cp["Event_A"].isin(fixed_ids))
        )
        df_conflicts = cp[df_mask].copy()

        # disp–disp conflicts (b)
        dd_mask = cp["Event_A"].isin(disp_ids) & cp["Event_B"].isin(disp_ids)
        dd_conflicts = cp[dd_mask].copy()

    else:
        print("  [WARNING] No conflict_pairs data. Using module-based conflicts only.")
        # Fallback: events in the same module conflict
        df_conflicts = pd.DataFrame(columns=["Event_A","Event_B","Shared_Students"])
        dd_conflicts = pd.DataFrame(columns=["Event_A","Event_B","Shared_Students"])

    print(f"  Displaced–Fixed conflict pairs: {len(df_conflicts):,}")
    print(f"  Displaced–Displaced conflict pairs: {len(dd_conflicts):,}")

    # Build fixed event time lookup: Event_ID → (day, start_hour, duration)
    fixed_time = {}
    for _, row in fixed_events.iterrows():
        if pd.notna(row.get("Day")) and pd.notna(row.get("Start_Hour")):
            fixed_time[row["Event_ID"]] = (row["Day"], row["Start_Hour"], row["Duration_min"])

    # ── Step 4: Build Xpress problem ─────────────────────────────────────────
    print(f"\n  Building Xpress MIP problem...")
    prob = xp.problem(name=f"TT_TIME_{scenario}")
    prob.controls.outputlog = 1 if verbose else 0
    prob.controls.maxtime   = -time_limit   # negative = wall-clock limit
    prob.controls.miprelstop= mip_gap
    prob.controls.threads   = 4

    # ── Step 5: Pre-compute blocked slots, then create Decision Variables ────
    #
    # FIX: Build the displaced→fixed conflict map BEFORE creating variables so
    # that slots already blocked by fixed-event conflicts are excluded upfront.
    # Previously, x-variables were created for ALL allowed slots and then
    # individually forced to 0 via x[e,t]==0 constraints.  When every allowed
    # slot of an event got blocked that way, the assignment constraint Σx = 1
    # became unsatisfiable → Xpress reported "infeasible due to row R1".
    # By pre-filtering we only create variables for genuinely viable slots and
    # skip events with no viable slots entirely.

    # Build disp→fixed conflict lookup
    df_dict = {}
    for _, row in df_conflicts.iterrows():
        ea, eb = str(row["Event_A"]), str(row["Event_B"])
        if ea in disp_ids and eb in fixed_ids:
            df_dict.setdefault(ea, set()).add(eb)
        elif eb in disp_ids and ea in fixed_ids:
            df_dict.setdefault(eb, set()).add(ea)

    # NOTE: variable names use integer indices (e_idx, t_idx) instead of
    # Event_ID strings to guarantee uniqueness — Event_IDs may share the same
    # prefix characters, which causes Xpress error ?1030 (duplicate column names)
    # when names are built from truncated IDs.
    x = {}
    event_slot_list = {}   # event_id → list of (var, slot_tuple)
    all_x_vars = []
    n_skipped_no_window  = 0   # events with no scenario-allowed slots
    n_skipped_all_blocked = 0  # events whose every slot is blocked by fixed conflicts

    # Build a stable integer index for every displaced event
    eid_to_idx = {eid: i for i, eid in enumerate(sorted(disp_ids))}

    for eid in disp_ids:
        if eid not in slot_map or len(slot_map[eid]) == 0:
            n_skipped_no_window += 1
            continue

        # Look up this event's duration for overlap checks
        disp_row = disp_events[disp_events["Event_ID"] == eid]
        if len(disp_row) == 0:
            continue
        disp_dur = disp_row.iloc[0]["Duration_min"]

        # Filter out slots that overlap with any conflicting fixed event
        fixed_set = df_dict.get(eid, set())
        viable_slots = []
        for slot in slot_map[eid]:
            d_day, d_start = slot
            blocked = any(
                fixed_eid in fixed_time and
                events_overlap(d_day, d_start, disp_dur,
                               *fixed_time[fixed_eid])
                for fixed_eid in fixed_set
            )
            if not blocked:
                viable_slots.append(slot)

        if len(viable_slots) == 0:
            n_skipped_all_blocked += 1
            continue   # skip: no feasible slot exists for this event

        e_idx = eid_to_idx[eid]
        event_slot_list[eid] = []
        for t_idx, slot in enumerate(viable_slots):
            var = xp.var(
                vartype=xp.binary,
                name=f"x_{e_idx}_{t_idx}",   # unique: integer event idx + slot idx
            )
            x[(eid, t_idx)] = var
            event_slot_list[eid].append((var, slot))
            all_x_vars.append(var)

    print(f"  Events skipped (no window slots):   {n_skipped_no_window:,}")
    print(f"  Events skipped (all slots blocked): {n_skipped_all_blocked:,}")
    print(f"  Events entering MIP:                {len(event_slot_list):,}")

    # Clash indicator variables z[e1, e2]
    z = {}
    all_z_vars = []
    dd_list = []  # list of (eid1, eid2, shared_students)

    z_idx = 0   # global counter – guarantees unique z variable names
    for _, row in dd_conflicts.iterrows():
        e1, e2 = str(row["Event_A"]), str(row["Event_B"])
        if e1 not in event_slot_list or e2 not in event_slot_list:
            continue
        shared = int(row["Shared_Students"])
        key = (e1, e2)
        zvar = xp.var(vartype=xp.binary, name=f"z_{z_idx}")   # unique integer index
        z[key] = zvar
        all_z_vars.append(zvar)
        dd_list.append((e1, e2, shared))
        z_idx += 1

    prob.addVariable(all_x_vars + all_z_vars)
    print(f"  Variables: {len(all_x_vars):,} x-vars, {len(all_z_vars):,} z-vars")

    # ── Step 6: Constraints ───────────────────────────────────────────────────
    n_assignment = 0
    n_clash_det  = 0

    # (1) Assignment: each event entering the MIP must use exactly one viable slot
    # (slots already blocked by fixed conflicts were excluded during variable creation)
    for eid, slot_vars in event_slot_list.items():
        prob.addConstraint(xp.Sum(v for v, _ in slot_vars) == 1)
        n_assignment += 1

    print(f"  Assignment constraints: {n_assignment:,}")
    print(f"  (Blocking constraints eliminated by pre-filtering viable slots)")

    # (3) Clash detection: if both e1 and e2 land on same slot → z = 1
    for (e1, e2, _) in dd_list:
        if e1 not in event_slot_list or e2 not in event_slot_list:
            continue
        key = (e1, e2)
        if key not in z:
            continue

        # Find slots that are common to both events (same day and start)
        slots_e1 = {slot: var for var, slot in event_slot_list[e1]}
        slots_e2 = {slot: var for var, slot in event_slot_list[e2]}
        common = set(slots_e1.keys()) & set(slots_e2.keys())

        for slot in common:
            # If both scheduled here, z must be 1
            # x[e1,t] + x[e2,t] <= 1 + z[e1,e2]
            prob.addConstraint(slots_e1[slot] + slots_e2[slot] <= 1 + z[key])
            n_clash_det += 1

    print(f"  Clash-detection constraints: {n_clash_det:,}")

    # ── Step 7: Objective ─────────────────────────────────────────────────────
    # Minimise total weighted clashes (weighted by number of shared students)
    obj_terms = []
    for (e1, e2, shared) in dd_list:
        key = (e1, e2)
        if key in z:
            obj_terms.append(shared * z[key])

    if obj_terms:
        prob.setObjective(xp.Sum(obj_terms), sense=xp.minimize)
    else:
        print("  [INFO] No clash variables – feasibility model only.")
        prob.setObjective(xp.Sum(0), sense=xp.minimize)

    # ── Step 8: Solve ─────────────────────────────────────────────────────────
    print(f"\n  Solving MIP (time limit: {time_limit}s, gap: {mip_gap*100:.0f}%)...")
    t0 = time.time()
    prob.solve()
    solve_time = time.time() - t0

    # ── Step 9: Extract Solution ──────────────────────────────────────────────
    # Use getProbStatus() which returns an integer and exists in all Xpress 9.x
    # versions.  getSolveStatus() was introduced in later sub-versions and is
    # absent in v9.7.0, causing an AttributeError even when the solve succeeded.
    # Safe constant lookup with getattr fallbacks covers version differences.
    status_int = prob.getProbStatus()

    _OPT  = getattr(xp, "mip_optimal",      6)
    _FEAS = getattr(xp, "mip_solution",      5)
    _NONE = getattr(xp, "mip_no_sol_found",  3)
    _LPO  = getattr(xp, "mip_lp_optimal",   2)

    _STATUS_NAMES = {
        getattr(xp, "mip_not_loaded",     0): "NOT_LOADED",
        getattr(xp, "mip_lp_not_optimal", 1): "LP_NOT_OPTIMAL",
        _LPO:  "LP_OPTIMAL_NO_INT",
        _NONE: "INFEASIBLE",
        _FEAS: "FEASIBLE",
        _OPT:  "OPTIMAL",
    }
    status_name = _STATUS_NAMES.get(status_int, f"UNKNOWN({status_int})")

    print(f"\n  Solve status:  {status_name}")
    print(f"  Solve time:    {solve_time:.1f}s")

    assignment_rows = []
    n_clashes_resolved = 0

    if status_int in (_FEAS, _OPT):
        obj_val = prob.getObjVal()
        print(f"  Objective (weighted clashes): {obj_val:.0f}")

        # Extract x values
        x_sol = {}
        for (eid, t_idx), var in x.items():
            x_sol[(eid, t_idx)] = round(prob.getSolution(var))

        for eid, slot_vars in event_slot_list.items():
            for t_idx, (var, (day, sh)) in enumerate(slot_vars):
                if x_sol.get((eid, t_idx), 0) == 1:
                    # Find the original row
                    orig = disp_events[disp_events["Event_ID"] == eid]
                    if len(orig) == 0:
                        continue
                    orig = orig.iloc[0]
                    assignment_rows.append({
                        "Event_ID":       eid,
                        "Module_Code":    orig["Module_Code"],
                        "Module_Name":    orig["Module_Name"],
                        "Event_Type":     orig["Event_Type"],
                        "Duration_min":   orig["Duration_min"],
                        "Event_Size":     orig["Event_Size"],
                        "Original_Day":   orig["Day"],
                        "Original_Start": orig["Start_Hour"],
                        "New_Day":        day,
                        "New_Start_Hour": sh,
                        "WholeClass":     orig["WholeClass"],
                        "Scenario":       scenario,
                    })

        # Count z values
        z_sol = {}
        for key, var in z.items():
            z_sol[key] = round(prob.getSolution(var))

        n_clashes_resolved = sum(z_sol.values())
        print(f"  Clash pairs after rescheduling: {n_clashes_resolved:,}")

    else:
        obj_val = None
        print("  [MIP] No feasible solution found within time limit.")

    assignment_df = pd.DataFrame(assignment_rows)
    if len(assignment_df) > 0:
        assignment_df.to_csv(save_dir / f"mip_assignment_{scenario}.csv", index=False)

    # ── Step 10: Summary ──────────────────────────────────────────────────────
    summary = {
        "Scenario":                  scenario,
        "Solve_Status":              status_name,
        "Solve_Time_s":              round(solve_time, 1),
        "N_Displaced_Events_MIP":    len(event_slot_list),
        "N_X_Variables":             len(all_x_vars),
        "N_Z_Variables":             len(all_z_vars),
        "N_Clash_Pairs_Before":      len(dd_list),
        "N_Clash_Pairs_After_MIP":   n_clashes_resolved,
        "Objective_Value":           obj_val,
        "N_Rescheduled_Events":      len(assignment_rows),
    }

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(save_dir / f"mip_summary_{scenario}.csv", index=False)

    print(f"\n  MIP Summary saved to mip_summary_{scenario}.csv")
    print(f"  Assignment saved to mip_assignment_{scenario}.csv")

    return {
        "status":        status_name,
        "objective":     obj_val,
        "n_displaced":   len(event_slot_list),
        "n_clashes":     n_clashes_resolved,
        "assignment_df": assignment_df,
        "summary":       summary,
    }



# Post-optimisation: Compute metrics from MIP assignment
def compute_mip_metrics(assignment_df: pd.DataFrame,
                        events: pd.DataFrame,
                        student_events: pd.DataFrame,
                        scenario: str,
                        out_dir: Path = None) -> dict:
    """
    Given MIP assignment (new timeslots for displaced events), compute:
      - Lunch break feasibility after rescheduling
      - Timeslot utilisation heatmap data
      - Number of events successfully placed in new window

    Parameters
    ----------
    assignment_df   : output from run_mip_scenario
    events          : full events DataFrame
    student_events  : student-event mapping
    scenario        : "S1_9am5pm" or "S2_NoFriPM"

    Returns
    -------
    dict of computed metrics
    """
    if assignment_df is None or len(assignment_df) == 0:
        return {"error": "No assignment data"}

    save_dir = out_dir or OUT_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    displaced_col = f"Displaced_{scenario.split('_')[0]}"

    # Build 'rescheduled' events dataframe with new timeslots
    fixed_events = events[~events[displaced_col]].copy()
    rescheduled  = assignment_df[["Event_ID","New_Day","New_Start_Hour","Duration_min"]].copy()
    rescheduled.columns = ["Event_ID","Day","Start_Hour","Duration_min"]
    rescheduled["End_Hour"] = rescheduled["Start_Hour"] + rescheduled["Duration_min"]/60.0

    # Combine fixed + rescheduled
    all_scheduled = pd.concat([
        fixed_events[["Event_ID","Day","Start_Hour","End_Hour","Duration_min","Event_Size","WholeClass"]],
        rescheduled[["Event_ID","Day","Start_Hour","End_Hour","Duration_min"]],
    ], ignore_index=True)

    # Lunch feasibility after rescheduling
    lunch_mask = (all_scheduled["Start_Hour"] < 14.0) & (all_scheduled["End_Hour"] > 12.0)
    lunch_event_ids = set(all_scheduled[lunch_mask]["Event_ID"])
    total_students  = student_events["AnonID"].nunique()
    busy_students   = student_events[student_events["Event_ID"].isin(lunch_event_ids)]["AnonID"].nunique()
    lunch_free_pct  = 100 * (total_students - busy_students) / max(total_students, 1)

    metrics = {
        "Scenario":                     scenario,
        "Events_Rescheduled":           len(assignment_df),
        "Lunch_Free_Pct_After_Reschedule": round(lunch_free_pct, 2),
        "Total_Students":               total_students,
        "Students_With_Lunch_Conflict": busy_students,
    }

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(save_dir / f"mip_metrics_{scenario}.csv", index=False)
    return metrics



# Run both scenarios
def run_all_mip_scenarios(data: dict,
                           max_events: int = MAX_EVENTS,
                           time_limit: int = SOLVER_TLIMIT,
                           out_dir: Path = None) -> dict:
    """
    Run MIP for S1_9am5pm and S2_NoFriPM.

    Parameters
    ----------
    data       : dict from data_preprocessing.run_preprocessing()
    max_events : MIP problem size limit
    time_limit : solver time limit per scenario
    out_dir    : Path, optional — directory where CSVs are saved (defaults to OUT_DIR)

    Returns
    -------
    dict: scenario → MIP result dict
    """
    events         = data["events"]
    student_events = data["student_events"]
    conflict_pairs = data.get("conflict_pairs")

    results = {}
    for scenario in ["S1_9am5pm", "S2_NoFriPM"]:
        res = run_mip_scenario(
            scenario=scenario,
            events=events,
            conflict_pairs=conflict_pairs,
            max_events=max_events,
            time_limit=time_limit,
            out_dir=out_dir,
        )
        results[scenario] = res

        # Compute post-optimisation metrics
        if res["assignment_df"] is not None and len(res["assignment_df"]) > 0:
            metrics = compute_mip_metrics(
                res["assignment_df"], events, student_events, scenario,
                out_dir=out_dir,
            )
            print(f"  Post-MIP Lunch Free: {metrics.get('Lunch_Free_Pct_After_Reschedule',0):.1f}%")

    return results


if __name__ == "__main__":
    import argparse
    from data_preprocessing import run_preprocessing

    parser = argparse.ArgumentParser(description="Run MIP model for TIME scenarios")
    parser.add_argument("--scenario", default="both",
                        choices=["S1_9am5pm","S2_NoFriPM","both"])
    parser.add_argument("--max-events", type=int, default=MAX_EVENTS)
    parser.add_argument("--time-limit", type=int, default=SOLVER_TLIMIT)
    parser.add_argument("--no-conflicts", action="store_true")
    args = parser.parse_args()

    data = run_preprocessing(build_conflicts=not args.no_conflicts)

    if args.scenario == "both":
        run_all_mip_scenarios(data, args.max_events, args.time_limit)
    else:
        events         = data["events"]
        conflict_pairs = data.get("conflict_pairs")
        run_mip_scenario(args.scenario, events, conflict_pairs,
                         args.max_events, args.time_limit)
