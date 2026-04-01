"""
mip_new.py  —  Extended Two-Phase MIP (Full-Objective: Disp–Disp + Disp–Fixed)

KEY DIFFERENCES from mip_model.py
──────────────────────────────────
1. NO HARD SLOT PRE-FILTERING
   The original model removed any slot that overlaps a conflicting fixed event
   (hard constraint). This model removes that filter — all in-window slots are
   candidate slots for every displaced event.

2. DISP–FIXED CLASH PENALTY IN OBJECTIVE
   For each (displaced event e, fixed event f) conflict pair with weight w[e,f]:
   every candidate slot t of e that overlaps f's slot contributes
       w[e,f] · x[e, t]
   to the objective as a LINEAR term (no new binary variable needed, since f's
   slot is known and fixed).

3. FULL OBJECTIVE
   Minimise:
     Σ_{(e1,e2)∈C_dd} w[e1,e2] · z[e1,e2]            (disp–disp, z-variable)
   + Σ_{(e,f)∈C_df}   w[e,f]   · Σ_{t: t∩f≠∅} x[e,t] (disp–fixed, linear)

   This allows the solver to trade off disp–disp vs disp–fixed clashes and find
   the globally optimal assignment across ALL conflict types.

4. EXTENDED SUMMARY METRICS
   Reports DD_Clash_Score (disp–disp only),
           DF_Clash_Score (disp–fixed only),
           Total_Clash_Score (sum of both).

SETS & PARAMETERS (per phase)
─────────────────────────────────────────────────────────────────────────────
E_disp    : displaced events for this phase
E_fixed   : in-window events (cannot be moved)
T_e       : ALL in-window timeslots for event e (no pre-filtering)
C_dd      : disp–disp conflict pairs  (shared students > 0)
C_df      : disp–fixed conflict pairs (shared students > 0)

DECISION VARIABLES
─────────────────────────────────────────────────────────────────────────────
x[e, t]   ∈ {0,1}   event e assigned to timeslot t
z[e1,e2]  ∈ {0,1}   clash indicator for disp–disp conflict pair (e1, e2)

CONSTRAINTS
─────────────────────────────────────────────────────────────────────────────
(1) Assignment:    Σ_{t ∈ T_e} x[e,t] = 1              ∀ e ∈ E_disp
(2) DD clash det:  x[e1,t1] + x[e2,t2] ≤ 1 + z[e1,e2]
                   ∀ overlapping (t1,t2) pairs for (e1,e2) ∈ C_dd

OBJECTIVE
─────────────────────────────────────────────────────────────────────────────
Minimise: Σ w[e1,e2]·z[e1,e2]  +  Σ w[e,f]·Σ_{t∩f≠∅} x[e,t]
"""

import xpress as xp
import pandas as pd
import numpy as np
from pathlib import Path
import time
import warnings
warnings.filterwarnings("ignore")

OUT_DIR   = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
DAY_INDEX = {d: i for i, d in enumerate(DAY_ORDER)}
ALL_HOURS = list(range(9, 18))

MAX_EVENTS_PHASE    = 500
SOLVER_TLIMIT_PHASE = 300
MIP_GAP             = 0.02


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers (identical to mip_model.py)
# ─────────────────────────────────────────────────────────────────────────────

def get_event_allowed_slots(duration_min: float, scenario: str) -> list:
    end_limit = 17.0 if scenario == "S1_9am5pm" else 18.0
    allowed = []
    for day in DAY_ORDER:
        for sh in ALL_HOURS:
            end_h = sh + duration_min / 60.0
            if end_h > end_limit:
                continue
            if scenario == "S2_NoFriPM" and day == "Friday" and end_h > 12.0:
                continue
            allowed.append((day, float(sh)))
    return allowed


def events_overlap(day1, start1, dur1, day2, start2, dur2) -> bool:
    if day1 != day2:
        return False
    end1 = start1 + dur1 / 60.0
    end2 = start2 + dur2 / 60.0
    return start1 < end2 and start2 < end1


def compute_mip_metrics(assignment_df, events, student_events, scenario, out_dir=None):
    """Post-optimisation lunch feasibility (identical to mip_model.py)."""
    if assignment_df is None or len(assignment_df) == 0:
        return {"error": "No assignment data"}
    save_dir = out_dir or OUT_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    displaced_col = f"Displaced_{scenario.split('_')[0]}"
    fixed_events  = events[~events[displaced_col]].copy()

    rescheduled = assignment_df[["Event_ID","New_Day","New_Start_Hour","Duration_min"]].copy()
    rescheduled.columns = ["Event_ID","Day","Start_Hour","Duration_min"]
    rescheduled["End_Hour"] = rescheduled["Start_Hour"] + rescheduled["Duration_min"] / 60.0

    all_scheduled = pd.concat([
        fixed_events[["Event_ID","Day","Start_Hour","End_Hour","Duration_min","Event_Size","WholeClass"]],
        rescheduled[["Event_ID","Day","Start_Hour","End_Hour","Duration_min"]],
    ], ignore_index=True)

    lunch_mask      = (all_scheduled["Start_Hour"] < 14.0) & (all_scheduled["End_Hour"] > 12.0)
    lunch_event_ids = set(all_scheduled[lunch_mask]["Event_ID"])
    total_students  = student_events["AnonID"].nunique()
    busy_students   = student_events[student_events["Event_ID"].isin(lunch_event_ids)]["AnonID"].nunique()
    lunch_free_pct  = 100 * (total_students - busy_students) / max(total_students, 1)

    metrics = {
        "Scenario":                        scenario,
        "Events_Rescheduled":              len(assignment_df),
        "Lunch_Free_Pct_After_Reschedule": round(lunch_free_pct, 2),
        "Total_Students":                  total_students,
        "Students_With_Lunch_Conflict":    busy_students,
    }
    pd.DataFrame([metrics]).to_csv(save_dir / f"mip_new_metrics_{scenario}.csv", index=False)
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Core single-phase solver  *** KEY CHANGES HERE ***
# ─────────────────────────────────────────────────────────────────────────────

def _run_single_mip_phase_new(
    phase_name:       str,
    scenario:         str,
    disp_events:      pd.DataFrame,
    fixed_events:     pd.DataFrame,
    conflict_pairs:   pd.DataFrame,
    extra_fixed_time: dict,
    max_events:       int,
    time_limit:       int,
    mip_gap:          float,
    verbose:          bool,
    save_dir:         Path,
) -> dict:
    """
    Run one MIP phase with the full (disp–disp + disp–fixed) objective.

    CHANGE vs _run_single_mip_phase in mip_model.py
    ────────────────────────────────────────────────
    • No hard slot pre-filtering for disp–fixed conflicts.
    • Linear disp–fixed penalty terms added to MIP objective.
    • Summary reports DD, DF, and Total clash scores separately.
    """
    print(f"\n{'─'*60}")
    print(f"  [{phase_name}]  scenario={scenario}  [FULL OBJECTIVE]")
    print(f"{'─'*60}")

    # Cap problem size
    if len(disp_events) > max_events:
        print(f"  Limiting to {max_events} events (by Event_Size)")
        disp_events = disp_events.nlargest(max_events, "Event_Size")

    n_before    = len(disp_events)
    disp_events = disp_events.drop_duplicates(subset=["Event_ID"])
    if len(disp_events) < n_before:
        print(f"  [INFO] Removed {n_before - len(disp_events)} duplicate Event_ID rows")

    disp_ids  = set(disp_events["Event_ID"].astype(str))
    fixed_ids = set(fixed_events["Event_ID"].astype(str))

    print(f"  Displaced events         : {len(disp_ids):,}")
    print(f"  Fixed events             : {len(fixed_ids):,}")
    print(f"  Extra-fixed (Phase 1)    : {len(extra_fixed_time):,}")

    # Allowed timeslot sets (ALL in-window slots, no pre-filtering)
    slot_map: dict = {}
    for _, row in disp_events.iterrows():
        eid = str(row["Event_ID"])
        slot_map[eid] = get_event_allowed_slots(row["Duration_min"], scenario)

    # Split conflict pairs into disp–disp and disp–fixed
    if conflict_pairs is not None and len(conflict_pairs) > 0:
        cp = conflict_pairs.copy()
        cp["Event_A"] = cp["Event_A"].astype(str)
        cp["Event_B"] = cp["Event_B"].astype(str)
        all_fixed_ids = fixed_ids | set(extra_fixed_time.keys())

        # disp–fixed: one side displaced, one side fixed
        df_mask = (
            (cp["Event_A"].isin(disp_ids) & cp["Event_B"].isin(all_fixed_ids)) |
            (cp["Event_B"].isin(disp_ids) & cp["Event_A"].isin(all_fixed_ids))
        )
        df_conflicts = cp[df_mask].copy()

        # disp–disp: both sides displaced
        dd_mask      = cp["Event_A"].isin(disp_ids) & cp["Event_B"].isin(disp_ids)
        dd_conflicts = cp[dd_mask].copy()
    else:
        print("  [WARNING] No conflict_pairs — objective has no clash terms.")
        df_conflicts = pd.DataFrame(columns=["Event_A","Event_B","Shared_Students"])
        dd_conflicts = pd.DataFrame(columns=["Event_A","Event_B","Shared_Students"])

    print(f"  Disp–Fixed conflict pairs : {len(df_conflicts):,}")
    print(f"  Disp–Disp  conflict pairs : {len(dd_conflicts):,}")

    # Fixed event time lookup
    fixed_time: dict = {}
    for _, row in fixed_events.iterrows():
        eid = str(row["Event_ID"])
        if pd.notna(row.get("Day")) and pd.notna(row.get("Start_Hour")):
            fixed_time[eid] = (row["Day"], float(row["Start_Hour"]), float(row["Duration_min"]))
    for eid, slot_info in extra_fixed_time.items():
        fixed_time[str(eid)] = slot_info

    all_fixed_ids_str = set(fixed_time.keys())

    # ── Build x decision variables (ALL in-window slots, NO pre-filtering) ──
    x:              dict = {}
    event_slot_list:     dict = {}
    all_x_vars:          list = []
    n_skipped_none:      int  = 0
    eid_to_idx = {eid: i for i, eid in enumerate(sorted(disp_ids))}

    disp_dur_map = {str(row["Event_ID"]): float(row["Duration_min"])
                    for _, row in disp_events.iterrows()}

    for _, row in disp_events.iterrows():
        eid = str(row["Event_ID"])
        dur = float(row["Duration_min"])

        if eid not in slot_map or not slot_map[eid]:
            n_skipped_none += 1
            continue

        # *** CHANGE: use ALL in-window slots (no disp-fixed pre-filtering) ***
        viable_slots = slot_map[eid]

        e_idx = eid_to_idx[eid]
        event_slot_list[eid] = []
        for t_idx, slot in enumerate(viable_slots):
            var = xp.var(vartype=xp.binary, name=f"x_{e_idx}_{t_idx}")
            x[(eid, t_idx)] = var
            event_slot_list[eid].append((var, slot))
            all_x_vars.append(var)

    print(f"  Skipped (no window slots) : {n_skipped_none:,}")
    print(f"  Events entering MIP        : {len(event_slot_list):,}")

    # ── Build z clash-indicator variables (disp–disp only) ──
    z:          dict = {}
    all_z_vars: list = []
    dd_list:    list = []
    z_idx = 0

    for _, row in dd_conflicts.iterrows():
        e1, e2 = str(row["Event_A"]), str(row["Event_B"])
        if e1 not in event_slot_list or e2 not in event_slot_list:
            continue
        shared = int(row["Shared_Students"])
        key    = (e1, e2)
        zvar   = xp.var(vartype=xp.binary, name=f"z_{z_idx}")
        z[key] = zvar
        all_z_vars.append(zvar)
        dd_list.append((e1, e2, shared))
        z_idx += 1

    # ── Build disp–fixed linear penalty terms ── *** NEW ***
    df_penalty_terms: list = []
    df_obj_pairs:     list = []   # (e_disp, e_fixed, shared) for post-solve scoring

    for _, row in df_conflicts.iterrows():
        ea, eb   = str(row["Event_A"]), str(row["Event_B"])
        shared   = int(row["Shared_Students"])

        # Determine which is displaced, which is fixed
        if ea in disp_ids and eb in all_fixed_ids_str:
            e_disp, e_fixed = ea, eb
        elif eb in disp_ids and ea in all_fixed_ids_str:
            e_disp, e_fixed = eb, ea
        else:
            continue

        if e_disp not in event_slot_list or e_fixed not in fixed_time:
            continue

        day_f, start_f, dur_f = fixed_time[e_fixed]
        dur_d = disp_dur_map.get(e_disp, 60.0)

        # Add w[e,f] * x[e,t] for each slot t of e_disp that overlaps f's slot
        for (var, (day_t, sh_t)) in event_slot_list[e_disp]:
            if events_overlap(day_t, sh_t, dur_d, day_f, start_f, dur_f):
                df_penalty_terms.append(shared * var)

        df_obj_pairs.append((e_disp, e_fixed, shared))

    print(f"  DD obj terms (z-vars)      : {len(dd_list):,}")
    print(f"  DF obj terms (linear)      : {len(df_penalty_terms):,}")

    # ── Build Xpress problem ──
    prob = xp.problem(name=f"TT_NEW_{phase_name}_{scenario}")
    prob.controls.outputlog  = 1 if verbose else 0
    prob.controls.maxtime    = -time_limit
    prob.controls.miprelstop = mip_gap
    prob.controls.threads    = 4

    prob.addVariable(all_x_vars + all_z_vars)
    print(f"  Variables: {len(all_x_vars):,} x-vars, {len(all_z_vars):,} z-vars")

    # Constraint (1): Assignment — each displaced event assigned to exactly one slot
    n_assignment = 0
    for eid, slot_vars in event_slot_list.items():
        prob.addConstraint(xp.Sum(v for v, _ in slot_vars) == 1)
        n_assignment += 1

    # Constraint (2): Disp–disp clash detection
    n_clash_det = 0
    for (e1, e2, _) in dd_list:
        if e1 not in event_slot_list or e2 not in event_slot_list:
            continue
        key = (e1, e2)
        if key not in z:
            continue
        dur1     = disp_dur_map.get(e1, 60.0)
        dur2     = disp_dur_map.get(e2, 60.0)
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

    print(f"  Assignment constraints  : {n_assignment:,}")
    print(f"  DD clash-det constraints: {n_clash_det:,}")

    # ── Objective: disp–disp (z-vars) + disp–fixed (linear) ── *** CHANGE ***
    dd_obj_terms = [shared * z[(e1, e2)] for (e1, e2, shared) in dd_list if (e1, e2) in z]
    obj_terms    = dd_obj_terms + df_penalty_terms

    if obj_terms:
        prob.setObjective(xp.Sum(obj_terms), sense=xp.minimize)
        print(f"  Objective: {len(dd_obj_terms)} DD terms + {len(df_penalty_terms)} DF terms")
    else:
        prob.setObjective(xp.Sum(0), sense=xp.minimize)
        print("  [INFO] No clash variables — feasibility model only.")

    # ── Solve ──
    print(f"\n  Solving (time limit: {time_limit}s, gap: {mip_gap*100:.0f}%) ...")
    t0         = time.time()
    prob.solve()
    solve_time = time.time() - t0

    # ── Extract solution ──
    status_int = prob.getProbStatus()
    _OPT  = getattr(xp, "mip_optimal",  6)
    _FEAS = getattr(xp, "mip_solution",  5)
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

    assignment_rows   = []
    dd_clashes_found  = 0
    df_clash_score_after = 0
    obj_val           = None
    assigned_slots:   dict = {}   # eid → (day, sh)

    if status_int in (_FEAS, _OPT):
        obj_val = prob.getObjVal()
        print(f"  Objective (total weighted clashes): {obj_val:.0f}")

        x_sol = {(eid, t_idx): round(prob.getSolution(var))
                 for (eid, t_idx), var in x.items()}

        for eid, slot_vars in event_slot_list.items():
            for t_idx, (var, (day, sh)) in enumerate(slot_vars):
                if x_sol.get((eid, t_idx), 0) == 1:
                    assigned_slots[eid] = (day, sh)
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

        # Post-solve: disp–disp clash count (from z variables)
        z_sol = {key: round(prob.getSolution(var)) for key, var in z.items()}
        dd_clashes_found = sum(z_sol.values())

        # Post-solve: disp–fixed clash score (recompute from assignments)
        for (e_disp, e_fixed, shared) in df_obj_pairs:
            if e_disp not in assigned_slots or e_fixed not in fixed_time:
                continue
            day_d, sh_d    = assigned_slots[e_disp]
            day_f, sh_f, dur_f = fixed_time[e_fixed]
            dur_d = disp_dur_map.get(e_disp, 60.0)
            if events_overlap(day_d, sh_d, dur_d, day_f, sh_f, dur_f):
                df_clash_score_after += shared

        dd_clash_score_after = sum(
            shared for (e1, e2, shared) in dd_list if z_sol.get((e1, e2), 0) == 1
        )
        print(f"  DD clash pairs after solve : {dd_clashes_found:,}")
        print(f"  DD clash score after solve : {dd_clash_score_after:,}")
        print(f"  DF clash score after solve : {df_clash_score_after:,}")
        print(f"  Total clash score          : {dd_clash_score_after + df_clash_score_after:,}")
    else:
        print(f"  [WARNING] No feasible solution found in {time_limit}s.")
        dd_clash_score_after = None

    summary = {
        "Phase":                  phase_name,
        "Scenario":               scenario,
        "Solve_Status":           status_name,
        "Solve_Time_s":           round(solve_time, 1),
        "N_Displaced_In":         len(event_slot_list),
        "N_X_Variables":          len(all_x_vars),
        "N_Z_Variables":          len(all_z_vars),
        "N_DF_Penalty_Terms":     len(df_penalty_terms),
        "N_DD_Clash_Pairs_Before": len(dd_list),
        "N_DF_Clash_Pairs":       len(df_obj_pairs),
        "N_DD_Clash_Pairs_After": dd_clashes_found,
        "DD_Clash_Score_After":   dd_clash_score_after,
        "DF_Clash_Score_After":   df_clash_score_after,
        "Total_Clash_Score_After": (
            (dd_clash_score_after or 0) + df_clash_score_after
            if dd_clash_score_after is not None else None
        ),
        "Objective_Value":        obj_val,
        "N_Rescheduled":          len(assignment_rows),
    }

    return {
        "status":          status_name,
        "objective":       obj_val,
        "n_displaced":     len(event_slot_list),
        "n_clashes":       dd_clashes_found,
        "dd_clash_score":  dd_clash_score_after,
        "df_clash_score":  df_clash_score_after,
        "assignment_df":   pd.DataFrame(assignment_rows),
        "assigned_slots":  assigned_slots,
        "summary":         summary,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Two-phase orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def _run_mip_two_phase(
    scenario:         str,
    semester_label:   str,
    events_sem:       pd.DataFrame,
    conflict_pairs:   pd.DataFrame,
    student_events:   pd.DataFrame,
    max_events:       int,
    time_limit:       int,
    mip_gap:          float,
    verbose:          bool,
    save_dir:         Path,
) -> dict:
    """
    Internal helper: run the two-phase MIP for one (scenario, semester) slice.

    Phase 1 handles WholeClass displaced events; Phase 2 handles SubGroup.
    Returns a dict with combined summary and assignment DataFrame.
    """
    displaced_col = f"Displaced_{scenario.split('_')[0]}"
    disp_mask     = events_sem[displaced_col]

    total_disp = int(disp_mask.sum())
    total_wc   = int((disp_mask & (events_sem["WholeClass"] == True)).sum())
    total_sg   = int((disp_mask & (events_sem["WholeClass"] == False)).sum())
    print(f"\n  [{semester_label}] Displaced: {total_disp:,}  (WC={total_wc:,}, SG={total_sg:,})")

    fixed_base = events_sem[~disp_mask].copy()
    wc_disp    = events_sem[disp_mask & (events_sem["WholeClass"] == True)].copy()

    # Phase 1: WholeClass
    p1 = _run_single_mip_phase_new(
        phase_name       = f"Phase1_WC_{semester_label}",
        scenario         = scenario,
        disp_events      = wc_disp,
        fixed_events     = fixed_base,
        conflict_pairs   = conflict_pairs,
        extra_fixed_time = {},
        max_events       = max_events,
        time_limit       = time_limit,
        mip_gap          = mip_gap,
        verbose          = verbose,
        save_dir         = save_dir,
    )

    if len(p1["assignment_df"]) > 0:
        p1["assignment_df"].to_csv(
            save_dir / f"mip_new_phase1_assignment_{scenario}_{semester_label}.csv",
            index=False)

    # Lock Phase 1 results for Phase 2
    extra_fixed_time: dict = {}
    p1_assigned_ids:  set  = set()

    for _, row in p1["assignment_df"].iterrows():
        eid = str(row["Event_ID"])
        extra_fixed_time[eid] = (row["New_Day"], float(row["New_Start_Hour"]),
                                  float(row["Duration_min"]))
        p1_assigned_ids.add(eid)

    for _, row in wc_disp.iterrows():
        eid = str(row["Event_ID"])
        if eid not in p1_assigned_ids:
            if pd.notna(row.get("Day")) and pd.notna(row.get("Start_Hour")):
                extra_fixed_time[eid] = (row["Day"], float(row["Start_Hour"]),
                                          float(row["Duration_min"]))

    print(f"  [{semester_label}] Phase 1 done — {len(p1_assigned_ids):,} WC events assigned.")

    # Phase 2: SubGroup
    sg_disp = events_sem[disp_mask & (events_sem["WholeClass"] == False)].copy()

    p2 = _run_single_mip_phase_new(
        phase_name       = f"Phase2_SG_{semester_label}",
        scenario         = scenario,
        disp_events      = sg_disp,
        fixed_events     = fixed_base,
        conflict_pairs   = conflict_pairs,
        extra_fixed_time = extra_fixed_time,
        max_events       = max_events,
        time_limit       = time_limit,
        mip_gap          = mip_gap,
        verbose          = verbose,
        save_dir         = save_dir,
    )

    if len(p2["assignment_df"]) > 0:
        p2["assignment_df"].to_csv(
            save_dir / f"mip_new_phase2_assignment_{scenario}_{semester_label}.csv",
            index=False)

    combined_df      = pd.concat([p1["assignment_df"], p2["assignment_df"]],
                                  ignore_index=True)
    total_obj        = (p1["objective"] or 0) + (p2["objective"] or 0)
    total_dd_clashes = p1["n_clashes"] + p2["n_clashes"]
    total_dd_score   = (p1["dd_clash_score"] or 0) + (p2["dd_clash_score"] or 0)
    total_df_score   = (p1["df_clash_score"] or 0) + (p2["df_clash_score"] or 0)

    def _rank(s):
        return {"OPTIMAL": 2, "FEASIBLE": 1}.get(s, 0)
    combined_status = p1["status"] if _rank(p1["status"]) <= _rank(p2["status"]) else p2["status"]

    sem_summary = {
        "Semester":                  semester_label,
        "Solve_Status":              combined_status,
        "N_Displaced_WholeClass":    p1["summary"]["N_Displaced_In"],
        "N_Displaced_SubGroup":      p2["summary"]["N_Displaced_In"],
        "N_Displaced_Total":         p1["summary"]["N_Displaced_In"] + p2["summary"]["N_Displaced_In"],
        "N_Rescheduled_WholeClass":  p1["summary"]["N_Rescheduled"],
        "N_Rescheduled_SubGroup":    p2["summary"]["N_Rescheduled"],
        "N_Rescheduled_Total":       p1["summary"]["N_Rescheduled"] + p2["summary"]["N_Rescheduled"],
        "DD_Clash_Score_P1":         p1["summary"]["DD_Clash_Score_After"],
        "DD_Clash_Score_P2":         p2["summary"]["DD_Clash_Score_After"],
        "DD_Clash_Score_Total":      total_dd_score,
        "DF_Clash_Score_P1":         p1["summary"]["DF_Clash_Score_After"],
        "DF_Clash_Score_P2":         p2["summary"]["DF_Clash_Score_After"],
        "DF_Clash_Score_Total":      total_df_score,
        "Total_Clash_Score":         total_dd_score + total_df_score,
        "Objective_Total":           total_obj,
    }

    print(f"  [{semester_label}] DD={total_dd_score:,}  DF={total_df_score:,}  "
          f"Total={total_dd_score + total_df_score:,}")

    return {
        "status":           combined_status,
        "objective":        total_obj,
        "n_displaced":      p1["n_displaced"] + p2["n_displaced"],
        "n_clashes":        total_dd_clashes,
        "dd_clash_score":   total_dd_score,
        "df_clash_score":   total_df_score,
        "total_clash_score": total_dd_score + total_df_score,
        "assignment_df":    combined_df,
        "sem_summary":      sem_summary,
        "phase1_result":    p1,
        "phase2_result":    p2,
    }


def run_mip_scenario_new(
    scenario:       str,
    events:         pd.DataFrame,
    conflict_pairs: pd.DataFrame,
    student_events: pd.DataFrame = None,
    max_events:     int   = MAX_EVENTS_PHASE,
    time_limit:     int   = SOLVER_TLIMIT_PHASE,
    mip_gap:        float = MIP_GAP,
    verbose:        bool  = True,
    out_dir:        Path  = None,
) -> dict:
    """
    Run the full-objective two-phase MIP for one scenario.

    Events are split by semester before optimisation: Semester 1 and Semester 2
    events are optimised independently (cross-semester pairs cannot conflict).
    Per-semester scores and a combined total score are reported.

    Phase 1: WholeClass displaced events  (disp–disp z + disp–fixed linear)
    Phase 2: SubGroup  displaced events   (same objective, Phase 1 results locked)
    """
    save_dir = out_dir or OUT_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    displaced_col = f"Displaced_{scenario.split('_')[0]}"

    print(f"\n{'='*60}")
    print(f"MIP NEW MODEL (full objective, two-phase, per-semester): {scenario}")
    print(f"{'='*60}")

    total_disp = int(events[displaced_col].sum())
    print(f"  Total displaced (all semesters): {total_disp:,}")

    # ── Split events by semester ──
    semesters = sorted(events["Semester"].dropna().unique())
    sem_results: dict = {}

    # Split conflict_pairs by semester (use Semester column if present)
    cp_has_semester = (conflict_pairs is not None and
                       "Semester" in conflict_pairs.columns)

    for sem in semesters:
        sem_label  = sem.replace(" ", "_")  # e.g. "Semester_1"
        events_sem = events[events["Semester"] == sem].copy()

        if cp_has_semester:
            cp_sem = conflict_pairs[conflict_pairs["Semester"] == sem].copy()
        else:
            # Fallback: filter by event IDs belonging to this semester
            sem_ids = set(events_sem["Event_ID"].astype(str))
            cp_sem  = conflict_pairs[
                conflict_pairs["Event_A"].astype(str).isin(sem_ids) &
                conflict_pairs["Event_B"].astype(str).isin(sem_ids)
            ].copy() if conflict_pairs is not None else None

        n_cp = len(cp_sem) if cp_sem is not None else 0
        print(f"\n{'─'*60}")
        print(f"  Semester: {sem}  |  Events: {len(events_sem):,}  |  "
              f"Conflict pairs: {n_cp:,}")

        sem_results[sem] = _run_mip_two_phase(
            scenario       = scenario,
            semester_label = sem_label,
            events_sem     = events_sem,
            conflict_pairs = cp_sem,
            student_events = student_events,
            max_events     = max_events,
            time_limit     = time_limit,
            mip_gap        = mip_gap,
            verbose        = verbose,
            save_dir       = save_dir,
        )

    # ── Aggregate across semesters ──
    all_assignments = pd.concat(
        [r["assignment_df"] for r in sem_results.values()],
        ignore_index=True
    )

    # Per-semester summary rows
    sem_summary_rows = []
    for sem, r in sem_results.items():
        row = dict(r["sem_summary"])
        row["Scenario"] = scenario
        sem_summary_rows.append(row)

    # Total (combined) summary
    total_dd_score   = sum(r["dd_clash_score"]   for r in sem_results.values())
    total_df_score   = sum(r["df_clash_score"]   for r in sem_results.values())
    total_obj        = sum(r["objective"]        for r in sem_results.values())
    total_n_clashes  = sum(r["n_clashes"]        for r in sem_results.values())
    total_n_displaced= sum(r["n_displaced"]      for r in sem_results.values())
    all_statuses     = [r["status"] for r in sem_results.values()]

    def _rank(s):
        return {"OPTIMAL": 2, "FEASIBLE": 1}.get(s, 0)
    combined_status = min(all_statuses, key=_rank)

    # Build combined summary (backward-compatible columns + per-semester breakdown)
    summary: dict = {
        "Scenario":              scenario,
        "Solve_Status":          combined_status,
        "N_Displaced_Total":     total_n_displaced,
        "N_Rescheduled_Total":   len(all_assignments),
        "DD_Clash_Score_Total":  total_dd_score,
        "DF_Clash_Score_Total":  total_df_score,
        "Total_Clash_Score":     total_dd_score + total_df_score,
        "Objective_Total":       total_obj,
        "N_DD_Clash_Pairs_After_Total": total_n_clashes,
    }
    # Add per-semester columns
    for sem, r in sem_results.items():
        sem_label = sem.replace(" ", "_")
        ss = r["sem_summary"]
        summary[f"DD_Clash_{sem_label}"]   = ss["DD_Clash_Score_Total"]
        summary[f"DF_Clash_{sem_label}"]   = ss["DF_Clash_Score_Total"]
        summary[f"Total_Clash_{sem_label}"]= ss["Total_Clash_Score"]
        summary[f"N_Displaced_{sem_label}"]= ss["N_Displaced_Total"]
        summary[f"N_Rescheduled_{sem_label}"] = ss["N_Rescheduled_Total"]

    # Save outputs
    if len(all_assignments) > 0:
        all_assignments.to_csv(
            save_dir / f"mip_new_assignment_{scenario}.csv", index=False)

    pd.DataFrame([summary]).to_csv(
        save_dir / f"mip_new_summary_{scenario}.csv", index=False)

    pd.DataFrame(sem_summary_rows).to_csv(
        save_dir / f"mip_new_summary_{scenario}_by_semester.csv", index=False)

    # Lunch feasibility (full combined schedule)
    if student_events is not None and len(all_assignments) > 0:
        metrics   = compute_mip_metrics(all_assignments, events, student_events,
                                         scenario, out_dir=save_dir)
        lunch_pct = metrics.get("Lunch_Free_Pct_After_Reschedule", 0)
        summary["Lunch_Free_Pct"] = lunch_pct
        print(f"  Post-MIP Lunch Free: {lunch_pct:.1f}%")

    print(f"\n  ── MIP NEW {scenario} complete (per-semester) ──")
    for sem, r in sem_results.items():
        print(f"  {sem}: DD={r['dd_clash_score']:,}  DF={r['df_clash_score']:,}  "
              f"Total={r['total_clash_score']:,}")
    print(f"  COMBINED: DD={total_dd_score:,}  DF={total_df_score:,}  "
          f"Total={total_dd_score + total_df_score:,}")

    return {
        "status":             combined_status,
        "objective":          total_obj,
        "n_displaced":        total_n_displaced,
        "n_clashes":          total_n_clashes,
        "dd_clash_score":     total_dd_score,
        "df_clash_score":     total_df_score,
        "total_clash_score":  total_dd_score + total_df_score,
        "assignment_df":      all_assignments,
        "summary":            summary,
        "sem_results":        sem_results,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Run both scenarios
# ─────────────────────────────────────────────────────────────────────────────

def run_all_mip_scenarios_new(
    data:       dict,
    max_events: int   = MAX_EVENTS_PHASE,
    time_limit: int   = SOLVER_TLIMIT_PHASE,
    mip_gap:    float = MIP_GAP,
    out_dir:    Path  = None,
) -> dict:
    """Run full-objective two-phase MIP for both S1_9am5pm and S2_NoFriPM."""
    events         = data["events"]
    student_events = data["student_events"]
    conflict_pairs = data.get("conflict_pairs")

    results = {}
    for scenario in ["S1_9am5pm", "S2_NoFriPM"]:
        res = run_mip_scenario_new(
            scenario       = scenario,
            events         = events,
            conflict_pairs = conflict_pairs,
            student_events = student_events,
            max_events     = max_events,
            time_limit     = time_limit,
            mip_gap        = mip_gap,
            out_dir        = out_dir,
        )
        results[scenario] = res
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Standalone entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from data_preprocessing import run_preprocessing

    parser = argparse.ArgumentParser(
        description="Run full-objective two-phase MIP (disp–disp + disp–fixed)"
    )
    parser.add_argument("--scenario",   default="both",
                        choices=["S1_9am5pm", "S2_NoFriPM", "both"])
    parser.add_argument("--max-events", type=int, default=MAX_EVENTS_PHASE)
    parser.add_argument("--time-limit", type=int, default=SOLVER_TLIMIT_PHASE)
    parser.add_argument("--mip-gap",    type=float, default=MIP_GAP)
    parser.add_argument("--no-conflicts", action="store_true")
    args = parser.parse_args()

    out = Path(__file__).resolve().parent / "outputs"
    data = run_preprocessing(
        build_conflicts=not args.no_conflicts,
        out_dir=out / "cleaned_data",
    )
    scenarios = (["S1_9am5pm", "S2_NoFriPM"] if args.scenario == "both"
                 else [args.scenario])
    for sc in scenarios:
        run_mip_scenario_new(
            scenario       = sc,
            events         = data["events"],
            conflict_pairs = data.get("conflict_pairs"),
            student_events = data["student_events"],
            max_events     = args.max_events,
            time_limit     = args.time_limit,
            mip_gap        = args.mip_gap,
            out_dir        = out,
        )
