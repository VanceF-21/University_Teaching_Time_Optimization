"""
heuristic_new.py  —  Extended Heuristic (Full Clash Score: Disp–Disp + Disp–Fixed)

KEY DIFFERENCES from heuristic_model.py
────────────────────────────────────────
1. FULL CLASH SCORE IN OBJECTIVE
   The original heuristic's greedy and local search already penalise disp–fixed
   clashes during placement (fixed events are pre-loaded into assignment and their
   neighbours in conflict_adj are checked).  However, the REPORTED clash scores
   and the LOCAL SEARCH convergence criterion used only disp–disp pairs.

   In this version:
     • compute_full_clash_score() counts ALL conflict pairs whose both members
       are currently in assignment (i.e. disp–disp + disp–fixed).
     • The local search optimises and reports this full score.
     • The summary reports both Full_Clash_Score and DD_Clash_Score separately.

2. FULL ASSIGNMENT PASSED TO COMPUTE_CLASH_SCORE
   In heuristic_model.py the score was computed over a filtered dict
   {eid: slot for eid in disp_ids}, excluding fixed events.
   Here we pass the full assignment (fixed + displaced) so disp–fixed pairs
   are also counted.

3. SEPARATE METRICS
   Summary reports:
     • Greedy_Full_Clash_Score   (disp–disp + disp–fixed, after greedy)
     • Greedy_DD_Clash_Score     (disp–disp only, for comparison)
     • LocalSearch_Full_Clash_Score
     • LocalSearch_DD_Clash_Score
     • Full_Improvement_Pct      (based on full clash score)

4. OUTPUT FILES
   Saved as heuristic_new_assignment_*.csv, heuristic_new_summary_*.csv, etc.
   (original heuristic_*.csv files are not overwritten)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import time
from collections import defaultdict
warnings.filterwarnings("ignore")

OUT_DIR   = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
DAY_INDEX = {d: i for i, d in enumerate(DAY_ORDER)}
ALL_HOURS = list(range(9, 18))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers (shared with heuristic_model.py)
# ─────────────────────────────────────────────────────────────────────────────

def get_allowed_slots(duration_min: float, scenario: str) -> list:
    end_limit = 17.0 if scenario == "S1_9am5pm" else 18.0
    slots = []
    for day in DAY_ORDER:
        for sh in ALL_HOURS:
            if sh + duration_min / 60.0 > end_limit:
                continue
            if scenario == "S2_NoFriPM" and day == "Friday" and sh + duration_min / 60.0 > 12.0:
                continue
            slots.append((day, float(sh)))
    return slots


def events_overlap(day1, start1, dur1, day2, start2, dur2) -> bool:
    if day1 != day2:
        return False
    end1 = start1 + dur1 / 60.0
    end2 = start2 + dur2 / 60.0
    return start1 < end2 and start2 < end1


def build_conflict_adjacency(conflict_pairs: pd.DataFrame, relevant_ids: set) -> tuple:
    adj     = defaultdict(set)
    weights = {}
    if conflict_pairs is None or len(conflict_pairs) == 0:
        return dict(adj), weights
    for _, row in conflict_pairs.iterrows():
        ea, eb = str(row["Event_A"]), str(row["Event_B"])
        shared = int(row.get("Shared_Students", 1))
        if ea in relevant_ids or eb in relevant_ids:
            adj[ea].add(eb)
            adj[eb].add(ea)
            key = tuple(sorted([ea, eb]))
            weights[key] = shared
    return dict(adj), weights


# ─────────────────────────────────────────────────────────────────────────────
# Clash scoring  *** KEY CHANGES HERE ***
# ─────────────────────────────────────────────────────────────────────────────

def compute_full_clash_score(
    assignment:       dict,
    event_info:       dict,
    conflict_adj:     dict,
    conflict_weights: dict,
) -> float:
    """
    Compute total weighted clash score using the FULL assignment dict
    (fixed + displaced events).

    *** CHANGE vs heuristic_model.py ***
    The original compute_clash_score iterated over a filtered dict containing
    only displaced events, so only disp–disp pairs were counted.
    This function uses the FULL assignment, counting disp–fixed pairs as well.

    Returns: Σ w[e1,e2] for every clashing conflict pair (e1,e2) where
             both e1 and e2 are in assignment.
    """
    score      = 0.0
    seen_pairs = set()

    for e1, (d1, s1) in assignment.items():
        dur1 = event_info.get(e1, 60.0)
        for e2 in conflict_adj.get(e1, []):
            if e2 not in assignment:
                continue
            pair = tuple(sorted([e1, e2]))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            d2, s2 = assignment[e2]
            dur2   = event_info.get(e2, 60.0)
            if events_overlap(d1, s1, dur1, d2, s2, dur2):
                score += conflict_weights.get(pair, 1)

    return score


def compute_dd_clash_score(
    assignment:       dict,
    event_info:       dict,
    conflict_adj:     dict,
    conflict_weights: dict,
    disp_ids:         set,
) -> float:
    """
    Compute clash score restricted to disp–disp pairs only.
    Equivalent to the original heuristic_model.py compute_clash_score.
    """
    filtered = {eid: slot for eid, slot in assignment.items() if eid in disp_ids}
    return compute_full_clash_score(filtered, event_info, conflict_adj, conflict_weights)


def compute_event_clash_contribution(
    event_id:         str,
    day:              str,
    start:            float,
    assignment:       dict,
    event_info:       dict,
    conflict_adj:     dict,
    conflict_weights: dict,
) -> float:
    """
    Clash contribution of placing event_id at (day, start) against current
    assignment (includes both fixed and displaced events).
    Unchanged from heuristic_model.py — already considers fixed events.
    """
    dur   = event_info.get(event_id, 60.0)
    score = 0.0
    for e2 in conflict_adj.get(event_id, []):
        if e2 not in assignment:
            continue
        d2, s2 = assignment[e2]
        dur2   = event_info.get(e2, 60.0)
        if events_overlap(day, start, dur, d2, s2, dur2):
            pair   = tuple(sorted([event_id, e2]))
            score += conflict_weights.get(pair, 1)
    return score


# ─────────────────────────────────────────────────────────────────────────────
# Algorithm 1: Greedy Assignment (unchanged logic, updated scoring calls)
# ─────────────────────────────────────────────────────────────────────────────

def greedy_assignment(
    disp_events:      pd.DataFrame,
    fixed_events:     pd.DataFrame,
    conflict_adj:     dict,
    conflict_weights: dict,
    scenario:         str,
) -> tuple:
    """
    Greedy most-constrained-first assignment.
    Fixed events pre-loaded into assignment so compute_event_clash_contribution
    already penalises disp–fixed overlaps during slot selection.
    (Logic identical to heuristic_model.py; scoring unchanged at this level.)
    """
    print(f"\n[Heuristic NEW] Greedy Assignment — {scenario}")
    t0 = time.time()

    event_info = {}
    for _, row in pd.concat([disp_events, fixed_events]).iterrows():
        event_info[row["Event_ID"]] = row["Duration_min"]

    # Fixed events occupy their current slots
    assignment = {}
    for _, row in fixed_events.iterrows():
        if pd.notna(row.get("Day")) and pd.notna(row.get("Start_Hour")):
            assignment[row["Event_ID"]] = (row["Day"], row["Start_Hour"])

    # Sort: most-constrained first
    disp_list = disp_events.copy()
    disp_list["n_conflicts"] = disp_list["Event_ID"].apply(
        lambda eid: len(conflict_adj.get(eid, []))
    )
    disp_list = disp_list.sort_values("n_conflicts", ascending=False)

    print(f"  Displaced events : {len(disp_list):,}")
    print(f"  Fixed events     : {len(fixed_events):,}")

    assignment_details = []
    total_clash        = 0.0

    for idx, (_, event_row) in enumerate(disp_list.iterrows()):
        eid     = event_row["Event_ID"]
        dur     = event_row["Duration_min"]
        allowed = get_allowed_slots(dur, scenario)

        if not allowed:
            assignment_details.append({
                "Event_ID": eid, "New_Day": None,
                "New_Start_Hour": None, "Clash_Score": -1, "Status": "NO_SLOT"
            })
            continue

        best_slot, best_score = None, float("inf")
        for (day, sh) in allowed:
            score = compute_event_clash_contribution(
                eid, day, sh, assignment, event_info, conflict_adj, conflict_weights
            )
            if score < best_score:
                best_score, best_slot = score, (day, sh)
            if score == 0:
                break

        assignment[eid] = best_slot
        total_clash    += best_score

        assignment_details.append({
            "Event_ID":       eid,
            "Module_Code":    event_row.get("Module_Code", ""),
            "Event_Type":     event_row.get("Event_Type", ""),
            "Duration_min":   dur,
            "Event_Size":     event_row.get("Event_Size", 0),
            "WholeClass":     event_row.get("WholeClass", False),
            "Original_Day":   event_row.get("Day", ""),
            "Original_Start": event_row.get("Start_Hour", ""),
            "New_Day":        best_slot[0] if best_slot else None,
            "New_Start_Hour": best_slot[1] if best_slot else None,
            "Clash_Score":    best_score,
            "N_Conflicts":    event_row["n_conflicts"],
            "Status":         "CLASH_FREE" if best_score == 0 else "WITH_CLASH",
        })

        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx+1:,}/{len(disp_list):,}, "
                  f"running clash score: {total_clash:.0f}")

    elapsed = time.time() - t0
    print(f"\n  Greedy complete in {elapsed:.1f}s | running clash score: {total_clash:.0f}")
    return assignment, pd.DataFrame(assignment_details), total_clash


# ─────────────────────────────────────────────────────────────────────────────
# Algorithm 2: Local Search  *** OPTIMISES FULL CLASH SCORE ***
# ─────────────────────────────────────────────────────────────────────────────

def local_search_new(
    assignment:       dict,
    event_info:       dict,
    conflict_adj:     dict,
    conflict_weights: dict,
    disp_ids:         set,
    scenario:         str,
    max_iterations:   int = 500,
    patience:         int = 20,
) -> tuple:
    """
    Local search with FULL clash score (disp–disp + disp–fixed) as objective.

    *** CHANGE vs heuristic_model.py local_search ***
    The original local search computed delta using only displaced events.
    Here, compute_event_clash_contribution already includes fixed events
    (they are in `assignment`), so the delta correctly reflects the change
    in both disp–disp AND disp–fixed clashes.

    We also initialise and report current_score from the full assignment,
    so the convergence criterion is the full objective.
    """
    print(f"\n[Heuristic NEW] Local Search (full objective) — {scenario}")
    t0 = time.time()

    allowed_slots = {
        eid: get_allowed_slots(event_info.get(eid, 60.0), scenario)
        for eid in disp_ids
    }

    # *** CHANGE: initialise from FULL assignment score ***
    current_score = compute_full_clash_score(
        assignment, event_info, conflict_adj, conflict_weights
    )
    print(f"  Initial FULL clash score: {current_score:.0f}")

    disp_list         = list(disp_ids)
    iteration_log     = [{"iteration": 0, "full_clash_score": current_score,
                          "improvement": 0}]
    no_improve_count  = 0
    total_improvements = 0

    for it in range(1, max_iterations + 1):
        improved = False
        np.random.shuffle(disp_list)

        for e1 in disp_list:
            if e1 not in assignment:
                continue
            d1, s1 = assignment[e1]

            for new_d, new_s in allowed_slots.get(e1, []):
                if (new_d, new_s) == (d1, s1):
                    continue

                # Incremental delta: contribution of e1 at old slot vs new slot
                # (includes fixed events via conflict_adj)
                old_contrib  = compute_event_clash_contribution(
                    e1, d1, s1, assignment, event_info, conflict_adj, conflict_weights
                )
                assignment[e1] = (new_d, new_s)
                new_contrib  = compute_event_clash_contribution(
                    e1, new_d, new_s, assignment, event_info, conflict_adj, conflict_weights
                )
                delta = new_contrib - old_contrib

                if delta < 0:
                    current_score  += delta
                    improved        = True
                    total_improvements += 1
                    break
                else:
                    assignment[e1] = (d1, s1)   # revert

        iteration_log.append({
            "iteration":        it,
            "full_clash_score": current_score,
            "improvement":      total_improvements,
        })

        if not improved:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(f"  Early stop at iter {it} (no improvement for {patience} iters)")
                break
        else:
            no_improve_count = 0

        if it % 50 == 0:
            print(f"  Iter {it}: full clash score = {current_score:.0f}")

    elapsed = time.time() - t0
    print(f"\n  Local search done in {elapsed:.1f}s")
    print(f"  Final FULL clash score  : {current_score:.0f}")
    print(f"  Total improvements made : {total_improvements}")
    return assignment, iteration_log


# ─────────────────────────────────────────────────────────────────────────────
# Lunch feasibility (unchanged from heuristic_model.py)
# ─────────────────────────────────────────────────────────────────────────────

def compute_lunch_feasibility(assignment, disp_events, fixed_events, student_events):
    rows = []
    for _, row in fixed_events.iterrows():
        if pd.notna(row.get("Day")) and pd.notna(row.get("Start_Hour")):
            rows.append({"Event_ID": row["Event_ID"], "Day": row["Day"],
                         "Start_Hour": row["Start_Hour"], "End_Hour": row["End_Hour"]})
    disp_id_set = set(disp_events["Event_ID"])
    for eid, (day, sh) in assignment.items():
        if eid in disp_id_set and day is not None:
            dur = disp_events[disp_events["Event_ID"] == eid]["Duration_min"].values
            dur = float(dur[0]) if len(dur) > 0 else 60.0
            rows.append({"Event_ID": eid, "Day": day,
                         "Start_Hour": sh, "End_Hour": sh + dur / 60.0})

    final_schedule = pd.DataFrame(rows)
    lunch_mask     = (final_schedule["Start_Hour"] < 14.0) & (final_schedule["End_Hour"] > 12.0)
    lunch_events   = set(final_schedule[lunch_mask]["Event_ID"])
    total = student_events["AnonID"].nunique()
    busy  = student_events[student_events["Event_ID"].isin(lunch_events)]["AnonID"].nunique()

    by_day = {}
    for day in DAY_ORDER:
        day_lunch = set(final_schedule[lunch_mask & (final_schedule["Day"] == day)]["Event_ID"])
        busy_day  = student_events[student_events["Event_ID"].isin(day_lunch)]["AnonID"].nunique()
        by_day[day] = round(100 * (total - busy_day) / max(total, 1), 2)

    return {
        "total_students": total,
        "free_students":  total - busy,
        "pct_free":       round(100 * (total - busy) / max(total, 1), 2),
        "by_day":         by_day,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main scenario runner  *** KEY CHANGES IN SCORING ***
# ─────────────────────────────────────────────────────────────────────────────

def _run_heuristic_semester(
    scenario:         str,
    semester_label:   str,
    events_sem:       pd.DataFrame,
    conflict_pairs:   pd.DataFrame,
    student_events:   pd.DataFrame,
    max_events_sem:   int,
    run_local_search: bool,
    ls_max_iter:      int,
    save_dir:         Path,
) -> dict:
    """
    Internal helper: run greedy + local search for one (scenario, semester) slice.
    Returns per-semester scores and assignment DataFrame.
    """
    displaced_col = "Displaced_S1" if scenario == "S1_9am5pm" else "Displaced_S2"
    disp_events   = events_sem[events_sem[displaced_col]].copy()
    fixed_events  = events_sem[~events_sem[displaced_col]].copy()

    print(f"\n  [{semester_label}] Displaced: {len(disp_events):,}  "
          f"| Processing up to {max_events_sem:,}")

    if len(disp_events) > max_events_sem:
        disp_events = disp_events.nlargest(max_events_sem, "Event_Size")

    # Build conflict adjacency restricted to this semester's events
    all_ids_sem = set(events_sem["Event_ID"])
    conflict_adj, conflict_weights = build_conflict_adjacency(conflict_pairs, all_ids_sem)
    event_info = {row["Event_ID"]: row["Duration_min"] for _, row in events_sem.iterrows()}

    # ── Greedy ──
    assignment, greedy_df, _ = greedy_assignment(
        disp_events, fixed_events, conflict_adj, conflict_weights, scenario
    )
    disp_ids_set = set(disp_events["Event_ID"])

    greedy_full_score = compute_full_clash_score(
        assignment, event_info, conflict_adj, conflict_weights
    )
    greedy_dd_score = compute_dd_clash_score(
        assignment, event_info, conflict_adj, conflict_weights, disp_ids_set
    )
    print(f"  [{semester_label}] Greedy FULL={greedy_full_score:.0f}  "
          f"DD={greedy_dd_score:.0f}  DF={greedy_full_score - greedy_dd_score:.0f}")

    # ── Local Search ──
    iteration_log = []
    ls_full_score = greedy_full_score
    ls_dd_score   = greedy_dd_score

    if run_local_search:
        assignment, iteration_log = local_search_new(
            assignment, event_info, conflict_adj, conflict_weights,
            disp_ids_set, scenario, max_iterations=ls_max_iter
        )
        ls_full_score = compute_full_clash_score(
            assignment, event_info, conflict_adj, conflict_weights
        )
        ls_dd_score = compute_dd_clash_score(
            assignment, event_info, conflict_adj, conflict_weights, disp_ids_set
        )
        print(f"  [{semester_label}] Post-LS FULL={ls_full_score:.0f}  "
              f"DD={ls_dd_score:.0f}")

    # ── Build assignment DataFrame ──
    assignment_rows = []
    for _, row in disp_events.iterrows():
        eid  = row["Event_ID"]
        slot = assignment.get(eid)
        assignment_rows.append({
            "Event_ID":       eid,
            "Module_Code":    row.get("Module_Code", ""),
            "Module_Name":    row.get("Module_Name", ""),
            "Event_Type":     row.get("Event_Type", ""),
            "Duration_min":   row["Duration_min"],
            "Event_Size":     row.get("Event_Size", 0),
            "WholeClass":     row.get("WholeClass", False),
            "Original_Day":   row.get("Day", ""),
            "Original_Start": row.get("Start_Hour", ""),
            "New_Day":        slot[0] if slot else None,
            "New_Start_Hour": slot[1] if slot else None,
            "Scenario":       scenario,
            "Semester":       semester_label,
        })

    assignment_df = pd.DataFrame(assignment_rows)

    # Save per-semester assignment
    assignment_df.to_csv(
        save_dir / f"heuristic_new_assignment_{scenario}_{semester_label}.csv",
        index=False)

    if iteration_log:
        pd.DataFrame(iteration_log).to_csv(
            save_dir / f"heuristic_new_iteration_log_{scenario}_{semester_label}.csv",
            index=False)

    full_impr_pct = round(
        100 * (greedy_full_score - ls_full_score) / max(greedy_full_score, 1), 2
    ) if run_local_search and greedy_full_score > 0 else 0

    sem_summary = {
        "Semester":                     semester_label,
        "N_Displaced_Total":            len(events_sem[events_sem[displaced_col]]),
        "N_Displaced_Processed":        len(disp_events),
        "N_Rescheduled":                len(assignment_df),
        "Greedy_Full_Clash_Score":      round(greedy_full_score, 0),
        "Greedy_DD_Clash_Score":        round(greedy_dd_score, 0),
        "LocalSearch_Full_Clash_Score": round(ls_full_score, 0) if run_local_search else None,
        "LocalSearch_DD_Clash_Score":   round(ls_dd_score, 0)   if run_local_search else None,
        "Full_Improvement_Pct":         full_impr_pct,
    }

    return {
        "assignment_df":     assignment_df,
        "sem_summary":       sem_summary,
        "iteration_log":     iteration_log,
        "greedy_full_score": greedy_full_score,
        "ls_full_score":     ls_full_score,
        "greedy_dd_score":   greedy_dd_score,
        "ls_dd_score":       ls_dd_score,
        "assignment":        assignment,
        "disp_events":       disp_events,
        "fixed_events":      fixed_events,
    }


def run_heuristic_scenario_new(
    scenario:         str,
    events:           pd.DataFrame,
    conflict_pairs:   pd.DataFrame,
    student_events:   pd.DataFrame,
    max_events:       int  = 2000,
    run_local_search: bool = True,
    ls_max_iter:      int  = 300,
    out_dir:          Path = None,
) -> dict:
    """
    Run greedy + local search with FULL clash score (disp–disp + disp–fixed).

    Events are split by semester before optimisation: Semester 1 and Semester 2
    events are optimised independently (cross-semester pairs cannot conflict).
    Per-semester scores and a combined total score are reported.

    Output files:
      heuristic_new_assignment_{scenario}_{Semester_X}.csv   (per semester)
      heuristic_new_assignment_{scenario}.csv                 (combined)
      heuristic_new_summary_{scenario}_by_semester.csv        (per semester)
      heuristic_new_summary_{scenario}.csv                    (combined total)
    """
    save_dir = out_dir or OUT_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    displaced_col = "Displaced_S1" if scenario == "S1_9am5pm" else "Displaced_S2"

    print(f"\n{'='*60}")
    print(f"HEURISTIC NEW MODEL (per-semester): {scenario}")
    print(f"{'='*60}")
    print(f"  Total displaced (all semesters): {events[displaced_col].sum():,}")

    # ── Split by semester ──
    semesters = sorted(events["Semester"].dropna().unique())
    cp_has_semester = (conflict_pairs is not None and
                       "Semester" in conflict_pairs.columns)

    # Distribute max_events budget proportionally across semesters
    total_disp = max(events[displaced_col].sum(), 1)
    sem_results: dict = {}
    all_iter_logs: list = []

    for sem in semesters:
        sem_label  = sem.replace(" ", "_")
        events_sem = events[events["Semester"] == sem].copy()

        if cp_has_semester:
            cp_sem = conflict_pairs[conflict_pairs["Semester"] == sem].copy()
        else:
            sem_ids = set(events_sem["Event_ID"].astype(str))
            cp_sem  = conflict_pairs[
                conflict_pairs["Event_A"].astype(str).isin(sem_ids) &
                conflict_pairs["Event_B"].astype(str).isin(sem_ids)
            ].copy() if conflict_pairs is not None else None

        # Proportional budget
        sem_disp   = int(events_sem[displaced_col].sum())
        max_ev_sem = max(1, round(max_events * sem_disp / total_disp))

        n_cp = len(cp_sem) if cp_sem is not None else 0
        print(f"\n{'─'*60}")
        print(f"  Semester: {sem}  |  Events: {len(events_sem):,}  |  "
              f"Conflict pairs: {n_cp:,}  |  Max displaced budget: {max_ev_sem:,}")

        sem_results[sem] = _run_heuristic_semester(
            scenario         = scenario,
            semester_label   = sem_label,
            events_sem       = events_sem,
            conflict_pairs   = cp_sem,
            student_events   = student_events,
            max_events_sem   = max_ev_sem,
            run_local_search = run_local_search,
            ls_max_iter      = ls_max_iter,
            save_dir         = save_dir,
        )
        if sem_results[sem]["iteration_log"]:
            all_iter_logs.extend(sem_results[sem]["iteration_log"])

    # ── Aggregate ──
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

    # Combined totals
    total_greedy_full = sum(r["greedy_full_score"] for r in sem_results.values())
    total_ls_full     = sum(r["ls_full_score"]     for r in sem_results.values())
    total_greedy_dd   = sum(r["greedy_dd_score"]   for r in sem_results.values())
    total_ls_dd       = sum(r["ls_dd_score"]       for r in sem_results.values())

    full_impr_pct = round(
        100 * (total_greedy_full - total_ls_full) / max(total_greedy_full, 1), 2
    ) if run_local_search and total_greedy_full > 0 else 0
    dd_impr_pct = round(
        100 * (total_greedy_dd - total_ls_dd) / max(total_greedy_dd, 1), 2
    ) if run_local_search and total_greedy_dd > 0 else 0

    # Lunch feasibility using full combined assignment
    # Approximate using the full events table merged with all assignments
    combined_assignment_dict: dict = {}
    for r in sem_results.values():
        combined_assignment_dict.update(r["assignment"])

    all_disp_events = pd.concat(
        [r["disp_events"] for r in sem_results.values()], ignore_index=True)
    all_fixed_events = pd.concat(
        [r["fixed_events"] for r in sem_results.values()], ignore_index=True)

    lunch = compute_lunch_feasibility(
        combined_assignment_dict, all_disp_events, all_fixed_events, student_events
    )

    # NO_SLOT stats
    n_noslot    = int(all_assignments["New_Day"].isna().sum())
    n_noslot_wc = int(all_assignments[all_assignments["WholeClass"] == True]["New_Day"].isna().sum())
    noslot_pct  = round(100 * n_noslot / max(len(all_assignments), 1), 2)

    # Slot balance CV across combined assignment
    slot_counts = defaultdict(int)
    for eid, slot in combined_assignment_dict.items():
        if slot and slot[0] is not None:
            slot_counts[slot] += 1
    cv = (np.std(list(slot_counts.values())) /
          max(np.mean(list(slot_counts.values())), 1)) if slot_counts else 0.0

    # Day distribution
    placed_df  = all_assignments[all_assignments["New_Day"].notna()]
    day_counts = placed_df.groupby("New_Day").size()
    day_dist   = {day: int(day_counts.get(day, 0)) for day in DAY_ORDER}

    # Combined summary (backward-compatible + per-semester columns)
    summary: dict = {
        "Scenario":                        scenario,
        "N_Displaced_Total":               int(events[displaced_col].sum()),
        "N_Displaced_Processed":           len(all_disp_events),
        "N_Rescheduled":                   len(all_assignments),
        "N_NoSlot":                        n_noslot,
        "N_NoSlot_WholeClass":             n_noslot_wc,
        "NoSlot_Pct":                      noslot_pct,
        "Greedy_Full_Clash_Score":         round(total_greedy_full, 0),
        "Greedy_DD_Clash_Score":           round(total_greedy_dd, 0),
        "LocalSearch_Full_Clash_Score":    round(total_ls_full, 0) if run_local_search else None,
        "LocalSearch_DD_Clash_Score":      round(total_ls_dd, 0)   if run_local_search else None,
        "Full_Improvement_Pct":            full_impr_pct,
        "DD_Improvement_Pct":              dd_impr_pct,
        "Lunch_Free_Pct":                  lunch["pct_free"],
        "Slot_Balance_CV":                 round(cv, 4),
    }
    # Per-semester breakdown columns
    for sem, r in sem_results.items():
        sem_label = sem.replace(" ", "_")
        ss = r["sem_summary"]
        summary[f"Greedy_Full_{sem_label}"] = ss["Greedy_Full_Clash_Score"]
        summary[f"LS_Full_{sem_label}"]     = ss["LocalSearch_Full_Clash_Score"]
        summary[f"Impr_Pct_{sem_label}"]    = ss["Full_Improvement_Pct"]
        summary[f"N_Displaced_{sem_label}"] = ss["N_Displaced_Total"]

    for day, pct in lunch["by_day"].items():
        summary[f"Lunch_Free_{day}"] = pct
    for day in DAY_ORDER:
        summary[f"Rescheduled_{day}"] = day_dist[day]

    # Save outputs
    all_assignments.to_csv(
        save_dir / f"heuristic_new_assignment_{scenario}.csv", index=False)
    pd.DataFrame([summary]).to_csv(
        save_dir / f"heuristic_new_summary_{scenario}.csv", index=False)
    pd.DataFrame(sem_summary_rows).to_csv(
        save_dir / f"heuristic_new_summary_{scenario}_by_semester.csv", index=False)

    if all_iter_logs:
        pd.DataFrame(all_iter_logs).to_csv(
            save_dir / f"heuristic_new_iteration_log_{scenario}.csv", index=False)

    print(f"\n[Heuristic NEW] {scenario} complete (per-semester)")
    for sem, r in sem_results.items():
        print(f"  {sem}: greedy={r['greedy_full_score']:.0f} → "
              f"LS={r['ls_full_score']:.0f}  "
              f"({r['sem_summary']['Full_Improvement_Pct']:.1f}% improvement)")
    print(f"  COMBINED: greedy={total_greedy_full:.0f} → "
          f"LS={total_ls_full:.0f}  ({full_impr_pct:.1f}% improvement)")
    print(f"  Lunch free: {lunch['pct_free']:.1f}%")

    return {
        "assignment_df":      all_assignments,
        "summary":            summary,
        "sem_summary_rows":   sem_summary_rows,
        "sem_results":        sem_results,
        "greedy_full_score":  total_greedy_full,
        "ls_full_score":      total_ls_full,
        "lunch":              lunch,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Run both scenarios
# ─────────────────────────────────────────────────────────────────────────────

def run_all_heuristic_scenarios_new(
    data:       dict,
    max_events: int  = 2000,
    run_ls:     bool = True,
    ls_iter:    int  = 300,
    out_dir:    Path = None,
) -> dict:
    """Run full-objective heuristic for both scenarios."""
    events         = data["events"]
    conflict_pairs = data.get("conflict_pairs")
    student_events = data["student_events"]

    results = {}
    for scenario in ["S1_9am5pm", "S2_NoFriPM"]:
        results[scenario] = run_heuristic_scenario_new(
            scenario         = scenario,
            events           = events,
            conflict_pairs   = conflict_pairs,
            student_events   = student_events,
            max_events       = max_events,
            run_local_search = run_ls,
            ls_max_iter      = ls_iter,
            out_dir          = out_dir,
        )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Standalone entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from data_preprocessing import run_preprocessing

    parser = argparse.ArgumentParser(
        description="Run full-objective heuristic (disp–disp + disp–fixed)")
    parser.add_argument("--scenario",   default="both",
                        choices=["S1_9am5pm", "S2_NoFriPM", "both"])
    parser.add_argument("--max-events", type=int, default=2000)
    parser.add_argument("--no-ls",      action="store_true")
    parser.add_argument("--ls-iter",    type=int, default=300)
    parser.add_argument("--no-conflicts", action="store_true")
    args = parser.parse_args()

    data = run_preprocessing(build_conflicts=not args.no_conflicts)

    if args.scenario == "both":
        run_all_heuristic_scenarios_new(
            data, args.max_events, not args.no_ls, args.ls_iter
        )
    else:
        run_heuristic_scenario_new(
            scenario         = args.scenario,
            events           = data["events"],
            conflict_pairs   = data.get("conflict_pairs"),
            student_events   = data["student_events"],
            max_events       = args.max_events,
            run_local_search = not args.no_ls,
            ls_max_iter      = args.ls_iter,
        )
