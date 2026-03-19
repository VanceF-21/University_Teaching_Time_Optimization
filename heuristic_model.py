"""
heuristic_model.py

Purpose:
    Heuristic methods for rescheduling displaced events within the proposed
    time windows, complementing the exact MIP approach.

    Two algorithms are implemented:

    ═══════════════════════════════════════════════════════════════════════════
    ALGORITHM 1: Greedy Assignment (Most-Constrained First)
    ═══════════════════════════════════════════════════════════════════════════
    Step 1: Sort displaced events by NUMBER OF CONFLICTS (descending).
            Events with more conflicts are harder to schedule — tackle first.

    Step 2: For each displaced event (in sorted order):
              For each candidate timeslot (in sorted order by day/hour):
                Count how many currently-assigned events this would clash with.
                Assign to the slot with the MINIMUM clash count.
              If a clash-free slot exists, use it (hard constraint satisfied).
              Otherwise, use the least-clashing slot (soft, counted as penalty).

    Step 3: Record assignment + number of clashes introduced.

    Complexity: O(|E_disp| × |T| × degree) where degree = avg conflicts per event.

    ═══════════════════════════════════════════════════════════════════════════
    ALGORITHM 2: Local Search (Swap Improvement)
    ═══════════════════════════════════════════════════════════════════════════
    Starting from the greedy solution, iteratively improve by swapping timeslots
    between pairs of displaced events.

    Step 1: Compute initial total clash score.
    Step 2: For each pair of events (e1, e2):
              Try swapping their assigned timeslots.
              If the swap reduces total clashes → accept (steepest descent).
    Step 3: Repeat until no improving swap found or max_iterations reached.

    This is a classic NEIGHBOURHOOD SEARCH with swap neighbourhood.
    Each iteration is O(|E_disp|² × |conflicts|).

    ═══════════════════════════════════════════════════════════════════════════
    METRICS
    ═══════════════════════════════════════════════════════════════════════════
    - Total clash pairs after heuristic
    - Improvement over initial timetable (displaced events cause 0 clashes if
      they can be rescheduled)
    - Lunch break feasibility after rescheduling
    - Timeslot utilisation balance (coefficient of variation across slots)

Outputs:
    outputs/heuristic_assignment_{scenario}.csv
    outputs/heuristic_summary_{scenario}.csv
    outputs/heuristic_iteration_log_{scenario}.csv   (local search progress)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import time
from collections import defaultdict
warnings.filterwarnings("ignore")

OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
DAY_INDEX = {d: i for i, d in enumerate(DAY_ORDER)}
ALL_HOURS = list(range(9, 18))   # Possible start hours: 9, 10, ..., 17


# ============================================================
# Helper: Get allowed slots for a scenario + duration
# ============================================================
def get_allowed_slots(duration_min: float, scenario: str) -> list:
    """
    Return sorted list of (day, start_hour) tuples allowed for this event.
    Same logic as mip_model.get_event_allowed_slots.
    """
    end_limit = 17.0 if scenario == "S1_9am5pm" else 18.0
    slots = []
    for day in DAY_ORDER:
        for sh in ALL_HOURS:
            if sh + duration_min / 60.0 > end_limit:
                continue
            if scenario == "S2_NoFriPM" and day == "Friday" and sh >= 12:
                continue
            slots.append((day, float(sh)))
    return slots


# ============================================================
# Helper: Check time overlap
# ============================================================
def events_overlap(day1, start1, dur1, day2, start2, dur2) -> bool:
    if day1 != day2:
        return False
    end1 = start1 + dur1 / 60.0
    end2 = start2 + dur2 / 60.0
    return start1 < end2 and start2 < end1


# ============================================================
# Helper: Build conflict adjacency (event → set of conflicting event IDs)
# ============================================================
def build_conflict_adjacency(conflict_pairs: pd.DataFrame,
                              relevant_ids: set) -> dict:
    """
    Build adjacency dict: event_id → set of conflicting event_ids.
    Only include events in `relevant_ids` for efficiency.
    Also store shared_students count: (e1, e2) → count.
    """
    adj = defaultdict(set)
    weights = {}

    if conflict_pairs is None or len(conflict_pairs) == 0:
        return adj, weights

    for _, row in conflict_pairs.iterrows():
        ea, eb = str(row["Event_A"]), str(row["Event_B"])
        shared = int(row.get("Shared_Students", 1))
        if ea in relevant_ids or eb in relevant_ids:
            adj[ea].add(eb)
            adj[eb].add(ea)
            key = tuple(sorted([ea, eb]))
            weights[key] = shared

    return dict(adj), weights


# ============================================================
# Clash Scoring Function
# ============================================================
def compute_clash_score(assignment: dict,
                         event_info: dict,
                         conflict_adj: dict,
                         conflict_weights: dict) -> float:
    """
    Compute total weighted clash score for a given assignment.

    assignment  : {event_id: (day, start_hour)}
    event_info  : {event_id: duration_min}
    conflict_adj: {event_id: set of conflicting event_ids}
    conflict_weights: {(e1,e2) sorted: shared_student_count}

    Returns sum of shared_students for each clashing pair.
    """
    score = 0.0
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
            dur2 = event_info.get(e2, 60.0)
            if events_overlap(d1, s1, dur1, d2, s2, dur2):
                score += conflict_weights.get(pair, 1)

    return score


def compute_event_clash_contribution(event_id: str,
                                      day: str,
                                      start: float,
                                      assignment: dict,
                                      event_info: dict,
                                      conflict_adj: dict,
                                      conflict_weights: dict) -> float:
    """
    Compute clash contribution of placing event_id at (day, start).
    Used during greedy selection.
    """
    dur = event_info.get(event_id, 60.0)
    score = 0.0
    for e2 in conflict_adj.get(event_id, []):
        if e2 not in assignment:
            continue
        d2, s2 = assignment[e2]
        dur2 = event_info.get(e2, 60.0)
        if events_overlap(day, start, dur, d2, s2, dur2):
            pair = tuple(sorted([event_id, e2]))
            score += conflict_weights.get(pair, 1)
    return score


# ============================================================
# Algorithm 1: Greedy Assignment
# ============================================================
def greedy_assignment(disp_events: pd.DataFrame,
                       fixed_events: pd.DataFrame,
                       conflict_adj: dict,
                       conflict_weights: dict,
                       scenario: str) -> tuple:
    """
    Greedy assignment: assign displaced events to allowed timeslots one by one,
    choosing the slot with minimum clash score. Most-constrained event first.

    Parameters
    ----------
    disp_events    : DataFrame of displaced events
    fixed_events   : DataFrame of in-window events (their slots are fixed)
    conflict_adj   : adjacency dict from build_conflict_adjacency
    conflict_weights: pair → shared_students
    scenario       : "S1_9am5pm" or "S2_NoFriPM"

    Returns
    -------
    assignment : dict {event_id: (day, start_hour)}
    clash_count: int (number of clashing pairs after assignment)
    """
    print(f"\n[Heuristic] Greedy Assignment — {scenario}")
    t0 = time.time()

    # Collect all event IDs + durations
    event_info = {}
    for _, row in pd.concat([disp_events, fixed_events]).iterrows():
        event_info[row["Event_ID"]] = row["Duration_min"]

    # Initial assignment: fixed events keep their current slot
    assignment = {}
    for _, row in fixed_events.iterrows():
        if pd.notna(row.get("Day")) and pd.notna(row.get("Start_Hour")):
            assignment[row["Event_ID"]] = (row["Day"], row["Start_Hour"])

    # Sort displaced events: most conflicts first (most constrained)
    disp_list = disp_events.copy()
    disp_list["n_conflicts"] = disp_list["Event_ID"].apply(
        lambda eid: len(conflict_adj.get(eid, []))
    )
    disp_list = disp_list.sort_values("n_conflicts", ascending=False)

    print(f"  Displaced events to schedule: {len(disp_list):,}")
    print(f"  Fixed events (reference): {len(fixed_events):,}")

    # Greedy assignment loop
    assignment_details = []
    total_clash = 0

    for idx, (_, event_row) in enumerate(disp_list.iterrows()):
        eid  = event_row["Event_ID"]
        dur  = event_row["Duration_min"]
        allowed = get_allowed_slots(dur, scenario)

        if not allowed:
            print(f"  [WARN] Event {eid} has no allowed slots!")
            assignment_details.append({
                "Event_ID": eid, "New_Day": None,
                "New_Start_Hour": None, "Clash_Score": -1,
                "Status": "NO_SLOT"
            })
            continue

        # Score each candidate slot
        best_slot  = None
        best_score = float("inf")

        for (day, sh) in allowed:
            score = compute_event_clash_contribution(
                eid, day, sh, assignment, event_info, conflict_adj, conflict_weights
            )
            if score < best_score:
                best_score = score
                best_slot  = (day, sh)
            if score == 0:
                break  # Clash-free slot found, no need to search further

        # Assign
        assignment[eid] = best_slot
        total_clash += best_score

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
            print(f"  Processed {idx+1:,}/{len(disp_list):,} events, "
                  f"running clash score: {total_clash:.0f}")

    elapsed = time.time() - t0
    clash_free_count = sum(1 for d in assignment_details if d["Status"] == "CLASH_FREE")
    print(f"\n  Greedy complete in {elapsed:.1f}s")
    print(f"  Clash-free placements: {clash_free_count:,}/{len(assignment_details):,}")
    print(f"  Total clash score: {total_clash:.0f}")

    return assignment, pd.DataFrame(assignment_details), total_clash


# ============================================================
# Algorithm 2: Local Search (Swap Improvement)
# ============================================================
def local_search(assignment: dict,
                  event_info: dict,
                  conflict_adj: dict,
                  conflict_weights: dict,
                  disp_ids: set,
                  scenario: str,
                  max_iterations: int = 500,
                  patience: int = 20) -> tuple:
    """
    Local search: iteratively swap timeslots between pairs of displaced events.
    Accepts only improving swaps (steepest descent).

    Parameters
    ----------
    assignment     : current {event_id: (day, start_hour)} assignment
    event_info     : {event_id: duration_min}
    conflict_adj   : adjacency dict
    conflict_weights: pair → shared_students
    disp_ids       : set of event IDs that CAN be moved
    scenario       : for getting allowed slot lists
    max_iterations : stop after this many swap attempts
    patience       : stop if no improvement for this many consecutive iterations

    Returns
    -------
    improved_assignment : dict
    iteration_log       : list of {iteration, clash_score, improvement}
    """
    print(f"\n[Heuristic] Local Search — {scenario}")
    t0 = time.time()

    # Build allowed slots per event
    allowed_slots = {}
    for eid in disp_ids:
        dur = event_info.get(eid, 60.0)
        allowed_slots[eid] = get_allowed_slots(dur, scenario)

    current_score = compute_clash_score(
        assignment, event_info, conflict_adj, conflict_weights
    )
    print(f"  Initial clash score: {current_score:.0f}")

    disp_list = list(disp_ids)
    iteration_log = [{"iteration": 0, "clash_score": current_score, "improvement": 0}]
    no_improve_count = 0
    total_improvements = 0

    for it in range(1, max_iterations + 1):
        improved = False

        # Shuffle order for exploration diversity
        np.random.shuffle(disp_list)

        for i, e1 in enumerate(disp_list):
            if e1 not in assignment:
                continue
            d1, s1 = assignment[e1]

            # Try: move e1 to a different allowed slot
            for new_d, new_s in allowed_slots.get(e1, []):
                if (new_d, new_s) == (d1, s1):
                    continue

                # Compute change in clash score if we move e1 from (d1,s1) to (new_d,new_s)
                old_contrib = compute_event_clash_contribution(
                    e1, d1, s1, assignment, event_info, conflict_adj, conflict_weights
                )
                assignment[e1] = (new_d, new_s)
                new_contrib = compute_event_clash_contribution(
                    e1, new_d, new_s, assignment, event_info, conflict_adj, conflict_weights
                )

                delta = new_contrib - old_contrib

                if delta < 0:
                    # Improvement: keep new slot
                    current_score += delta
                    improved = True
                    total_improvements += 1
                    break
                else:
                    # Revert
                    assignment[e1] = (d1, s1)

        iteration_log.append({
            "iteration":   it,
            "clash_score": current_score,
            "improvement": total_improvements,
        })

        if not improved:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(f"  Early stop at iteration {it} (no improvement for {patience} iters)")
                break
        else:
            no_improve_count = 0

        if it % 50 == 0:
            print(f"  Iteration {it}: clash score = {current_score:.0f}")

    elapsed = time.time() - t0
    print(f"\n  Local search complete in {elapsed:.1f}s")
    print(f"  Final clash score: {current_score:.0f}")
    print(f"  Total improvements made: {total_improvements}")

    return assignment, iteration_log


# ============================================================
# Lunch Break Analysis post-heuristic
# ============================================================
def compute_lunch_feasibility(assignment: dict,
                               disp_events: pd.DataFrame,
                               fixed_events: pd.DataFrame,
                               student_events: pd.DataFrame) -> dict:
    """
    Compute % of students with a free lunch window (12:00-14:00) after
    the heuristic rescheduling.

    Returns dict with total_students, free_students, pct_free, by_day.
    """
    # Build final event schedule (fixed + rescheduled)
    rows = []
    for _, row in fixed_events.iterrows():
        if pd.notna(row.get("Day")) and pd.notna(row.get("Start_Hour")):
            rows.append({
                "Event_ID":   row["Event_ID"],
                "Day":        row["Day"],
                "Start_Hour": row["Start_Hour"],
                "End_Hour":   row["End_Hour"],
            })
    for eid, (day, sh) in assignment.items():
        if eid in set(disp_events["Event_ID"]) and day is not None:
            dur = disp_events[disp_events["Event_ID"]==eid]["Duration_min"].values
            dur = float(dur[0]) if len(dur) > 0 else 60.0
            rows.append({
                "Event_ID":   eid,
                "Day":        day,
                "Start_Hour": sh,
                "End_Hour":   sh + dur/60.0,
            })

    final_schedule = pd.DataFrame(rows)

    # Events overlapping 12-14
    lunch_mask = (final_schedule["Start_Hour"] < 14.0) & (final_schedule["End_Hour"] > 12.0)
    lunch_events = set(final_schedule[lunch_mask]["Event_ID"])

    total = student_events["AnonID"].nunique()
    busy  = student_events[student_events["Event_ID"].isin(lunch_events)]["AnonID"].nunique()
    free  = total - busy

    by_day = {}
    for day in ["Monday","Tuesday","Wednesday","Thursday","Friday"]:
        day_lunch = set(final_schedule[lunch_mask & (final_schedule["Day"]==day)]["Event_ID"])
        busy_day  = student_events[student_events["Event_ID"].isin(day_lunch)]["AnonID"].nunique()
        by_day[day] = round(100 * (total - busy_day) / max(total,1), 2)

    return {
        "total_students": total,
        "free_students":  free,
        "pct_free":       round(100 * free / max(total,1), 2),
        "by_day":         by_day,
    }


# ============================================================
# Main: Run heuristic for a scenario
# ============================================================
def run_heuristic_scenario(scenario: str,
                             events: pd.DataFrame,
                             conflict_pairs: pd.DataFrame,
                             student_events: pd.DataFrame,
                             max_events: int = 2000,
                             run_local_search: bool = True,
                             ls_max_iter: int = 300,
                             out_dir: Path = None) -> dict:
    """
    Run greedy + local search for a given scenario.

    Parameters
    ----------
    scenario          : "S1_9am5pm" or "S2_NoFriPM"
    events            : full events DataFrame
    conflict_pairs    : from data_preprocessing
    student_events    : student-event mapping
    max_events        : max displaced events to process
    run_local_search  : whether to run local search after greedy
    ls_max_iter       : local search iteration limit

    Returns
    -------
    dict with assignment_df, summary, iteration_log
    """
    save_dir = out_dir or OUT_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    displaced_col = "Displaced_S1" if scenario == "S1_9am5pm" else "Displaced_S2"

    disp_events  = events[events[displaced_col]].copy()
    fixed_events = events[~events[displaced_col]].copy()

    print(f"\n{'='*60}")
    print(f"HEURISTIC MODEL: {scenario}")
    print(f"{'='*60}")
    print(f"  Total displaced events: {len(disp_events):,}")
    print(f"  Processing up to {max_events:,} events")

    # Limit for speed
    if len(disp_events) > max_events:
        disp_events = disp_events.nlargest(max_events, "Event_Size")

    # Build conflict adjacency
    all_ids = set(events["Event_ID"])
    conflict_adj, conflict_weights = build_conflict_adjacency(conflict_pairs, all_ids)

    # Build event_info
    event_info = {row["Event_ID"]: row["Duration_min"]
                  for _, row in events.iterrows()}

    # ── Phase 1: Greedy ──────────────────────────────────────────────────────
    assignment, greedy_df, greedy_clash = greedy_assignment(
        disp_events, fixed_events, conflict_adj, conflict_weights, scenario
    )

    # ── Phase 2: Local Search ────────────────────────────────────────────────
    iteration_log = []
    ls_clash = greedy_clash

    if run_local_search:
        disp_ids = set(disp_events["Event_ID"])
        assignment, iteration_log = local_search(
            assignment, event_info, conflict_adj, conflict_weights,
            disp_ids, scenario, max_iterations=ls_max_iter
        )
        ls_clash = compute_clash_score(
            {eid: slot for eid, slot in assignment.items()
             if eid in disp_ids},
            event_info, conflict_adj, conflict_weights
        )
        print(f"  Clash score after local search: {ls_clash:.0f}")

    # ── Build assignment DataFrame ───────────────────────────────────────────
    disp_ids = set(disp_events["Event_ID"])
    assignment_rows = []
    for _, row in disp_events.iterrows():
        eid = row["Event_ID"]
        slot = assignment.get(eid)
        assignment_rows.append({
            "Event_ID":        eid,
            "Module_Code":     row.get("Module_Code",""),
            "Module_Name":     row.get("Module_Name",""),
            "Event_Type":      row.get("Event_Type",""),
            "Duration_min":    row["Duration_min"],
            "Event_Size":      row.get("Event_Size", 0),
            "WholeClass":      row.get("WholeClass", False),
            "Original_Day":    row.get("Day",""),
            "Original_Start":  row.get("Start_Hour",""),
            "New_Day":         slot[0] if slot else None,
            "New_Start_Hour":  slot[1] if slot else None,
            "Scenario":        scenario,
        })

    assignment_df = pd.DataFrame(assignment_rows)
    assignment_df.to_csv(save_dir / f"heuristic_assignment_{scenario}.csv", index=False)

    # ── Lunch feasibility ────────────────────────────────────────────────────
    lunch = compute_lunch_feasibility(
        assignment, disp_events, fixed_events, student_events
    )
    print(f"\n  Lunch free (after heuristic): {lunch['pct_free']:.1f}% "
          f"({lunch['free_students']:,}/{lunch['total_students']:,} students)")

    # ── Timeslot load balance ────────────────────────────────────────────────
    slot_counts = defaultdict(int)
    for eid, slot in assignment.items():
        if slot and slot[0] is not None:
            slot_counts[slot] += 1
    cv_slots = np.std(list(slot_counts.values())) / max(np.mean(list(slot_counts.values())), 1)

    # ── Q1/Q2: NO_SLOT analysis (events that cannot fit ANY timeslot) ────────
    n_noslot    = int(assignment_df["New_Day"].isna().sum())
    n_noslot_wc = int(
        assignment_df[assignment_df["WholeClass"] == True]["New_Day"].isna().sum()
    )
    noslot_pct  = round(100 * n_noslot / max(len(assignment_df), 1), 2)
    print(f"\n  NO_SLOT events (cannot fit window): {n_noslot:,} "
          f"({noslot_pct:.1f}%)  — WholeClass: {n_noslot_wc:,}")

    # ── Q3: WholeClass-specific clash-free rate (from greedy phase) ──────────
    wc_greedy       = greedy_df[greedy_df["WholeClass"] == True]
    wc_placed       = int((wc_greedy["Status"] != "NO_SLOT").sum())
    wc_clash_free   = int((wc_greedy["Status"] == "CLASH_FREE").sum())
    wc_clash_pct    = round(100 * wc_clash_free / max(wc_placed, 1), 2)
    print(f"  WholeClass greedy clash-free: {wc_clash_free:,}/{wc_placed:,} "
          f"({wc_clash_pct:.1f}%)")

    # ── Q5: Rescheduled events distribution by day ───────────────────────────
    placed_df   = assignment_df[assignment_df["New_Day"].notna()]
    day_counts  = placed_df.groupby("New_Day").size()
    day_dist    = {day: int(day_counts.get(day, 0)) for day in DAY_ORDER}
    print(f"  Day distribution of rescheduled events: {day_dist}")

    # ── Summary ──────────────────────────────────────────────────────────────
    summary = {
        "Scenario":                        scenario,
        "N_Displaced_Total":               len(events[events[displaced_col]]),
        "N_Displaced_Processed":           len(disp_events),
        "N_Rescheduled":                   len(assignment_df),
        # Q1/Q2 — feasibility
        "N_NoSlot":                        n_noslot,
        "N_NoSlot_WholeClass":             n_noslot_wc,
        "NoSlot_Pct":                      noslot_pct,
        # Q3 — clash quality
        "Greedy_Clash_Score":              round(greedy_clash, 0),
        "LocalSearch_Clash_Score":         round(ls_clash, 0) if run_local_search else None,
        "Improvement_Pct":                 round(100*(greedy_clash-ls_clash)/max(greedy_clash,1), 2)
                                            if run_local_search else 0,
        "Clash_Free_Placements":           int((assignment_df["New_Day"].notna()).sum()),
        "Greedy_ClashFree_WholeClass":     wc_clash_free,
        "Greedy_ClashFree_WholeClass_Pct": wc_clash_pct,
        # Q4 — lunch
        "Lunch_Free_Pct":                  lunch["pct_free"],
        # Q5 — slot balance
        "Slot_Balance_CV":                 round(cv_slots, 4),
    }
    for day, pct in lunch["by_day"].items():
        summary[f"Lunch_Free_{day}"] = pct
    # Q5 — day distribution of rescheduled events
    for day in DAY_ORDER:
        summary[f"Rescheduled_{day}"] = day_dist[day]

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(save_dir / f"heuristic_summary_{scenario}.csv", index=False)

    # Save iteration log
    if iteration_log:
        pd.DataFrame(iteration_log).to_csv(
            save_dir / f"heuristic_iteration_log_{scenario}.csv", index=False
        )

    print(f"\n[Heuristic] Results saved for {scenario}")
    print(summary_df[["Scenario","N_Rescheduled","Greedy_Clash_Score",
                       "LocalSearch_Clash_Score","Lunch_Free_Pct"]].to_string(index=False))

    return {
        "assignment_df": assignment_df,
        "summary":       summary,
        "iteration_log": iteration_log,
        "lunch":         lunch,
    }


# ============================================================
# Run both scenarios
# ============================================================
def run_all_heuristic_scenarios(data: dict,
                                 max_events: int = 2000,
                                 run_ls: bool = True,
                                 ls_iter: int = 300,
                                 out_dir: Path = None) -> dict:
    """
    Run heuristic for S1_9am5pm and S2_NoFriPM.

    Parameters
    ----------
    out_dir : Path, optional — directory where CSVs are saved (defaults to OUT_DIR)
    """
    events          = data["events"]
    conflict_pairs  = data.get("conflict_pairs")
    student_events  = data["student_events"]

    results = {}
    for scenario in ["S1_9am5pm", "S2_NoFriPM"]:
        res = run_heuristic_scenario(
            scenario=scenario,
            events=events,
            conflict_pairs=conflict_pairs,
            student_events=student_events,
            max_events=max_events,
            run_local_search=run_ls,
            ls_max_iter=ls_iter,
            out_dir=out_dir,
        )
        results[scenario] = res

    return results


if __name__ == "__main__":
    import argparse
    from data_preprocessing import run_preprocessing

    parser = argparse.ArgumentParser(description="Run Heuristic model")
    parser.add_argument("--scenario", default="both",
                        choices=["S1_9am5pm","S2_NoFriPM","both"])
    parser.add_argument("--max-events", type=int, default=2000)
    parser.add_argument("--no-ls",      action="store_true",
                        help="Skip local search (greedy only)")
    parser.add_argument("--ls-iter",    type=int, default=300)
    parser.add_argument("--no-conflicts", action="store_true")
    args = parser.parse_args()

    data = run_preprocessing(build_conflicts=not args.no_conflicts)

    if args.scenario == "both":
        run_all_heuristic_scenarios(
            data, args.max_events, not args.no_ls, args.ls_iter
        )
    else:
        run_heuristic_scenario(
            scenario=args.scenario,
            events=data["events"],
            conflict_pairs=data.get("conflict_pairs"),
            student_events=data["student_events"],
            max_events=args.max_events,
            run_local_search=not args.no_ls,
            ls_max_iter=args.ls_iter,
        )
