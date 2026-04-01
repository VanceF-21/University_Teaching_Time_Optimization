"""
baseline_analysis.py

Purpose:
    Compute baseline (current timetable) metrics and scenario-level summary
    statistics to answer Research Questions Q1-Q5:

    Q1: Feasibility of Mon-Fri 9am-5pm
    Q2: Feasibility of eliminating Friday 12pm-6pm
    Q3: Clashes under each scenario
    Q4: Lunch break (12-2pm) availability for students
    Q5: Timeslot and room utilisation changes

Outputs (saved to outputs/):
    - baseline_timeslot_utilisation.csv
    - baseline_room_utilisation.csv
    - scenario_displaced_summary.csv
    - lunch_break_analysis.csv
    - utilisation_comparison.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


# 1. Timeslot Utilisation
def compute_timeslot_utilisation(events: pd.DataFrame, label: str = "Baseline") -> pd.DataFrame:
    """
    For each (Day, Start_Hour) timeslot, count the number of events scheduled there.
    Also computes:
      - Room-hours consumed (event_size × duration_min / 60)
      - WholeClass event count
      - Student-contact-hours (event_size × duration_min / 60)

    Parameters
    ----------
    events : DataFrame with Day, Start_Hour, Duration_min, Event_Size, WholeClass
    label  : label for this dataset (e.g. "Baseline", "S1_9am5pm")

    Returns
    -------
    DataFrame with columns: Day, Start_Hour, Num_Events, Num_WholeClass,
                             Student_Contact_Hours, Scenario
    """
    df = events.dropna(subset=["Day", "Start_Hour"]).copy()
    df["Contact_Hours"] = df["Event_Size"] * df["Duration_min"] / 60.0

    grp = (
        df.groupby(["Day", "Day_Idx", "Start_Hour"])
        .agg(
            Num_Events=("Event_ID", "count"),
            Num_WholeClass=("WholeClass", "sum"),
            Student_Contact_Hours=("Contact_Hours", "sum"),
        )
        .reset_index()
        .sort_values(["Day_Idx", "Start_Hour"])
    )
    grp["Scenario"] = label
    return grp


# 2. Room Utilisation
def compute_room_utilisation(events: pd.DataFrame,
                              rooms: pd.DataFrame,
                              label: str = "Baseline") -> pd.DataFrame:
    """
    Compute room-level utilisation:
      - Utilisation rate = Event_Size / Room_Capacity
      - Number of events per room
      - Average fill rate

    Parameters
    ----------
    events : DataFrame with Room, Event_Size, Duration_min, Day, Start_Hour
    rooms  : DataFrame with Room_ID, Capacity, Campus, Room_Type

    Returns
    -------
    DataFrame with per-room summary statistics
    """
    # Merge events with room capacity
    merged = events.merge(
        rooms[["Room_ID", "Capacity", "Campus", "Room_Type"]],
        left_on="Room",
        right_on="Room_ID",
        how="left",
    )
    merged = merged.dropna(subset=["Capacity", "Room"])
    merged["Fill_Rate"] = (merged["Event_Size"] / merged["Capacity"]).clip(0, 1)

    agg_dict = {
        "Num_Events":           ("Event_ID", "count"),
        "Avg_Fill_Rate":        ("Fill_Rate", "mean"),
        "Total_Student_Events": ("Event_Size", "sum"),
        "Capacity":             ("Capacity", "first"),
    }
    # Only include Campus/Room_Type if the merge brought them in
    if "Campus_y" in merged.columns:
        merged["Campus"]    = merged["Campus_y"]
        merged["Room_Type"] = merged.get("Room_Type_y", merged.get("Room_Type", "Unknown"))
    elif "Campus" in merged.columns:
        pass   # already there from events
    else:
        merged["Campus"]    = "Unknown"
        merged["Room_Type"] = "Unknown"

    agg_dict["Campus"]    = ("Campus", "first")
    agg_dict["Room_Type"] = ("Room_Type", "first")

    room_stats = (
        merged.groupby("Room")
        .agg(**agg_dict)
        .reset_index()
    )
    room_stats["Scenario"] = label
    return room_stats


# 3. Displaced Event Summary
def compute_displaced_summary(events: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise the number and proportion of displaced events under each scenario.
    Also breaks down by Event_Type and WholeClass status.

    Returns
    -------
    DataFrame: Scenario, Metric, Value
    """
    total = len(events)
    records = []

    for col, name in [("Displaced_S1", "S1_9am5pm"),
                      ("Displaced_S2", "S2_NoFriPM")]:
        disp = events[events[col]]
        pct  = 100 * len(disp) / total

        records.append({
            "Scenario":                name,
            "Total_Events":            total,
            "Displaced_Events":        len(disp),
            "Displaced_Pct":           round(pct, 2),
            "Displaced_WholeClass":    int(disp["WholeClass"].sum()),
            "Displaced_SubGroup":      int((~disp["WholeClass"]).sum()),
            "Displaced_Unique_Modules": disp["Module_Code"].nunique(),
        })

        # Per day breakdown
        for day in DAY_ORDER:
            day_disp = disp[disp["Day"] == day]
            records.append({
                "Scenario":                f"{name}_{day}",
                "Total_Events":            len(events[events["Day"] == day]),
                "Displaced_Events":        len(day_disp),
                "Displaced_Pct":           round(
                    100 * len(day_disp) / max(len(events[events["Day"] == day]), 1), 2),
                "Displaced_WholeClass":    int(day_disp["WholeClass"].sum()),
                "Displaced_SubGroup":      int((~day_disp["WholeClass"]).sum()),
                "Displaced_Unique_Modules": day_disp["Module_Code"].nunique(),
            })

    return pd.DataFrame(records)


# 4. Lunch Break Feasibility
def compute_lunch_break_feasibility(events: pd.DataFrame,
                                     student_events: pd.DataFrame,
                                     scenario_displaced_col: str = "Displaced_S0"
                                     ) -> dict:
    """
    Determine what fraction of students have a FREE LUNCH WINDOW (12:00-14:00)
    under a given scenario.

    Logic:
      - For scenario Sn, remove events that are displaced (they would be rescheduled
        or dropped). Remaining events form the 'in-window' schedule.
      - For each student: check if any of their in-window events overlap [12, 14].
      - A student has a 'free lunch' if they have NO events between 12:00 and 14:00.
      - 'Overlap' means: Start_Hour < 14 AND End_Hour > 12

    Parameters
    ----------
    events                : full events DataFrame (with End_Hour, Start_Hour, Day, Displaced_* cols)
    student_events        : student-event mapping (AnonID, Event_ID)
    scenario_displaced_col: e.g. "Displaced_S0" → only keep non-displaced events

    Returns
    -------
    dict with keys:
        total_students, lunch_free_students, lunch_free_pct,
        by_day (dict: day → pct of students free at lunch on that specific day)
    """
    # In-window events (not displaced)
    in_window_events = events[~events[scenario_displaced_col]].copy()

    # Events that overlap the lunch window 12-14
    lunch_mask = (in_window_events["Start_Hour"] < 14.0) & (in_window_events["End_Hour"] > 12.0)
    lunch_events = in_window_events[lunch_mask]["Event_ID"].unique()

    # Students with at least one lunch event
    se = student_events.copy()
    students_with_lunch_event = set(
        se[se["Event_ID"].isin(lunch_events)]["AnonID"].unique()
    )
    all_students = set(se["AnonID"].unique())
    students_free = all_students - students_with_lunch_event

    total = len(all_students)
    free  = len(students_free)

    # By day: what fraction of students are free at lunch on each specific day
    by_day = {}
    for day in DAY_ORDER:
        day_lunch_events = in_window_events[
            lunch_mask & (in_window_events["Day"] == day)
        ]["Event_ID"].unique()
        students_busy_that_day = set(
            se[se["Event_ID"].isin(day_lunch_events)]["AnonID"].unique()
        )
        free_that_day = all_students - students_busy_that_day
        by_day[day] = round(100 * len(free_that_day) / max(total, 1), 2)

    return {
        "total_students":      total,
        "lunch_free_students": free,
        "lunch_free_pct":      round(100 * free / max(total, 1), 2),
        "by_day":              by_day,
    }


# 5. Clash Detection (schedule-based, not MIP-based)
def _share_module_code(code1: str, code2: str) -> bool:
    """
    Return True if code1 and code2 share at least one module identifier.

    Some events are jointly taught and carry a comma-separated Module_Code
    such as "BVMS08060_SS1_YR_2024/5, BVMS08061_SS1_YR_2024/5". A student
    may be registered to both the joint event and the individual module event,
    which would otherwise be treated as two different modules and generate a
    spurious clash. This function treats any overlap in module code sets as
    "same module", so such pairs are skipped during clash detection.

    Examples
    --------
    _share_module_code("A, B", "A")    → True  (shared: A)
    _share_module_code("A, B", "B, C") → True  (shared: B)
    _share_module_code("A", "B")       → False
    """
    set1 = {c.strip() for c in code1.split(",")}
    set2 = {c.strip() for c in code2.split(",")}
    return bool(set1 & set2)


def detect_clashes(events: pd.DataFrame,
                   student_events: pd.DataFrame,
                   scenario_displaced_col: str = "Displaced_S0") -> dict:
    """
    Count scheduling clashes in the WEEKLY timetable under a scenario.

    KEY INSIGHT: Each Event_ID in the data represents a SINGLE WEEKLY OCCURRENCE
    (one specific week) of a recurring timetable slot. For example, a lecture
    "Monday 9am, Semester 1" may appear as 9 separate Event_IDs (one per teaching week).

    To detect clashes in the WEEKLY schedule, we:
      1. Deduplicate events to unique (Module_Code, Day, Start_Hour, End_Hour, Semester)
         per student — each represents a recurring weekly timeslot.
      2. Check if any two events from DIFFERENT modules for the same student overlap.
      3. Cross-semester comparisons are excluded (Sem1 vs Sem2 can't clash).

    Joint module handling:
      Some events carry a comma-separated Module_Code (e.g. "MOD_A, MOD_B"), meaning
      one session counts toward multiple modules. Two events are treated as the same
      module if their module code sets share at least one identifier, preventing false
      clashes between a joint event and its constituent single-module counterpart.

    Severity levels:
      - Severity 3 (most severe): WholeClass vs WholeClass
      - Severity 2: WholeClass vs SubGroup
      - Severity 1: SubGroup vs SubGroup

    Returns
    -------
    dict:
        total_clash_pairs, severity_3_WC_WC, severity_2_WC_SG, severity_1_SG_SG,
        students_with_clashes, student_clash_pct
    """
    in_window = events[~events[scenario_displaced_col]][
        ["Event_ID", "Module_Code", "Day", "Start_Hour", "End_Hour", "WholeClass", "Semester"]
    ].copy()

    # Build a canonical slot ID: one representative Event_ID per
    # (Module_Code, Day, Start_Hour, Semester) recurring timeslot.
    # Multiple Event_IDs sharing the same slot (different teaching weeks of
    # the same recurring event) collapse to this single representative so that
    # the same structural clash is never counted more than once.
    canonical = (
        in_window.sort_values("Event_ID")
        .groupby(["Module_Code", "Day", "Start_Hour", "Semester"])
        .agg(Canon_EID=("Event_ID", "first"),
             End_Hour=("End_Hour", "first"),
             WholeClass=("WholeClass", "first"))
        .reset_index()
    )

    # Merge student-events with event times.
    # Use only Event_ID from student_events to avoid Semester column collision.
    se_times = student_events[["AnonID", "Event_ID"]].merge(
        in_window[["Event_ID", "Module_Code", "Day", "Start_Hour", "End_Hour", "WholeClass", "Semester"]],
        on="Event_ID", how="inner"
    )
    # Rename Semester from events table to Sem
    se_times.rename(columns={"Semester": "Sem"}, inplace=True)

    # Attach canonical slot ID
    se_times = se_times.merge(
        canonical[["Module_Code", "Day", "Start_Hour", "Semester", "Canon_EID"]],
        left_on=["Module_Code", "Day", "Start_Hour", "Sem"],
        right_on=["Module_Code", "Day", "Start_Hour", "Semester"],
        how="left"
    ).drop(columns=["Semester"], errors="ignore")

    # Deduplicate to unique recurring weekly timeslots per student:
    # one row per (AnonID, Canon_EID) — collapses multi-week occurrences
    # and ensures each structural slot appears once per student.
    weekly_slots = (
        se_times.groupby(["AnonID", "Canon_EID", "Module_Code", "Day",
                          "Start_Hour", "End_Hour", "Sem", "WholeClass"])
        .first()
        .reset_index()
    )

    print(f"  Weekly unique slots (after deduplication): {len(weekly_slots):,}")

    # Group by (student, day, semester) and check for overlapping slots.
    # Clash pairs are tracked as unique (Canon_EID_A, Canon_EID_B) tuples —
    # each structural clash counted exactly once, regardless of how many
    # students share it or how many weeks it recurs.
    students_with_any_clash = set()
    unique_clash_pairs = {}   # (canon_eid_a, canon_eid_b) -> severity
    sev = {1: 0, 2: 0, 3: 0}

    for (student, day, sem), grp in weekly_slots.groupby(["AnonID", "Day", "Sem"]):
        slots = grp[["Canon_EID", "Module_Code", "Start_Hour", "End_Hour", "WholeClass"]].values.tolist()
        n = len(slots)
        for i in range(n):
            for j in range(i + 1, n):
                s1 = slots[i]   # [canon_eid, mod, start, end, wc]
                s2 = slots[j]
                # Skip same module or jointly-taught siblings
                if _share_module_code(s1[1], s2[1]):
                    continue
                # Check time overlap
                if s1[2] < s2[3] and s2[2] < s1[3]:
                    students_with_any_clash.add(student)
                    pair_key = tuple(sorted([s1[0], s2[0]]))
                    if pair_key not in unique_clash_pairs:
                        wc1, wc2 = bool(s1[4]), bool(s2[4])
                        level = 3 if (wc1 and wc2) else 2 if (wc1 or wc2) else 1
                        unique_clash_pairs[pair_key] = level
                        sev[level] += 1

    total_pairs = len(unique_clash_pairs)

    total_students = student_events["AnonID"].nunique()
    return {
        "total_clash_pairs":     total_pairs,
        "severity_3_WC_WC":      sev[3],
        "severity_2_WC_SG":      sev[2],
        "severity_1_SG_SG":      sev[1],
        "students_with_clashes": len(students_with_any_clash),
        "student_clash_pct":     round(100 * len(students_with_any_clash) / max(total_students, 1), 2),
    }


# 6a. NO_SLOT Feasibility (Q1/Q2)
def compute_noslot_feasibility(events: pd.DataFrame) -> pd.DataFrame:
    """
    For each scenario, determine which displaced events have NO valid timeslot,
    i.e. their duration is too long to fit within the allowed teaching window.

    Logic mirrors heuristic_model.get_allowed_slots():
      S1_9am5pm  : earliest start = 9am, must end <= 17:00 → max duration = 8h = 480 min
      S2_NoFriPM : earliest start = 9am, must end <= 18:00 → max duration = 9h = 540 min
                   (Friday events must finish by 12:00, but Mon-Thu still allow up to 9h,
                   so the binding constraint for NO_SLOT remains 540 min)

    Returns
    -------
    DataFrame: Scenario, Total_Displaced, N_NoSlot, NoSlot_Pct,
               N_NoSlot_WholeClass, N_NoSlot_SubGroup,
               plus one column per Event_Type that has any NO_SLOT events.
    """
    # Thresholds: max feasible duration (minutes) per scenario
    WINDOW_MAX_DUR = {
        "S1_9am5pm":  480.0,   # 17 - 9 = 8 h
        "S2_NoFriPM": 540.0,   # 18 - 9 = 9 h (Mon-Thu window; displaced Fri events can move to Mon-Thu)
    }
    COL_MAP = {"S1_9am5pm": "Displaced_S1", "S2_NoFriPM": "Displaced_S2"}

    records = []
    for scenario, max_dur in WINDOW_MAX_DUR.items():
        col  = COL_MAP[scenario]
        disp = events[events[col]].copy()
        disp["NoSlot"] = disp["Duration_min"] > max_dur

        no_slot = disp[disp["NoSlot"]]
        n_total = len(disp)
        n_ns    = len(no_slot)

        row = {
            "Scenario":            scenario,
            "Total_Displaced":     n_total,
            "Max_Duration_Min":    max_dur,
            "N_NoSlot":            n_ns,
            "NoSlot_Pct":          round(100 * n_ns / max(n_total, 1), 2),
            "N_NoSlot_WholeClass": int(no_slot["WholeClass"].sum()),
            "N_NoSlot_SubGroup":   int((~no_slot["WholeClass"]).sum()),
        }
        # Top event types with NO_SLOT events
        if n_ns > 0:
            by_type = (
                no_slot.groupby("Event_Type").size()
                       .sort_values(ascending=False)
                       .head(10)
            )
            for etype, cnt in by_type.items():
                row[f"NoSlot_{etype.replace(' ', '_')}"] = int(cnt)

        records.append(row)
        print(f"  {scenario}: {n_ns:,}/{n_total:,} displaced events have NO valid slot "
              f"({row['NoSlot_Pct']:.1f}%)  — WholeClass: {row['N_NoSlot_WholeClass']:,}")

    return pd.DataFrame(records)


# 6b. Hourly Load Comparison by Day (Q5)
def compute_hourly_load_comparison(events: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the number of events scheduled in each (Day, Start_Hour) slot
    under the baseline and both proposed scenarios.

    For each scenario, only IN-WINDOW events are counted (displaced events
    are assumed to be rescheduled and are excluded from the original slot).
    This shows how the hour-by-hour teaching load is distributed across the
    week before any optimisation is applied — the 'pressure' on each slot.

    Returns
    -------
    DataFrame: Scenario, Day, Start_Hour, Num_Events, Num_WholeClass,
               Student_Contact_Hours
    """
    records = []
    for col, name in [("Displaced_S0", "S0_Baseline"),
                      ("Displaced_S1", "S1_9am5pm"),
                      ("Displaced_S2", "S2_NoFriPM")]:
        in_window = events[~events[col]].dropna(subset=["Day", "Start_Hour"]).copy()
        in_window["Contact_Hours"] = (
            in_window["Event_Size"] * in_window["Duration_min"] / 60.0
        )
        grp = (
            in_window.groupby(["Day", "Start_Hour"])
            .agg(
                Num_Events=("Event_ID", "count"),
                Num_WholeClass=("WholeClass", "sum"),
                Student_Contact_Hours=("Contact_Hours", "sum"),
            )
            .reset_index()
        )
        grp["Scenario"] = name
        records.append(grp)

    result = pd.concat(records, ignore_index=True)
    # Add day ordering for easy sorting
    day_idx = {d: i for i, d in enumerate(DAY_ORDER)}
    result["Day_Idx"] = result["Day"].map(day_idx)
    result = result.sort_values(["Scenario", "Day_Idx", "Start_Hour"]).drop(columns="Day_Idx")
    return result


# 6c. Utilisation Comparison Across Scenarios
def compute_utilisation_comparison(events: pd.DataFrame) -> pd.DataFrame:
    """
    Compare room/timeslot utilisation across baseline and two scenarios.
    For each scenario, we consider only the in-window events.

    Metrics:
      - Total event-hours (sum of Duration_min/60 for all events)
      - Total available teaching slots (days × hours × rooms)
      - Utilisation rate = event-hours / available-hours
    """
    records = []

    # Total unique rooms
    total_rooms = events["Room"].nunique()

    for col, name in [("Displaced_S0", "S0_Baseline"),
                      ("Displaced_S1", "S1_9am5pm"),
                      ("Displaced_S2", "S2_NoFriPM")]:
        in_window = events[~events[col]]

        # Available UNIQUE (Room, Timeslot) combinations per scenario
        # = Number of rooms × number of allowed hourly slots
        if name == "S0_Baseline":
            avail_slots = 9 * 5      # 9am-6pm × 5 days = 45 slots
        elif name == "S1_9am5pm":
            avail_slots = 8 * 5      # 9am-5pm × 5 days = 40 slots
        else:                         # S2: Mon-Thu 9am-6pm (9×4=36) + Fri 9am-12pm (3 slots ending by 12:00)
            avail_slots = 9 * 4 + 3  # = 39 slots

        total_avail_room_slots = avail_slots * total_rooms

        # Room-slot utilisation: count unique (Room, Day, Start_Hour) occupied
        occupied = in_window.dropna(subset=["Room", "Day", "Start_Hour"])
        occupied_pairs = occupied.drop_duplicates(subset=["Room", "Day", "Start_Hour"])
        n_occupied_slots = len(occupied_pairs)

        event_hours     = (in_window["Duration_min"] / 60.0).sum()
        student_contact = (in_window["Event_Size"] * in_window["Duration_min"] / 60.0).sum()

        records.append({
            "Scenario":              name,
            "In_Window_Events":      len(in_window),
            "Displaced_Events":      len(events) - len(in_window),
            "Total_Event_Hours":     round(event_hours, 1),
            "Student_Contact_Hours": round(student_contact, 1),
            "Available_Room_Slots":  total_avail_room_slots,
            "Occupied_Room_Slots":   n_occupied_slots,
            "Room_Utilisation_Pct":  round(100 * n_occupied_slots / max(total_avail_room_slots, 1), 2),
            "Avg_Events_Per_Slot":   round(len(in_window) / avail_slots, 2),
        })

    return pd.DataFrame(records)


# 7. Main Baseline Analysis
def run_baseline_analysis(data: dict, out_dir: Path = None) -> dict:
    """
    Run all baseline analyses and save results.

    Parameters
    ----------
    data    : dict returned by data_preprocessing.run_preprocessing()
    out_dir : Path, optional
        Directory where output CSVs are saved. Defaults to OUT_DIR.

    Returns
    -------
    dict of result DataFrames
    """
    save_dir = out_dir or OUT_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("BASELINE ANALYSIS")
    print("=" * 60)

    events         = data["events"]
    student_events = data["student_events"]
    rooms          = data["rooms"]

    results = {}

    # Timeslot utilisation (baseline)
    print("\n[baseline] Computing timeslot utilisation...")
    ts_util = compute_timeslot_utilisation(events, label="Baseline")
    ts_util.to_csv(save_dir / "baseline_timeslot_utilisation.csv", index=False)
    results["timeslot_util"] = ts_util
    print(f"  Peak timeslot: {ts_util.loc[ts_util['Num_Events'].idxmax(), ['Day', 'Start_Hour', 'Num_Events']].to_dict()}")

    # Room utilisation (baseline)
    print("\n[baseline] Computing room utilisation...")
    room_util = compute_room_utilisation(events, rooms, label="Baseline")
    room_util.to_csv(save_dir / "baseline_room_utilisation.csv", index=False)
    results["room_util"] = room_util
    print(f"  Avg room fill rate: {room_util['Avg_Fill_Rate'].mean():.2%}")
    print(f"  Rooms with > 90% fill: {(room_util['Avg_Fill_Rate'] > 0.9).sum():,}")

    # Displaced events summary
    print("\n[baseline] Computing displaced events summary...")
    displaced_summary = compute_displaced_summary(events)
    displaced_summary.to_csv(save_dir / "scenario_displaced_summary.csv", index=False)
    results["displaced_summary"] = displaced_summary
    print(displaced_summary[displaced_summary["Scenario"].isin(["S1_9am5pm", "S2_NoFriPM"])][
        ["Scenario", "Displaced_Events", "Displaced_Pct", "Displaced_WholeClass"]
    ].to_string(index=False))

    # Lunch break feasibility
    print("\n[baseline] Computing lunch break feasibility...")
    lunch_records = []
    for col, name in [("Displaced_S0", "S0_Baseline"),
                      ("Displaced_S1", "S1_9am5pm"),
                      ("Displaced_S2", "S2_NoFriPM")]:
        res = compute_lunch_break_feasibility(events, student_events, col)
        row = {
            "Scenario":            name,
            "Total_Students":      res["total_students"],
            "Lunch_Free_Students": res["lunch_free_students"],
            "Lunch_Free_Pct":      res["lunch_free_pct"],
        }
        for day, pct in res["by_day"].items():
            row[f"Free_{day}"] = pct
        lunch_records.append(row)
        print(f"  {name}: {res['lunch_free_pct']:.1f}% students free for lunch "
              f"({res['lunch_free_students']:,}/{res['total_students']:,})")

    lunch_df = pd.DataFrame(lunch_records)
    lunch_df.to_csv(save_dir / "lunch_break_analysis.csv", index=False)
    results["lunch_break"] = lunch_df

    # Clash detection (on current timetable, no rescheduling)
    print("\n[baseline] Detecting clashes in current timetable...")
    clash_records = []
    for col, name in [("Displaced_S0", "S0_Baseline"),
                      ("Displaced_S1", "S1_9am5pm"),
                      ("Displaced_S2", "S2_NoFriPM")]:
        print(f"  Computing clashes for {name}...")
        clash_res = detect_clashes(events, student_events, col)
        clash_res["Scenario"] = name
        clash_records.append(clash_res)
        print(f"  {name}: {clash_res['total_clash_pairs']:,} clash pairs, "
              f"{clash_res['student_clash_pct']:.1f}% students affected")

    clash_df = pd.DataFrame(clash_records)
    clash_df.to_csv(save_dir / "clash_analysis.csv", index=False)
    results["clashes"] = clash_df

    # Utilisation comparison
    print("\n[baseline] Computing utilisation comparison...")
    util_comp = compute_utilisation_comparison(events)
    util_comp.to_csv(save_dir / "utilisation_comparison.csv", index=False)
    results["utilisation_comparison"] = util_comp
    print(util_comp[["Scenario", "In_Window_Events", "Displaced_Events",
                     "Room_Utilisation_Pct"]].to_string(index=False))

    # Q1/Q2: NO_SLOT feasibility (events too long to fit any slot)
    print("\n[baseline] Computing NO_SLOT feasibility per scenario...")
    noslot_df = compute_noslot_feasibility(events)
    noslot_df.to_csv(save_dir / "noslot_feasibility.csv", index=False)
    results["noslot_feasibility"] = noslot_df

    # Q5: Hourly load by day across scenarios
    print("\n[baseline] Computing hourly load comparison by day...")
    hourly_load = compute_hourly_load_comparison(events)
    hourly_load.to_csv(save_dir / "hourly_load_comparison.csv", index=False)
    results["hourly_load"] = hourly_load
    for sc in ["S0_Baseline", "S1_9am5pm", "S2_NoFriPM"]:
        sc_df = hourly_load[hourly_load["Scenario"] == sc]
        if len(sc_df):
            peak = sc_df.loc[sc_df["Num_Events"].idxmax()]
            print(f"  {sc} peak slot: {peak['Day']} {int(peak['Start_Hour'])}:00 "
                  f"— {int(peak['Num_Events'])} events")

    print("\n[baseline] Baseline analysis complete.\n")
    return results


if __name__ == "__main__":
    from data_preprocessing import run_preprocessing
    data = run_preprocessing(build_conflicts=False)
    run_baseline_analysis(data)
