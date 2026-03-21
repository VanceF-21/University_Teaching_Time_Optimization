"""
data_preprocessing.py

Purpose:
    Load, clean, and preprocess all raw Excel data files.
    Outputs:
        - events.csv          : cleaned event schedule with parsed timeslots
        - student_events.csv  : student-to-event mapping (filtered)
        - conflict_pairs.csv  : pairs of events that share at least one student
        - rooms.csv           : cleaned room data

Key Design Decisions:
    - Timeslots are represented as (Day, Start_Hour) pairs.
    - Conflict pairs are built by inverting the student -> events mapping.
    - Online/placeholder events are excluded from analysis.
    - Only events with valid (non-null) Timeslot are retained.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# Paths
# ============================================================
DATA_DIR  = Path(__file__).resolve().parent / "data"
OUT_DIR   = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# ============================================================
# Constants
# ============================================================
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
DAY_INDEX = {d: i for i, d in enumerate(DAY_ORDER)}

# Teaching-hour boundaries (baseline)
BASELINE_START = 9.0   # 9:00
BASELINE_END   = 18.0  # up to 18:00 (last slot can start at 17:00 + 1h)

# Scenario definitions (used in scenario_analysis, mip_model, heuristic_model)
# Each scenario defines which (day, start_hour) combinations are ALLOWED.
# Events outside allowed slots are "displaced" and must be rescheduled.
SCENARIOS = {
    "S0_Baseline": {
        "description": "Current timetable, no restrictions",
        "allowed_days": DAY_ORDER,
        "start_min": 9.0,
        "end_max": 18.0,          # events can start up to 17 (end by 18)
        "exclude_fri_pm": False,
    },
    "S1_9am5pm": {
        "description": "Reduce core teaching to Mon-Fri 9am-5pm",
        "allowed_days": DAY_ORDER,
        "start_min": 9.0,
        "end_max": 17.0,          # events must FINISH by 17:00
        "exclude_fri_pm": False,
    },
    "S2_NoFriPM": {
        "description": "Eliminate Friday teaching from 12pm-6pm (Mon-Thu unchanged)",
        "allowed_days": DAY_ORDER,
        "start_min": 9.0,
        "end_max": 18.0,
        "exclude_fri_pm": True,   # no events on Fri >= 12:00
    },
}

# ============================================================
# 1. Load & Clean Events
# ============================================================
def load_events() -> pd.DataFrame:
    """
    Load '2024-5 Event Module Room.xlsx' and perform:
      - Column renaming for convenience
      - Timeslot parsing  → Day, Start_Hour, End_Hour
      - Day index encoding → Day_Idx (0=Mon … 4=Fri)
      - Filtering out online/placeholder events (no physical room needed)
      - Filtering to events with a valid Timeslot (physically scheduled)

    Returns
    -------
    pd.DataFrame with columns:
        Event_ID, Module_Code, Module_Name, Module_Department,
        Event_Name, Event_Type, Duration_min, Event_Size,
        Timeslot, Day, Start_Hour, End_Hour, Day_Idx,
        WholeClass, Room, Room_Type, Building, Campus,
        Semester, Room_Lock, Num_Weeks
    """
    print("[data_preprocessing] Loading Event Module Room...")
    raw = pd.read_excel(
        DATA_DIR / "2024-5 Event Module Room.xlsx",
        sheet_name="2024-5 Event Module Room",
        dtype={"Module Code": str, "Event ID": str},
    )

    # Rename columns to snake-case friendly names
    rename_map = {
        "Module Department":   "Module_Department",
        "Module Code":         "Module_Code",
        "Module Name":         "Module_Name",
        "Event ID":            "Event_ID",
        "Event Name":          "Event_Name",
        "Event Type":          "Event_Type",
        "Duration (minutes)":  "Duration_min",
        "Event Size":          "Event_Size",
        "Timeslot":            "Timeslot",
        "WholeClass":          "WholeClass",
        "Online Delivery":     "Online_Delivery",
        "Number of Weeks":     "Num_Weeks",
        "Weeks":               "Weeks",
        "Room":                "Room",
        "Room type 2":         "Room_Type2",
        "Room Type 1":         "Room_Type1",
        "Building":            "Building",
        "Campus":              "Campus",
        "Semester":            "Semester",
        "Room Lock":           "Room_Lock",
    }
    raw.rename(columns=rename_map, inplace=True)
    

    # Exclude purely online events (they have no physical room/timeslot impact)
    online_mask = raw["Online_Delivery"].notna()
    raw_in_person = raw[~online_mask].copy()
    print(f"  Total rows: {len(raw):,} | After removing online: {len(raw_in_person):,}")

    # Keep only events with a valid timeslot
    raw_scheduled = raw_in_person[raw_in_person["Timeslot"].notna()].copy()
    print(f"  After requiring valid Timeslot: {len(raw_scheduled):,}")

    # Parse timeslot → Day, Start_Hour
    def parse_timeslot(ts):
        parts = str(ts).strip().split()
        if len(parts) < 2:
            return None, None
        day = parts[0]
        hh, mm = map(int, parts[1].split(":"))
        return day, hh + mm / 60.0

    parsed = raw_scheduled["Timeslot"].apply(
        lambda x: pd.Series(parse_timeslot(x), index=["Day", "Start_Hour"])
    )
    raw_scheduled = pd.concat([raw_scheduled, parsed], axis=1)

    # Compute End_Hour
    raw_scheduled["End_Hour"] = (
        raw_scheduled["Start_Hour"] + raw_scheduled["Duration_min"] / 60.0
    )

    # Day index (Monday=0 … Friday=4)
    raw_scheduled["Day_Idx"] = raw_scheduled["Day"].map(DAY_INDEX)

    # Drop rows where Day is not Mon-Fri (e.g. Saturday/Sunday edge cases)
    valid_days = raw_scheduled["Day"].isin(DAY_ORDER)
    raw_scheduled = raw_scheduled[valid_days].copy()
    print(f"  After filtering to Mon-Fri: {len(raw_scheduled):,}")

    print(f"  WholeClass events: {raw_scheduled['WholeClass'].sum():,}")
    print(f"  Unique Event IDs:  {raw_scheduled['Event_ID'].nunique():,}")

    return raw_scheduled.reset_index(drop=True)


# ============================================================
# 2. Load Student–Event Mapping
# ============================================================
def load_student_events() -> pd.DataFrame:
    """
    Load '2024-5 Student Programme Module Event.xlsx'.

    Returns
    -------
    pd.DataFrame with columns:
        AnonID, Department, Programme, Programme_Code_Year,
        Course_Name, Course_ID, Event_ID, Semester
    """
    print("[data_preprocessing] Loading Student Programme Module Event...")
    raw = pd.read_excel(
        DATA_DIR / "2024-5 Student Programme Module Event.xlsx",
        sheet_name="2024-5 Student Programme Module",
        dtype={"Event ID": str, "AnonID": str},
    )
    rename_map = {
        "AnonID":               "AnonID",
        "Department":           "Department",
        "Programme":            "Programme",
        "Programme Code-Year":  "Programme_Code_Year",
        "Course Name":          "Course_Name",
        "Course ID":            "Course_ID",
        "Event ID":             "Event_ID",
        "Semester":             "Semester",
    }
    raw.rename(columns=rename_map, inplace=True)
    raw = raw.dropna(subset=["AnonID", "Event_ID"])
    raw = raw.drop_duplicates(subset=["AnonID", "Event_ID"])
    print(f"  Rows: {len(raw):,} | Students: {raw['AnonID'].nunique():,} | Events: {raw['Event_ID'].nunique():,}")
    return raw.reset_index(drop=True)


# ============================================================
# 3. Load Rooms
# ============================================================
def load_rooms() -> pd.DataFrame:
    """
    Load 'Rooms and Room Types.xlsx' sheet 'Room'.

    Returns
    -------
    pd.DataFrame with Room_ID, Description, Capacity, Campus, Room_Type
    """
    print("[data_preprocessing] Loading Rooms...")
    raw = pd.read_excel(
        DATA_DIR / "Rooms and Room Types.xlsx",
        sheet_name="Room",
    )
    rename_map = {
        "Id":                 "Room_ID",
        "Description":        "Description",
        "Capacity":           "Capacity",
        "Building":           "Building_Code",
        "Building.1":         "Building_Name",
        "Campus":             "Campus",
        "Central/Local":      "Central_Local",
        "Room Type":          "Room_Type",
        "Specialist room type": "Specialist_Type",
        "Has a 24-5 Event":   "Has_24_5_Event",
    }
    raw.rename(columns=rename_map, inplace=True)
    raw["Capacity"] = pd.to_numeric(raw["Capacity"], errors="coerce")
    print(f"  Rooms: {len(raw):,} | Capacity range: {raw['Capacity'].min():.0f} – {raw['Capacity'].max():.0f}")
    return raw.reset_index(drop=True)


# ============================================================
# 4. Load Programme-Course mapping
# ============================================================
def load_programme_course() -> pd.DataFrame:
    """
    Load 'Programme-Course.xlsx' sheet 'CourseModule'.
    Returns columns: CourseId (programme), ModuleId (course), Compulsory
    """
    print("[data_preprocessing] Loading Programme-Course mapping...")
    raw = pd.read_excel(DATA_DIR / "Programme-Course.xlsx", sheet_name="CourseModule")
    print(f"  Rows: {len(raw):,} | Compulsory: {raw['Compulsory'].sum():,}")
    return raw.reset_index(drop=True)


# ============================================================
# 5. Build Conflict Pairs
# ============================================================
def build_conflict_pairs(student_events: pd.DataFrame,
                         events: pd.DataFrame,
                         max_students_for_full_build: int = 30_000
                         ) -> pd.DataFrame:
    """
    Build a table of (Event_A, Event_B) pairs that share at least one student.
    This is the 'conflict graph' used in both the MIP and heuristic models.

    Strategy:
      - Group by AnonID → list of Event_IDs that student attends
      - For each student, generate all pairs (e1, e2) from their event list
      - De-duplicate pairs (canonical order: e1 < e2)

    For large datasets this can be memory-intensive, so we:
      (a) work only on events that appear in the events DataFrame (in-person, scheduled)
      (b) use efficient set operations

    Returns
    -------
    pd.DataFrame with columns:
        Event_A, Event_B, Shared_Students (count)
    """
    print("[data_preprocessing] Building conflict pairs (this may take a minute)...")

    # Filter student_events to only in-person scheduled events
    valid_event_ids = set(events["Event_ID"].unique())
    se_filtered = student_events[student_events["Event_ID"].isin(valid_event_ids)].copy()
    print(f"  Filtered student-events: {len(se_filtered):,}")

    # Build event → set of students (for counting shared students later)
    event_students = se_filtered.groupby("Event_ID")["AnonID"].apply(set)

    # Build student → list of events
    student_event_lists = (
        se_filtered.groupby("AnonID")["Event_ID"]
        .apply(list)
        .reset_index()
    )

    # Generate all conflict pairs
    pair_counts: dict = {}
    processed = 0
    for _, row in student_event_lists.iterrows():
        evs = sorted(row["Event_ID"])
        for i in range(len(evs)):
            for j in range(i + 1, len(evs)):
                key = (evs[i], evs[j])
                pair_counts[key] = pair_counts.get(key, 0) + 1
        processed += 1
        if processed % 5000 == 0:
            print(f"    Processed {processed:,} / {len(student_event_lists):,} students, "
                  f"pairs so far: {len(pair_counts):,}")

    print(f"  Total conflict pairs found: {len(pair_counts):,}")

    df_pairs = pd.DataFrame(
        [(a, b, c) for (a, b), c in pair_counts.items()],
        columns=["Event_A", "Event_B", "Shared_Students"],
    )
    return df_pairs


# ============================================================
# 6. Scenario Tagging
# ============================================================
def tag_displaced_events(events: pd.DataFrame) -> pd.DataFrame:
    """
    For each scenario, add a boolean column indicating whether each event
    is 'displaced' (falls outside the allowed time window).

    S0_Baseline  : no events displaced (reference)
    S1_9am5pm    : displaced if End_Hour > 17.0
    S2_NoFriPM   : displaced if Day=="Friday" AND End_Hour > 12.0
    """
    events = events.copy()

    # Baseline: nothing displaced
    events["Displaced_S0"] = False

    # Scenario 1: 9am-5pm — event must FINISH by 17:00
    events["Displaced_S1"] = events["End_Hour"] > 17.0

    # Scenario 2: No Friday PM — Friday events must finish by 12:00
    events["Displaced_S2"] = ( (events["Day"] == "Friday") & (events["End_Hour"] > 12.0))

    print("[data_preprocessing] Displaced event counts:")
    print(f"  S1 (9am-5pm):    {events['Displaced_S1'].sum():,} events displaced")
    print(f"  S2 (No Fri PM):  {events['Displaced_S2'].sum():,} events displaced")

    return events


# ============================================================
# 7. Compute Allowed Timeslots per Scenario
# ============================================================
def get_allowed_slots(scenario_key: str,
                      duration_min: float = 60.0) -> list:
    """
    Return a list of (Day, Start_Hour) tuples that are ALLOWED under the
    given scenario for an event of `duration_min` minutes.

    The end of the event = Start_Hour + duration_min/60 must be <= end_max.
    For Scenario 2, Friday events must finish by 12:00.
    """
    sc = SCENARIOS[scenario_key]
    slots = []
    for day in sc["allowed_days"]:
        # Candidate start hours (whole hours from baseline_start to ...)
        for sh in range(int(sc["start_min"]), int(sc["end_max"])):
            end_h = sh + duration_min / 60.0
            # Must end by end_max
            if end_h > sc["end_max"]:
                continue
            # No Friday PM: Friday events must finish by 12:00
            if sc["exclude_fri_pm"] and day == "Friday" and end_h > 12.0:
                continue
            slots.append((day, float(sh)))
    return slots


# ============================================================
# 8. Main pipeline
# ============================================================
def run_preprocessing(build_conflicts: bool = True, out_dir: Path = None) -> dict:
    """
    Full preprocessing pipeline. Saves output CSVs to out_dir (or OUT_DIR).

    Parameters
    ----------
    build_conflicts : bool
        If True (default), build and save the conflict-pairs table.
        Set to False for quick re-runs when conflict_pairs.csv already exists.
    out_dir : Path, optional
        Directory where output CSVs are saved. Defaults to OUT_DIR.

    Returns
    -------
    dict with keys: events, student_events, rooms, conflict_pairs, programme_course
    """
    save_dir = out_dir or OUT_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PREPROCESSING PIPELINE")
    print("=" * 60)

    # Load
    events          = load_events()
    student_events  = load_student_events()
    rooms           = load_rooms()
    programme_course = load_programme_course()

    # Tag displaced events per scenario
    events = tag_displaced_events(events)

    # Save events
    events.to_csv(save_dir / "events.csv", index=False)
    print(f"\n[data_preprocessing] Saved events.csv ({len(events):,} rows)")

    # Save student_events (filter to valid events only)
    valid_ids = set(events["Event_ID"].unique())
    se_valid = student_events[student_events["Event_ID"].isin(valid_ids)].copy()
    se_valid.to_csv(save_dir / "student_events.csv", index=False)
    print(f"[data_preprocessing] Saved student_events.csv ({len(se_valid):,} rows)")

    # Save rooms
    rooms.to_csv(save_dir / "rooms.csv", index=False)
    print(f"[data_preprocessing] Saved rooms.csv ({len(rooms):,} rows)")

    # Save programme-course
    programme_course.to_csv(save_dir / "programme_course.csv", index=False)
    print(f"[data_preprocessing] Saved programme_course.csv ({len(programme_course):,} rows)")

    conflict_pairs = None
    conflict_pairs_path = save_dir / "conflict_pairs.csv"

    if build_conflicts:
        conflict_pairs = build_conflict_pairs(se_valid, events)
        conflict_pairs.to_csv(conflict_pairs_path, index=False)
        print(f"[data_preprocessing] Saved conflict_pairs.csv ({len(conflict_pairs):,} rows)")
    elif conflict_pairs_path.exists():
        conflict_pairs = pd.read_csv(conflict_pairs_path, dtype=str)
        conflict_pairs["Shared_Students"] = conflict_pairs["Shared_Students"].astype(int)
        print(f"[data_preprocessing] Loaded conflict_pairs.csv ({len(conflict_pairs):,} rows)")
    else:
        print("[data_preprocessing] WARNING: conflict_pairs.csv not found. "
              "Run with build_conflicts=True to generate.")

    print("\n[data_preprocessing] Preprocessing complete.\n")
    return {
        "events":           events,
        "student_events":   se_valid,
        "rooms":            rooms,
        "programme_course": programme_course,
        "conflict_pairs":   conflict_pairs,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run preprocessing pipeline")
    parser.add_argument("--no-conflicts", action="store_true",
                        help="Skip building conflict pairs (fast mode)")
    args = parser.parse_args()
    run_preprocessing(build_conflicts=not args.no_conflicts)
