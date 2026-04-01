"""
room_utilisation_analysis.py
─────────────────────────────────────────────────────────────────────────────
Exploratory analysis of room capacity sufficiency across campuses and
semesters, before and after timetable rescheduling under S1 / S2.

Key argument:
  Although room-capacity constraints were excluded from the MIP model
  (because including hard room constraints frequently caused infeasibility),
  we demonstrate here that feasible room assignments exist after rescheduling:
  at each rescheduled timeslot there are sufficient free rooms of adequate
  capacity to accommodate the displaced events placed there.

The analysis is scoped to the three main campuses (Central, Kings Buildings,
Holyrood), which together account for ≥93% of all timetabled events.

Outputs (saved to room_analysis/figures/):
  Fig 1 – Room capacity distribution vs event-size distribution by campus
  Fig 2 – Heatmap: % rooms occupied per timeslot (baseline vs after S2)
  Fig 3 – Per-event: number of free feasible rooms at new timeslot
  Fig 4 – Greedy room-assignment feasibility rate by campus and scenario
  Fig 5 – Semester breakdown of feasibility rate (Sem 1 vs Sem 2)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ─── Paths ───────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "outputs" / "cleaned_data"
RUN_DIR  = ROOT / "outputs" / "run_20260330_162212"
OUT_DIR  = Path(__file__).resolve().parent / "figures"
OUT_DIR.mkdir(exist_ok=True)

# ─── Constants ───────────────────────────────────────────────────────────────
MAIN_CAMPUSES  = ["Central", "Kings Buildings", "Holyrood"]
CAMPUS_COLOURS = {
    "Central":         "#2196F3",
    "Kings Buildings": "#4CAF50",
    "Holyrood":        "#FF9800",
}
DAY_ORDER  = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
HOUR_SLOTS = list(range(9, 18))

# ─── Load data ───────────────────────────────────────────────────────────────
print("Loading data …")
events = pd.read_csv(DATA_DIR / "events.csv")
rooms  = pd.read_csv(DATA_DIR / "rooms.csv")
events["Campus"] = events["Campus"].str.strip()
rooms["Campus"]  = rooms["Campus"].str.strip()

h_s1 = pd.read_csv(RUN_DIR / "heuristic_new_assignment_S1_9am5pm.csv")
h_s2 = pd.read_csv(RUN_DIR / "heuristic_new_assignment_S2_NoFriPM.csv")
print(f"  events={len(events):,}  rooms={len(rooms):,}  "
      f"h_s1={len(h_s1):,}  h_s2={len(h_s2):,}")

# ─── Helpers ─────────────────────────────────────────────────────────────────

def overlaps_slot(ev_day, ev_start, ev_dur, slot_day, slot_start):
    """Does an event overlap the integer hour [slot_start, slot_start+1)?"""
    if ev_day != slot_day:
        return False
    ev_end = ev_start + ev_dur / 60.0
    return ev_start < (slot_start + 1) and ev_end > slot_start


def build_room_occupancy(ev_df: pd.DataFrame) -> dict:
    """
    Returns dict: (day, hour) → set of occupied Room_IDs.
    Only events with a non-null room are considered.
    """
    occ = {(d, h): set() for d in DAY_ORDER for h in HOUR_SLOTS}
    sub = ev_df[ev_df["Room"].notna() & (ev_df["Room"] != "0")].copy()
    for _, row in sub.iterrows():
        day  = row.get("Day")
        sh   = float(row.get("Start_Hour", 9))
        dur  = float(row.get("Duration_min", 60))
        rm   = str(row["Room"])
        if pd.isna(day):
            continue
        for h in HOUR_SLOTS:
            if overlaps_slot(day, sh, dur, day, h):
                occ[(day, h)].add(rm)
    return occ


def get_free_feasible_rooms(campus: str,
                             day: str,
                             start_h: float,
                             event_size: float,
                             occ_dict: dict,
                             rooms_df: pd.DataFrame) -> list:
    """
    Returns list of Room_IDs on the given campus that are:
      (a) free at the given timeslot, AND
      (b) have capacity >= event_size.
    """
    campus_rooms = rooms_df[rooms_df["Campus"] == campus]
    occupied = set()
    for h in HOUR_SLOTS:
        if overlaps_slot(day, start_h, 60, day, h):   # approximate 1h block
            occupied |= occ_dict.get((day, h), set())

    feasible = campus_rooms[
        (~campus_rooms["Room_ID"].isin(occupied)) &
        (campus_rooms["Capacity"] >= event_size)
    ]["Room_ID"].tolist()
    return feasible


# ─── Build fixed-event occupancy for each scenario ───────────────────────────
print("Building room occupancy for fixed events …")

def fixed_events_for_scenario(events_df, assignment_df, displaced_col):
    disp_ids = set(assignment_df["Event_ID"].astype(str))
    fixed = events_df[
        (~events_df["Event_ID"].astype(str).isin(disp_ids)) &
        (events_df[displaced_col] == False)
    ].copy()
    return fixed

fixed_s1 = fixed_events_for_scenario(events, h_s1, "Displaced_S1")
fixed_s2 = fixed_events_for_scenario(events, h_s2, "Displaced_S2")
occ_s1   = build_room_occupancy(fixed_s1)
occ_s2   = build_room_occupancy(fixed_s2)
occ_base = build_room_occupancy(events)   # baseline: all events
print(f"  Fixed S1={len(fixed_s1):,}  Fixed S2={len(fixed_s2):,}")

# ─── Event info lookup ───────────────────────────────────────────────────────
ev_info = (
    events.drop_duplicates("Event_ID")
    .set_index("Event_ID")[["Campus","Event_Size","Duration_min","Semester"]]
    .to_dict("index")
)

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 1  –  Room capacity vs event size distribution by campus  (CDF overlay)
# ═══════════════════════════════════════════════════════════════════════════════
print("\nGenerating Fig 1 – capacity vs event-size CDF …")

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)

for ax, campus in zip(axes, MAIN_CAMPUSES):
    colour = CAMPUS_COLOURS[campus]

    # Event sizes on this campus (events with rooms)
    ev_sizes = events[
        (events["Campus"] == campus) & events["Event_Size"].notna()
    ]["Event_Size"].dropna().values

    # Room capacities on this campus
    rm_caps = rooms[rooms["Campus"] == campus]["Capacity"].dropna().values

    # Plot CDFs
    for data, label, ls, alpha in [
        (np.sort(ev_sizes), "Event size (enrolment)", "-",   0.9),
        (np.sort(rm_caps),  "Room capacity (seats)",  "--",  0.9),
    ]:
        cdf = np.arange(1, len(data)+1) / len(data)
        ax.plot(data, cdf, linestyle=ls, color=colour, alpha=alpha,
                linewidth=1.8, label=label)

    # Shade region where rooms are sufficient (capacity ≥ p90 event size)
    p90 = np.percentile(ev_sizes, 90)
    ax.axvline(p90, color="grey", linestyle=":", linewidth=1.2,
               label=f"90th pct event size ({int(p90)})")

    ax.set_xlim(0, min(np.percentile(rm_caps, 99), 500))
    ax.set_xlabel("Student seats / event enrolment", fontsize=9)
    ax.set_title(campus, fontsize=10, fontweight="bold")
    ax.set_ylabel("Cumulative fraction" if campus == "Central" else "", fontsize=9)
    ax.legend(fontsize=7.5)
    ax.grid(linestyle="--", alpha=0.35)

    n_ev  = len(ev_sizes)
    n_rm  = len(rm_caps)
    ax.text(0.97, 0.08,
            f"Events: {n_ev:,}\nRooms: {n_rm}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color="grey")

# fig.suptitle(
#     "Cumulative distribution of event enrolment sizes vs room capacities by campus\n"
#     "(the room-capacity CDF lies to the right of the event-size CDF,\n"
#     " confirming that rooms can accommodate the events placed in them)",
#     fontsize=10, y=1.02)
plt.tight_layout()
fig.savefig(OUT_DIR / "fig1_cdf_eventsize_vs_roomcap.pdf", dpi=150, bbox_inches="tight")
fig.savefig(OUT_DIR / "fig1_cdf_eventsize_vs_roomcap.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved fig1")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 2  –  Heatmap: % of campus rooms OCCUPIED per (Day × Hour)
#   Three rows: Baseline / After S1 / After S2
#   Three cols: Central / Kings Buildings / Holyrood
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Fig 2 – room occupancy heatmaps …")

def occupancy_heatmap(occ_dict: dict, rooms_df: pd.DataFrame,
                      campus: str) -> pd.DataFrame:
    """% of campus rooms occupied at each (day, hour) slot."""
    n_rooms = len(rooms_df[rooms_df["Campus"] == campus])
    if n_rooms == 0:
        return pd.DataFrame(0, index=DAY_ORDER, columns=HOUR_SLOTS)
    mat = {}
    for day in DAY_ORDER:
        row = {}
        for h in HOUR_SLOTS:
            campus_rooms_in_campus = rooms_df[rooms_df["Campus"]==campus]["Room_ID"]
            occupied_on_campus = occ_dict.get((day, h), set()) & set(campus_rooms_in_campus)
            row[h] = 100 * len(occupied_on_campus) / n_rooms
        mat[day] = row
    return pd.DataFrame(mat).T.reindex(index=DAY_ORDER, columns=HOUR_SLOTS)


fig, axes = plt.subplots(3, 3, figsize=(14, 9))
row_labels  = ["Baseline", "After S1 rescheduling", "After S2 rescheduling"]
occ_sets    = [occ_base, occ_s1, occ_s2]

for r, (label, occ_d) in enumerate(zip(row_labels, occ_sets)):
    for c, campus in enumerate(MAIN_CAMPUSES):
        ax  = axes[r, c]
        mat = occupancy_heatmap(occ_d, rooms, campus)

        im = ax.imshow(mat.values, aspect="auto", origin="upper",
                       cmap="Blues", vmin=0, vmax=100)

        ax.set_xticks(range(len(HOUR_SLOTS)))
        ax.set_xticklabels([f"{h:02d}:00" for h in HOUR_SLOTS],
                           rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(DAY_ORDER)))
        ax.set_yticklabels([d[:3] for d in DAY_ORDER], fontsize=8)

        for i in range(len(DAY_ORDER)):
            for j in range(len(HOUR_SLOTS)):
                v = mat.values[i, j]
                if v > 5:
                    ax.text(j, i, f"{v:.0f}%", ha="center", va="center",
                            fontsize=6, color="white" if v > 65 else "black")

        if r == 0:
            n_rms = len(rooms[rooms["Campus"] == campus])
            ax.set_title(f"{campus}\n({n_rms} rooms)",
                         fontsize=9, fontweight="bold")
        if c == 0:
            ax.set_ylabel(label, fontsize=8, fontweight="bold")

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label="% rooms occupied")

# fig.suptitle(
#     "Percentage of rooms occupied per timeslot by campus\n"
#     "(white/light cells = available rooms for rescheduled events)",
#     fontsize=11, y=1.005)
plt.tight_layout()
fig.savefig(OUT_DIR / "fig2_room_occupancy_heatmap.pdf", dpi=150, bbox_inches="tight")
fig.savefig(OUT_DIR / "fig2_room_occupancy_heatmap.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved fig2")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 3  –  Per-event: number of free feasible rooms at new timeslot
#           (for a random subsample of 500 rescheduled events per scenario)
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Fig 3 – per-event feasible room count …")

def count_feasible_rooms_for_assignment(assignment_df, occ_dict, rooms_df,
                                         sample_n=600, seed=42):
    """
    For each rescheduled event in assignment_df, count how many rooms on
    the same campus are free AND have capacity >= event_size at the new slot.
    Returns a list of (campus, n_feasible_rooms) tuples.
    """
    rng    = np.random.RandomState(seed)
    subset = assignment_df.dropna(subset=["New_Day","New_Start_Hour"]).copy()
    if len(subset) > sample_n:
        subset = subset.sample(sample_n, random_state=rng)

    results = []
    for _, row in subset.iterrows():
        eid  = str(row["Event_ID"])
        info = ev_info.get(eid, {})
        campus = info.get("Campus", "Unknown")
        if campus not in MAIN_CAMPUSES:
            continue
        ev_size = float(info.get("Event_Size", 0))
        new_day = row["New_Day"]
        new_h   = float(row["New_Start_Hour"])
        dur     = float(info.get("Duration_min", 60))

        # Rooms on this campus that are free at the new timeslot
        campus_rooms = rooms_df[rooms_df["Campus"] == campus]
        occupied = set()
        for h in HOUR_SLOTS:
            if overlaps_slot(new_day, new_h, dur, new_day, h):
                occupied |= occ_dict.get((new_day, h), set())

        n_free_feasible = len(campus_rooms[
            (~campus_rooms["Room_ID"].isin(occupied)) &
            (campus_rooms["Capacity"] >= ev_size)
        ])
        results.append({"Campus": campus, "N_Feasible": n_free_feasible,
                        "Event_Size": ev_size})
    return pd.DataFrame(results)


print("  Counting feasible rooms for S1 …")
ff_s1 = count_feasible_rooms_for_assignment(h_s1, occ_s1, rooms)
print("  Counting feasible rooms for S2 …")
ff_s2 = count_feasible_rooms_for_assignment(h_s2, occ_s2, rooms)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, (label, ff_df) in zip(axes, [("S1 (Mon–Fri 9am–5pm)", ff_s1),
                                      ("S2 (No Fri PM)",       ff_s2)]):
    for campus in MAIN_CAMPUSES:
        sub = ff_df[ff_df["Campus"] == campus]["N_Feasible"].values
        if len(sub) == 0:
            continue
        bins = range(0, min(int(sub.max())+10, 200), 5)
        ax.hist(sub, bins=bins, alpha=0.55, label=campus,
                color=CAMPUS_COLOURS[campus])
        pct_ok = 100 * (sub > 0).mean()
        med    = int(np.median(sub))
        ax.axvline(med, color=CAMPUS_COLOURS[campus], linestyle="--",
                   linewidth=1.2)
        ax.text(med+1, ax.get_ylim()[1]*0.02, f"med={med}",
                color=CAMPUS_COLOURS[campus], fontsize=8, va="bottom")

    ax.set_xlabel("Number of free feasible rooms at new timeslot", fontsize=9)
    ax.set_ylabel("Number of rescheduled events", fontsize=9)
    ax.set_title(f"Scenario {label}", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Annotation: % with at least one feasible room
    for campus in MAIN_CAMPUSES:
        sub = ff_df[ff_df["Campus"] == campus]["N_Feasible"].values
        if len(sub):
            pct = 100*(sub>0).mean()
            print(f"    {label}  {campus:20s}: {pct:.1f}% have ≥1 room")

# fig.suptitle(
#     "Number of free, capacity-sufficient rooms available to each rescheduled event\n"
#     "(vertical dashed lines = median; events with ≥1 bar can be feasibly assigned)",
#     fontsize=10)
plt.tight_layout()
fig.savefig(OUT_DIR / "fig3_per_event_feasible_rooms.pdf", dpi=150, bbox_inches="tight")
fig.savefig(OUT_DIR / "fig3_per_event_feasible_rooms.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved fig3")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 4  –  Greedy room-assignment feasibility rate  (bar chart)
#   For each (campus × scenario), what % of rescheduled events can receive
#   a room via a greedy largest-event-first assignment?
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Fig 4 – greedy feasibility rate …")

def greedy_room_assignment_feasibility(assignment_df, occ_dict, rooms_df,
                                        ev_info_dict):
    """
    For each new timeslot, greedily assign rooms (largest event first, smallest
    sufficient room). Returns per-campus feasibility rate.
    """
    # Group rescheduled events by (campus, new_day, new_start_hour)
    slot_events = {}
    for _, row in assignment_df.dropna(subset=["New_Day","New_Start_Hour"]).iterrows():
        eid    = str(row["Event_ID"])
        info   = ev_info_dict.get(eid, {})
        campus = info.get("Campus","Unknown")
        if campus not in MAIN_CAMPUSES:
            continue
        nd  = row["New_Day"]
        nh  = float(row["New_Start_Hour"])
        dur = float(info.get("Duration_min", 60))
        key = (campus, nd, nh)
        slot_events.setdefault(key, []).append({
            "eid":    eid,
            "size":   float(info.get("Event_Size", 0)),
            "dur":    dur,
            "sem":    info.get("Semester",""),
        })

    campus_assigned   = {c: 0 for c in MAIN_CAMPUSES}
    campus_total      = {c: 0 for c in MAIN_CAMPUSES}
    sem_results       = {}  # (campus, semester) → [assigned, total]

    for (campus, nd, nh), ev_list in slot_events.items():
        # Rooms on campus that are free at this slot
        campus_rm = rooms_df[rooms_df["Campus"] == campus].copy()
        occupied  = set()
        for h in HOUR_SLOTS:
            if overlaps_slot(nd, nh, 60, nd, h):
                occupied |= occ_dict.get((nd, h), set())
        free_rm = campus_rm[~campus_rm["Room_ID"].isin(occupied)].copy()
        free_rm = free_rm.sort_values("Capacity")

        # Sort events: largest first
        ev_list_sorted = sorted(ev_list, key=lambda e: e["size"], reverse=True)
        free_caps = list(free_rm["Capacity"].values)   # sorted ascending

        for ev in ev_list_sorted:
            sem    = ev["sem"]
            key_cs = (campus, sem)
            campus_total[campus] += 1
            sem_results.setdefault(key_cs, [0, 0])[1] += 1

            # Find smallest room with capacity >= ev["size"]
            idx = next(
                (i for i, cap in enumerate(free_caps) if cap >= ev["size"]),
                None
            )
            if idx is not None:
                campus_assigned[campus] += 1
                sem_results[key_cs][0]  += 1
                free_caps.pop(idx)  # room is now occupied

    rates = {c: (campus_assigned[c] / campus_total[c] * 100
                 if campus_total[c] > 0 else 0)
             for c in MAIN_CAMPUSES}
    return rates, sem_results, campus_total


print("  Running greedy assignment for S1 …")
r_s1, sr_s1, tot_s1 = greedy_room_assignment_feasibility(h_s1, occ_s1, rooms, ev_info)
print("  Running greedy assignment for S2 …")
r_s2, sr_s2, tot_s2 = greedy_room_assignment_feasibility(h_s2, occ_s2, rooms, ev_info)

print("\n  Greedy feasibility rates:")
for c in MAIN_CAMPUSES:
    print(f"    {c:20s}  S1={r_s1[c]:.1f}%  S2={r_s2[c]:.1f}%")

fig, ax = plt.subplots(figsize=(8, 5))
x      = np.arange(len(MAIN_CAMPUSES))
bar_w  = 0.32
col_s1 = "#1976D2"
col_s2 = "#E65100"

b1 = ax.bar(x - bar_w/2,
            [r_s1[c] for c in MAIN_CAMPUSES], bar_w,
            label="Scenario S1", color=col_s1, alpha=0.85)
b2 = ax.bar(x + bar_w/2,
            [r_s2[c] for c in MAIN_CAMPUSES], bar_w,
            label="Scenario S2", color=col_s2, alpha=0.85)

for bars in [b1, b2]:
    for bar in bars:
        h_val = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h_val + 0.3,
                f"{h_val:.1f}%", ha="center", va="bottom", fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(MAIN_CAMPUSES, fontsize=11)
ax.set_ylabel("Events successfully assigned a room (%)", fontsize=10)
ax.set_ylim(0, 108)
ax.axhline(100, color="black", linestyle=":", linewidth=1.0, alpha=0.4)
ax.set_title("Greedy room-assignment feasibility rate after rescheduling\n"
             "(events sorted largest-first; smallest sufficient free room assigned)",
             fontsize=10)
ax.legend(fontsize=10)
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
fig.savefig(OUT_DIR / "fig4_greedy_feasibility_rate.pdf", dpi=150)
fig.savefig(OUT_DIR / "fig4_greedy_feasibility_rate.png", dpi=150)
plt.close(fig)
print("  Saved fig4")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 5  –  Semester breakdown of feasibility
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Fig 5 – semester breakdown …")

def sem_rates(sr_dict, campus_list):
    records = []
    for campus in campus_list:
        for sem in ["Semester 1", "Semester 2"]:
            key = (campus, sem)
            vals = sr_dict.get(key, [0, 0])
            rate = vals[0]/vals[1]*100 if vals[1] > 0 else 0
            records.append({"Campus": campus, "Semester": sem,
                             "Rate": rate, "n": vals[1]})
    return pd.DataFrame(records)

df_sem_s1 = sem_rates(sr_s1, MAIN_CAMPUSES)
df_sem_s2 = sem_rates(sr_s2, MAIN_CAMPUSES)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, (scen_label, df_sem) in zip(axes, [
        ("S1 (Mon–Fri 9am–5pm)", df_sem_s1),
        ("S2 (No Fri PM)",       df_sem_s2)]):

    x     = np.arange(len(MAIN_CAMPUSES))
    bar_w = 0.32
    sems  = ["Semester 1", "Semester 2"]
    cols  = ["#7986CB", "#B0BEC5"]

    for i, (sem, colour) in enumerate(zip(sems, cols)):
        sub = df_sem[df_sem["Semester"] == sem].set_index("Campus").reindex(MAIN_CAMPUSES)
        bars = ax.bar(x + (i - 0.5)*bar_w, sub["Rate"].values, bar_w,
                      label=sem, color=colour, alpha=0.9)
        for bar, n_ev in zip(bars, sub["n"].values):
            h_val = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h_val + 0.4,
                    f"{h_val:.0f}%\n(n={int(n_ev)})",
                    ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(MAIN_CAMPUSES, fontsize=10)
    ax.set_ylabel("Greedy feasibility rate (%)", fontsize=9)
    ax.set_ylim(0, 115)
    ax.axhline(100, color="black", linestyle=":", linewidth=1.0, alpha=0.5)
    ax.set_title(f"Scenario {scen_label}", fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

# fig.suptitle(
#     "Greedy room-assignment feasibility rate by campus and semester\n"
#     "(n = number of rescheduled events on that campus in that semester)",
#     fontsize=10)
plt.tight_layout()
fig.savefig(OUT_DIR / "fig5_semester_feasibility.pdf", dpi=150, bbox_inches="tight")
fig.savefig(OUT_DIR / "fig5_semester_feasibility.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved fig5")


# ─── Summary CSV ─────────────────────────────────────────────────────────────
print("\nSaving summary tables …")

# Greedy rates
summary_rows = []
for campus in MAIN_CAMPUSES:
    summary_rows.append({
        "Campus":          campus,
        "Greedy_Rate_S1":  round(r_s1[campus], 1),
        "Greedy_Rate_S2":  round(r_s2[campus], 1),
        "Total_Rooms":     len(rooms[rooms["Campus"]==campus]),
        "Total_Capacity":  int(rooms[rooms["Campus"]==campus]["Capacity"].sum()),
    })
pd.DataFrame(summary_rows).to_csv(OUT_DIR / "feasibility_summary.csv", index=False)
print(pd.DataFrame(summary_rows).to_string(index=False))

# Per-event feasible room count stats
for label, ff_df in [("S1", ff_s1), ("S2", ff_s2)]:
    stat = ff_df.groupby("Campus")["N_Feasible"].agg(
        ["median","mean",lambda x: (x>0).mean()*100]).round(1)
    stat.columns = ["Median_Feasible","Mean_Feasible","Pct_With_Feasible"]
    print(f"\nPer-event feasible room stats ({label}):")
    print(stat.to_string())

print("\n✓ All figures saved to", OUT_DIR)
