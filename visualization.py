"""
visualization.py

Purpose:
    Generate all publication-quality charts and visualisations for the project.
    Charts are saved to outputs/figures/.

Chart catalogue:
    1.  heatmap_events_by_day_hour.png       - event density across week (baseline)
    2.  heatmap_wholeclass_by_day_hour.png   - WholeClass event density (baseline)
    3.  displaced_events_bar.png             - displaced counts S1 vs S2
    4.  displaced_by_day.png                 - displaced events broken down by day
    5.  displaced_by_eventtype.png           - displaced by event type
    6.  clash_comparison.png                 - clash severity: baseline vs S1 vs S2
    7.  lunch_break_comparison.png           - lunch free % across scenarios
    8.  lunch_by_day.png                     - lunch free by day of week
    9.  room_utilisation_histogram.png       - fill rate distribution (baseline)
    10. utilisation_comparison.png           - room utilisation: baseline vs S1 vs S2
    11. heuristic_iteration_curve.png        - local search convergence (S1 & S2)
    12. mip_vs_heuristic_comparison.png      - clash reduction: MIP vs heuristic
    13. timeslot_load_before_after.png       - slot load baseline vs rescheduled
    14. end_hour_distribution.png            - event end-time distribution
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

OUT_DIR  = Path(__file__).resolve().parent / "outputs"
FIG_DIR  = OUT_DIR / "figures"   # legacy fallback; not auto-created

# Runtime override: set by run_all_visualisations() when out_dir is provided
_active_fig_dir: Path = None

DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
SCENARIO_COLORS = {
    "S0_Baseline": "#4C72B0",
    "S1_9am5pm":   "#DD8452",
    "S2_NoFriPM":  "#55A868",
}
sns.set_theme(style="whitegrid", font_scale=1.1)


# ============================================================
# Utility: Save figure
# ============================================================
def save_fig(fig, name: str, dpi: int = 150):
    d = _active_fig_dir if _active_fig_dir is not None else FIG_DIR
    path = d / name
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"  [viz] Saved {name}")


# ============================================================
# 1 & 2. Event Density Heatmaps
# ============================================================
def plot_event_heatmaps(events: pd.DataFrame):
    """
    Heatmap of number of events per (Day, Start_Hour) slot.
    One for ALL events, one for WholeClass only.
    """
    hours = list(range(9, 18))

    for label, mask in [("All Events", events.index == events.index),
                         ("WholeClass Lectures", events["WholeClass"] == True)]:
        subset = events[mask].dropna(subset=["Day","Start_Hour"])
        pivot = subset.groupby(["Day","Start_Hour"])["Event_ID"].count().unstack(fill_value=0)

        # Reindex to standard order
        pivot = pivot.reindex(index=DAY_ORDER, columns=hours, fill_value=0)

        fig, ax = plt.subplots(figsize=(12, 5))
        sns.heatmap(
            pivot, ax=ax, cmap="YlOrRd", annot=True, fmt="d",
            linewidths=0.5, cbar_kws={"label": "Number of Events"},
        )
        ax.set_title(f"Event Density by Day and Hour — {label} (Baseline)", fontsize=14)
        ax.set_xlabel("Start Hour")
        ax.set_ylabel("Day of Week")
        ax.set_xticklabels([f"{h}:00" for h in hours], rotation=45)

        fname = ("heatmap_events_by_day_hour.png" if "All" in label
                 else "heatmap_wholeclass_by_day_hour.png")
        save_fig(fig, fname)


# ============================================================
# 3. Displaced Events Bar Chart
# ============================================================
def plot_displaced_bar(events: pd.DataFrame):
    """
    Bar chart comparing total displaced events in S1 vs S2.
    """
    data = {
        "Scenario": ["S1: 9am-5pm", "S2: No Fri PM"],
        "All Events": [events["Displaced_S1"].sum(), events["Displaced_S2"].sum()],
        "WholeClass":  [(events["Displaced_S1"] & events["WholeClass"]).sum(),
                        (events["Displaced_S2"] & events["WholeClass"]).sum()],
        "Tutorials":   [(events["Displaced_S1"] & ~events["WholeClass"]).sum(),
                        (events["Displaced_S2"] & ~events["WholeClass"]).sum()],
    }
    df = pd.DataFrame(data).set_index("Scenario")

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(df))
    w = 0.25
    ax.bar(x - w, df["All Events"], w*2, label="Total", color="#4C72B0", alpha=0.85)
    ax.bar(x,      df["WholeClass"], w,   label="WholeClass", color="#DD8452", alpha=0.85)
    ax.bar(x + w,  df["Tutorials"], w,   label="SubGroup/Tutorial", color="#55A868", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(df.index, fontsize=12)
    ax.set_ylabel("Number of Displaced Events")
    ax.set_title("Displaced Events by Scenario", fontsize=14)
    ax.legend()

    # Annotate totals
    for xi, val in zip(x - w, df["All Events"]):
        ax.text(xi, val + 20, str(val), ha="center", fontsize=10, fontweight="bold")

    total = len(events)
    ax.set_ylim(0, max(df["All Events"]) * 1.15)
    ax.text(0.98, 0.97, f"Total events: {total:,}", transform=ax.transAxes,
            ha="right", va="top", fontsize=10, color="grey")

    save_fig(fig, "displaced_events_bar.png")


# ============================================================
# 4. Displaced Events by Day
# ============================================================
def plot_displaced_by_day(events: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    for ax, (col, title) in zip(axes, [("Displaced_S1", "S1: 9am–5pm"),
                                         ("Displaced_S2", "S2: No Fri PM")]):
        per_day = events[events[col]].groupby("Day")["Event_ID"].count().reindex(DAY_ORDER, fill_value=0)
        colors  = [SCENARIO_COLORS["S1_9am5pm"] if col=="Displaced_S1" else SCENARIO_COLORS["S2_NoFriPM"]] * 5
        bars = ax.bar(per_day.index, per_day.values, color=colors, alpha=0.85, edgecolor="white")
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Day of Week")
        ax.set_ylabel("Displaced Events")
        for bar, val in zip(bars, per_day.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    str(val), ha="center", va="bottom", fontsize=10)
        ax.set_xticklabels(per_day.index, rotation=30, ha="right")

    plt.suptitle("Displaced Events by Day of Week", fontsize=14, y=1.02)
    plt.tight_layout()
    save_fig(fig, "displaced_by_day.png")


# ============================================================
# 5. Displaced Events by Event Type
# ============================================================
def plot_displaced_by_eventtype(events: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (col, title) in zip(axes, [("Displaced_S1", "S1: 9am–5pm"),
                                         ("Displaced_S2", "S2: No Fri PM")]):
        by_type = (events[events[col]].groupby("Event_Type")["Event_ID"]
                   .count().sort_values(ascending=True).tail(12))
        by_type.plot(kind="barh", ax=ax, color=SCENARIO_COLORS["S1_9am5pm" if "S1" in col else "S2_NoFriPM"],
                     alpha=0.85, edgecolor="white")
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Number of Displaced Events")
        ax.set_ylabel("Event Type")

    plt.suptitle("Displaced Events by Event Type", fontsize=14, y=1.02)
    plt.tight_layout()
    save_fig(fig, "displaced_by_eventtype.png")


# ============================================================
# 6. Clash Comparison
# ============================================================
def plot_clash_comparison(clash_df: pd.DataFrame):
    if clash_df is None or len(clash_df) == 0:
        print("  [viz] No clash data to plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Total clash pairs
    ax = axes[0]
    colors = [SCENARIO_COLORS[s] for s in clash_df["Scenario"]]
    bars = ax.bar(clash_df["Scenario"], clash_df["total_clash_pairs"],
                  color=colors, alpha=0.85, edgecolor="white")
    ax.set_title("Total Clash Pairs per Scenario", fontsize=13)
    ax.set_ylabel("Number of Student Clash Pairs")
    ax.set_xticklabels(clash_df["Scenario"], rotation=20, ha="right")
    for bar, val in zip(bars, clash_df["total_clash_pairs"]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+10,
                f"{val:,}", ha="center", fontsize=10)

    # Students affected
    ax = axes[1]
    bars = ax.bar(clash_df["Scenario"], clash_df["student_clash_pct"],
                  color=colors, alpha=0.85, edgecolor="white")
    ax.set_title("% Students with Scheduling Clashes", fontsize=13)
    ax.set_ylabel("% Students Affected")
    ax.set_ylim(0, min(clash_df["student_clash_pct"].max() * 1.3, 100))
    ax.set_xticklabels(clash_df["Scenario"], rotation=20, ha="right")
    for bar, val in zip(bars, clash_df["student_clash_pct"]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                f"{val:.1f}%", ha="center", fontsize=10)

    plt.suptitle("Scheduling Clashes Under Each Scenario\n(Existing timetable, no rescheduling)",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    save_fig(fig, "clash_comparison.png")


# ============================================================
# 7. Lunch Break Comparison
# ============================================================
def plot_lunch_comparison(lunch_df: pd.DataFrame):
    if lunch_df is None or len(lunch_df) == 0:
        print("  [viz] No lunch data to plot")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [SCENARIO_COLORS.get(s, "#999999") for s in lunch_df["Scenario"]]
    bars = ax.bar(lunch_df["Scenario"], lunch_df["Lunch_Free_Pct"],
                  color=colors, alpha=0.85, edgecolor="white", width=0.5)

    ax.axhline(50, color="red", linestyle="--", alpha=0.7, label="50% threshold")
    ax.set_ylim(0, 105)
    ax.set_ylabel("% Students with Free Lunch (12–14)")
    ax.set_title("Lunch Break Feasibility by Scenario (Q4)", fontsize=14)
    ax.set_xticklabels(lunch_df["Scenario"], rotation=20, ha="right")
    ax.legend()

    for bar, val in zip(bars, lunch_df["Lunch_Free_Pct"]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                f"{val:.1f}%", ha="center", fontsize=11, fontweight="bold")

    save_fig(fig, "lunch_break_comparison.png")


# ============================================================
# 8. Lunch Free by Day
# ============================================================
def plot_lunch_by_day(lunch_df: pd.DataFrame):
    if lunch_df is None:
        return
    days = ["Monday","Tuesday","Wednesday","Thursday","Friday"]
    day_cols = [f"Free_{d}" for d in days]

    # Check all columns exist
    available = [c for c in day_cols if c in lunch_df.columns]
    if not available:
        return

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(days))
    w = 0.22
    for i, (_, row) in enumerate(lunch_df.iterrows()):
        vals = [row.get(c, 0) for c in day_cols]
        color = SCENARIO_COLORS.get(row["Scenario"], "#999999")
        offset = (i - 1) * w
        bars = ax.bar(x + offset, vals, w, label=row["Scenario"],
                      color=color, alpha=0.85, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(days)
    ax.set_ylabel("% Students Free for Lunch (12–14)")
    ax.set_title("Lunch Break Availability by Day and Scenario", fontsize=14)
    ax.set_ylim(0, 110)
    ax.legend()
    save_fig(fig, "lunch_by_day.png")


# ============================================================
# 9. Room Fill Rate Histogram
# ============================================================
def plot_room_utilisation_hist(room_util: pd.DataFrame):
    if room_util is None:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    baseline = room_util[room_util["Scenario"] == "Baseline"]["Avg_Fill_Rate"].dropna()
    ax.hist(baseline * 100, bins=25, color="#4C72B0", alpha=0.8, edgecolor="white")
    ax.axvline(baseline.mean() * 100, color="red", linestyle="--", linewidth=2,
               label=f"Mean = {baseline.mean()*100:.1f}%")
    ax.set_xlabel("Average Room Fill Rate (%)")
    ax.set_ylabel("Number of Rooms")
    ax.set_title("Distribution of Room Fill Rates (Baseline)", fontsize=14)
    ax.legend()
    save_fig(fig, "room_utilisation_histogram.png")


# ============================================================
# 10. Utilisation Comparison
# ============================================================
def plot_utilisation_comparison(util_df: pd.DataFrame):
    if util_df is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Events remaining
    ax = axes[0]
    colors = [SCENARIO_COLORS.get(s, "#999999") for s in util_df["Scenario"]]
    bars = ax.bar(util_df["Scenario"], util_df["In_Window_Events"], color=colors,
                  alpha=0.85, edgecolor="white")
    ax.set_title("In-Window Events per Scenario (Q5)", fontsize=13)
    ax.set_ylabel("Number of Events")
    for bar, val in zip(bars, util_df["In_Window_Events"]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+50,
                f"{val:,}", ha="center", fontsize=10)

    # Room utilisation %
    ax = axes[1]
    bars = ax.bar(util_df["Scenario"], util_df["Room_Utilisation_Pct"], color=colors,
                  alpha=0.85, edgecolor="white")
    ax.set_title("Room Utilisation Rate per Scenario (Q5)", fontsize=13)
    ax.set_ylabel("Room Utilisation (%)")
    for bar, val in zip(bars, util_df["Room_Utilisation_Pct"]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                f"{val:.1f}%", ha="center", fontsize=10)

    plt.suptitle("Timeslot & Room Utilisation Comparison", fontsize=14, y=1.02)
    plt.tight_layout()
    save_fig(fig, "utilisation_comparison.png")


# ============================================================
# 11. Heuristic Iteration Curves
# ============================================================
def plot_heuristic_iterations(heuristic_results: dict):
    if not heuristic_results:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, scenario in zip(axes, ["S1_9am5pm", "S2_NoFriPM"]):
        if scenario not in heuristic_results:
            continue
        log = heuristic_results[scenario].get("iteration_log", [])
        if not log:
            ax.set_title(f"{scenario}: No iteration log")
            continue
        df_log = pd.DataFrame(log)
        ax.plot(df_log["iteration"], df_log["clash_score"],
                color=SCENARIO_COLORS.get(scenario, "#555"), linewidth=2)
        ax.set_title(f"Local Search Convergence — {scenario}", fontsize=12)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Clash Score (weighted)")
        ax.fill_between(df_log["iteration"], df_log["clash_score"],
                         alpha=0.15, color=SCENARIO_COLORS.get(scenario, "#555"))

    plt.suptitle("Heuristic Local Search Convergence", fontsize=14, y=1.02)
    plt.tight_layout()
    save_fig(fig, "heuristic_iteration_curve.png")


# ============================================================
# 12. MIP vs Heuristic Comparison
# ============================================================
def plot_mip_vs_heuristic(mip_results: dict, heuristic_results: dict):
    scenarios = ["S1_9am5pm", "S2_NoFriPM"]
    methods   = ["Greedy", "Local Search", "MIP"]
    clash_data = {m: [] for m in methods}

    for sc in scenarios:
        hr = heuristic_results.get(sc, {})
        mr = mip_results.get(sc, {})

        clash_data["Greedy"].append(hr.get("summary", {}).get("Greedy_Clash_Score", 0))
        clash_data["Local Search"].append(hr.get("summary", {}).get("LocalSearch_Clash_Score", 0) or 0)
        clash_data["MIP"].append(mr.get("n_clashes", 0) or 0)

    x = np.arange(len(scenarios))
    w = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    for i, (method, vals) in enumerate(clash_data.items()):
        ax.bar(x + (i-1)*w, vals, w, label=method,
               color=colors[i], alpha=0.85, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_"," ") for s in scenarios])
    ax.set_ylabel("Weighted Clash Score")
    ax.set_title("Clash Reduction: MIP vs Heuristic Methods", fontsize=14)
    ax.legend()
    save_fig(fig, "mip_vs_heuristic_comparison.png")


# ============================================================
# 13. Timeslot Load Before/After
# ============================================================
def plot_timeslot_load_comparison(events: pd.DataFrame, heuristic_results: dict):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    hours = list(range(9, 18))

    for row_idx, (col, scenario_key, title_suffix) in enumerate([
        ("Displaced_S1", "S1_9am5pm", "S1: 9am–5pm"),
        ("Displaced_S2", "S2_NoFriPM", "S2: No Fri PM"),
    ]):
        # Before (baseline)
        ax_before = axes[row_idx][0]
        subset = events.dropna(subset=["Day","Start_Hour"])
        pivot = subset.groupby(["Day","Start_Hour"])["Event_ID"].count().unstack(fill_value=0)
        pivot = pivot.reindex(index=DAY_ORDER, columns=hours, fill_value=0)
        sns.heatmap(pivot, ax=ax_before, cmap="YlOrRd", annot=True, fmt="d",
                    linewidths=0.5, cbar_kws={"label": "Events"})
        ax_before.set_title(f"Baseline Schedule — {title_suffix}", fontsize=11)
        ax_before.set_xlabel("Start Hour")
        ax_before.set_ylabel("Day")
        ax_before.set_xticklabels([f"{h}" for h in hours], rotation=45)

        # After rescheduling (heuristic)
        ax_after  = axes[row_idx][1]
        hr = heuristic_results.get(scenario_key, {})
        adf = hr.get("assignment_df")

        if adf is not None and len(adf) > 0:
            fixed_events = events[~events[col]].dropna(subset=["Day","Start_Hour"])
            reschedule_rows = adf.dropna(subset=["New_Day","New_Start_Hour"])

            combined = pd.concat([
                fixed_events[["Day","Start_Hour","Event_ID"]],
                reschedule_rows[["New_Day","New_Start_Hour","Event_ID"]].rename(
                    columns={"New_Day":"Day","New_Start_Hour":"Start_Hour"})
            ])
            pivot2 = combined.groupby(["Day","Start_Hour"])["Event_ID"].count().unstack(fill_value=0)
            pivot2 = pivot2.reindex(index=DAY_ORDER, columns=hours, fill_value=0)
            sns.heatmap(pivot2, ax=ax_after, cmap="YlOrRd", annot=True, fmt="d",
                        linewidths=0.5, cbar_kws={"label": "Events"})
            ax_after.set_title(f"After Heuristic Rescheduling — {title_suffix}", fontsize=11)
        else:
            ax_after.text(0.5, 0.5, "No rescheduling data", ha="center", va="center",
                          transform=ax_after.transAxes, fontsize=12)
            ax_after.set_title(f"After Heuristic — {title_suffix}", fontsize=11)

        ax_after.set_xlabel("Start Hour")
        ax_after.set_ylabel("Day")
        ax_after.set_xticklabels([f"{h}" for h in hours], rotation=45)

    plt.suptitle("Timeslot Event Load: Before vs After Rescheduling", fontsize=14, y=1.01)
    plt.tight_layout()
    save_fig(fig, "timeslot_load_before_after.png")


# ============================================================
# 14. End-Hour Distribution
# ============================================================
def plot_end_hour_distribution(events: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5))

    for col, label, color in [("Displaced_S0", "All Events", "#4C72B0"),
                                ("Displaced_S1", "Displaced S1", "#DD8452"),
                                ("Displaced_S2", "Displaced S2", "#55A868")]:
        if col == "Displaced_S0":
            subset = events.dropna(subset=["End_Hour"])
        else:
            subset = events[events[col]].dropna(subset=["End_Hour"])

        ax.hist(subset["End_Hour"], bins=30, alpha=0.5, label=label, color=color, edgecolor="white")

    ax.axvline(17, color="red", linestyle="--", linewidth=2, label="17:00 (5pm)")
    ax.axvline(18, color="orange", linestyle="--", linewidth=2, label="18:00 (6pm)")
    ax.set_xlabel("Event End Hour")
    ax.set_ylabel("Number of Events")
    ax.set_title("Distribution of Event End Times", fontsize=14)
    ax.legend()
    save_fig(fig, "end_hour_distribution.png")


# ============================================================
# Master: Run all visualisations
# ============================================================
def run_all_visualisations(data: dict,
                            baseline_results: dict,
                            mip_results: dict = None,
                            heuristic_results: dict = None,
                            out_dir: Path = None):
    """
    Generate all charts from available data.

    Parameters
    ----------
    data               : from data_preprocessing.run_preprocessing()
    baseline_results   : from baseline_analysis.run_baseline_analysis()
    mip_results        : from mip_model.run_all_mip_scenarios() (optional)
    heuristic_results  : from heuristic_model.run_all_heuristic_scenarios() (optional)
    out_dir            : Path, optional — parent run directory; figures go to out_dir/figures
    """
    global _active_fig_dir
    fig_dir = (out_dir / "figures") if out_dir is not None else FIG_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)
    _active_fig_dir = fig_dir

    print(f"\n{'='*60}")
    print("VISUALISATION")
    print(f"{'='*60}")
    print(f"  Saving figures to {fig_dir}")

    events = data["events"]

    # ── Baseline heatmaps ────────────────────────────────────────────────────
    print("\n[viz] Generating event density heatmaps...")
    plot_event_heatmaps(events)

    # ── Displaced events ─────────────────────────────────────────────────────
    print("[viz] Generating displaced event charts...")
    plot_displaced_bar(events)
    plot_displaced_by_day(events)
    plot_displaced_by_eventtype(events)

    # ── Clashes ──────────────────────────────────────────────────────────────
    clash_df = baseline_results.get("clashes")
    if clash_df is not None:
        print("[viz] Generating clash comparison chart...")
        plot_clash_comparison(clash_df)

    # ── Lunch breaks ─────────────────────────────────────────────────────────
    lunch_df = baseline_results.get("lunch_break")
    if lunch_df is not None:
        print("[viz] Generating lunch break charts...")
        plot_lunch_comparison(lunch_df)
        plot_lunch_by_day(lunch_df)

    # ── Room utilisation ─────────────────────────────────────────────────────
    room_util = baseline_results.get("room_util")
    if room_util is not None:
        print("[viz] Generating room utilisation charts...")
        plot_room_utilisation_hist(room_util)

    util_comp = baseline_results.get("utilisation_comparison")
    if util_comp is not None:
        plot_utilisation_comparison(util_comp)

    # ── End hour distribution ────────────────────────────────────────────────
    print("[viz] Generating end-hour distribution...")
    plot_end_hour_distribution(events)

    # ── Heuristic results ────────────────────────────────────────────────────
    if heuristic_results:
        print("[viz] Generating heuristic charts...")
        plot_heuristic_iterations(heuristic_results)
        plot_timeslot_load_comparison(events, heuristic_results)

        # MIP vs Heuristic
        if mip_results:
            print("[viz] Generating MIP vs Heuristic comparison...")
            plot_mip_vs_heuristic(mip_results, heuristic_results)

    print(f"\n[viz] All figures saved to {fig_dir}")
    _active_fig_dir = None   # reset after run
    return fig_dir


if __name__ == "__main__":
    from data_preprocessing import run_preprocessing
    from baseline_analysis import run_baseline_analysis

    data    = run_preprocessing(build_conflicts=False)
    results = run_baseline_analysis(data)
    run_all_visualisations(data, results)
