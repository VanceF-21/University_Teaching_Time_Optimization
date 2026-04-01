"""
sensitivity.py

Parameter Sensitivity Analysis for the Two-Phase MIP Model.

Strategy: One-At-a-Time (OAT) sweep — vary one parameter at a time while
holding the other two at their defaults, for both scheduling scenarios.

Parameters analysed
────────────────────
  max_events  : per-phase cap on displaced events (Top-N by Event_Size)
                Default 500.  Sweep: [100, 200, 300, 400, 500, 700]
  time_limit  : Xpress wall-clock limit per phase (seconds)
                Default 300.  Sweep: [30, 60, 120, 180, 300]
  mip_gap     : MIP relative optimality gap tolerance
                Default 0.02. Sweep: [0.005, 0.01, 0.02, 0.05, 0.10]

Metrics collected per run
──────────────────────────
  Objective       : total weighted clash score (P1 + P2)
  Solve_Time_s    : wall-clock solve time (P1 + P2)
  N_Clashes       : clash pairs remaining after optimisation
  N_Rescheduled   : events successfully assigned
  Solve_Status    : OPTIMAL / FEASIBLE / INFEASIBLE

Outputs  (saved to out_dir/sensitivity/)
─────────────────────────────────────────
  sensitivity_max_events.csv
  sensitivity_time_limit.csv
  sensitivity_mip_gap.csv
  sensitivity_runtime_comparison.csv
  figures/sens_max_events.png
  figures/sens_time_limit.png
  figures/sens_mip_gap.png
  figures/sens_pareto.png
  figures/sens_runtime_comparison.png
  figures/sens_heatmap.png
"""

import time
import warnings
import traceback
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore")

# Default parameter values
DEFAULT_MAX_EVENTS = 500
DEFAULT_TIME_LIMIT = 300
DEFAULT_MIP_GAP    = 0.02

# Sweep ranges
SWEEP_MAX_EVENTS  = [200, 300, 400, 500, 600, 700, 800, 900, 1000]
SWEEP_TIME_LIMITS = [30, 60, 120, 180, 300, 400, 500]
SWEEP_MIP_GAPS    = [0.005, 0.01, 0.02, 0.05, 0.10]

SCENARIOS = ["S1_9am5pm", "S2_NoFriPM"]

SCENARIO_LABELS = {
    "S1_9am5pm":  "S1 (9am–5pm)",
    "S2_NoFriPM": "S2 (No Fri PM)",
}

COLORS = {
    "S1_9am5pm":  "#2196F3",   # blue
    "S2_NoFriPM": "#FF5722",   # orange
}


def _run_mip_once(
    scenario: str,
    events: pd.DataFrame,
    conflict_pairs: pd.DataFrame,
    student_events: pd.DataFrame,
    max_events: int,
    time_limit: int,
    mip_gap: float,
    tmp_dir: Path,
) -> list:
    """
    Run one two-phase MIP solve (per-semester) and return a list of flat metric
    dicts — one row per semester plus one "Combined" summary row.

    Each row contains:
        Scenario, Semester, max_events, time_limit, mip_gap,
        Total_Clash_Score, DD_Clash_Score, DF_Clash_Score,
        Solve_Time_s, N_Clashes, N_Rescheduled, Solve_Status
    """
    param_base = {
        "Scenario":         scenario,
        "max_events":       max_events,
        "time_limit":       time_limit,
        "mip_gap":          mip_gap,
    }
    err_row = dict(param_base, Semester="Combined",
                   Total_Clash_Score=np.nan, DD_Clash_Score=np.nan,
                   DF_Clash_Score=np.nan, Solve_Time_s=np.nan,
                   N_Clashes=np.nan, N_Rescheduled=0, Solve_Status="ERROR")
    try:
        from mip_new import run_mip_scenario_new
        t0  = time.time()
        res = run_mip_scenario_new(
            scenario       = scenario,
            events         = events,
            conflict_pairs = conflict_pairs,
            student_events = student_events,
            max_events     = max_events,
            time_limit     = time_limit,
            mip_gap        = mip_gap,
            verbose        = False,
            out_dir        = tmp_dir,
        )
        elapsed = round(time.time() - t0, 1)

        rows = []
        for sem, sr in res.get("sem_results", {}).items():
            rows.append(dict(
                param_base,
                Semester         = sem,
                Total_Clash_Score= sr.get("total_clash_score", np.nan),
                DD_Clash_Score   = sr.get("dd_clash_score",    np.nan),
                DF_Clash_Score   = sr.get("df_clash_score",    np.nan),
                Solve_Time_s     = elapsed,   # total elapsed for both semesters
                N_Clashes        = sr.get("n_clashes",         np.nan),
                N_Rescheduled    = len(sr["assignment_df"]) if sr.get("assignment_df") is not None else 0,
                Solve_Status     = sr.get("status", "UNKNOWN"),
            ))
        # Combined row
        rows.append(dict(
            param_base,
            Semester         = "Combined",
            Total_Clash_Score= res.get("total_clash_score", np.nan),
            DD_Clash_Score   = res.get("dd_clash_score",    np.nan),
            DF_Clash_Score   = res.get("df_clash_score",    np.nan),
            Solve_Time_s     = elapsed,
            N_Clashes        = res.get("n_clashes",         np.nan),
            N_Rescheduled    = len(res["assignment_df"]) if res.get("assignment_df") is not None else 0,
            Solve_Status     = res.get("status", "UNKNOWN"),
        ))
        return rows

    except ImportError:
        print("  [sensitivity] xpress / mip_new not available — skipping MIP run.")
        err_row["Solve_Status"] = "SKIPPED"
    except Exception as e:
        print(f"  [sensitivity] MIP run failed ({scenario}, max_events={max_events}, "
              f"time_limit={time_limit}, mip_gap={mip_gap}): {e}")
        traceback.print_exc()
    return [err_row]


def _run_heuristic_once(
    scenario: str,
    events: pd.DataFrame,
    conflict_pairs: pd.DataFrame,
    student_events: pd.DataFrame,
    tmp_dir: Path,
) -> list:
    """
    Run one greedy + local-search heuristic solve (per-semester).
    Returns a list of flat metric dicts — one per semester plus "Combined".
    使用 DEFAULT_MAX_EVENTS 与 MIP 保持相同问题规模，确保 runtime 对比公平。
    """
    err_row = {
        "Scenario": scenario, "Semester": "Combined", "Method": "Heuristic (Greedy+LS)",
        "Total_Clash_Score": np.nan, "DD_Clash_Score": np.nan, "DF_Clash_Score": np.nan,
        "Solve_Time_s": np.nan, "N_Rescheduled": 0, "Solve_Status": "ERROR",
    }
    try:
        from heuristic_new import run_heuristic_scenario_new
        t0  = time.time()
        res = run_heuristic_scenario_new(
            scenario         = scenario,
            events           = events,
            conflict_pairs   = conflict_pairs,
            student_events   = student_events,
            max_events       = DEFAULT_MAX_EVENTS,
            run_local_search = True,
            ls_max_iter      = 300,
            out_dir          = tmp_dir,
        )
        elapsed = round(time.time() - t0, 1)

        rows = []
        for sem, sr in res.get("sem_results", {}).items():
            ls  = sr.get("ls_full_score",     sr.get("greedy_full_score", np.nan))
            dd  = sr.get("ls_dd_score",       sr.get("greedy_dd_score",   np.nan))
            gls = sr.get("greedy_full_score", np.nan)
            df_score = (ls - dd) if (not np.isnan(ls) and not np.isnan(dd)) else np.nan
            rows.append({
                "Scenario":          scenario,
                "Semester":          sem,
                "Method":            "Heuristic (Greedy+LS)",
                "Total_Clash_Score": ls,
                "DD_Clash_Score":    dd,
                "DF_Clash_Score":    df_score,
                "Greedy_Score":      gls,
                "Solve_Time_s":      elapsed,
                "N_Rescheduled":     len(sr["assignment_df"]) if sr.get("assignment_df") is not None else 0,
                "Solve_Status":      "FEASIBLE",
            })
        summary = res.get("summary", {})
        ls_full = summary.get("LocalSearch_Full_Clash_Score",
                               summary.get("Greedy_Full_Clash_Score", np.nan))
        ls_dd   = summary.get("LocalSearch_DD_Clash_Score",
                               summary.get("Greedy_DD_Clash_Score",   np.nan))
        rows.append({
            "Scenario":          scenario,
            "Semester":          "Combined",
            "Method":            "Heuristic (Greedy+LS)",
            "Total_Clash_Score": ls_full,
            "DD_Clash_Score":    ls_dd,
            "DF_Clash_Score":    (ls_full - ls_dd) if (not np.isnan(ls_full) and not np.isnan(ls_dd)) else np.nan,
            "Greedy_Score":      summary.get("Greedy_Full_Clash_Score", np.nan),
            "Solve_Time_s":      elapsed,
            "N_Rescheduled":     len(res["assignment_df"]) if res.get("assignment_df") is not None else 0,
            "Solve_Status":      "FEASIBLE",
        })
        return rows
    except Exception as e:
        print(f"  [sensitivity] Heuristic run failed ({scenario}): {e}")
        traceback.print_exc()
    return [err_row]



# OAT Sweep functions
def _sweep_max_events(
    events: pd.DataFrame,
    conflict_pairs: pd.DataFrame,
    student_events: pd.DataFrame,
    out_dir: Path,
) -> pd.DataFrame:
    """Sweep max_events; hold time_limit and mip_gap at defaults."""
    tmp_dir = out_dir / "_tmp_sens"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print("\n[Sensitivity] Sweeping max_events …")
    rows = []
    for sc in SCENARIOS:
        for me in SWEEP_MAX_EVENTS:
            print(f"  scenario={sc}  max_events={me}")
            rows.extend(_run_mip_once(
                scenario       = sc,
                events         = events,
                conflict_pairs = conflict_pairs,
                student_events = student_events,
                max_events     = me,
                time_limit     = DEFAULT_TIME_LIMIT,
                mip_gap        = DEFAULT_MIP_GAP,
                tmp_dir        = tmp_dir,
            ))

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "sensitivity_max_events.csv", index=False)
    print(f"  Saved → {out_dir / 'sensitivity_max_events.csv'}")
    return df


def _sweep_time_limit(
    events: pd.DataFrame,
    conflict_pairs: pd.DataFrame,
    student_events: pd.DataFrame,
    out_dir: Path,
) -> pd.DataFrame:
    """Sweep time_limit; hold max_events and mip_gap at defaults."""
    tmp_dir = out_dir / "_tmp_sens"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print("\n[Sensitivity] Sweeping time_limit …")
    rows = []
    for sc in SCENARIOS:
        for tl in SWEEP_TIME_LIMITS:
            print(f"  scenario={sc}  time_limit={tl}s")
            rows.extend(_run_mip_once(
                scenario       = sc,
                events         = events,
                conflict_pairs = conflict_pairs,
                student_events = student_events,
                max_events     = DEFAULT_MAX_EVENTS,
                time_limit     = tl,
                mip_gap        = DEFAULT_MIP_GAP,
                tmp_dir        = tmp_dir,
            ))

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "sensitivity_time_limit.csv", index=False)
    print(f"  Saved → {out_dir / 'sensitivity_time_limit.csv'}")
    return df


def _sweep_mip_gap(
    events: pd.DataFrame,
    conflict_pairs: pd.DataFrame,
    student_events: pd.DataFrame,
    out_dir: Path,
) -> pd.DataFrame:
    """Sweep mip_gap; hold max_events and time_limit at defaults."""
    tmp_dir = out_dir / "_tmp_sens"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print("\n[Sensitivity] Sweeping mip_gap …")
    rows = []
    for sc in SCENARIOS:
        for gap in SWEEP_MIP_GAPS:
            print(f"  scenario={sc}  mip_gap={gap:.3f}")
            rows.extend(_run_mip_once(
                scenario       = sc,
                events         = events,
                conflict_pairs = conflict_pairs,
                student_events = student_events,
                max_events     = DEFAULT_MAX_EVENTS,
                time_limit     = DEFAULT_TIME_LIMIT,
                mip_gap        = gap,
                tmp_dir        = tmp_dir,
            ))

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "sensitivity_mip_gap.csv", index=False)
    print(f"  Saved → {out_dir / 'sensitivity_mip_gap.csv'}")
    return df


def run_parameter_sweeps(data: dict, out_dir: Path) -> dict:
    """
    Run all three OAT parameter sweeps.

    Returns
    -------
    dict with keys 'max_events', 'time_limit', 'mip_gap',
    each containing the corresponding sweep DataFrame.
    """
    events         = data["events"]
    conflict_pairs = data.get("conflict_pairs")
    student_events = data["student_events"]

    out_dir.mkdir(parents=True, exist_ok=True)

    df_me  = _sweep_max_events(events, conflict_pairs, student_events, out_dir)
    df_tl  = _sweep_time_limit(events, conflict_pairs, student_events, out_dir)
    df_gap = _sweep_mip_gap(events, conflict_pairs, student_events, out_dir)

    return {
        "max_events": df_me,
        "time_limit": df_tl,
        "mip_gap":    df_gap,
    }



# Runtime comparison: MIP vs Heuristic
def run_runtime_comparison(data: dict, out_dir: Path) -> pd.DataFrame:
    """
    Compare MIP (default params) vs Heuristic (Greedy+LS) on runtime and
    solution quality.

    Returns a DataFrame with one row per (scenario × method).
    """
    events         = data["events"]
    conflict_pairs = data.get("conflict_pairs")
    student_events = data["student_events"]
    tmp_dir        = out_dir / "_tmp_sens"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print("\n[Sensitivity] Runtime comparison: MIP vs Heuristic …")
    rows = []
    for sc in SCENARIOS:
        # MIP with default parameters
        print(f"  MIP  scenario={sc}")
        mip_rows = _run_mip_once(
            scenario       = sc,
            events         = events,
            conflict_pairs = conflict_pairs,
            student_events = student_events,
            max_events     = DEFAULT_MAX_EVENTS,
            time_limit     = DEFAULT_TIME_LIMIT,
            mip_gap        = DEFAULT_MIP_GAP,
            tmp_dir        = tmp_dir,
        )
        for r in mip_rows:
            r["Method"] = f"MIP (Xpress)"
        rows.extend(mip_rows)

        # Heuristic
        print(f"  Heur scenario={sc}")
        heur_rows = _run_heuristic_once(
            scenario       = sc,
            events         = events,
            conflict_pairs = conflict_pairs,
            student_events = student_events,
            tmp_dir        = tmp_dir,
        )
        for r in heur_rows:
            r["max_events"] = DEFAULT_MAX_EVENTS
            r["time_limit"] = DEFAULT_TIME_LIMIT
            r["mip_gap"]    = DEFAULT_MIP_GAP
        rows.extend(heur_rows)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "sensitivity_runtime_comparison.csv", index=False)
    print(f"  Saved → {out_dir / 'sensitivity_runtime_comparison.csv'}")
    return df



# Plotting helpers
def _savefig(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ── Subplot layout constants ───────────────────────────────────────────────────
SEMESTERS     = ["Semester 1", "Semester 2"]
SEM_COLORS    = {"Semester 1": "#1976D2", "Semester 2": "#E64A19"}   # blue / orange
SEM_LINESTYLE = {"Semester 1": "-",       "Semester 2": "--"}
METHOD_COLORS = {"MIP (Xpress)": "#1565C0", "Heuristic (Greedy+LS)": "#E65100"}
METHOD_HATCH  = {"MIP (Xpress)": "",        "Heuristic (Greedy+LS)": "///"}


def _make_2x2_axes(title: str):
    """Return (fig, axes_2x2) for a 2-scenario × 2-semester grid."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey="row")
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    for row_i, sc in enumerate(SCENARIOS):
        for col_j, sem in enumerate(SEMESTERS):
            axes[row_i, col_j].set_title(
                f"{SCENARIO_LABELS[sc]} — {sem}", fontsize=11, fontweight="bold"
            )
    return fig, axes


def _plot_max_events_sweep(df: pd.DataFrame, fig_dir: Path) -> None:
    """2×2 grid: rows=scenario, cols=semester. Each panel: Total_Clash_Score vs max_events."""
    data = df[df["Semester"] != "Combined"] if "Semester" in df.columns else df
    if data.empty:
        return

    fig, axes = _make_2x2_axes(
        f"Sensitivity — max_events  (time_limit={DEFAULT_TIME_LIMIT}s, gap={DEFAULT_MIP_GAP*100:.0f}%)"
    )

    for row_i, sc in enumerate(SCENARIOS):
        for col_j, sem in enumerate(SEMESTERS):
            ax  = axes[row_i, col_j]
            sub = data[(data["Scenario"] == sc) & (data["Semester"] == sem)].dropna(
                subset=["Total_Clash_Score"]
            ).sort_values("max_events")
            if not sub.empty:
                ax.plot(sub["max_events"], sub["Total_Clash_Score"],
                        marker="o", linewidth=2, color=SEM_COLORS[sem], label="Clash Score")
                ax2 = ax.twinx()
                sub2 = data[(data["Scenario"] == sc) & (data["Semester"] == sem)].dropna(
                    subset=["Solve_Time_s"]
                ).sort_values("max_events")
                ax2.plot(sub2["max_events"], sub2["Solve_Time_s"],
                         marker="s", linewidth=1.5, linestyle="--",
                         color="grey", alpha=0.7, label="Solve Time")
                ax2.set_ylabel("Solve Time (s)", fontsize=9, color="grey")
                ax2.tick_params(axis="y", labelcolor="grey")
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")
            ax.axvline(DEFAULT_MAX_EVENTS, color="grey", linestyle=":", linewidth=1,
                       label=f"Default ({DEFAULT_MAX_EVENTS})")
            ax.set_xlabel("max_events", fontsize=10)
            ax.set_ylabel("Total Clash Score (↓ better)", fontsize=10)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _savefig(fig, fig_dir / "sens_max_events.png")


def _plot_time_limit_sweep(df: pd.DataFrame, fig_dir: Path) -> None:
    """2×2 grid: rows=scenario, cols=semester. Each panel: Total_Clash_Score vs time_limit."""
    data = df[df["Semester"] != "Combined"] if "Semester" in df.columns else df
    if data.empty:
        return

    fig, axes = _make_2x2_axes(
        f"Sensitivity — time_limit  (max_events={DEFAULT_MAX_EVENTS}, gap={DEFAULT_MIP_GAP*100:.0f}%)"
    )

    for row_i, sc in enumerate(SCENARIOS):
        for col_j, sem in enumerate(SEMESTERS):
            ax  = axes[row_i, col_j]
            sub = data[(data["Scenario"] == sc) & (data["Semester"] == sem)].dropna(
                subset=["Total_Clash_Score"]
            ).sort_values("time_limit")
            if not sub.empty:
                ax.plot(sub["time_limit"], sub["Total_Clash_Score"],
                        marker="s", linewidth=2, color=SEM_COLORS[sem], label="Clash Score")
                ax2 = ax.twinx()
                sub2 = data[(data["Scenario"] == sc) & (data["Semester"] == sem)].dropna(
                    subset=["Solve_Time_s"]
                ).sort_values("time_limit")
                ax2.plot(sub2["time_limit"], sub2["Solve_Time_s"],
                         marker="o", linewidth=1.5, linestyle="--",
                         color="grey", alpha=0.7, label="Solve Time")
                ax2.set_ylabel("Solve Time (s)", fontsize=9, color="grey")
                ax2.tick_params(axis="y", labelcolor="grey")
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")
            ax.axvline(DEFAULT_TIME_LIMIT, color="grey", linestyle=":", linewidth=1)
            ax.set_xlabel("time_limit (s per phase)", fontsize=10)
            ax.set_ylabel("Total Clash Score (↓ better)", fontsize=10)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _savefig(fig, fig_dir / "sens_time_limit.png")


def _plot_mip_gap_sweep(df: pd.DataFrame, fig_dir: Path) -> None:
    """2×2 grid: rows=scenario, cols=semester. Each panel: Total_Clash_Score vs mip_gap."""
    data = df[df["Semester"] != "Combined"] if "Semester" in df.columns else df
    if data.empty:
        return

    fig, axes = _make_2x2_axes(
        f"Sensitivity — mip_gap  (max_events={DEFAULT_MAX_EVENTS}, time_limit={DEFAULT_TIME_LIMIT}s)"
    )

    for row_i, sc in enumerate(SCENARIOS):
        for col_j, sem in enumerate(SEMESTERS):
            ax  = axes[row_i, col_j]
            sub = data[(data["Scenario"] == sc) & (data["Semester"] == sem)].dropna(
                subset=["Total_Clash_Score"]
            ).sort_values("mip_gap")
            if not sub.empty:
                ax.semilogx(sub["mip_gap"] * 100, sub["Total_Clash_Score"],
                            marker="^", linewidth=2, color=SEM_COLORS[sem], label="Clash Score")
                ax2 = ax.twinx()
                sub2 = data[(data["Scenario"] == sc) & (data["Semester"] == sem)].dropna(
                    subset=["Solve_Time_s"]
                ).sort_values("mip_gap")
                ax2.semilogx(sub2["mip_gap"] * 100, sub2["Solve_Time_s"],
                             marker="o", linewidth=1.5, linestyle="--",
                             color="grey", alpha=0.7, label="Solve Time")
                ax2.set_ylabel("Solve Time (s)", fontsize=9, color="grey")
                ax2.tick_params(axis="y", labelcolor="grey")
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
            ax.axvline(DEFAULT_MIP_GAP * 100, color="grey", linestyle=":", linewidth=1)
            ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
            ax.set_xlabel("mip_gap (%)", fontsize=10)
            ax.set_ylabel("Total Clash Score (↓ better)", fontsize=10)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _savefig(fig, fig_dir / "sens_mip_gap.png")


def _plot_pareto(sweep_results: dict, fig_dir: Path) -> None:
    """
    Quality vs Runtime Pareto scatter — 2×2 grid (scenario × semester).
    Each point = one parameter-value run; marker encodes which parameter was swept.
    """
    frames = []
    for param_name, df in sweep_results.items():
        cols = ["Scenario", "Semester", "Total_Clash_Score", "Solve_Time_s", "Solve_Status"]
        tmp = df[[c for c in cols if c in df.columns]].copy()
        tmp["Param"] = param_name
        frames.append(tmp)
    if not frames:
        return

    all_data = pd.concat(frames, ignore_index=True)
    all_data = all_data[all_data.get("Semester", pd.Series(["Combined"])) != "Combined"]
    all_data = all_data.dropna(subset=["Total_Clash_Score", "Solve_Time_s"])
    if all_data.empty:
        return

    fig, axes = _make_2x2_axes("Quality vs Runtime — Pareto View (all sweeps)")
    markers   = {"max_events": "o", "time_limit": "s", "mip_gap": "^"}
    param_colors = {"max_events": "#0288D1", "time_limit": "#388E3C", "mip_gap": "#F57C00"}

    for row_i, sc in enumerate(SCENARIOS):
        for col_j, sem in enumerate(SEMESTERS):
            ax  = axes[row_i, col_j]
            sub = all_data[(all_data["Scenario"] == sc) & (all_data["Semester"] == sem)]
            for param, grp in sub.groupby("Param"):
                ax.scatter(grp["Solve_Time_s"], grp["Total_Clash_Score"],
                           color=param_colors.get(param, "grey"),
                           marker=markers.get(param, "o"),
                           s=60, alpha=0.8, label=param)
            ax.set_xlabel("Solve Time (s)", fontsize=10)
            ax.set_ylabel("Total Clash Score", fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _savefig(fig, fig_dir / "sens_pareto.png")


def _plot_runtime_comparison(df_cmp: pd.DataFrame, fig_dir: Path) -> None:
    """
    2×2 grid (scenario × semester): grouped bars MIP vs Heuristic for
    Solve_Time_s and Total_Clash_Score side by side within each panel.
    """
    data = df_cmp[df_cmp.get("Semester", pd.Series(["x"])) != "Combined"] \
           if "Semester" in df_cmp.columns else df_cmp
    if data.empty:
        return

    fig, axes = _make_2x2_axes("MIP vs Heuristic — Runtime & Solution Quality")
    methods = [m for m in ["MIP (Xpress)", "Heuristic (Greedy+LS)"]
               if m in data["Method"].unique()]
    metrics = [("Solve_Time_s", "Solve Time (s)"), ("Total_Clash_Score", "Total Clash Score")]
    width   = 0.3
    x       = np.arange(len(metrics))

    for row_i, sc in enumerate(SCENARIOS):
        for col_j, sem in enumerate(SEMESTERS):
            ax  = axes[row_i, col_j]
            n   = len(methods)
            for i, method in enumerate(methods):
                sub = data[(data["Scenario"] == sc) &
                           (data["Semester"] == sem) &
                           (data["Method"] == method)]
                vals = [sub[col].values[0] if len(sub) > 0 and col in sub.columns
                        else np.nan for col, _ in metrics]
                offset = (i - (n - 1) / 2) * width
                bars = ax.bar(x + offset, vals, width,
                              label=method,
                              color=METHOD_COLORS.get(method, "grey"),
                              hatch=METHOD_HATCH.get(method, ""),
                              alpha=0.85)
                for bar, v in zip(bars, vals):
                    if not np.isnan(v):
                        ax.text(bar.get_x() + bar.get_width() / 2,
                                bar.get_height() * 1.01,
                                f"{v:,.0f}", ha="center", va="bottom", fontsize=7)
            ax.set_xticks(x)
            ax.set_xticklabels([lbl for _, lbl in metrics], fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    _savefig(fig, fig_dir / "sens_runtime_comparison.png")


def plot_runtime_comparison_standalone(df_cmp: pd.DataFrame, fig_dir: Path) -> None:
    """
    Standalone runtime-only comparison figure.
    Left panel  : Solve_Time_s grouped by (Scenario × Semester), bars = Method.
    Right panel : Total_Clash_Score grouped by (Scenario × Semester), bars = Method.
    Each panel is a single grouped-bar chart with one bar-group per
    (Scenario, Semester) combination.
    """
    data = df_cmp[df_cmp["Semester"] != "Combined"].copy() \
           if "Semester" in df_cmp.columns else df_cmp.copy()
    if data.empty:
        print("  [runtime standalone] No data — skipping.")
        return

    methods  = [m for m in ["MIP (Xpress)", "Heuristic (Greedy+LS)"]
                if m in data["Method"].unique()]
    combos   = [(sc, sem) for sc in SCENARIOS for sem in SEMESTERS]
    x_labels = [f"{SCENARIO_LABELS[sc]}\n{sem}" for sc, sem in combos]
    x        = np.arange(len(combos))
    width    = 0.35
    n        = len(methods)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("MIP vs Heuristic: Runtime Comparison by Scenario & Semester",
                 fontsize=13, fontweight="bold")

    for ax, (col, ylabel) in zip(axes, [("Solve_Time_s", "Solve Time (seconds)"),
                                         ("Total_Clash_Score", "Total Clash Score (↓ better)")]):
        for i, method in enumerate(methods):
            vals = []
            for sc, sem in combos:
                sub = data[(data["Scenario"] == sc) &
                           (data["Semester"]  == sem) &
                           (data["Method"]    == method)]
                vals.append(sub[col].values[0] if len(sub) > 0 and col in sub.columns
                            else np.nan)
            offset = (i - (n - 1) / 2) * width
            bars   = ax.bar(x + offset, vals, width,
                            label=method,
                            color=METHOD_COLORS.get(method, "grey"),
                            hatch=METHOD_HATCH.get(method, ""),
                            alpha=0.85, edgecolor="white")
            for bar, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() * 1.01,
                            f"{v:,.0f}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    _savefig(fig, fig_dir / "runtime_comparison_standalone.png")
    print(f"  [runtime standalone] Saved → {fig_dir / 'runtime_comparison_standalone.png'}")


def _plot_heatmap(
    events: pd.DataFrame,
    conflict_pairs: pd.DataFrame,
    student_events: pd.DataFrame,
    out_dir: Path,
    fig_dir: Path,
    scenario: str = "S1_9am5pm",
) -> None:
    """
    Approximate 2-D sensitivity: max_events × time_limit → Objective.
    Runs a small sub-grid to stay tractable.
    """
    me_vals  = [200, 400, 700]
    tl_vals  = [60, 180, 300]
    tmp_dir  = out_dir / "_tmp_sens"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[Sensitivity] 2-D heatmap ({scenario}) …")
    matrix = np.full((len(me_vals), len(tl_vals)), np.nan)

    for i, me in enumerate(me_vals):
        for j, tl in enumerate(tl_vals):
            print(f"  max_events={me}  time_limit={tl}s")
            row = _run_mip_once(
                scenario       = scenario,
                events         = events,
                conflict_pairs = conflict_pairs,
                student_events = student_events,
                max_events     = me,
                time_limit     = tl,
                mip_gap        = DEFAULT_MIP_GAP,
                tmp_dir        = tmp_dir,
            )
            matrix[i, j] = row["Total_Clash_Score"]

    # Save raw data
    hm_df = pd.DataFrame(
        matrix,
        index   = [f"me={m}" for m in me_vals],
        columns = [f"tl={t}s" for t in tl_vals],
    )
    hm_df.to_csv(out_dir / f"sensitivity_heatmap_{scenario}.csv")

    if np.all(np.isnan(matrix)):
        print("  [heatmap] All values NaN — skipping plot.")
        return

    # Plot
    vmin = np.nanmin(matrix)
    vmax = np.nanmax(matrix)
    if vmin == vmax:
        vmax = vmin + 1   # avoid zero-range colormap

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd",
                   vmin=vmin, vmax=vmax, origin="upper")
    plt.colorbar(im, ax=ax, label="Weighted Clash Score")

    ax.set_xticks(range(len(tl_vals)))
    ax.set_xticklabels([f"{t}s" for t in tl_vals], fontsize=10)
    ax.set_yticks(range(len(me_vals)))
    ax.set_yticklabels([str(m) for m in me_vals], fontsize=10)
    ax.set_xlabel("time_limit (seconds per phase)", fontsize=11)
    ax.set_ylabel("max_events (events per phase)",  fontsize=11)
    ax.set_title(
        f"2-D Sensitivity: max_events × time_limit\n"
        f"Scenario: {SCENARIO_LABELS.get(scenario, scenario)}  |  gap={DEFAULT_MIP_GAP*100:.0f}%",
        fontsize=12, fontweight="bold"
    )

    # Annotate cells
    for i in range(len(me_vals)):
        for j in range(len(tl_vals)):
            val = matrix[i, j]
            text = f"{val:.0f}" if not np.isnan(val) else "N/A"
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=10, color="black")

    plt.tight_layout()
    _savefig(fig, fig_dir / f"sens_heatmap_{scenario}.png")



# Main entry point
def run_sensitivity_analysis(
    data: dict,
    out_dir: Path = None,
    run_heatmap: bool = True,
    heatmap_scenario: str = "S1_9am5pm",
) -> dict:
    """
    Full sensitivity analysis pipeline.

    Steps
    -----
    1. OAT sweeps for max_events, time_limit, mip_gap.
    2. Runtime comparison: MIP (default) vs Heuristic.
    3. Six plots: per-parameter sweep (×3), Pareto, runtime comparison, heatmap.

    Parameters
    ----------
    data             : dict from data_preprocessing.run_preprocessing()
    out_dir          : directory for CSV and figure outputs
                       (default: outputs/sensitivity/)
    run_heatmap      : whether to run the 2-D heatmap sub-grid (can be slow)
    heatmap_scenario : which scenario to use for the 2-D heatmap

    Returns
    -------
    dict with keys:
        sweep_results   – dict of DataFrames per parameter
        comparison_df   – runtime comparison DataFrame
    """
    base_dir = out_dir or (Path(__file__).resolve().parent / "outputs" / "sensitivity")
    base_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = base_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 65)
    print("  SENSITIVITY ANALYSIS (MIP Parameter Study)")
    print("=" * 65)
    print(f"  Output directory: {base_dir}")

    # ── Step 1: OAT sweeps ───────────────────────────────────────────────────
    sweep_results = run_parameter_sweeps(data, out_dir=base_dir)

    # ── Step 2: Runtime comparison ───────────────────────────────────────────
    comparison_df = run_runtime_comparison(data, out_dir=base_dir)

    # ── Step 3: Plots ────────────────────────────────────────────────────────
    print("\n[Sensitivity] Generating plots …")

    _plot_max_events_sweep(sweep_results["max_events"], fig_dir)
    _plot_time_limit_sweep(sweep_results["time_limit"], fig_dir)
    _plot_mip_gap_sweep(sweep_results["mip_gap"], fig_dir)
    _plot_pareto(sweep_results, fig_dir)
    _plot_runtime_comparison(comparison_df, fig_dir)
    plot_runtime_comparison_standalone(comparison_df, fig_dir)

    if run_heatmap:
        _plot_heatmap(
            events         = data["events"],
            conflict_pairs = data.get("conflict_pairs"),
            student_events = data["student_events"],
            out_dir        = base_dir,
            fig_dir        = fig_dir,
            scenario       = heatmap_scenario,
        )

    print("\n[Sensitivity] Analysis complete.")
    print(f"  CSVs  → {base_dir}")
    print(f"  Plots → {fig_dir}")

    return {
        "sweep_results": sweep_results,
        "comparison_df": comparison_df,
    }



# ─────────────────────────────────────────────────────────────────────────────
# Runtime Analysis: MIP vs Heuristic across max_events (no time limit)
# ─────────────────────────────────────────────────────────────────────────────

# Xpress: setting maxtime=0 disables the wall-clock limit entirely.
# We represent "no time limit" as time_limit=0 → prob.controls.maxtime = 0.
_NO_TIME_LIMIT = 0


def _run_mip_no_tlimit(
    scenario: str,
    events: pd.DataFrame,
    conflict_pairs: pd.DataFrame,
    student_events: pd.DataFrame,
    max_events: int,
    mip_gap: float,
    tmp_dir: Path,
) -> list:
    """
    Run one two-phase MIP solve (per-semester) with NO time limit.
    Returns a list of dicts — one per semester + "Combined".
    """
    err = {"Scenario": scenario, "Model": "MIP", "Semester": "Combined",
           "max_events": max_events, "Solve_Time_s": np.nan,
           "Total_Clash_Score": np.nan, "DD_Clash_Score": np.nan,
           "DF_Clash_Score": np.nan, "N_Clashes": np.nan,
           "N_Rescheduled": 0, "Solve_Status": "ERROR"}
    try:
        from mip_new import run_mip_scenario_new
        t0  = time.time()
        res = run_mip_scenario_new(
            scenario       = scenario,
            events         = events,
            conflict_pairs = conflict_pairs,
            student_events = student_events,
            max_events     = max_events,
            time_limit     = _NO_TIME_LIMIT,
            mip_gap        = mip_gap,
            verbose        = False,
            out_dir        = tmp_dir,
        )
        elapsed = round(time.time() - t0, 2)
        rows = []
        for sem, sr in res.get("sem_results", {}).items():
            rows.append({"Scenario": scenario, "Model": "MIP", "Semester": sem,
                         "max_events": max_events, "Solve_Time_s": elapsed,
                         "Total_Clash_Score": sr.get("total_clash_score", np.nan),
                         "DD_Clash_Score":    sr.get("dd_clash_score",    np.nan),
                         "DF_Clash_Score":    sr.get("df_clash_score",    np.nan),
                         "N_Clashes":         sr.get("n_clashes",         np.nan),
                         "N_Rescheduled":     len(sr["assignment_df"]) if sr.get("assignment_df") is not None else 0,
                         "Solve_Status":      sr.get("status", "UNKNOWN")})
        rows.append({"Scenario": scenario, "Model": "MIP", "Semester": "Combined",
                     "max_events": max_events, "Solve_Time_s": elapsed,
                     "Total_Clash_Score": res.get("total_clash_score", np.nan),
                     "DD_Clash_Score":    res.get("dd_clash_score",    np.nan),
                     "DF_Clash_Score":    res.get("df_clash_score",    np.nan),
                     "N_Clashes":         res.get("n_clashes",         np.nan),
                     "N_Rescheduled":     len(res["assignment_df"]) if res.get("assignment_df") is not None else 0,
                     "Solve_Status":      res.get("status", "UNKNOWN")})
        return rows
    except ImportError:
        print(f"  [runtime] xpress/mip_new not available — skipping MIP "
              f"(scenario={scenario}, max_events={max_events}).")
        err["Solve_Status"] = "SKIPPED"
    except Exception as e:
        print(f"  [runtime] MIP failed (scenario={scenario}, "
              f"max_events={max_events}): {e}")
        traceback.print_exc()
    return [err]


def _run_heuristic_no_tlimit(
    scenario: str,
    events: pd.DataFrame,
    conflict_pairs: pd.DataFrame,
    student_events: pd.DataFrame,
    max_events: int,
    ls_max_iter: int,
    tmp_dir: Path,
) -> list:
    """
    Run one Greedy + Local-Search heuristic solve (per-semester).
    No time limit is imposed. Returns a list of dicts — one per semester + "Combined".
    """
    err = {"Scenario": scenario, "Model": "Heuristic", "Semester": "Combined",
           "max_events": max_events, "Solve_Time_s": np.nan,
           "Total_Clash_Score": np.nan, "DD_Clash_Score": np.nan,
           "N_Rescheduled": 0, "Solve_Status": "ERROR"}
    try:
        from heuristic_new import run_heuristic_scenario_new
        t0  = time.time()
        res = run_heuristic_scenario_new(
            scenario         = scenario,
            events           = events,
            conflict_pairs   = conflict_pairs,
            student_events   = student_events,
            max_events       = max_events,
            run_local_search = True,
            ls_max_iter      = ls_max_iter,
            out_dir          = tmp_dir,
        )
        elapsed = round(time.time() - t0, 2)
        rows = []
        for sem, sr in res.get("sem_results", {}).items():
            ls  = sr.get("ls_full_score",   sr.get("greedy_full_score", np.nan))
            dd  = sr.get("ls_dd_score",     sr.get("greedy_dd_score",   np.nan))
            rows.append({"Scenario": scenario, "Model": "Heuristic", "Semester": sem,
                         "max_events": max_events, "Solve_Time_s": elapsed,
                         "Total_Clash_Score": ls, "DD_Clash_Score": dd,
                         "N_Rescheduled": len(sr["assignment_df"]) if sr.get("assignment_df") is not None else 0,
                         "Solve_Status": "FEASIBLE"})
        sm = res.get("summary", {})
        ls_full = sm.get("LocalSearch_Full_Clash_Score", sm.get("Greedy_Full_Clash_Score", np.nan))
        ls_dd   = sm.get("LocalSearch_DD_Clash_Score",   sm.get("Greedy_DD_Clash_Score",   np.nan))
        rows.append({"Scenario": scenario, "Model": "Heuristic", "Semester": "Combined",
                     "max_events": max_events, "Solve_Time_s": elapsed,
                     "Total_Clash_Score": ls_full, "DD_Clash_Score": ls_dd,
                     "N_Rescheduled": len(res["assignment_df"]) if res.get("assignment_df") is not None else 0,
                     "Solve_Status": "FEASIBLE"})
        return rows
    except Exception as e:
        print(f"  [runtime] Heuristic failed (scenario={scenario}, "
              f"max_events={max_events}): {e}")
        traceback.print_exc()
    return [err]


def run_runtime_vs_max_events(
    data: dict,
    sweep: list | None = None,
    mip_gap: float = DEFAULT_MIP_GAP,
    ls_max_iter: int = 300,
    out_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Analyse and compare MIP vs Heuristic **run-time** across a range of
    ``max_events`` values WITHOUT imposing any solver time limit.

    This isolates the pure scaling behaviour of each model:
    the MIP is allowed to run to proven optimality (or convergence within
    ``mip_gap``), while the heuristic runs until local-search convergence
    or ``ls_max_iter`` iterations.

    Parameters
    ----------
    data        : dict from ``data_preprocessing.run_preprocessing()``
    sweep       : list of max_events values to test.
                  Defaults to [50, 100, 200, 300, 500, 750, 1000].
    mip_gap     : MIP relative optimality gap tolerance (default 0.02 = 2 %).
    ls_max_iter : Heuristic local-search iteration cap (default 300).
    out_dir     : directory for CSV + figure outputs.
                  Defaults to outputs/runtime_analysis/.

    Returns
    -------
    pd.DataFrame
        One row per (scenario, model, max_events) with columns:
        Scenario, Model, max_events, Solve_Time_s, Objective,
        N_Clashes, N_Rescheduled, Solve_Status.

    Side-effects
    ------------
    Saves to ``out_dir``:
        runtime_vs_max_events.csv
        figures/runtime_vs_max_events_time.png
        figures/runtime_vs_max_events_quality.png
        figures/runtime_vs_max_events_ratio.png
        figures/runtime_vs_max_events_summary.png
    """
    if sweep is None:
        sweep = [50, 100, 200, 300, 500, 750, 1000]

    base_dir = out_dir or (
        Path(__file__).resolve().parent / "outputs" / "runtime_analysis"
    )
    base_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = base_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = base_dir / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    events         = data["events"]
    conflict_pairs = data.get("conflict_pairs")
    student_events = data["student_events"]

    print("\n" + "=" * 65)
    print("  RUNTIME ANALYSIS: MIP vs Heuristic (no time limit)")
    print(f"  max_events sweep : {sweep}")
    print(f"  mip_gap          : {mip_gap*100:.1f}%")
    print(f"  ls_max_iter      : {ls_max_iter}")
    print(f"  Output dir       : {base_dir}")
    print("=" * 65)

    rows = []
    total_runs = len(SCENARIOS) * len(sweep) * 2   # ×2 for MIP + Heuristic
    run_idx    = 0

    for sc in SCENARIOS:
        for me in sweep:
            run_idx += 1
            print(f"\n[{run_idx}/{total_runs}] MIP  | scenario={sc} | max_events={me}")
            mip_row = _run_mip_no_tlimit(
                scenario       = sc,
                events         = events,
                conflict_pairs = conflict_pairs,
                student_events = student_events,
                max_events     = me,
                mip_gap        = mip_gap,
                tmp_dir        = tmp_dir,
            )
            rows.append(mip_row)
            print(f"  → time={mip_row['Solve_Time_s']:.1f}s  "
                  f"obj={mip_row['Objective']}  "
                  f"status={mip_row['Solve_Status']}")

            run_idx += 1
            print(f"\n[{run_idx}/{total_runs}] Heur | scenario={sc} | max_events={me}")
            heur_row = _run_heuristic_no_tlimit(
                scenario       = sc,
                events         = events,
                conflict_pairs = conflict_pairs,
                student_events = student_events,
                max_events     = me,
                ls_max_iter    = ls_max_iter,
                tmp_dir        = tmp_dir,
            )
            rows.append(heur_row)
            print(f"  → time={heur_row['Solve_Time_s']:.1f}s  "
                  f"obj={heur_row['Objective']}  "
                  f"status={heur_row['Solve_Status']}")

    df = pd.DataFrame(rows)
    csv_path = base_dir / "runtime_vs_max_events.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[runtime] Results saved → {csv_path}")

    # ── Generate plots ────────────────────────────────────────────────────────
    _plot_runtime_analysis(df, fig_dir, sweep)

    print(f"\n[runtime] All figures saved → {fig_dir}")
    print("[runtime] Analysis complete.\n")
    return df


# ── Plotting helpers for runtime analysis ─────────────────────────────────────

_MODEL_COLORS  = {"MIP": "#1565C0", "Heuristic": "#E65100"}
_MODEL_MARKERS = {"MIP": "o",       "Heuristic": "s"}
_SC_DASH       = {"S1_9am5pm": "-", "S2_NoFriPM": "--"}


def _plot_runtime_analysis(df: pd.DataFrame, fig_dir: Path,
                            sweep: list) -> None:
    """Generate four figures summarising the runtime analysis."""
    if df.empty:
        print("  [plot] Empty DataFrame — skipping plots.")
        return

    # ── Figure 1: Solve Time vs max_events (one line per model×scenario) ─────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(
        "Runtime vs max_events — MIP (no time limit) vs Heuristic",
        fontsize=13, fontweight="bold",
    )

    for model in ["MIP", "Heuristic"]:
        for sc in SCENARIOS:
            sub = df[(df["Model"] == model) & (df["Scenario"] == sc)].copy()
            sub = sub.dropna(subset=["Solve_Time_s"]).sort_values("max_events")
            if sub.empty:
                continue
            label = f"{model} — {SCENARIO_LABELS[sc]}"
            color = _MODEL_COLORS[model]
            ls    = _SC_DASH[sc]
            marker = _MODEL_MARKERS[model]

            axes[0].plot(sub["max_events"], sub["Solve_Time_s"],
                         color=color, linestyle=ls, marker=marker,
                         linewidth=2, label=label)

    axes[0].set_xlabel("max_events (events per phase)", fontsize=11)
    axes[0].set_ylabel("Solve Time (seconds)", fontsize=11)
    axes[0].set_title("Solve Time", fontsize=11)
    axes[0].legend(fontsize=8, loc="upper left")
    axes[0].grid(True, alpha=0.3)

    # ── Same on log-y scale ───────────────────────────────────────────────────
    for model in ["MIP", "Heuristic"]:
        for sc in SCENARIOS:
            sub = df[(df["Model"] == model) & (df["Scenario"] == sc)].copy()
            sub = sub.dropna(subset=["Solve_Time_s"]).sort_values("max_events")
            if sub.empty:
                continue
            label  = f"{model} — {SCENARIO_LABELS[sc]}"
            color  = _MODEL_COLORS[model]
            ls     = _SC_DASH[sc]
            marker = _MODEL_MARKERS[model]
            vals   = sub["Solve_Time_s"].clip(lower=0.01)
            axes[1].semilogy(sub["max_events"], vals,
                             color=color, linestyle=ls, marker=marker,
                             linewidth=2, label=label)

    axes[1].set_xlabel("max_events (events per phase)", fontsize=11)
    axes[1].set_ylabel("Solve Time (seconds, log scale)", fontsize=11)
    axes[1].set_title("Solve Time (log scale)", fontsize=11)
    axes[1].legend(fontsize=8, loc="upper left")
    axes[1].grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    _savefig(fig, fig_dir / "runtime_vs_max_events_time.png")

    # ── Figure 2: Solution Quality (Objective) vs max_events ─────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(
        "Solution Quality vs max_events — MIP vs Heuristic",
        fontsize=13, fontweight="bold",
    )

    for ax_idx, sc in enumerate(SCENARIOS):
        ax = axes[ax_idx]
        for model in ["MIP", "Heuristic"]:
            sub = df[(df["Model"] == model) & (df["Scenario"] == sc)].copy()
            sub = sub.dropna(subset=["Objective"]).sort_values("max_events")
            if sub.empty:
                continue
            ax.plot(sub["max_events"], sub["Objective"],
                    color=_MODEL_COLORS[model],
                    marker=_MODEL_MARKERS[model],
                    linewidth=2, label=model)

        ax.set_xlabel("max_events (events per phase)", fontsize=11)
        ax.set_ylabel("Weighted Clash Score (↓ better)", fontsize=11)
        ax.set_title(SCENARIO_LABELS[sc], fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _savefig(fig, fig_dir / "runtime_vs_max_events_quality.png")

    # ── Figure 3: MIP / Heuristic runtime ratio ───────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title(
        "Runtime Ratio: MIP / Heuristic vs max_events",
        fontsize=13, fontweight="bold",
    )
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=1,
               label="Ratio = 1 (equal speed)")

    for sc in SCENARIOS:
        mip_s  = df[(df["Model"] == "MIP") & (df["Scenario"] == sc)
                    ].set_index("max_events")["Solve_Time_s"]
        heur_s = df[(df["Model"] == "Heuristic") & (df["Scenario"] == sc)
                    ].set_index("max_events")["Solve_Time_s"]
        common = sorted(set(mip_s.index) & set(heur_s.index))
        if not common:
            continue
        ratio = [mip_s[me] / max(heur_s[me], 0.001) for me in common]
        ax.plot(common, ratio,
                color=COLORS[sc], marker="o", linewidth=2,
                label=SCENARIO_LABELS[sc])

    ax.set_xlabel("max_events (events per phase)", fontsize=11)
    ax.set_ylabel("Runtime Ratio  (MIP time / Heuristic time)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _savefig(fig, fig_dir / "runtime_vs_max_events_ratio.png")

    # ── Figure 4: 2×2 summary dashboard ──────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Runtime Analysis Dashboard — MIP vs Heuristic (no time limit)",
        fontsize=14, fontweight="bold",
    )

    # (0,0) — runtime, linear
    ax = axes[0, 0]
    for model in ["MIP", "Heuristic"]:
        for sc in SCENARIOS:
            sub = df[(df["Model"] == model) & (df["Scenario"] == sc)
                     ].dropna(subset=["Solve_Time_s"]).sort_values("max_events")
            if sub.empty:
                continue
            ax.plot(sub["max_events"], sub["Solve_Time_s"],
                    color=_MODEL_COLORS[model], linestyle=_SC_DASH[sc],
                    marker=_MODEL_MARKERS[model], linewidth=2,
                    label=f"{model} {SCENARIO_LABELS[sc]}")
    ax.set_xlabel("max_events"); ax.set_ylabel("Solve Time (s)")
    ax.set_title("(a) Solve Time vs max_events (linear)"); ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (0,1) — runtime, log
    ax = axes[0, 1]
    for model in ["MIP", "Heuristic"]:
        for sc in SCENARIOS:
            sub = df[(df["Model"] == model) & (df["Scenario"] == sc)
                     ].dropna(subset=["Solve_Time_s"]).sort_values("max_events")
            if sub.empty:
                continue
            ax.semilogy(sub["max_events"],
                        sub["Solve_Time_s"].clip(lower=0.01),
                        color=_MODEL_COLORS[model], linestyle=_SC_DASH[sc],
                        marker=_MODEL_MARKERS[model], linewidth=2,
                        label=f"{model} {SCENARIO_LABELS[sc]}")
    ax.set_xlabel("max_events"); ax.set_ylabel("Solve Time (s, log)")
    ax.set_title("(b) Solve Time vs max_events (log scale)"); ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, which="both")

    # (1,0) — quality (objective) per scenario
    ax = axes[1, 0]
    for model in ["MIP", "Heuristic"]:
        for sc in SCENARIOS:
            sub = df[(df["Model"] == model) & (df["Scenario"] == sc)
                     ].dropna(subset=["Objective"]).sort_values("max_events")
            if sub.empty:
                continue
            ax.plot(sub["max_events"], sub["Objective"],
                    color=_MODEL_COLORS[model], linestyle=_SC_DASH[sc],
                    marker=_MODEL_MARKERS[model], linewidth=2,
                    label=f"{model} {SCENARIO_LABELS[sc]}")
    ax.set_xlabel("max_events"); ax.set_ylabel("Weighted Clash Score (↓)")
    ax.set_title("(c) Solution Quality vs max_events"); ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (1,1) — runtime ratio
    ax = axes[1, 1]
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=1,
               label="Ratio = 1")
    for sc in SCENARIOS:
        mip_s  = df[(df["Model"] == "MIP") & (df["Scenario"] == sc)
                    ].set_index("max_events")["Solve_Time_s"]
        heur_s = df[(df["Model"] == "Heuristic") & (df["Scenario"] == sc)
                    ].set_index("max_events")["Solve_Time_s"]
        common = sorted(set(mip_s.index) & set(heur_s.index))
        if not common:
            continue
        ratio = [mip_s[me] / max(heur_s[me], 0.001) for me in common]
        ax.plot(common, ratio, color=COLORS[sc], marker="o",
                linewidth=2, label=SCENARIO_LABELS[sc])
    ax.set_xlabel("max_events"); ax.set_ylabel("MIP time / Heuristic time")
    ax.set_title("(d) Runtime Ratio MIP / Heuristic"); ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _savefig(fig, fig_dir / "runtime_vs_max_events_summary.png")


# Standalone entry point
if __name__ == "__main__":
    import argparse
    from data_preprocessing import run_preprocessing

    parser = argparse.ArgumentParser(
        description="MIP parameter sensitivity analysis + runtime comparison"
    )
    parser.add_argument(
        "--no-heatmap", action="store_true",
        help="Skip the 2-D heatmap sub-grid (saves time)"
    )
    parser.add_argument(
        "--heatmap-scenario", default="S1_9am5pm",
        choices=["S1_9am5pm", "S2_NoFriPM"],
        help="Scenario to use for the 2-D heatmap"
    )
    parser.add_argument(
        "--skip-conflicts", action="store_true",
        help="Load existing conflict_pairs.csv instead of rebuilding"
    )
    parser.add_argument(
        "--runtime-only", action="store_true",
        help="Run ONLY the runtime-vs-max_events analysis (skip OAT sweeps)"
    )
    parser.add_argument(
        "--runtime-sweep", type=int, nargs="+",
        default=[50, 100, 200, 300, 500, 750, 1000],
        metavar="N",
        help="max_events values for runtime analysis "
             "(default: 50 100 200 300 500 750 1000)"
    )
    parser.add_argument(
        "--mip-gap", type=float, default=DEFAULT_MIP_GAP,
        help=f"MIP optimality gap for runtime analysis (default: {DEFAULT_MIP_GAP})"
    )
    parser.add_argument(
        "--ls-iter", type=int, default=300,
        help="Heuristic local-search max iterations (default: 300)"
    )
    args = parser.parse_args()

    _CLEANED = Path(__file__).resolve().parent / "outputs" / "cleaned_data"
    _CLEANED.mkdir(parents=True, exist_ok=True)

    conflict_path   = _CLEANED / "conflict_pairs.csv"
    build_conflicts = not (args.skip_conflicts and conflict_path.exists())

    data = run_preprocessing(
        build_conflicts = build_conflicts,
        out_dir         = _CLEANED,
    )

    if args.runtime_only:
        # ── Runtime-only mode ─────────────────────────────────────────────────
        run_runtime_vs_max_events(
            data        = data,
            sweep       = args.runtime_sweep,
            mip_gap     = args.mip_gap,
            ls_max_iter = args.ls_iter,
        )
    else:
        # ── Full sensitivity + runtime analysis ───────────────────────────────
        run_sensitivity_analysis(
            data             = data,
            run_heatmap      = not args.no_heatmap,
            heatmap_scenario = args.heatmap_scenario,
        )
        run_runtime_vs_max_events(
            data        = data,
            sweep       = args.runtime_sweep,
            mip_gap     = args.mip_gap,
            ls_max_iter = args.ls_iter,
        )
