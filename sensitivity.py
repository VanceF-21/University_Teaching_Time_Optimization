"""
param_sensitivity.py

Parameter sensitivity analysis and runtime comparison for the
University Teaching Time Optimization pipeline.

Two main analyses:
─────────────────────────────────────────────────────────────────────
A) MIP PARAMETER SWEEPS
   For mip_model.py (two-phase: WholeClass + SubGroup),
   independently vary three parameters and record quality + runtime:

     • max_events  : problem size limit  [50, 100, 200, 300, 400, 500, 700, 1000]
     • time_limit  : solver wall-clock s  [30, 60, 120, 180, 300, 480, 600]
     • mip_gap     : relative opt. gap   [0.005, 0.01, 0.02, 0.05, 0.10]

   Metrics recorded per run:
     solve_time_s, objective_value (weighted clash score),
     n_clashes, n_rescheduled, solve_status

B) RUNTIME COMPARISON (3 methods, both scenarios)
     1. MIP Model        – mip_model.py  (two-phase: WholeClass + SubGroup)
     2. Heuristic greedy – greedy only, no local search
     3. Heuristic full   – greedy + local search (200 iterations)

   A grouped bar chart is produced comparing total wall-clock time.

─────────────────────────────────────────────────────────────────────

Visualisations produced (all saved to <out_dir>/sensitivity/figures/):
  1. max_events_vs_quality.png    – clash score & solve time vs max_events
  2. time_limit_vs_quality.png    – clash score & gap vs time_limit
  3. mip_gap_vs_quality.png       – clash score & solve time vs mip_gap
  4. quality_vs_time_scatter.png  – Pareto scatter (quality vs runtime)
  5. runtime_comparison.png       – grouped bar chart (3 methods × 2 scenarios)
  6. sensitivity_heatmap.png      – heatmap: max_events × time_limit → objective

CSVs also saved to <out_dir>/sensitivity/:
  sensitivity_max_events.csv
  sensitivity_time_limit.csv
  sensitivity_mip_gap.csv
  runtime_comparison.csv

Usage:
  python param_sensitivity.py                      # full sweep (slow)
  python param_sensitivity.py --quick              # reduced grid (fast demo)
  python param_sensitivity.py --scenario S1_9am5pm # one scenario only
  python param_sensitivity.py --skip-sweep         # runtime comparison only
  python param_sensitivity.py --skip-runtime       # parameter sweeps only
"""

import argparse
import time
import warnings
import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR  = BASE_DIR / "outputs"



# Colour / style constants (aligned with visualization.py)
sns.set_theme(style="whitegrid", font_scale=1.1)

SCENARIO_COLORS = {
    "S1_9am5pm":  "#DD8452",
    "S2_NoFriPM": "#55A868",
}
METHOD_COLORS = {
    "MIP Model":        "#4C72B0",
    "Heuristic Greedy": "#8172B2",
    "Heuristic Full":   "#64B5CD",
}
METHOD_ORDER = ["MIP Model", "Heuristic Greedy", "Heuristic Full"]

# Default sweep grids
GRID_MAX_EVENTS  = [100, 200, 300, 400, 500, 700, 1000]
GRID_TIME_LIMIT  = [30, 60, 120, 180, 300, 480, 600]
GRID_MIP_GAP     = [0.005, 0.01, 0.02, 0.05, 0.10]

# Quick-mode grids (fewer points)
GRID_MAX_EVENTS_QUICK = [50, 150, 300, 500, 700]
GRID_TIME_LIMIT_QUICK = [30, 120, 300, 600]
GRID_MIP_GAP_QUICK    = [0.01, 0.05, 0.10]



def parse_args():
    p = argparse.ArgumentParser(
        description="MIP parameter sensitivity analysis + runtime comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--quick",        action="store_true",
                   help="Use reduced grid (fewer sweep points, faster)")
    p.add_argument("--scenario",     default="both",
                   choices=["S1_9am5pm", "S2_NoFriPM", "both"],
                   help="Which scenario(s) to analyse (default: both)")
    p.add_argument("--skip-sweep",   action="store_true",
                   help="Skip parameter sweep; only run runtime comparison")
    p.add_argument("--skip-runtime", action="store_true",
                   help="Skip runtime comparison; only run parameter sweeps")
    p.add_argument("--skip-conflicts", action="store_true",
                   help="Reuse existing conflict_pairs.csv (faster startup)")
    p.add_argument("--heur-iter",    type=int, default=200,
                   help="Local search iterations for heuristic full mode (default: 200)")
    return p.parse_args()



def make_sens_dirs() -> tuple:
    """Create output dirs for sensitivity analysis and return (sens_dir, fig_dir)."""
    ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    sens_dir = OUT_DIR / f"sensitivity_{ts}"
    fig_dir  = sens_dir / "figures"
    sens_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    return sens_dir, fig_dir


def save_fig(fig: plt.Figure, name: str, fig_dir: Path, dpi: int = 150):
    path = fig_dir / name
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"  [sens] Saved figure → {path.relative_to(BASE_DIR)}")



def _run_mip_once(scenario: str,
                  events: pd.DataFrame,
                  conflict_pairs,
                  max_events: int,
                  time_limit: int,
                  mip_gap: float,
                  tmp_dir: Path) -> dict:
    """Run a single MIP solve (two-phase) and return a flat metrics dict."""
    t0 = time.time()
    try:
        from mip_model import run_mip_scenario
        res       = run_mip_scenario(
            scenario=scenario,
            events=events,
            conflict_pairs=conflict_pairs,
            max_events=max_events,
            time_limit=time_limit,
            mip_gap=mip_gap,
            verbose=False,
            out_dir=tmp_dir,
        )
        wall_time = time.time() - t0
        summary   = res.get("summary", {})
        obj     = summary.get("Objective_Total",    res.get("objective"))
        ncl     = (summary.get("N_Clash_Pairs_After_P1", 0) or 0) + \
                  (summary.get("N_Clash_Pairs_After_P2", 0) or 0)
        nres    = summary.get("N_Rescheduled_Total", res.get("n_displaced", 0))
        ndisp   = summary.get("N_Displaced_Total",   res.get("n_displaced", 0))
        solve_t = (summary.get("Phase1_Solve_Time_s", 0) or 0) + \
                  (summary.get("Phase2_Solve_Time_s", 0) or 0)
        return {
            "Model":         "MIP Model",
            "Scenario":      scenario,
            "Max_Events":    max_events,
            "Time_Limit_s":  time_limit,
            "MIP_Gap":       mip_gap,
            "Solve_Status":  res.get("status", "UNKNOWN"),
            "Solve_Time_s":  round(float(solve_t or wall_time), 2),
            "Wall_Time_s":   round(wall_time, 2),
            "Objective":     round(float(obj), 1) if obj is not None else np.nan,
            "N_Clashes":     int(ncl or 0),
            "N_Rescheduled": int(nres or 0),
            "N_Displaced":   int(ndisp or 0),
        }
    except Exception as exc:
        wall_time = time.time() - t0
        print(f"    [WARN] MIP/{scenario} max={max_events} "
              f"tlim={time_limit} gap={mip_gap} → {exc}")
        return {
            "Model":         "MIP Model",
            "Scenario":      scenario,
            "Max_Events":    max_events,
            "Time_Limit_s":  time_limit,
            "MIP_Gap":       mip_gap,
            "Solve_Status":  "ERROR",
            "Solve_Time_s":  round(wall_time, 2),
            "Wall_Time_s":   round(wall_time, 2),
            "Objective":     np.nan,
            "N_Clashes":     np.nan,
            "N_Rescheduled": np.nan,
            "N_Displaced":   np.nan,
        }



def _run_heuristic_once(scenario: str,
                         data: dict,
                         run_ls: bool,
                         ls_iter: int,
                         tmp_dir: Path) -> dict:
    """Run one heuristic scenario and return a flat metrics dict."""
    t0 = time.time()
    try:
        from heuristic_model import run_heuristic_scenario
        res = run_heuristic_scenario(
            scenario=scenario,
            events=data["events"],
            conflict_pairs=data.get("conflict_pairs"),
            student_events=data["student_events"],
            max_events=5000,
            run_local_search=run_ls,
            ls_max_iter=ls_iter,
            out_dir=tmp_dir,
        )
        wall_time = time.time() - t0
        sm = res.get("summary", {})
        return {
            "Model":         "Heuristic Full" if run_ls else "Heuristic Greedy",
            "Scenario":      scenario,
            "Wall_Time_s":   round(wall_time, 2),
            "Objective":     sm.get("LocalSearch_Clash_Score",
                                    sm.get("Greedy_Clash_Score", np.nan)),
            "N_Rescheduled": sm.get("N_Rescheduled", np.nan),
        }
    except Exception as exc:
        print(f"    [WARN] heuristic/{scenario} ls={run_ls} → {exc}")
        return {
            "Model":         "Heuristic Full" if run_ls else "Heuristic Greedy",
            "Scenario":      scenario,
            "Wall_Time_s":   round(time.time() - t0, 2),
            "Objective":     np.nan,
            "N_Rescheduled": np.nan,
        }



def run_parameter_sweeps(data: dict,
                          scenarios: list,
                          quick: bool,
                          sens_dir: Path,
                          fig_dir: Path) -> dict:
    """
    Run three independent parameter sweeps (max_events, time_limit, mip_gap).

    Each sweep varies ONE parameter while holding the others at their defaults:
      default max_events  = 300
      default time_limit  = 180 s
      default mip_gap     = 0.02

    Returns a dict of DataFrames: { "max_events": df, "time_limit": df, "mip_gap": df }
    """
    DEFAULT_MAX_EVENTS = 500
    DEFAULT_TIME_LIMIT = 300
    DEFAULT_MIP_GAP    = 0.02

    grid_me  = GRID_MAX_EVENTS_QUICK  if quick else GRID_MAX_EVENTS
    grid_tl  = GRID_TIME_LIMIT_QUICK  if quick else GRID_TIME_LIMIT
    grid_gap = GRID_MIP_GAP_QUICK     if quick else GRID_MIP_GAP

    events         = data["events"]
    conflict_pairs = data.get("conflict_pairs")
    tmp_dir        = sens_dir / "_tmp_mip"
    tmp_dir.mkdir(exist_ok=True)

    sweep_results = {"max_events": [], "time_limit": [], "mip_gap": []}

    total_runs = (len(grid_me) + len(grid_tl) + len(grid_gap)) * len(scenarios)
    run_no = 0
    print(f"\n  Total MIP sweep runs planned: {total_runs}")

    for sc in scenarios:

        # Sweep 1: max_events
        print(f"\n  [{sc}] Sweeping max_events …")
        for me in grid_me:
            run_no += 1
            print(f"    Run {run_no}/{total_runs}  max_events={me} …", end=" ", flush=True)
            row = _run_mip_once(sc, events, conflict_pairs,
                                me, DEFAULT_TIME_LIMIT, DEFAULT_MIP_GAP, tmp_dir)
            row["Sweep"] = "max_events"
            sweep_results["max_events"].append(row)
            print(f"status={row['Solve_Status']}  obj={row['Objective']}  "
                  f"t={row['Solve_Time_s']}s")

        # Sweep 2: time_limit
        print(f"\n  [{sc}] Sweeping time_limit …")
        for tl in grid_tl:
            run_no += 1
            print(f"    Run {run_no}/{total_runs}  time_limit={tl}s …", end=" ", flush=True)
            row = _run_mip_once(sc, events, conflict_pairs,
                                DEFAULT_MAX_EVENTS, tl, DEFAULT_MIP_GAP, tmp_dir)
            row["Sweep"] = "time_limit"
            sweep_results["time_limit"].append(row)
            print(f"status={row['Solve_Status']}  obj={row['Objective']}  "
                  f"t={row['Solve_Time_s']}s")

        # Sweep 3: mip_gap
        print(f"\n  [{sc}] Sweeping mip_gap …")
        for gap in grid_gap:
            run_no += 1
            print(f"    Run {run_no}/{total_runs}  mip_gap={gap} …", end=" ", flush=True)
            row = _run_mip_once(sc, events, conflict_pairs,
                                DEFAULT_MAX_EVENTS, DEFAULT_TIME_LIMIT, gap, tmp_dir)
            row["Sweep"] = "mip_gap"
            sweep_results["mip_gap"].append(row)
            print(f"status={row['Solve_Status']}  obj={row['Objective']}  "
                  f"t={row['Solve_Time_s']}s")

    # Consolidate into DataFrames and save
    dfs = {}
    for sweep_key, rows in sweep_results.items():
        df = pd.DataFrame(rows)
        dfs[sweep_key] = df
        csv_path = sens_dir / f"sensitivity_{sweep_key}.csv"
        df.to_csv(csv_path, index=False)
        print(f"  [sens] CSV saved → {csv_path.relative_to(BASE_DIR)}")

    # Produce charts
    _plot_max_events_sweep(dfs["max_events"],   fig_dir)
    _plot_time_limit_sweep(dfs["time_limit"],   fig_dir)
    _plot_mip_gap_sweep(dfs["mip_gap"],         fig_dir)
    _plot_quality_vs_time(dfs["max_events"],    fig_dir)
    _plot_sensitivity_heatmap(dfs["max_events"], dfs["time_limit"], fig_dir)

    return dfs



# Runtime comparison
def run_runtime_comparison(data: dict,
                            scenarios: list,
                            ls_iter: int,
                            sens_dir: Path,
                            fig_dir: Path) -> pd.DataFrame:
    """
    Run all three methods once at default settings and compare wall-clock time.

    Methods:
      MIP Model        max_events=500, time_limit=300, gap=0.02
      Heuristic Greedy greedy only
      Heuristic Full   greedy + local search (ls_iter iterations)
    """
    events         = data["events"]
    conflict_pairs = data.get("conflict_pairs")
    tmp_dir        = sens_dir / "_tmp_rt"
    tmp_dir.mkdir(exist_ok=True)

    rows = []
    methods = [
        ("mip",       False),
        ("greedy",    False),
        ("heuristic", True),
    ]

    for sc in scenarios:
        for mtype, is_heuristic in methods:
            if is_heuristic or mtype == "greedy":
                # Heuristic
                run_ls = (mtype == "heuristic")
                print(f"  [runtime] Heuristic ({'full' if run_ls else 'greedy'}) / {sc} …",
                      end=" ", flush=True)
                row = _run_heuristic_once(sc, data, run_ls, ls_iter, tmp_dir)
            else:
                # MIP Model
                print(f"  [runtime] MIP Model / {sc} …", end=" ", flush=True)
                row = _run_mip_once(sc, events, conflict_pairs,
                                    500, 300, 0.02, tmp_dir)
            rows.append(row)
            print(f"t={row['Wall_Time_s']}s  obj={row.get('Objective', 'N/A')}")

    df = pd.DataFrame(rows)
    csv_path = sens_dir / "runtime_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"  [sens] CSV saved → {csv_path.relative_to(BASE_DIR)}")

    _plot_runtime_comparison(df, fig_dir)
    return df



# Plotting functions
def _add_status_markers(ax, df: pd.DataFrame, x_col: str, y_col: str):
    """Overlay ✗ markers on points where Solve_Status is not OPTIMAL/FEASIBLE."""
    if "Solve_Status" not in df.columns:
        return
    bad = df[~df["Solve_Status"].isin(["OPTIMAL", "FEASIBLE", "FEASIBLE_TIME_LIMIT",
                                        "LP_OPTIMAL_NO_INT"])]
    if len(bad):
        ax.scatter(bad[x_col], bad[y_col], marker="X", s=120,
                   color="red", zorder=5, label="Not solved")


def _twin_axes_plot(ax_left, x_vals, y_left, y_right,
                    label_left, label_right,
                    color_left="#4C72B0", color_right="#C44E52",
                    marker="o"):
    """Plot two y-axes sharing the same x-axis."""
    ax_right = ax_left.twinx()
    ax_left.plot(x_vals,  y_left,  color=color_left,  marker=marker,
                 linewidth=2, markersize=7, label=label_left)
    ax_right.plot(x_vals, y_right, color=color_right, marker="s",
                  linewidth=2, markersize=7, linestyle="--", label=label_right)
    ax_left.set_ylabel(label_left,  color=color_left,  fontsize=11)
    ax_right.set_ylabel(label_right, color=color_right, fontsize=11)
    ax_left.tick_params(axis="y",  labelcolor=color_left)
    ax_right.tick_params(axis="y", labelcolor=color_right)
    # Combined legend
    lines_l, labs_l = ax_left.get_legend_handles_labels()
    lines_r, labs_r = ax_right.get_legend_handles_labels()
    ax_left.legend(lines_l + lines_r, labs_l + labs_r,
                   loc="upper right", fontsize=9)
    return ax_right


# 1. max_events sweep
def _plot_max_events_sweep(df: pd.DataFrame, fig_dir: Path):
    """Two-panel: (a) clash score vs max_events, (b) solve time vs max_events."""
    if df.empty:
        return

    scenarios = sorted(df["Scenario"].unique())
    models    = sorted(df["Model"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Parameter Sensitivity: Problem Size (max_events)", fontsize=14, y=1.01)

    linestyles = {"MIP Model": "-"}
    markers    = {"S1_9am5pm": "o", "S2_NoFriPM": "s"}

    for ax, (y_col, ylabel, title) in zip(
        axes,
        [("Objective",   "Weighted Clash Score",  "(a) Solution Quality vs Problem Size"),
         ("Solve_Time_s","Solve Time (s)",         "(b) Runtime vs Problem Size")]
    ):
        for model in models:
            for sc in scenarios:
                sub = df[(df["Model"] == sc) | (df["Scenario"] == sc)]
                sub = df[(df["Model"] == model) & (df["Scenario"] == sc)].copy()
                sub = sub.sort_values("Max_Events")
                valid = sub.dropna(subset=[y_col])
                if valid.empty:
                    continue
                color = SCENARIO_COLORS.get(sc, "#888888")
                ls    = linestyles.get(model, "-")
                mk    = markers.get(sc, "o")
                ax.plot(valid["Max_Events"], valid[y_col],
                        color=color, linestyle=ls, marker=mk,
                        linewidth=2, markersize=7,
                        label=f"{model} / {sc}")

        ax.set_xlabel("max_events (problem size limit)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=8, loc="upper left")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    fig.tight_layout()
    save_fig(fig, "max_events_vs_quality.png", fig_dir)


# 2. time_limit sweep
def _plot_time_limit_sweep(df: pd.DataFrame, fig_dir: Path):
    """Two-panel: clash score vs time_limit, and solve_time vs time_limit."""
    if df.empty:
        return

    scenarios = sorted(df["Scenario"].unique())
    models    = sorted(df["Model"].unique())
    linestyles = {"MIP Model": "-"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Parameter Sensitivity: Solver Time Limit (time_limit)", fontsize=14, y=1.01)

    for ax, (y_col, ylabel, title) in zip(
        axes,
        [("Objective",    "Weighted Clash Score", "(a) Solution Quality vs Time Limit"),
         ("Solve_Time_s", "Actual Solve Time (s)", "(b) Actual Runtime vs Time Limit")]
    ):
        for model in models:
            for sc in scenarios:
                sub = df[(df["Model"] == model) & (df["Scenario"] == sc)].copy()
                sub = sub.sort_values("Time_Limit_s")
                valid = sub.dropna(subset=[y_col])
                if valid.empty:
                    continue
                color = SCENARIO_COLORS.get(sc, "#888888")
                ls    = linestyles.get(model, "-")
                ax.plot(valid["Time_Limit_s"], valid[y_col],
                        color=color, linestyle=ls, marker="o",
                        linewidth=2, markersize=7,
                        label=f"{model} / {sc}")

        # Diagonal reference line for panel (b): actual_time = time_limit
        if y_col == "Solve_Time_s" and not df["Time_Limit_s"].dropna().empty:
            xlim_max = df["Time_Limit_s"].max() * 1.05
            ax.plot([0, xlim_max], [0, xlim_max], "k:", linewidth=1,
                    alpha=0.5, label="actual = limit (bound)")

        ax.set_xlabel("time_limit (s)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    save_fig(fig, "time_limit_vs_quality.png", fig_dir)


# 3. mip_gap sweep
def _plot_mip_gap_sweep(df: pd.DataFrame, fig_dir: Path):
    """Two-panel: clash score vs mip_gap, and solve_time vs mip_gap."""
    if df.empty:
        return

    scenarios = sorted(df["Scenario"].unique())
    models    = sorted(df["Model"].unique())
    linestyles = {"MIP Model": "-"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Parameter Sensitivity: Optimality Gap (mip_gap)", fontsize=14, y=1.01)

    for ax, (y_col, ylabel, title) in zip(
        axes,
        [("Objective",    "Weighted Clash Score", "(a) Solution Quality vs MIP Gap"),
         ("Solve_Time_s", "Solve Time (s)",        "(b) Runtime vs MIP Gap")]
    ):
        for model in models:
            for sc in scenarios:
                sub = df[(df["Model"] == model) & (df["Scenario"] == sc)].copy()
                sub = sub.sort_values("MIP_Gap")
                valid = sub.dropna(subset=[y_col])
                if valid.empty:
                    continue
                color = SCENARIO_COLORS.get(sc, "#888888")
                ls    = linestyles.get(model, "-")
                ax.plot(valid["MIP_Gap"] * 100, valid[y_col],
                        color=color, linestyle=ls, marker="o",
                        linewidth=2, markersize=7,
                        label=f"{model} / {sc}")

        ax.set_xlabel("MIP Gap (%)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=8)

    fig.tight_layout()
    save_fig(fig, "mip_gap_vs_quality.png", fig_dir)


# 4. Quality vs Runtime scatter (Pareto)
def _plot_quality_vs_time(df: pd.DataFrame, fig_dir: Path):
    """
    Scatter plot: x = solve time, y = clash score.
    Each point = one solver run; coloured by scenario, shaped by model.
    Shows the quality–time trade-off (Pareto frontier).
    """
    if df.empty:
        return

    valid = df.dropna(subset=["Objective", "Solve_Time_s"])
    if valid.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    markers_by_model = {"MIP Model": "o"}

    for _, row in valid.iterrows():
        color = SCENARIO_COLORS.get(row["Scenario"], "#888888")
        mk    = markers_by_model.get(row["Model"], "D")
        ax.scatter(row["Solve_Time_s"], row["Objective"],
                   color=color, marker=mk, s=80, alpha=0.85, edgecolors="white")

    # Proxy artists for legend
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker="o", color=c, markersize=9, linestyle="none",
               label=sc)
        for sc, c in SCENARIO_COLORS.items()
        if sc in valid["Scenario"].values
    ] + [
        Line2D([0], [0], marker=mk, color="#555", markersize=9, linestyle="none",
               label=m)
        for m, mk in markers_by_model.items()
        if m in valid["Model"].values
    ]
    ax.legend(handles=legend_elems, fontsize=9, loc="upper right")

    ax.set_xlabel("Solve Time (s)", fontsize=12)
    ax.set_ylabel("Weighted Clash Score (lower is better)", fontsize=12)
    ax.set_title("Quality vs Runtime Trade-off (MIP Sweep)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "quality_vs_time_scatter.png", fig_dir)


# 5. Runtime comparison bar chart
def _plot_runtime_comparison(df: pd.DataFrame, fig_dir: Path):
    """
    Grouped bar chart: x = method, grouped by scenario,
    y = wall-clock runtime (s).
    Secondary twin axis shows clash score.
    """
    if df.empty:
        return

    scenarios = sorted(df["Scenario"].unique())
    # Standardise method column name
    method_col = "Model"
    methods    = [m for m in METHOD_ORDER if m in df[method_col].values]
    if not methods:
        return

    x     = np.arange(len(methods))
    width = 0.35
    n_sc  = len(scenarios)
    offsets = np.linspace(-(n_sc - 1) * width / 2,
                            (n_sc - 1) * width / 2, n_sc)

    fig, ax1 = plt.subplots(figsize=(11, 6))
    ax2 = ax1.twinx()

    bars_plotted = []
    for i, (sc, offset) in enumerate(zip(scenarios, offsets)):
        sub    = df[df["Scenario"] == sc]
        times  = [sub[sub[method_col] == m]["Wall_Time_s"].mean()
                  for m in methods]
        scores = [sub[sub[method_col] == m]["Objective"].mean()
                  for m in methods]

        color  = SCENARIO_COLORS.get(sc, f"C{i}")
        b = ax1.bar(x + offset, times, width * 0.9,
                    label=f"{sc} — runtime", color=color, alpha=0.80)
        bars_plotted.append(b)

        # Clash score as line on secondary axis
        valid_idx = [j for j, s in enumerate(scores) if not np.isnan(s)]
        if valid_idx:
            ax2.plot([x[j] + offset for j in valid_idx],
                     [scores[j] for j in valid_idx],
                     color=color, marker="D", markersize=8,
                     linewidth=1.5, linestyle="--",
                     label=f"{sc} — clash score")

    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=11)
    ax1.set_xlabel("Method", fontsize=12)
    ax1.set_ylabel("Wall-Clock Runtime (s)", fontsize=12)
    ax2.set_ylabel("Weighted Clash Score (lower is better)",
                   fontsize=12, color="#555")

    ax1.set_title("Runtime & Solution Quality Comparison: All Methods",
                  fontsize=13, fontweight="bold")

    # Combined legend
    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2,
               loc="upper left", fontsize=9, ncol=2)

    fig.tight_layout()
    save_fig(fig, "runtime_comparison.png", fig_dir)


# 6. Sensitivity heatmap: max_events × time_limit
def _plot_sensitivity_heatmap(df_me: pd.DataFrame,
                               df_tl: pd.DataFrame,
                               fig_dir: Path):
    """
    Heatmap of objective value over a grid of (max_events, time_limit).
    Combines rows from both sweep DataFrames to approximate the 2D grid.
    Only uses DEFAULT values for the other parameter.
    """
    DEFAULT_TL = 180
    DEFAULT_ME = 300

    # Rows from max_events sweep: use their time_limit (= DEFAULT_TL)
    # Rows from time_limit sweep: use their max_events (= DEFAULT_ME)
    combined_rows = []

    if not df_me.empty:
        for _, r in df_me.iterrows():
            combined_rows.append({
                "Max_Events":   r["Max_Events"],
                "Time_Limit_s": DEFAULT_TL,
                "Scenario":     r["Scenario"],
                "Model":        r["Model"],
                "Objective":    r["Objective"],
            })
    if not df_tl.empty:
        for _, r in df_tl.iterrows():
            combined_rows.append({
                "Max_Events":   DEFAULT_ME,
                "Time_Limit_s": r["Time_Limit_s"],
                "Scenario":     r["Scenario"],
                "Model":        r["Model"],
                "Objective":    r["Objective"],
            })

    if not combined_rows:
        return

    grid_df = pd.DataFrame(combined_rows).dropna(subset=["Objective"])
    scenarios = sorted(grid_df["Scenario"].unique())
    models    = sorted(grid_df["Model"].unique())

    n_rows = len(scenarios) * len(models)
    if n_rows == 0:
        return

    fig, axes = plt.subplots(len(scenarios), len(models),
                              figsize=(6 * len(models), 4.5 * len(scenarios)),
                              squeeze=False)
    fig.suptitle("Objective Heatmap: max_events × time_limit",
                 fontsize=14, y=1.02)

    for ri, sc in enumerate(scenarios):
        for ci, model in enumerate(models):
            ax = axes[ri][ci]
            sub = grid_df[(grid_df["Scenario"] == sc) & (grid_df["Model"] == model)]

            if sub.empty:
                ax.set_visible(False)
                continue

            pivot = sub.pivot_table(index="Max_Events", columns="Time_Limit_s",
                                     values="Objective", aggfunc="mean")
            if pivot.empty:
                ax.set_visible(False)
                continue

            sns.heatmap(pivot, ax=ax, cmap="YlOrRd_r", annot=True, fmt=".0f",
                        linewidths=0.5, cbar_kws={"label": "Clash Score"})
            ax.set_title(f"{model} / {sc}", fontsize=11)
            ax.set_xlabel("Time Limit (s)", fontsize=10)
            ax.set_ylabel("Max Events",     fontsize=10)

    fig.tight_layout()
    save_fig(fig, "sensitivity_heatmap.png", fig_dir)



def main():
    args = parse_args()

    # Check Xpress availability
    xpress_ok = False
    if not args.skip_sweep or not args.skip_runtime or True:
        try:
            import xpress  # noqa: F401
            xpress_ok = True
        except ImportError:
            print("[ERROR] FICO Xpress is not installed / not on PATH.")
            print("        MIP sweep and MIP runtime comparisons will be skipped.")
            print("        Install Xpress and re-run to enable MIP analyses.")

    # Load data
    print("\n>>> Loading data …")
    from data_preprocessing import run_preprocessing
    cleaned_dir = OUT_DIR / "cleaned_data"
    cleaned_dir.mkdir(exist_ok=True)
    conflict_path   = cleaned_dir / "conflict_pairs.csv"
    build_conflicts = not (args.skip_conflicts and conflict_path.exists())
    if not build_conflicts:
        print("  Reusing existing conflict_pairs.csv")
    data = run_preprocessing(build_conflicts=build_conflicts, out_dir=cleaned_dir)

    # Create output dirs
    sens_dir, fig_dir = make_sens_dirs()
    print(f"\n  Sensitivity outputs → {sens_dir.relative_to(BASE_DIR)}")

    scenarios = (["S1_9am5pm", "S2_NoFriPM"] if args.scenario == "both"
                 else [args.scenario])

    # Parameter sweeps
    if not args.skip_sweep:
        if not xpress_ok:
            print("\n>>> PARAMETER SWEEPS — skipped (Xpress unavailable)")
        else:
            print("\n>>> PARAMETER SWEEPS")
            print(f"  Scenarios : {scenarios}")
            print(f"  Quick mode: {args.quick}")
            run_parameter_sweeps(data, scenarios,
                                 args.quick, sens_dir, fig_dir)
    else:
        print("\n>>> PARAMETER SWEEPS — skipped (--skip-sweep)")

    # Runtime comparison
    if not args.skip_runtime:
        print("\n>>> RUNTIME COMPARISON (all methods)")
        if not xpress_ok:
            print("  [INFO] Xpress unavailable — running heuristic methods only.")
        run_runtime_comparison(data, scenarios, args.heur_iter, sens_dir, fig_dir)
    else:
        print("\n>>> RUNTIME COMPARISON — skipped (--skip-runtime)")

    print(f"\n>>> Done. All outputs in:\n    {sens_dir}")


if __name__ == "__main__":
    main()
