# University Teaching Time Optimization

[![中文文档](https://img.shields.io/badge/文档-中文版-FF4B4B?style=for-the-badge&logo=googletranslate&logoColor=white)](./README_zh.md)
[![Google Drive](https://img.shields.io/badge/Outputs-Google%20Drive-4285F4?style=for-the-badge&logo=googledrive&logoColor=white)](https://drive.google.com/drive/folders/1tPd5ysSP1J02JN9CroGGa4z9xvUDSTTO?usp=drive_link)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FICO Xpress](https://img.shields.io/badge/Solver-FICO%20Xpress-FF6600?style=for-the-badge&logo=databricks&logoColor=white)](https://www.fico.com/en/products/fico-xpress-optimization)

Analytical pipeline for evaluating timetable restructuring scenarios at university [Group 17].

## Research Questions

| Q | Question |
|---|----------|
| Q1 | Would it be possible to reduce core teaching hours to Mon–Fri 9am–5pm? |
| Q2 | Would it be possible to eliminate teaching on Friday 12pm–6pm? |
| Q3 | Could core teaching/lectures still be delivered without clashes in each scenario? |
| Q4 | Could we provide a lunch break for majority of students between 12pm–2pm? |
| Q5 | How would these changes affect timeslot/room utilisation throughout the week? |

## Scenarios

| ID | Description | Answers |
|----|-------------|---------|
| S1 | Restrict all teaching to Mon–Fri, 9am–5pm | Q1, Q3, Q4, Q5 |
| S2 | Eliminate Friday teaching after 12pm | Q2, Q3, Q4, Q5 |

## Project Structure

```
.
├── main.py                  # Pipeline entry point
├── data_preprocessing.py    # Load & clean Excel data, build conflict pairs
├── baseline_analysis.py     # Descriptive stats for current timetable
├── mip_model.py             # FICO Xpress Binary Integer Programme
├── heuristic_model.py       # Greedy + local search optimiser
├── visualization.py         # All charts and figures
├── data/                    # Raw Excel input files
└── outputs/
    ├── cleaned_data/        # Preprocessed CSVs (shared across runs)
    └── run_YYYYMMDD_HHMMSS/ # Per-run results, logs, and figures
```

## Requirements

- Python 3.9+
- `pandas`, `numpy`, `matplotlib`, `seaborn`, `openpyxl`
- Python Xpress (for MIP model only)

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn openpyxl xpress
```

## Usage

```bash
# Full pipeline
python main.py

# Skip MIP (heuristic only — no Xpress needed)
python main.py --skip-mip

# Skip conflict pair rebuild (faster re-runs)
python main.py --skip-conflicts

# Limit MIP problem size and solver time
python main.py --mip-events 200 --mip-time 120
```

## Output Files

Each run creates a timestamped folder under `outputs/`. All outputs are also on Google Drive.

```
outputs/run_YYYYMMDD_HHMMSS/
├── pipeline_log_*.txt                    # Full console log
├── figures/
│   ├── event_heatmap_*.png
│   ├── clash_comparison.png
│   └── ...
├── baseline_timeslot_utilisation.csv     # Event count per (day, hour) — baseline
├── baseline_room_utilisation.csv         # Per-room fill rate — baseline
├── scenario_displaced_summary.csv        # Displaced events per scenario & day
├── noslot_feasibility.csv                # Q1/Q2: events too long to fit any slot
├── hourly_load_comparison.csv            # Q5: event load by (scenario, day, hour)
├── lunch_break_analysis.csv              # Q4: lunch-free student % per scenario
├── clash_analysis.csv                    # Q3: clash pairs & severity (WC/SG)
├── utilisation_comparison.csv            # Q5: room utilisation across scenarios
├── mip_summary_S1_9am5pm.csv            # MIP results — S1
├── mip_summary_S2_NoFriPM.csv           # MIP results — S2
├── heuristic_summary_S1_9am5pm.csv      # Heuristic results — S1
├── heuristic_summary_S2_NoFriPM.csv     # Heuristic results — S2
└── final_summary.csv                     # Consolidated Q1–Q5 summary
```

Preprocessed data is saved once to `outputs/cleaned_data/` and reused across runs.

## Key Metrics by Research Question

### Q1 / Q2 — Feasibility of reducing teaching window

| Metric | Source | Description |
|--------|--------|-------------|
| `Displaced_Events` / `Displaced_Pct` | `scenario_displaced_summary.csv` | Events falling outside the new window |
| `Displaced_WholeClass` | `scenario_displaced_summary.csv` | Core (whole-class) events affected |
| `N_NoSlot` / `NoSlot_Pct` | `noslot_feasibility.csv` | Events whose duration exceeds the window (S1: >480 min; S2: >540 min) — **cannot be scheduled at all** |
| `N_NoSlot_WholeClass` | `noslot_feasibility.csv` | Whole-class events with no valid slot |
| `N_NoSlot` (heuristic) | `heuristic_summary_*.csv` | Confirmed NO_SLOT count after running the optimiser |

### Q3 — Clash-free delivery of core teaching

| Metric | Source | Description |
|--------|--------|-------------|
| `severity_3_WC_WC` | `clash_analysis.csv` | Whole-class vs whole-class clash pairs (most severe) |
| `student_clash_pct` | `clash_analysis.csv` | % of students with at least one clash |
| `LocalSearch_Clash_Score` | `heuristic_summary_*.csv` | Weighted clash score after optimisation |
| `Greedy_ClashFree_WholeClass` / `_Pct` | `heuristic_summary_*.csv` | Whole-class events placed without any clash (greedy phase) |
| `N_Clash_Pairs_After_MIP` | `mip_summary_*.csv` | Residual clash pairs after exact optimisation |

### Q4 — Lunch break (12–2pm) for majority of students

| Metric | Source | Description |
|--------|--------|-------------|
| `Lunch_Free_Pct` | `lunch_break_analysis.csv` | % of students free 12–2pm (baseline & both scenarios) |
| `Free_{Day}` | `lunch_break_analysis.csv` | Per-day lunch-free percentage |
| `Lunch_Free_Pct` (heuristic) | `heuristic_summary_*.csv` | Lunch-free % after rescheduling displaced events |

### Q5 — Timeslot & room utilisation

| Metric | Source | Description |
|--------|--------|-------------|
| `Num_Events` by (Day, Hour) | `hourly_load_comparison.csv` | Hour-by-hour event load across all three scenarios |
| `Room_Utilisation_Pct` | `utilisation_comparison.csv` | % of available room-slots occupied per scenario |
| `Slot_Balance_CV` | `heuristic_summary_*.csv` | Coefficient of variation of slot load (lower = more balanced) |
| `Rescheduled_{Day}` | `heuristic_summary_*.csv` | How many displaced events land on each day after rescheduling |
