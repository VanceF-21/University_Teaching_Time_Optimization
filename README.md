# University Teaching Time Optimization

[![中文文档](https://img.shields.io/badge/文档-中文版-FF4B4B?style=for-the-badge&logo=googletranslate&logoColor=white)](./README_zh.md)
[![Google Drive](https://img.shields.io/badge/Outputs-Google%20Drive-4285F4?style=for-the-badge&logo=googledrive&logoColor=white)](https://drive.google.com/drive/folders/1tPd5ysSP1J02JN9CroGGa4z9xvUDSTTO?usp=drive_link)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FICO Xpress](https://img.shields.io/badge/Solver-FICO%20Xpress-FF6600?style=for-the-badge&logo=databricks&logoColor=white)](https://www.fico.com/en/products/fico-xpress-optimization)

Analytical pipeline for evaluating timetable restructuring scenarios at university.

## Scenarios

| ID | Description |
|----|-------------|
| S1 | Restrict all teaching to Mon–Fri, 9am–5pm |
| S2 | Eliminate Friday teaching after 12pm |

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
- [FICO Xpress](https://www.fico.com/en/products/fico-xpress-optimization) (for MIP model only)

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn openpyxl
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

## Output

Each run creates a timestamped folder under `outputs/`:

```
outputs/run_20260226_143022/
├── pipeline_log_20260226_143022.txt
├── figures/
│   ├── event_heatmap_*.png
│   ├── clash_comparison.png
│   └── ...
├── baseline_timeslot_utilisation.csv
├── mip_summary_S1_9am5pm.csv
├── heuristic_summary_S2_NoFriPM.csv
└── final_summary.csv
```

Preprocessed data is saved once to `outputs/cleaned_data/` and reused across runs.
