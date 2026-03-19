# 大学教学时间优化

[![English](https://img.shields.io/badge/Docs-English-FF4B4B?style=for-the-badge&logo=googletranslate&logoColor=white)](./README.md)
[![Google Drive](https://img.shields.io/badge/输出文件-Google%20Drive-4285F4?style=for-the-badge&logo=googledrive&logoColor=white)](https://drive.google.com/drive/folders/1tPd5ysSP1J02JN9CroGGa4z9xvUDSTTO?usp=drive_link)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FICO Xpress](https://img.shields.io/badge/求解器-FICO%20Xpress-FF6600?style=for-the-badge&logo=databricks&logoColor=white)](https://www.fico.com/en/products/fico-xpress-optimization)

针对英国大学课表调整方案的分析流水线 [第17组]。

## 研究问题

| 编号 | 问题 |
|------|------|
| Q1 | 能否将核心教学时间缩减至周一至周五 9:00–17:00？ |
| Q2 | 能否取消周五 12:00 之后的所有教学活动？ |
| Q3 | 在各方案下，核心教学/讲座能否实现无冲突排课？ |
| Q4 | 能否为大多数学生提供 12:00–14:00 的午休时段？ |
| Q5 | 以上调整将如何影响全周的时间槽与教室利用率？ |

## 研究场景

| 编号 | 说明 | 对应研究问题 |
|------|------|-------------|
| S1 | 将所有教学活动限制在周一至周五 9:00–17:00 | Q1、Q3、Q4、Q5 |
| S2 | 取消周五 12:00 之后的教学活动 | Q2、Q3、Q4、Q5 |

## 项目结构

```
.
├── main.py                  # 流水线入口
├── data_preprocessing.py    # 加载与清洗 Excel 数据，构建冲突对
├── baseline_analysis.py     # 当前课表的描述性统计分析
├── mip_model.py             # FICO Xpress 整数规划模型
├── heuristic_model.py       # 贪心 + 局部搜索启发式算法
├── visualization.py         # 图表生成
├── data/                    # 原始 Excel 输入文件
└── outputs/
    ├── cleaned_data/        # 预处理后的 CSV（跨运行共享）
    └── run_YYYYMMDD_HHMMSS/ # 每次运行的结果、日志与图表
```

## 环境要求

- Python 3.9+
- `pandas`、`numpy`、`matplotlib`、`seaborn`、`openpyxl`
- Python Xpress（仅 MIP 模型需要）

安装依赖：

```bash
pip install pandas numpy matplotlib seaborn openpyxl
```

## 使用方法

```bash
# 完整流水线
python main.py

# 跳过 MIP（仅运行启发式，无需 Xpress）
python main.py --skip-mip

# 跳过冲突对构建（加速重复运行）
python main.py --skip-conflicts

# 限制 MIP 规模与求解时间
python main.py --mip-events 200 --mip-time 120
```

## 输出文件说明

每次运行在 `outputs/` 下生成一个带时间戳的子文件夹，所有输出同步至 Google Drive。

```
outputs/run_YYYYMMDD_HHMMSS/
├── pipeline_log_*.txt                    # 完整控制台日志
├── figures/
│   ├── event_heatmap_*.png
│   ├── clash_comparison.png
│   └── ...
├── baseline_timeslot_utilisation.csv     # 基线每（天、时）的事件数
├── baseline_room_utilisation.csv         # 基线各教室填充率
├── scenario_displaced_summary.csv        # 各场景及各天的被移位事件统计
├── noslot_feasibility.csv                # Q1/Q2：超出时窗无法安排的事件
├── hourly_load_comparison.csv            # Q5：各场景逐小时事件负载
├── lunch_break_analysis.csv              # Q4：各场景午休空闲学生比例
├── clash_analysis.csv                    # Q3：冲突对数量与严重性分级
├── utilisation_comparison.csv            # Q5：各场景教室利用率对比
├── mip_summary_S1_9am5pm.csv            # MIP 结果 — S1
├── mip_summary_S2_NoFriPM.csv           # MIP 结果 — S2
├── heuristic_summary_S1_9am5pm.csv      # 启发式结果 — S1
├── heuristic_summary_S2_NoFriPM.csv     # 启发式结果 — S2
└── final_summary.csv                     # Q1–Q5 综合汇总
```

预处理数据保存在 `outputs/cleaned_data/`，跨运行复用，无需重复生成。

## 各研究问题对应指标

### Q1 / Q2 — 缩减教学时窗的可行性

| 指标 | 来源文件 | 说明 |
|------|---------|------|
| `Displaced_Events` / `Displaced_Pct` | `scenario_displaced_summary.csv` | 落在新时窗外的事件数及占比 |
| `Displaced_WholeClass` | `scenario_displaced_summary.csv` | 受影响的整班课（核心课）数量 |
| `N_NoSlot` / `NoSlot_Pct` | `noslot_feasibility.csv` | 时长超出时窗、**无法安排进任何槽位**的事件（S1: >480分钟；S2: >540分钟） |
| `N_NoSlot_WholeClass` | `noslot_feasibility.csv` | 其中整班课的数量 |
| `N_NoSlot`（启发式） | `heuristic_summary_*.csv` | 经优化器实际确认的无法安排事件数 |

### Q3 — 核心教学能否无冲突排课

| 指标 | 来源文件 | 说明 |
|------|---------|------|
| `severity_3_WC_WC` | `clash_analysis.csv` | 整班课 vs 整班课的冲突对数（最严重） |
| `student_clash_pct` | `clash_analysis.csv` | 存在冲突的学生比例 |
| `LocalSearch_Clash_Score` | `heuristic_summary_*.csv` | 优化后的加权冲突分数 |
| `Greedy_ClashFree_WholeClass` / `_Pct` | `heuristic_summary_*.csv` | 贪心阶段零冲突安置的整班课数及比例 |
| `N_Clash_Pairs_After_MIP` | `mip_summary_*.csv` | 精确优化后的残余冲突对数 |

### Q4 — 为大多数学生提供 12–14 时午休

| 指标 | 来源文件 | 说明 |
|------|---------|------|
| `Lunch_Free_Pct` | `lunch_break_analysis.csv` | 午休空闲学生比例（基线及两个场景） |
| `Free_{Day}` | `lunch_break_analysis.csv` | 各天的午休空闲比例 |
| `Lunch_Free_Pct`（启发式） | `heuristic_summary_*.csv` | 重排被移位事件后的午休空闲比例 |

### Q5 — 时间槽与教室利用率变化

| 指标 | 来源文件 | 说明 |
|------|---------|------|
| `Num_Events` 按（天、时） | `hourly_load_comparison.csv` | 三个场景下各小时的事件数分布 |
| `Room_Utilisation_Pct` | `utilisation_comparison.csv` | 各场景可用教室槽位的占用率 |
| `Slot_Balance_CV` | `heuristic_summary_*.csv` | 槽位负载变异系数（越低越均衡） |
| `Rescheduled_{Day}` | `heuristic_summary_*.csv` | 重排后各天承接的被移位事件数 |
