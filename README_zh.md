# 大学教学时间优化

[![English](https://img.shields.io/badge/Docs-English-FF4B4B?style=for-the-badge&logo=googletranslate&logoColor=white)](./README.md)
[![Google Drive](https://img.shields.io/badge/输出文件-Google%20Drive-4285F4?style=for-the-badge&logo=googledrive&logoColor=white)](https://drive.google.com/drive/folders/1tPd5ysSP1J02JN9CroGGa4z9xvUDSTTO?usp=drive_link)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FICO Xpress](https://img.shields.io/badge/求解器-FICO%20Xpress-FF6600?style=for-the-badge&logo=databricks&logoColor=white)](https://www.fico.com/en/products/fico-xpress-optimization)

针对英国大学课表调整方案的分析流水线。

## 研究场景

| 编号 | 说明 |
|------|------|
| S1 | 将所有教学活动限制在周一至周五 9:00–17:00 |
| S2 | 取消周五 12:00 之后的教学活动 |

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
- [FICO Xpress](https://www.fico.com/en/products/fico-xpress-optimization)（仅 MIP 模型需要）

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

## 输出说明

每次运行在 `outputs/` 下生成一个带时间戳的子文件夹：

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

预处理数据保存在 `outputs/cleaned_data/`，跨运行复用，无需重复生成。
