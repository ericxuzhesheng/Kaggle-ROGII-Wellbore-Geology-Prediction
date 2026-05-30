# Kaggle ROGII 井筒地质预测 | Kaggle ROGII Wellbore Geology Prediction

<p align="center">
  <a href="#zh"><img src="https://img.shields.io/badge/语言-中文-E84D3D?style=for-the-badge&labelColor=3B3F47" alt="中文"></a>
  &nbsp;
  <a href="#en"><img src="https://img.shields.io/badge/Language-English-2F73C9?style=for-the-badge&labelColor=3B3F47" alt="English"></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/任务-TVT预测-F2C94C?style=for-the-badge" alt="TVT Prediction">
  <img src="https://img.shields.io/badge/框架-残差锚GBDT-7AC943?style=for-the-badge" alt="Residual-Anchor GBDT">
  <img src="https://img.shields.io/badge/最佳公榜-RMSE 9.791-9B51E0?style=for-the-badge" alt="Best Public LB">
</p>

---

<a id="zh"></a>

## 中文

### 项目概览

本仓库记录 **Kaggle ROGII 井筒地质预测** 竞赛的完整解决方案。任务目标是预测水平井隐藏区间的 **TVT（真实垂直厚度）** 值，并以 `id,tvt` 格式提交包含 14,151 行预测结果的 `submission.csv`。

解决方案从保守的 LightGBM 基线出发，历经九个版本的迭代，逐步演化为具备正则化、防泄漏、多模型集成特性的鲁棒预测体系。核心方法为**残差锚 GBDT 框架**：模型预测 `TVT − last_known_TVT_input` 残差，再通过最后已知锚点重建最终 TVT，而非直接回归原始目标值。

---

### 核心创新

**残差锚设计**：将目标分解为"最后已知锚点 + 学习残差"两部分。锚点通过已揭露的 TVT 上下文捕获大部分信号，残差部分则由模型学习水平井隐藏区间的轨迹驱动偏移，大幅降低标签噪声并防止间接泄漏。

其他关键设计原则：

- **井级别 GroupKFold**：使用打乱顺序的井分组交叉验证，避免行级别随机切分导致的乐观估计
- **`alpha × tau` 后处理**：基于最后已知 TVT 的锚式后处理，不引入 Savitzky-Golay 平滑或 `w_pf` 混合搜索等脆弱技巧
- **对抗验证**：通过 `scripts/adversarial_validation.py` 监测训练集与测试集分布偏移，绝对坐标为强分离特征，通过正则化和集成多样性控制而非直接排除
- **集成多样性**：LightGBM、XGBoost、CatBoost 三模型集成，而非单一重模型

---

### Notebook 演进路径

| Notebook | 说明 | 公榜 RMSE |
|---|---|---:|
| `kaggle_kernel_v1_lightgbm.ipynb` | 保守 LightGBM 基线 | 27.247 |
| `kaggle_kernel_v3_residual_lgbm.ipynb` | 首个残差锚 Notebook | 14.527 |
| `kaggle_kernel_v4_residual_optimized.ipynb` | 门控混合、残差裁剪、种子集成 | 13.764 |
| `kaggle_kernel_v5_lgb_xgb_residual.ipynb` | LightGBM + XGBoost 残差集成 | 10.013 |
| `kaggle_kernel_v6_regularized_robust_ensemble.ipynb` | 抗过拟合主线；打乱 GroupKFold、CatBoost 多样性 | **9.791** |
| `kaggle_kernel_v7_full_retrain.ipynb` | 三项泄漏修复、Sakoe-Chiba DTW 特征、全量重训 | — |
| `kaggle_kernel_v8_robust.ipynb` | 日志驱动鲁棒性修复；分层 GroupKFold、更宽 pp 网格 | 10.143 |
| `kaggle_kernel_v9_no_dtw.ipynb` | 移除 DTW（v8 Nelder-Mead 确认 LGB 权重 ≈ 0） | — |

> 公榜覆盖约 26% 测试数据，最终结果使用另外 74%。模型选择以井级别 OOF 和隐藏区间诊断为主，公榜仅作提交可行性的完整性核查。

---

### 验证体系

- **分组键**：文件名井 ID
- **切分器**：打乱顺序的 `GroupKFold`（井级别，非行级别）
- **评估指标**：RMSE
- **对抗验证**：`scripts/adversarial_validation.py` 诊断训练/测试分布偏移
- **核心原则**：公榜结果仅作完整性核查，优先使用井级别 OOF 和隐藏区间诊断选模型

---

### 仓库结构

```
kaggle_kernel_v*.ipynb      # Kaggle 提交 Notebook（自包含，无网络调用）
scripts/
  _build_v*.py              # Notebook 构建器（从 Python 源生成 notebook JSON）
  adversarial_validation.py # 训练/测试分布偏移诊断
  train_baseline.py         # 保守本地基线训练
  make_submission.py        # 基线提交文件生成
src/
  data_utils.py             # 数据加载工具函数
docs/
  competition_notes.md      # 竞赛相关备注
```

原始竞赛数据（`data/`）不提交至仓库。请从 [Kaggle 竞赛页面](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/data) 下载，放置于 `data/train/`、`data/test/` 和 `data/sample_submission.csv`。

---

### Kaggle 提交流程

所有最终提交通过重新运行 Kaggle Notebook 生成，不直接上传本地 CSV。

1. 打开目标 Notebook（推荐 `kaggle_kernel_v6_regularized_robust_ensemble.ipynb`）
2. 使用 **Add Input** 挂载竞赛数据集
3. 运行所有单元格 —— 最后一个单元格会断言 `submission.csv` 已写入 `/kaggle/working/`
4. **Save Version → Save & Run All**，然后提交 Notebook 输出

Notebook 从 `/kaggle/input/rogii-wellbore-geology-prediction/`（或自动检测含 `sample_submission.csv` 的挂载目录）读取数据，运行时不发起任何网络请求。

**Kaggle 代码约束（强制）**：CPU/GPU 运行时 ≤ 9 小时；提交时禁止访问互联网；外部数据仅通过 Kaggle 数据集附加；输出文件必须命名为 `submission.csv`。

---

### 快速开始

**环境配置**

```bash
pip install -r requirements.txt
```

**本地脚本**

```bash
python scripts/inspect_data.py       # 数据概览
python scripts/run_eda.py            # 探索性数据分析
python scripts/train_baseline.py     # 本地基线训练
python scripts/make_submission.py    # 生成基线提交文件
```

---

<a id="en"></a>

## English

### Project Overview

This repository documents the complete solution for the **Kaggle ROGII Wellbore Geology Prediction** competition. The task is to predict hidden **TVT (True Vertical Thickness)** values for horizontal-well intervals and produce a `submission.csv` in `id,tvt` format against 14,151 test rows.

The solution progresses through nine notebook versions, evolving from a conservative LightGBM baseline into a regularized, leakage-aware, multi-model ensemble. The primary approach is the **residual-anchor GBDT framework**: models predict the residual `TVT − last_known_TVT_input` and reconstruct the final TVT from the last known anchor, rather than regressing the raw target directly.

---

### Key Innovation

**Residual-anchor design**: the target is decomposed into a last-known anchor plus a learned residual. The anchor captures most of the signal from the revealed TVT context, while the residual captures trajectory-driven deviation into the hidden interval. This substantially reduces label noise and prevents indirect leakage through naive use of revealed TVT inputs.

Additional design principles:

- **Well-level shuffled `GroupKFold`**: realistic cross-validation by well group — no row-level random splits
- **`alpha × tau` post-processing**: anchor-based post-processing anchored to the last known TVT, without fragile tricks like Savitzky-Golay smoothing or `w_pf` blend search
- **Adversarial validation**: `scripts/adversarial_validation.py` monitors train/test distribution shift; absolute coordinates are strong separating features, controlled by regularization and ensemble diversity rather than excluded
- **Ensemble diversity**: LightGBM, XGBoost, and CatBoost combined for diversity rather than a single heavy model

---

### Notebook Progression

| Notebook | Description | Public LB RMSE |
|---|---|---:|
| `kaggle_kernel_v1_lightgbm.ipynb` | Conservative LightGBM baseline | 27.247 |
| `kaggle_kernel_v3_residual_lgbm.ipynb` | First residual-anchor notebook | 14.527 |
| `kaggle_kernel_v4_residual_optimized.ipynb` | Gated blending, residual clipping, seed ensemble | 13.764 |
| `kaggle_kernel_v5_lgb_xgb_residual.ipynb` | LightGBM + XGBoost residual ensemble | 10.013 |
| `kaggle_kernel_v6_regularized_robust_ensemble.ipynb` | Anti-overfitting line; shuffled GroupKFold, CatBoost diversity | **9.791** |
| `kaggle_kernel_v7_full_retrain.ipynb` | Three leakage fixes, Sakoe-Chiba DTW features, full-data retrain | — |
| `kaggle_kernel_v8_robust.ipynb` | Log-driven robustness fixes; stratified GroupKFold, wider pp grid | 10.143 |
| `kaggle_kernel_v9_no_dtw.ipynb` | DTW removed after v8 Nelder-Mead confirmed LGB weight ≈ 0 | — |

> The Public LB covers approximately 26% of test data; hidden-LB generalization is the primary optimization target. Public LB is used only as a notebook-rerun sanity check.

---

### Validation Design

- **Group key**: filename well ID
- **Splitter**: shuffled `GroupKFold` (well-level, not row-level)
- **Metric**: RMSE
- **Adversarial validation**: `scripts/adversarial_validation.py` diagnoses train/test drift; absolute coordinates strongly separate train and test and are controlled by regularization rather than excluded
- **Guiding principle**: Public LB is a rerun sanity check only — prefer well-level OOF and hidden-interval diagnostics for model selection

---

### Repository Structure

```
kaggle_kernel_v*.ipynb      # Kaggle submission notebooks (self-contained, no internet)
scripts/
  _build_v*.py              # Notebook builders (emit notebook JSON from Python source)
  adversarial_validation.py # Train/test distribution diagnostic
  train_baseline.py         # Conservative offline baseline
  make_submission.py        # Baseline submission generator
src/
  data_utils.py             # Shared data loading utilities
docs/
  competition_notes.md      # Competition-specific notes
```

Raw competition data (`data/`) is not committed. Download from the [Kaggle competition page](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/data) and place under `data/train/`, `data/test/`, and `data/sample_submission.csv`.

---

### Kaggle Submission Workflow

All final submissions are generated by rerunning a Kaggle Notebook — not by uploading a local CSV directly.

1. Open the target notebook (recommended: `kaggle_kernel_v6_regularized_robust_ensemble.ipynb`)
2. Use **Add Input** to attach the competition dataset
3. Run all cells — the final cell asserts `submission.csv` is written to `/kaggle/working/`
4. **Save Version → Save & Run All**, then submit the notebook output

Notebooks read from `/kaggle/input/rogii-wellbore-geology-prediction/` (or auto-detect a mounted directory containing `sample_submission.csv`) and never make network calls at run-time.

**Kaggle Code Requirements (hard constraints)**: CPU/GPU run-time ≤ 9 hours; internet access disabled at submission; external data only via attached Kaggle datasets; output file must be named `submission.csv`.

---

### Quick Start

**Environment setup**

```bash
pip install -r requirements.txt
```

**Local scripts**

```bash
python scripts/inspect_data.py       # Data overview
python scripts/run_eda.py            # Exploratory data analysis
python scripts/train_baseline.py     # Local baseline training
python scripts/make_submission.py    # Generate baseline submission
```

---

### References

- Competition: [ROGII Wellbore Geology Prediction](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction)
- Modeling policy: `CLAUDE.md`, `AGENTS.md`
