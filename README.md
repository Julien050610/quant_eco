<p align="center">
  <h1 align="center">HMM Regime Detector</h1>
  <p align="center">
    <em>隐马尔可夫市场状态感知雷达 — 零未来泄露的量化交易状态机</em>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/python-%3E%3D3.10-blue?logo=python&logoColor=white" alt="Python 3.10+">
    <img src="https://img.shields.io/badge/hmmlearn-%3E%3D0.3.0-orange" alt="hmmlearn">
    <img src="https://img.shields.io/badge/build-passing-brightgreen" alt="Build Status">
    <img src="https://img.shields.io/badge/license-MIT-green" alt="License MIT">
    <img src="https://img.shields.io/badge/status-stable-blueviolet" alt="Status Stable">
    <img src="https://img.shields.io/badge/last_updated-2026--05-brightgreen" alt="Last Updated">
  </p>
</p>

---

## 📋 项目简介

**HMM Regime Detector** 是一个基于隐马尔可夫模型（GaussianHMM）的市场状态感知模块，能够将任意资产的 OHLCV 时序数据自动划分为**恐慌态 (PANIC_OFF)**、**震荡态 (OBSERVE)** 和 **趋势态 (TREND_ON)** 三种系统状态，为量化交易策略提供可靠的"市场环境掩码"。

> 将你的 OHLCV 数据喂给模型，拿回三个状态开关（0/1/2），告诉你市场当前是恐慌、震荡还是趋势。

---

## 核心亮点

### 机制一：滚动窗口独立标准化 — 杜绝前视偏差

```python
# 每个 504 天窗口完全独立处理
for window in rolling_windows:
    scaler = StandardScaler()
    scaler.fit(X_train)           # ✅ 仅用训练集统计量
    X_predict = scaler.transform(X_pred)  # ✅ 从源头杜绝未来数据泄露
```

**问题**：传统方法在整个历史数据上全局标准化，第 T 天的特征被第 T+1 天及未来的统计量"偷看"。  
**解决**：每个滚动窗口内独立进行 Winsorize(1%/99%) + Z-Score 标准化，训练集独占统计参数。

### 机制二：手写纯前向递推 — 替代 predict_proba

```python
# 仅使用 t 时刻及之前的概率信息
prior = train_final_posterior
for t in range(T_test):
    prior_pred = prior @ transmat            # 状态转移预测
    log_posterior = log(prior_pred) + log_emit[t]  # 发射更新
    posterior = LogSumExp_normalize(log_posterior)
    prior = posterior                         # 递推
```

**问题**：`hmmlearn.predict_proba()` 默认执行 Forward-Backward 双向算法，后向传递从序列末尾回传信息，泄露了未来数据。  
**解决**：手动实现纯前向递推算法，仅利用到当前时刻 t 及之前的信息，从数学上消除后向泄露。

### 机制三：施密特触发器 — 消除状态闪烁

```
TREND_ON 维持: P_TREND ≥ 0.50  (宽松保持)
TREND_ON 进入: P_TREND ≥ 0.75  (严格进入)
PANIC_OFF 触发: P_PANIC ≥ 0.60
从 PANIC_OFF 恢复: 需要 P_TREND ≥ 0.75 (强信号)
```

**问题**：HMM 后验概率在阈值边界附近震荡，导致状态频繁跳变（闪烁），引发无效交易信号。  
**解决**：引入非对称迟滞比较器（Schmitt Trigger），进入和维持使用不同阈值，系统状态一旦锁定就不会被轻微波动干扰，保证状态切换的稳定性与可解释性。

### Bug 修复历史（V2.1 已全部修复）

| Bug | 问题 | 根因 | 修复方式 |
|-----|------|------|---------|
| **#1 状态机死锁** | 97.6% 输出 OBSERVE，几乎不开仓 | hmmlearn 状态标签随机初始化，固定映射错位 | 按 F1 对数收益率高斯均值**动态排序重对齐** |
| **#2 数值下溢死锁** | 状态锁死无法跃迁 | 长序列累乘概率 → log(0) = -inf → NaN 传播 | 引入 `EPS=1e-300` 兜底 + LogSumExp 重归一化 |

---

## 数据契约 (Data Contract)

### 输入格式

模型期望接收一个 **`{symbol: DataFrame}` 字典**，每个 DataFrame 必须包含以下列：

| 列名 | 类型 | 必须 | 说明 |
|------|------|------|------|
| `open` | float64 | ✅ | 开盘价 |
| `high` | float64 | ✅ | 最高价 |
| `low`  | float64 | ✅ | 最低价 |
| `close`| float64 | ✅ | 收盘价 |
| `volume`| int64/float64 | ✅ | 成交量 |

**索引要求**：所有 DataFrame 必须共享 **`DatetimeIndex`**（交易日级别），模型会自动对齐公共交易日。

### 输出格式

| 字段 | 类型 | 值范围 | 说明 |
|------|------|--------|------|
| `index` | `DatetimeIndex` | — | 交易日日期 |
| `regime_action` | `int32` | **0 / 1 / 2** | 市场状态编码 |

### 状态编码表

| 编码 | 名称 | 含义 | 对策略影响 |
|:----:|------|------|-----------|
| **0** | `PANIC_OFF` | 恐慌关闭 | 强制平仓 + 禁止开仓 |
| **1** | `OBSERVE`  | 观察等待 | 正常交易，劫持阈值 × 1.2 |
| **2** | `TREND_ON` | 趋势开启 | 正常交易，默认参数 |

> 输出经过严格的 `_validate_output()` 自动化校验：类型、索引、列名、值域、缺失值五大维度全覆盖。

---

## 快速启动

### 前置条件

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) 或 [Anaconda](https://www.anaconda.com/)（推荐 Python 3.10+）

### 步骤一：创建 Conda 环境

```bash
# 从 environment.yml 创建隔离环境
conda env create -f environment.yml

# 激活环境
conda activate hmm_detector_env
```

### 步骤二：运行示例脚本

```bash
# 生成合成 OHLCV 数据 → 运行完整 HMM 流程 → 输出状态结果
python demo_universal.py
```

预计终端输出（部分示意）：

```
HMM Regime Detector — Universal Demo
========================================
[Step 1] Generating synthetic OHLCV data...
[Step 2] Initializing HMMRegimeRadar...
[Step 3] Computing 4-dim cross-sectional features...
[Step 4] Running rolling-window HMM training...
[Step 5] Applying hysteresis comparator...

RESULTS
  Regime actions (last 20 days):
  0 = PANIC_OFF | 1 = OBSERVE | 2 = TREND_ON
  ...
  State distribution:
    PANIC_OFF (0):   123  (23.5%)
    OBSERVE   (1):   288  (55.0%)
    TREND_ON  (2):   113  (21.5%)
```

### 在你的代码中使用

```python
import numpy as np
import pandas as pd
from hmm_regime_radar import HMMRegimeRadar

# 1. 准备你的 OHLCV 数据（字典格式）
panel_dict = {
    "AAPL": ohlcv_df_aapl,   # 含 open/high/low/close/volume 列
    "MSFT": ohlcv_df_msft,   # DatetimeIndex 索引
    # ... 至少 1~3 个以上标的效果更佳
}

# 2. 创建 HMM 雷达实例
radar = HMMRegimeRadar(
    etf_pool=list(panel_dict.keys()),
    window=504,   # 训练窗口（≈2年）
    step=20,      # 滚动步长（≈1个月）
)

# 3. 计算特征 → 训练 → 输出状态
feature_df = radar._compute_features(panel_dict)
proba_df = radar._rolling_hmm_train(feature_df)
regime_series = radar._hysteresis_comparator(proba_df)

# 4. 查看结果
print(regime_series.tail(20))
print(radar.get_state_summary())
```

---

## 目录结构

```
hmm-regime-detector/
├── hmm_regime_radar.py      # 核心模块（仅依赖 numpy/pandas/hmmlearn/sklearn）
├── demo_universal.py        # 通用示例脚本（合成数据 + 完整流程演示）
├── environment.yml          # Conda 环境隔离文件
└── README.md                # 项目文档（就是本文件）
```

---

## 依赖清单

| 依赖 | 版本要求 | 作用 |
|------|---------|------|
| `python` | ≥ 3.10 | 运行环境 |
| `numpy` | ≥ 1.23 | 向量化数值计算 |
| `pandas` | ≥ 1.5  | 时序数据处理 |
| `scikit-learn` | ≥ 1.2 | StandardScaler 标准化 |
| `hmmlearn` | ≥ 0.3.0 | GaussianHMM 隐马尔可夫模型 |

---

## License

本项目采用 **MIT License** 开源。欢迎 Fork、Star、Issue 和 Pull Request！

---

<p align="center">
  <sub>Wish to help the community with the code</sub>
</p>
