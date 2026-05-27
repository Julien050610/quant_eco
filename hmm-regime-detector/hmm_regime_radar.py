"""
================================================================================
模块 1.5 — 隐马尔可夫市场状态感知雷达 (HMM Regime Radar)  V2.1
================================================================================
核心定位：
  作为现有量化系统的"市场状态感知层"，独立运行于模块 1 (Data Pipeline)
  与模块 2 (Signal Generator) 之间，输出干净的系统状态掩码。

关键改进（V2.1）：
  - Bug #1 修复：训练集前向递推中使用对齐后的对数似然函数，消除了因
    标签映射错乱导致的状态机卡死问题。
  - Bug #2 修复：在 _forward_algorithm 中引入 EPS 兜底，防止 log(0) = -inf
    导致的数值下溢和状态锁死。

依赖：
  - hmmlearn (pip install hmmlearn)
  - numpy, pandas, scikit-learn
  - data_pipeline_v2.py (仅通过其公开接口获取数据)

核心机制：
  1. 截面特征中位数降维：12只ETF × 4维特征 → 横截面中位数 → N×4 全局特征
  2. 滚动HMM训练（窗口504天/步长20天），窗口内独立 fit→transform 标准化
  3. 纯前向递推计算样本外后验概率，而非 predict_proba()
  4. 状态强制重对齐：按F1高斯均值排序
  5. 迟滞比较器（施密特触发器）：非对称阈值避免状态闪烁

输出契约：
  - 类型: pd.Series, index=DatetimeIndex, values='regime_action' (int: 0/1/2)
  - 0=PANIC_OFF(恐慌关闭), 1=OBSERVE(观察), 2=TREND_ON(趋势开启)
================================================================================
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from data_pipeline_v2 import MultiETFDataPipeline


# =============================================================================
# 常量定义
# =============================================================================

# 默认ETF标的池（12只核心ETF，与Data Pipeline保持一致）
DEFAULT_ETF_POOL: List[str] = [
    "510300", "159915", "588000", "510500",
    "512480", "515980", "515700",
    "512880", "512800",
    "515220", "512690", "512170",
]

# HMM 参数
HMM_N_COMPONENTS: int = 3          # 状态数：0=恐慌态, 1=震荡态, 2=趋势态
HMM_COVARIANCE_TYPE: str = "full"  # 协方差类型
HMM_N_ITER: int = 1000             # EM最大迭代次数
HMM_RANDOM_STATE: int = 42         # 随机种子（保证可复现）

# 滚动窗口参数
ROLLING_WINDOW: int = 504          # 窗口大小（交易日 ≈ 2年）
ROLLING_STEP: int = 20             # 滚动步长（约1个月）

# ATR 计算参数
ATR_PERIOD: int = 14

# 效率系数参数
ER_PERIOD: int = 14

# 量能移动平均参数
VOLUME_MA_PERIOD: int = 14

# 极值截断参数（1%-99% Winsorize）
WINSORIZE_LOWER: float = 0.01
WINSORIZE_UPPER: float = 0.99

# 迟滞比较器阈值
# 进入趋势态
TREND_ENTER_THRESHOLD: float = 0.75
# 维持趋势态
TREND_HOLD_THRESHOLD: float = 0.50
# 恐慌态触发
PANIC_TRIGGER_THRESHOLD: float = 0.60

# 动作指令编码
PANIC_OFF: int = 0   # 恐慌关闭（禁止交易）
OBSERVE: int = 1     # 观察（灰度地带）
TREND_ON: int = 2    # 趋势开启（允许交易）

# 状态索引映射（重对齐后）
STATE_PANIC: int = 0
STATE_OSCILLATE: int = 1
STATE_TREND: int = 2

# 前向算法的数值稳定性常量
LOG_SUM_EXP_MIN: float = -745.0  # exp(-745) ≈ 5e-324，接近 float64 下溢边界
LOG_ZERO: float = -np.inf        # 表示对数概率为0
EPS: float = 1e-300              # V2.1 新增：防止 log(0) = -inf 的极小值


# =============================================================================
# Logger 配置
# =============================================================================
def _setup_logger(name: str = "HMMRegimeRadar") -> logging.Logger:
    """配置日志输出"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger


# =============================================================================
# HMMRegimeRadar 类
# =============================================================================
class HMMRegimeRadar:
    """
    隐马尔可夫市场状态感知雷达。

    功能：
      1. 从 Data Pipeline 获取12只ETF的对齐OHLCV数据
      2. 计算4维截面中位数特征（对数收益率、归一化波动率、效率系数、相对量能）
      3. 滚动窗口HMM训练 + 窗口内独立标准化 + 纯前向递推
      4. 状态标签强制重对齐 + 迟滞比较器输出稳定的交易状态掩码

    消除未来函数泄露的关键设计：
      - 特征标准化在滚动窗口内进行，每个窗口的 fit 仅使用训练集数据
      - 使用纯前向递推算法替代 predict_proba()，不依赖后向传递

    Parameters
    ----------
    etf_pool : List[str], optional
        ETF代码列表，默认12只核心ETF
    window : int, optional
        滚动窗口大小，默认504个交易日
    step : int, optional
        滚动步长，默认20个交易日
    logger : logging.Logger, optional
        外部日志实例
    """

    def __init__(
        self,
        etf_pool: Optional[List[str]] = None,
        window: int = ROLLING_WINDOW,
        step: int = ROLLING_STEP,
        logger: Optional[logging.Logger] = None,
    ):
        self.etf_pool = etf_pool or DEFAULT_ETF_POOL.copy()
        self.window = window
        self.step = step
        self.logger = logger or _setup_logger()

        # 缓存：最近一次运行的原始特征、HMM模型、重对齐映射等
        self._cached_features: Optional[pd.DataFrame] = None
        self._cached_probas: Optional[pd.DataFrame] = None
        self._cached_regime: Optional[pd.Series] = None
        self._cached_models: List = []

        self.logger.info(
            "HMMRegimeRadar V2.1 初始化完成 | ETF数量: %d | 窗口: %d | 步长: %d",
            len(self.etf_pool), self.window, self.step,
        )

    # ======================================================================
    # 向量化辅助函数（全部使用 NumPy 向量化操作，零 for 循环）
    # ======================================================================
    @staticmethod
    def _vectorized_sma(arr: np.ndarray, period: int) -> np.ndarray:
        """
        完全向量化的简单移动平均（使用滑动窗口视图，零 for 循环）。

        Parameters
        ----------
        arr : np.ndarray
            输入数组，1D
        period : int
            窗口大小

        Returns
        -------
        np.ndarray
            SMA 结果，前 period-1 个值为 NaN
        """
        out = np.full_like(arr, np.nan, dtype=np.float64)
        n = len(arr)
        if n < period:
            return out

        from numpy.lib.stride_tricks import sliding_window_view
        windows = sliding_window_view(arr, window_shape=period)
        out[period - 1:] = np.nanmean(windows, axis=1)

        return out

    @staticmethod
    def _vectorized_rolling_sum(arr: np.ndarray, period: int) -> np.ndarray:
        """
        完全向量化的滚动求和（使用 cumsum，零 for 循环）。

        Parameters
        ----------
        arr : np.ndarray
            输入数组，1D
        period : int
            窗口大小

        Returns
        -------
        np.ndarray
            滚动求和结果，前 period-1 个值为 NaN
        """
        out = np.full_like(arr, np.nan, dtype=np.float64)
        n = len(arr)
        if n < period:
            return out

        clean_arr = np.nan_to_num(arr, nan=0.0)
        cumsum = np.cumsum(clean_arr)
        out[period - 1] = cumsum[period - 1]
        out[period:] = cumsum[period:] - cumsum[:-period]

        if np.any(np.isnan(arr)):
            from numpy.lib.stride_tricks import sliding_window_view
            windows = sliding_window_view(arr, window_shape=period)
            has_nan = np.any(np.isnan(windows), axis=1)
            nan_mask = np.full(n, False)
            nan_mask[period - 1:] = has_nan
            out[nan_mask] = np.nan

        return out

    # ======================================================================
    # 第一步：特征工程与截面降维（仅原始特征，不做标准化）
    # ======================================================================
    def _compute_features(self, panel_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        计算12只ETF的4维截面中位数特征。

        【消除未来函数泄露 #1】V2.0 改动：
          本方法仅计算原始特征值（对数收益率、归一化波动率、效率系数、相对量能），
          不再包含 Winsorize 极值截断和 Z-Score 标准化。
          标准化操作已下放至 _rolling_hmm_train() 的滚动窗口循环内，
          每个训练窗口独立 fit→transform，从源头杜绝全局标准化带来的未来数据泄露。

        Parameters
        ----------
        panel_dict : Dict[str, pd.DataFrame]
            {symbol: df} 字典，每个df含open,high,low,close,volume列

        Returns
        -------
        pd.DataFrame
            N×4 全局特征矩阵，列名=['F1','F2','F3','F4']，索引为DatetimeIndex
            注意：返回的为原始特征，未经极值截断和 Z-Score 标准化
        """
        self.logger.info("开始计算截面特征（原始值，不含标准化）...")

        # ---- Step 1: 对齐所有ETF的日期索引（取共同交易日）----
        aligned_sets = [set(df.index) for df in panel_dict.values()]
        if not aligned_sets:
            raise ValueError("无任何ETF数据可用")

        aligned_dates = sorted(set.intersection(*aligned_sets))
        if not aligned_dates:
            raise ValueError("所有ETF无共同交易日，无法构建特征矩阵")

        n_dates = len(aligned_dates)
        n_etfs = len(self.etf_pool)
        n_features = 4
        date_index = pd.DatetimeIndex(aligned_dates)

        self.logger.info(
            "共同交易日数: %d | 对齐前ETF数量: %d", n_dates, n_etfs,
        )

        # ---- Step 2: 预先分配 3D 数组 (n_dates, n_etfs, n_features) ----
        features_3d = np.full((n_dates, n_etfs, n_features), np.nan, dtype=np.float64)

        # ---- Step 3: 对每只ETF全向量化计算4个特征 ----
        for i, symbol in enumerate(self.etf_pool):
            if symbol not in panel_dict:
                self.logger.warning("[%s] 数据缺失，跳过", symbol)
                continue

            df = panel_dict[symbol]
            df_aligned = df.reindex(date_index)

            close = df_aligned["close"].to_numpy(dtype=np.float64)
            high = df_aligned["high"].to_numpy(dtype=np.float64)
            low = df_aligned["low"].to_numpy(dtype=np.float64)
            volume = df_aligned["volume"].to_numpy(dtype=np.float64)

            # --- F1: 对数收益率 ln(C_t / C_{t-1}) ---
            f1 = np.full(n_dates, np.nan)
            f1[1:] = np.log(close[1:] / close[:-1])

            # --- F2: 归一化波动率 ln(ATR_14 / C_t) ---
            high_low = high[1:] - low[1:]
            high_close_prev = np.abs(high[1:] - close[:-1])
            low_close_prev = np.abs(low[1:] - close[:-1])
            tr = np.full(n_dates, np.nan)
            tr[1:] = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))

            atr = self._vectorized_sma(tr, ATR_PERIOD)
            f2 = np.full(n_dates, np.nan)
            valid_f2 = (close > 0) & (atr > 0) & (~np.isnan(atr)) & (~np.isnan(close))
            f2[valid_f2] = np.log(atr[valid_f2] / close[valid_f2])

            # --- F3: 效率系数 |C_t - C_{t-14}| / sum_{i=0}^{13} |C_{t-i} - C_{t-i-1}| ---
            numerator = np.full(n_dates, np.nan)
            numerator[ER_PERIOD:] = np.abs(close[ER_PERIOD:] - close[:-ER_PERIOD])

            abs_returns = np.abs(np.diff(close, prepend=close[0]))
            denominator = self._vectorized_rolling_sum(abs_returns, ER_PERIOD)

            f3 = np.full(n_dates, np.nan)
            valid_f3 = (denominator > 1e-12) & (~np.isnan(denominator))
            f3[valid_f3] = numerator[valid_f3] / denominator[valid_f3]

            # --- F4: 对数相对量能 ln(V_t / MA(V, 14)) ---
            vol_ma = self._vectorized_sma(volume, VOLUME_MA_PERIOD)
            f4 = np.full(n_dates, np.nan)
            valid_f4 = (volume > 0) & (vol_ma > 0) & (~np.isnan(vol_ma))
            f4[valid_f4] = np.log(volume[valid_f4] / vol_ma[valid_f4])

            # ---- 存入 3D 数组 ----
            features_3d[:, i, 0] = f1
            features_3d[:, i, 1] = f2
            features_3d[:, i, 2] = f3
            features_3d[:, i, 3] = f4

        # ---- Step 4: 横截面中位数降维（沿 ETF 轴取中位数）----
        # 忽略全NaN切片的警告（仅部分ETF数据缺失的日期会自动被忽略）
        with np.errstate(all="ignore"):
            median_features = np.nanmedian(features_3d, axis=1)  # shape: (n_dates, 4)

        # ---- Step 4b: 将 -inf / inf 替换为 NaN（前向安全）----
        median_features = np.where(np.isfinite(median_features), median_features, np.nan)

        # ---- 构建返回 DataFrame ----
        feature_df = pd.DataFrame(
            median_features,
            index=date_index,
            columns=["F1", "F2", "F3", "F4"],
        )

        self.logger.info(
            "原始截面特征计算完成 | 维度: %s | 时间范围: %s ~ %s",
            feature_df.shape,
            feature_df.index[0].strftime("%Y-%m-%d"),
            feature_df.index[-1].strftime("%Y-%m-%d"),
        )

        return feature_df

    # ======================================================================
    # 纯前向递推算法 (Forward Algorithm) — 消除 predict_proba 的后向泄露
    # ======================================================================
    @staticmethod
    def _forward_algorithm(
        X_test: np.ndarray,
        transmat: np.ndarray,
        log_likelihood_fn,
        prior_init: np.ndarray,
    ) -> np.ndarray:
        """
        纯前向递推算法 (Forward Pass) 计算样本外后验概率。

        【消除未来函数泄露 #2】
        hmmlearn 的 predict_proba() 底层默认执行全序列的前向-后向算法
        (Forward-Backward Algorithm)，其中后向传递 (Backward Pass)
        从序列末尾向开始回传信息。若将未来20天的数据整体传入 predict_proba()，
        第1天的概率结果会"偷看"第2~20天的未来数据。

        本方法手动实现纯前向递推：
          - 仅使用到当前时刻 t 及之前的信息
          - 从训练集最后一天的状态概率初始化
          - 逐天递推：状态转移预测 → 发射概率更新 → 贝叶斯归一化
          - 完全杜绝对未来数据的任何依赖

        V2.1 修复：引入 EPS=1e-300 防止 log(0) = -inf 导致状态锁死。

        Parameters
        ----------
        X_test : np.ndarray, shape (T_test, n_features)
            样本外测试期的特征数据
        transmat : np.ndarray, shape (n_states, n_states)
            HMM 状态转移矩阵
        log_likelihood_fn : callable
            函数签名 log_likelihood_fn(X) → np.ndarray (T, n_states)
            返回每个时刻每个状态的对数发射概率
        prior_init : np.ndarray, shape (n_states,)
            初始先验概率 P(S_{t-1})，即训练集最后一天的推断状态概率

        Returns
        -------
        np.ndarray, shape (T_test, n_states)
            逐天的后验状态概率矩阵，每行和为1
        """
        n_states = transmat.shape[0]
        T_test = len(X_test)

        # 获取对数发射概率矩阵 log_emission[t, s] = log P(O_t | S_t = s)
        # 使用 log_likelihood_fn 获取每个状态的对数似然
        log_emission = log_likelihood_fn(X_test)  # (T_test, n_states)

        # 后验概率存储
        posteriors = np.zeros((T_test, n_states), dtype=np.float64)

        # 当前先验：来自训练集最后一天的推断状态概率
        prior = prior_init.copy()  # P(S_{t-1})

        for t in range(T_test):
            # ---- Step 1: 状态转移预测 ----
            # prior_t_pred[s] = sum_{s_prev} P(S_{t-1}=s_prev) * transmat[s_prev, s]
            prior_pred = np.dot(prior, transmat)  # (n_states,)

            # ---- Step 2: 引入当日发射概率 ----
            # 获取当日对数发射概率 log P(O_t | S_t = s)
            log_emit_t = log_emission[t, :]       # (n_states,)

            # V2.1 修复：使用 EPS 防止 log(0) = -inf
            # 将 prior_pred 中的 0 转换为极小正值，确保 log 安全
            prior_pred_safe = np.maximum(prior_pred, EPS)
            log_prior_pred = np.log(prior_pred_safe)

            # log(posterior) = log(prior_pred) + log(emission)
            log_posterior_unnorm = log_prior_pred + log_emit_t

            # ---- Step 3: Log-Sum-Exp 数值稳定归一化 ----
            # 找到最大值做移位，防止 exp 下溢
            max_log = np.max(log_posterior_unnorm)
            if max_log == LOG_ZERO or np.isnan(max_log):
                # 完全下溢，退回均匀分布
                posterior_t = np.ones(n_states, dtype=np.float64) / n_states
            else:
                # 移位后 exp，确保不会下溢
                shifted = log_posterior_unnorm - max_log
                # 对极小值主动约束，防止 nan 传播
                shifted = np.clip(shifted, LOG_SUM_EXP_MIN, None)
                posterior_t = np.exp(shifted)
                posterior_sum = np.sum(posterior_t)
                if posterior_sum > 0:
                    posterior_t /= posterior_sum
                else:
                    # 异常兜底：均匀分布
                    posterior_t = np.ones(n_states, dtype=np.float64) / n_states

            posteriors[t, :] = posterior_t

            # ---- Step 4: 更新先验供下一递推使用 ----
            prior = posterior_t.copy()

        return posteriors

    # ======================================================================
    # 第二步：滚动 HMM 训练 + 窗口内标准化 + 纯前向递推
    # ======================================================================
    def _rolling_hmm_train(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        滚动窗口 HMM 训练，窗口内独立标准化 + 纯前向递推。

        【消除未来函数泄露 #1】窗口内独立标准化：
          在每个 504 天的训练窗口中，使用 train_data 进行 fit
          （计算分位数、均值、方差），然后使用该规则去 transform
          当期的 predict_data。同一预测期 (20天) 的所有数据共享
          相同的标准化参数，但不会用到窗口外的未来数据。

        【消除未来函数泄露 #2】纯前向递推代替 predict_proba()：
          每个窗口训练完模型后，利用训练集的 Viterbi/后向概率获取
          最后一天的推断状态作为初始先验，然后手动实现前向递推
          计算预测期每一天的后验概率。

        V2.1 修复 Bug #1：
          训练集的前向递推改用对齐后的对数似然函数，
          确保对数发射概率与 transmat_aligned 在标签顺序上严格对应。

        流程：
          1. 以 window=504, step=20 滚动切分训练集和预测集
          2. 每个窗口内：对 train_data 独立进行 Winsorize + StandardScaler fit
          3. 使用训练集的标准化规则 transform predict_data
          4. 训练 GaussianHMM(n_components=3)
          5. 按 F1（对数收益率）的高斯均值进行状态重对齐
          6. 使用纯前向递推算法计算预测期后验概率

        Parameters
        ----------
        features : pd.DataFrame
            N×4 全局特征矩阵 (列: F1, F2, F3, F4)，原始值（未标准化）

        Returns
        -------
        pd.DataFrame
            后验概率矩阵，列=['P_PANIC','P_OSCILLATE','P_TREND']，
            索引为预测期的 DatetimeIndex
        """
        from hmmlearn.hmm import GaussianHMM

        self.logger.info("开始滚动 HMM 训练（窗口内标准化 + 前向递推）...")
        self.logger.info(
            "  窗口大小: %d | 步长: %d | 总天数: %d",
            self.window, self.step, len(features),
        )

        X = features.to_numpy(dtype=np.float64)
        dates = features.index.to_numpy()
        n_total = len(X)

        # 存储所有预测期的后验概率
        proba_list: List[pd.DataFrame] = []
        model_list: List = []

        # NOTE: HMM 的 fit() 必须逐窗口调用，这是 hmmlearn 的 API 限制
        # 此处的 while 循环是框架层级的迭代，不属于数据层面的 for 循环
        start_idx = 0
        window_count = 0
        while start_idx + self.window < n_total:
            train_end = start_idx + self.window
            predict_end = min(train_end + self.step, n_total)

            # ---------------------------------------------------------------
            # 训练数据切片（原始特征，未标准化）
            # ---------------------------------------------------------------
            X_train_raw = X[start_idx:train_end].copy()
            X_predict_raw = X[train_end:predict_end].copy()

            # 跳过有 NaN 的行（训练集和预测集分别处理）
            # 训练集：去除非有限行
            finite_train_mask = np.all(np.isfinite(X_train_raw), axis=1)
            if not np.any(finite_train_mask):
                self.logger.warning(
                    "窗口 [%d:%d] 训练集全为 NaN/Inf，跳过",
                    start_idx, train_end,
                )
                start_idx += self.step
                continue

            X_train_finite = X_train_raw[finite_train_mask]

            # ---------------------------------------------------------------
            # 【消除未来函数泄露 #1】窗口内独立 Winsorize + StandardScaler
            # 仅使用 train_data 进行 fit，完全杜绝未来数据的统计信息偷看
            # ---------------------------------------------------------------
            # Step A: Winsorize 极值截断（1%-99%），仅基于训练集的分位数
            n_features = X_train_finite.shape[1]
            X_train_winsor = X_train_finite.copy()
            lower_bounds = np.zeros(n_features)
            upper_bounds = np.zeros(n_features)

            for j in range(n_features):
                col = X_train_finite[:, j]
                lower = np.nanpercentile(col, WINSORIZE_LOWER * 100)
                upper = np.nanpercentile(col, WINSORIZE_UPPER * 100)
                lower_bounds[j] = lower
                upper_bounds[j] = upper
                X_train_winsor[:, j] = np.clip(col, lower, upper)

            # Step B: Z-Score 标准化，仅基于训练集的均值/标准差
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_winsor)

            # ---------------------------------------------------------------
            # 将训练集的有限行按原始位置放回（用于 fit HMM）
            # ---------------------------------------------------------------
            X_train = np.full_like(X_train_raw, 0.0, dtype=np.float64)
            X_train[finite_train_mask] = X_train_scaled

            # 处理训练集中的非有限位：用列均值填充
            col_means = np.nanmean(X_train_scaled, axis=0)
            col_means = np.where(np.isnan(col_means), 0.0, col_means)
            X_train[~finite_train_mask] = col_means

            # ---------------------------------------------------------------
            # 使用训练集的标准化规则 transform 预测集
            # 先 Winsorize（用训练集的分位数），再 StandardScaler（用训练集的参数）
            # ---------------------------------------------------------------
            if len(X_predict_raw) > 0:
                X_predict = X_predict_raw.copy()
                pred_finite_mask = np.all(np.isfinite(X_predict), axis=1)

                if np.any(pred_finite_mask):
                    # 用训练集的分位数做极值截断
                    X_predict_finite = X_predict[pred_finite_mask]
                    X_predict_winsor = X_predict_finite.copy()
                    for j in range(n_features):
                        X_predict_winsor[:, j] = np.clip(
                            X_predict_finite[:, j],
                            lower_bounds[j],
                            upper_bounds[j],
                        )
                    # 用训练集的 scaler 做 transform
                    X_predict_scaled = scaler.transform(X_predict_winsor)
                    X_predict[pred_finite_mask] = X_predict_scaled

                # 填充非有限行
                X_predict[~pred_finite_mask] = col_means
            else:
                X_predict = np.empty((0, n_features), dtype=np.float64)

            # ---------------------------------------------------------------
            # 训练 HMM（使用窗口内标准化后的训练数据）
            # ---------------------------------------------------------------
            try:
                model = GaussianHMM(
                    n_components=HMM_N_COMPONENTS,
                    covariance_type=HMM_COVARIANCE_TYPE,
                    n_iter=HMM_N_ITER,
                    random_state=HMM_RANDOM_STATE,
                    tol=1e-4,
                )
                model.fit(X_train)
            except Exception as e:
                self.logger.error(
                    "窗口 [%d:%d] HMM 训练失败: %s", start_idx, train_end, str(e),
                )
                start_idx += self.step
                continue

            # ---------------------------------------------------------------
            # 状态强制重对齐（按 F1 高斯均值排序）
            # F1 是第 0 列特征（对数收益率）
            # ---------------------------------------------------------------
            f1_means = model.means_[:, 0]  # shape=(3,)
            sorted_indices = np.argsort(f1_means)  # 升序: 最小→0, 中间→1, 最大→2

            reorder_map = {
                sorted_indices[0]: STATE_PANIC,       # 最小均值 → 恐慌态 (State 0)
                sorted_indices[1]: STATE_OSCILLATE,    # 中间均值 → 震荡态 (State 1)
                sorted_indices[2]: STATE_TREND,        # 最大均值 → 趋势态 (State 2)
            }

            # 创建重对齐后的转移矩阵
            transmat_aligned = np.zeros_like(model.transmat_)
            for orig_label, new_label in reorder_map.items():
                for orig_label2, new_label2 in reorder_map.items():
                    transmat_aligned[new_label, new_label2] = (
                        model.transmat_[orig_label, orig_label2]
                    )

            # 构建映射的逆映射用于包装器的标签重对齐
            reorder_map_inv = {v: k for k, v in reorder_map.items()}

            # ---------------------------------------------------------------
            # 【消除未来函数泄露 #2】纯前向递推
            # V2.1 修复 Bug #1：训练集和预测集统一使用对齐后的对数似然函数
            # ---------------------------------------------------------------
            if len(X_predict) > 0:
                # 获取训练集最后一天的推断状态概率作为初始先验
                # 使用训练集最后 1 天的数据，通过模型推断状态概率
                if np.any(finite_train_mask):
                    # -------------------------------------------------------
                    # V2.1 修复：统一使用对齐后的对数似然函数
                    # 构建对齐包装器（训练集和预测集共用同一套包装器）
                    # -------------------------------------------------------
                    def _make_log_likelihood_fn(hmm_model, r_map_inv):
                        """创建重对齐后的对数发射似然函数"""
                        orig_likelihood_fn = hmm_model._compute_log_likelihood

                        def _log_likelihood_wrapper(X_data):
                            """对输入特征返回 (T, n_states) 的对数发射概率（已重对齐）"""
                            log_lik_orig = orig_likelihood_fn(X_data)  # (T, 3) 原始标签顺序
                            # 重对齐到标准标签顺序 [恐慌,震荡,趋势]
                            log_lik_aligned = np.zeros_like(log_lik_orig)
                            for orig_label, new_label in r_map_inv.items():
                                log_lik_aligned[:, new_label] = log_lik_orig[:, orig_label]
                            return log_lik_aligned

                        return _log_likelihood_wrapper

                    aligned_log_likelihood_fn = _make_log_likelihood_fn(
                        model, reorder_map_inv
                    )

                    # -------------------------------------------------------
                    # V2.1 修复：# 使用对齐后的对数似然函数计算训练集发射概率
                    # 确保与 transmat_aligned 在标签顺序上严格对应
                    # -------------------------------------------------------
                    train_log_emission_aligned = aligned_log_likelihood_fn(X_train)

                    # 计算训练集转移矩阵的平稳分布作为初始先验
                    # （因为我们不知道训练集第一天之前的状态）
                    from numpy.linalg import eig
                    eigvals, eigvecs = eig(transmat_aligned.T)
                    stationary_idx = np.argmin(np.abs(eigvals - 1.0))
                    stationary_dist = np.real(
                        eigvecs[:, stationary_idx] / np.sum(eigvecs[:, stationary_idx])
                    )
                    # 确保非负
                    stationary_dist = np.maximum(stationary_dist, 0)
                    stationary_dist /= np.sum(stationary_dist)

                    # 用平稳分布初始化，然后对训练集所有逐天做前向递推
                    # 以获得训练集最后一天的推断状态概率
                    prior = stationary_dist.copy()

                    for t_idx in range(len(X_train)):
                        prior_pred = np.dot(prior, transmat_aligned)
                        # V2.1 修复：使用 EPS 防止 log(0) = -inf
                        prior_pred_safe = np.maximum(prior_pred, EPS)
                        log_prior_pred = np.log(prior_pred_safe)
                        log_posterior_unnorm = log_prior_pred + train_log_emission_aligned[t_idx, :]

                        max_log = np.max(log_posterior_unnorm)
                        if max_log == LOG_ZERO or np.isnan(max_log):
                            posterior = np.ones(HMM_N_COMPONENTS, dtype=np.float64) / HMM_N_COMPONENTS
                        else:
                            shifted = log_posterior_unnorm - max_log
                            shifted = np.clip(shifted, LOG_SUM_EXP_MIN, None)
                            posterior = np.exp(shifted)
                            post_sum = np.sum(posterior)
                            if post_sum > 0:
                                posterior /= post_sum
                            else:
                                posterior = np.ones(HMM_N_COMPONENTS, dtype=np.float64) / HMM_N_COMPONENTS
                        prior = posterior.copy()

                    # 此时 prior 已经是训练集最后一天的推断状态概率
                    train_final_posterior = prior.copy()

                else:
                    # 异常兜底：平稳分布
                    train_final_posterior = stationary_dist.copy()

                # -----------------------------------------------------------
                # 核心：纯前向递推计算预测期后验概率（使用对齐后的似然函数）
                # -----------------------------------------------------------
                # 执行纯前向递推
                forward_probas = self._forward_algorithm(
                    X_predict,                       # 标准化后的预测期特征 (T_predict, n_features)
                    transmat_aligned,                # 重对齐后的转移矩阵 (3, 3)
                    aligned_log_likelihood_fn,        # V2.1: 使用对齐后的对数似然函数
                    train_final_posterior,            # 初始先验 (3,)
                )

                # 构建 DataFrame
                pred_dates = dates[train_end:predict_end]
                proba_df = pd.DataFrame(
                    forward_probas,
                    index=pd.DatetimeIndex(pred_dates),
                    columns=["P_PANIC", "P_OSCILLATE", "P_TREND"],
                )
                proba_list.append(proba_df)
                model_list.append(model)
                window_count += 1

            start_idx += self.step

        if not proba_list:
            raise RuntimeError("滚动HMM训练未产生任何预测结果，请检查数据长度")

        # ---- 合并所有预测片段 ----
        final_proba = pd.concat(proba_list, axis=0)

        # 去重（保留首次出现的，因为滚动窗口边界可能重叠）
        final_proba = final_proba[~final_proba.index.duplicated(keep="first")]

        # 按日期排序
        final_proba.sort_index(inplace=True)

        self._cached_models = model_list

        self.logger.info(
            "滚动 HMM 训练完成 | 预测天数: %d | 窗口数: %d | 时间范围: %s ~ %s",
            len(final_proba), window_count,
            final_proba.index[0].strftime("%Y-%m-%d"),
            final_proba.index[-1].strftime("%Y-%m-%d"),
        )

        return final_proba

    # ======================================================================
    # 第三步：迟滞比较器（施密特触发器）
    # ======================================================================
    @staticmethod
    def _hysteresis_comparator(proba_df: pd.DataFrame) -> pd.Series:
        """
        基于后验概率的迟滞比较器，输出稳定状态序列。

        施密特触发器逻辑：
          - 初始状态为 OBSERVE (1)
          - 进入 TREND_ON (2): P_TREND >= 0.75
          - 维持 TREND_ON: P_TREND >= 0.50 AND P_PANIC < 0.60
          - 进入 PANIC_OFF (0): P_PANIC >= 0.60
          - 从 PANIC_OFF 转 TREND_ON: P_TREND >= 0.75（需要强信号）
          - 默认/灰色地带: OBSERVE (1)

        Parameters
        ----------
        proba_df : pd.DataFrame
            后验概率矩阵，列含 P_PANIC, P_OSCILLATE, P_TREND

        Returns
        -------
        pd.Series
            交易状态序列，index=DatetimeIndex, values=0/1/2
        """
        P_PANIC = proba_df["P_PANIC"].to_numpy()
        P_TREND = proba_df["P_TREND"].to_numpy()
        dates = proba_df.index

        n = len(proba_df)
        actions = np.full(n, OBSERVE, dtype=np.int32)

        # 初始判定
        if n > 0:
            if P_TREND[0] >= TREND_ENTER_THRESHOLD:
                actions[0] = TREND_ON
            elif P_PANIC[0] >= PANIC_TRIGGER_THRESHOLD:
                actions[0] = PANIC_OFF
            else:
                actions[0] = OBSERVE

        # 逐日递推迟滞逻辑
        # NOTE: 这是状态机逻辑，必须逐日递推（依赖前一日状态），
        # 无法用纯向量化替代，但逻辑复杂度为 O(n)，效率可接受
        for i in range(1, n):
            current_action = actions[i - 1]

            if current_action == TREND_ON:
                # 维持 TREND_ON: P_TREND >= 0.50 且 P_PANIC < 0.60
                if P_TREND[i] >= TREND_HOLD_THRESHOLD and P_PANIC[i] < PANIC_TRIGGER_THRESHOLD:
                    actions[i] = TREND_ON
                elif P_TREND[i] >= TREND_ENTER_THRESHOLD:
                    actions[i] = TREND_ON
                else:
                    actions[i] = OBSERVE

            elif current_action == PANIC_OFF:
                # 从恐慌态恢复需要强趋势信号
                if P_TREND[i] >= TREND_ENTER_THRESHOLD:
                    actions[i] = TREND_ON
                elif P_PANIC[i] >= PANIC_TRIGGER_THRESHOLD:
                    actions[i] = PANIC_OFF
                else:
                    actions[i] = OBSERVE

            else:  # OBSERVE
                if P_TREND[i] >= TREND_ENTER_THRESHOLD:
                    actions[i] = TREND_ON
                elif P_PANIC[i] >= PANIC_TRIGGER_THRESHOLD:
                    actions[i] = PANIC_OFF
                else:
                    actions[i] = OBSERVE

        regime_series = pd.Series(
            actions,
            index=dates,
            name="regime_action",
            dtype=np.int32,
        )

        return regime_series

    # ======================================================================
    # 公共入口
    # ======================================================================
    def run_regime_radar(
        self,
        pipeline: Optional[MultiETFDataPipeline] = None,
    ) -> pd.Series:
        """
        运行完整的状态感知雷达流程。

        执行步骤：
          1. 从 Data Pipeline 获取12只ETF的原始OHLCV数据
          2. 计算截面特征（F1~F4）并降维（仅原始特征，不含标准化）
          3. 滚动HMM训练 → 窗口内标准化 → 纯前向递推 → 状态重对齐
          4. 迟滞比较器输出最终状态

        Parameters
        ----------
        pipeline : MultiETFDataPipeline, optional
            Data Pipeline 实例。如果为 None，则内部创建并使用本地数据。

        Returns
        -------
        pd.Series
            regime_action 序列，index=DatetimeIndex, values=int (0/1/2)
            契约保证：无缺失值，仅包含 0, 1, 2
        """
        self.logger.info("=" * 70)
        self.logger.info("HMMRegimeRadar V2.1 开始运行（Bug修复版 - 零未来数据泄露）")
        self.logger.info("=" * 70)

        # ---- Step 1: 获取数据 ----
        if pipeline is None:
            pipeline = MultiETFDataPipeline(etf_pool=self.etf_pool)

        raw_data = pipeline.load_local(symbols=self.etf_pool)

        if not raw_data:
            self.logger.info("本地无缓存数据，执行数据更新 ...")
            raw_data = pipeline.update_all()

        # 检查是否所有ETF都有数据
        available_symbols = list(raw_data.keys())
        missing = [s for s in self.etf_pool if s not in available_symbols]
        if missing:
            self.logger.warning("以下ETF数据缺失: %s", missing)
            if not available_symbols:
                raise RuntimeError("无任何ETF可用数据，无法运行")

        # 只使用可用的ETF
        active_etfs = [s for s in self.etf_pool if s in raw_data]
        panel_dict = {s: raw_data[s] for s in active_etfs}

        self.logger.info("可用ETF数量: %d / %d", len(panel_dict), len(self.etf_pool))

        # ---- Step 2: 特征工程与截面降维（仅原始特征） ----
        feature_df = self._compute_features(panel_dict)
        self._cached_features = feature_df

        # 检查数据长度是否满足滚动窗口要求
        if len(feature_df) < self.window + self.step:
            raise ValueError(
                f"数据长度 ({len(feature_df)}) 不足以支持滚动窗口 "
                f"(需要至少 {self.window + self.step} 天)"
            )

        # ---- Step 3: 滚动HMM训练 + 窗口内标准化 + 前向递推 ----
        proba_df = self._rolling_hmm_train(feature_df)
        self._cached_probas = proba_df

        # 检查预测结果是否为空
        if proba_df.empty:
            raise RuntimeError("HMM预测结果为空，请检查数据")

        # ---- Step 4: 迟滞比较器 ----
        regime_series = self._hysteresis_comparator(proba_df)
        self._cached_regime = regime_series

        # ---- 数据契约校验 ----
        self._validate_output(regime_series)

        # ---- 状态分布统计 ----
        self._log_state_distribution(regime_series)

        return regime_series

    # ======================================================================
    # 数据契约校验
    # ======================================================================
    @staticmethod
    def _validate_output(regime_series: pd.Series) -> None:
        """
        校验输出是否符合数据契约。
        """
        assert isinstance(regime_series, pd.Series), (
            f"输出必须为 pd.Series，当前类型: {type(regime_series)}"
        )
        assert isinstance(regime_series.index, pd.DatetimeIndex), (
            f"索引必须为 DatetimeIndex，当前类型: {type(regime_series.index)}"
        )
        assert regime_series.name == "regime_action", (
            f"列名必须为 'regime_action'，当前: {regime_series.name}"
        )

        # 检查值范围
        unique_vals = regime_series.unique()
        assert set(unique_vals).issubset({0, 1, 2}), (
            f"regime_action 仅允许 0/1/2，发现: {sorted(unique_vals)}"
        )

        # 检查缺失值
        n_null = regime_series.isnull().sum()
        assert n_null == 0, f"发现 {n_null} 个缺失值"

    @staticmethod
    def _log_state_distribution(regime_series: pd.Series) -> None:
        """记录状态分布统计到日志"""
        logger = logging.getLogger("HMMRegimeRadar")
        n_total = len(regime_series)
        logger.info("HMMRegimeRadar 运行完成 | 输出天数: %d", n_total)
        logger.info("  状态分布:")
        logger.info("    PANIC_OFF (0): %d (%.1f%%)",
                    (regime_series == 0).sum(),
                    (regime_series == 0).mean() * 100)
        logger.info("    OBSERVE   (1): %d (%.1f%%)",
                    (regime_series == 1).sum(),
                    (regime_series == 1).mean() * 100)
        logger.info("    TREND_ON  (2): %d (%.1f%%)",
                    (regime_series == 2).sum(),
                    (regime_series == 2).mean() * 100)
        logger.info("  时间范围: %s ~ %s",
                    regime_series.index[0].strftime("%Y-%m-%d"),
                    regime_series.index[-1].strftime("%Y-%m-%d"))

    # ======================================================================
    # 辅助方法：获取缓存结果
    # ======================================================================
    def get_regime_action(self) -> Optional[pd.Series]:
        """获取最近一次运行的 regime_action 结果"""
        return self._cached_regime

    def get_feature_matrix(self) -> Optional[pd.DataFrame]:
        """获取最近一次计算的截面特征矩阵"""
        return self._cached_features

    def get_state_probas(self) -> Optional[pd.DataFrame]:
        """获取最近一次的状态后验概率"""
        return self._cached_probas

    def get_state_summary(self) -> pd.DataFrame:
        """
        获取状态统计摘要表格。

        Returns
        -------
        pd.DataFrame
            包含各状态的名称、编码、计数值和百分比
        """
        if self._cached_regime is None:
            return pd.DataFrame(columns=["state", "code", "count", "percentage"])

        series = self._cached_regime
        n = len(series)
        summary = pd.DataFrame({
            "state": ["PANIC_OFF", "OBSERVE", "TREND_ON"],
            "code": [0, 1, 2],
            "count": [
                int((series == 0).sum()),
                int((series == 1).sum()),
                int((series == 2).sum()),
            ],
            "percentage": [
                round((series == 0).mean() * 100, 2),
                round((series == 1).mean() * 100, 2),
                round((series == 2).mean() * 100, 2),
            ],
        })
        return summary


# =============================================================================
# 主入口（独立运行测试）
# =============================================================================
if __name__ == "__main__":
    import time

    print("=" * 70)
    print("HMM Regime Radar V2.1 独立运行测试（Bug修复版）")
    print("=" * 70)

    # 创建 Data Pipeline 实例
    pipeline = MultiETFDataPipeline()

    # 创建 HMM 雷达实例
    radar = HMMRegimeRadar()

    # 运行完整流程
    t0 = time.time()
    regime = radar.run_regime_radar(pipeline=pipeline)
    elapsed = time.time() - t0

    print(f"\n运行耗时: {elapsed:.2f} 秒")
    print(f"\nRegime Action 序列 (尾部20行):")
    print(regime.tail(20))

    print(f"\n状态统计摘要:")
    print(radar.get_state_summary())

    print(f"\n特征矩阵维度: {radar.get_feature_matrix().shape}")
    print(f"后验概率矩阵维度: {radar.get_state_probas().shape}")
