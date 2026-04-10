"""
ForecastPro Agent核心模块
实现AI驱动的需求预测任务，包含数据摄取、基准建模、高级模型探索、评估诊断和管理报告生成。
"""

import pandas as pd
import numpy as np
import warnings
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import json

# 数据科学库
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# 时间序列库
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import acf, pacf, adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

# 可视化
import matplotlib.pyplot as plt
import seaborn as sns

# 深度学习库（可选）
try:
    import tensorflow as tf
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

try:
    from neuralforecast import NeuralForecast
    from neuralforecast.models import LSTM, NHITS, TFT
    HAS_NEURALFORECAST = True
except ImportError:
    HAS_NEURALFORECAST = False

try:
    from darts import TimeSeries
    from darts.models import (
        ExponentialSmoothing as DartsExponentialSmoothing,
        ARIMA as DartsARIMA,
        RandomForest as DartsRandomForest,
        XGBModel,
        RNNModel,
        TFTModel,
        NHiTSModel
    )
    HAS_DARTS = True
except ImportError:
    HAS_DARTS = False

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')


class ForecastProAgent:
    """
    ForecastPro AI需求预测Agent

    专门负责为企业设计和执行AI驱动的需求预测任务，核心目标是减少因预测不准
    导致的缺货、库存积压或产能闲置，从而提高运营效率。
    """

    def __init__(self,
                 data_path: Optional[str] = None,
                 time_col: str = 'date',
                 demand_col: str = 'demand',
                 freq: str = 'D',
                 random_seed: int = 42):
        """
        初始化ForecastPro Agent

        参数:
            data_path: 数据文件路径 (CSV/Excel)
            time_col: 时间列名称
            demand_col: 需求列名称
            freq: 时间频率 ('D'=日, 'W'=周, 'M'=月, 'Q'=季, 'Y'=年)
            random_seed: 随机种子
        """
        self.data_path = data_path
        self.time_col = time_col
        self.demand_col = demand_col
        self.freq = freq
        self.random_seed = random_seed
        np.random.seed(random_seed)

        # 数据属性
        self.data = None
        self.X = None
        self.y = None
        self.features = None
        self.covariates = None
        self.train_data = None
        self.test_data = None

        # 模型存储
        self.baseline_models = {}
        self.advanced_models = {}
        self.model_results = {}
        self.best_model = None

        # 报告存储
        self.report = {}

        # 配置
        self.test_size = 0.2  # 20%测试集

        # 变量术语映射 - 根据实际变量名适配报告术语
        self.variable_labels = {
            # 需求相关
            'demand': '需求',
            'quantity': '数量',
            'volume': '成交量',
            'units_sold': '销售数量',
            # 销售/收入相关
            'sales': '销售额',
            'revenue': '收入',
            'sales_revenue': '销售收入',
            'value': '价值',
            'Value': '价值',
            # 消耗相关
            'consumption': '消耗量',
            'Consumption': '消耗量',
            # 成本相关
            'cost': '成本',
            'Cost': '成本',
            # 价格相关
            'price': '价格',
            'Price': '价格',
            # 通用目标变量
            'y': '目标变量',
            'target': '目标变量'
        }

        print("=" * 60)
        print("ForecastPro AI需求预测Agent初始化完成")
        print("=" * 60)

    def load_data(self, data_path: Optional[str] = None) -> pd.DataFrame:
        """
        Step 1: 数据摄取与画像

        数据读取：接受CSV、Excel等格式的历史数据
        特征识别：自动识别时间索引、需求变量以及协变量
        质量检查：检查并报告缺失值、异常值和不规则时间步长

        返回:
            处理后的DataFrame
        """
        if data_path is None:
            data_path = self.data_path

        if data_path is None:
            raise ValueError("请提供数据文件路径")

        print(f"正在加载数据: {data_path}")

        # 根据文件扩展名选择读取方法
        file_ext = Path(data_path).suffix.lower()

        if file_ext == '.csv':
            df = pd.read_csv(data_path)
        elif file_ext in ['.xlsx', '.xls']:
            df = pd.read_excel(data_path)
        else:
            raise ValueError(f"不支持的文件格式: {file_ext}")

        print(f"数据加载成功: {df.shape[0]} 行, {df.shape[1]} 列")

        # 数据预处理：删除不必要的列
        # 删除完全为空的列
        df = df.dropna(axis=1, how='all')

        # 删除可能是索引列的未命名列（如'Unnamed: 0'）
        unnamed_cols = [col for col in df.columns if 'unnamed' in str(col).lower()]
        if unnamed_cols:
            print(f"删除未命名列: {unnamed_cols}")
            df = df.drop(columns=unnamed_cols)

        # 自动识别时间列
        time_candidates = ['date', 'time', 'timestamp', 'datetime', 'Date', 'Time', 'TxnDate', 'TxnTime',
                          'Date', 'TIME', 'TIMESTAMP', 'Datetime', 'transaction_date', 'TransactionDate']
        if self.time_col not in df.columns:
            for col in time_candidates:
                if col in df.columns:
                    self.time_col = col
                    print(f"自动识别时间列: {self.time_col}")
                    break

        # 自动识别需求列
        demand_candidates = ['demand', 'sales', 'quantity', 'volume', 'y', 'target', 'consumption', 'Consumption', 'value', 'Value',
                            'sales_revenue', 'units_sold', 'revenue', 'Revenue', 'consumption', 'Consumption',
                            'cost', 'Cost', 'price', 'Price']
        if self.demand_col not in df.columns:
            for col in demand_candidates:
                if col in df.columns:
                    self.demand_col = col
                    print(f"自动识别需求列: {self.demand_col}")
                    break

        # 转换为时间序列
        if self.time_col in df.columns:
            # 检查是否有分离的日期和时间列
            if self.time_col == 'TxnDate' and 'TxnTime' in df.columns:
                # 合并日期和时间
                print("检测到分离的日期和时间列，合并为datetime...")
                df['datetime'] = pd.to_datetime(df['TxnDate'] + ' ' + df['TxnTime'])
                df = df.set_index('datetime')
                # 删除原始的日期和时间列，避免它们成为协变量
                if 'TxnDate' in df.columns:
                    df = df.drop(columns=['TxnDate'])
                if 'TxnTime' in df.columns:
                    df = df.drop(columns=['TxnTime'])
                self.time_col = 'datetime'  # 更新时间列名称
            else:
                # 标准时间列处理
                df[self.time_col] = pd.to_datetime(df[self.time_col])
                df = df.set_index(self.time_col)

            # 检查是否有重复的时间索引
            if df.index.duplicated().any():
                print(f"警告: 时间索引中有 {df.index.duplicated().sum()} 个重复值")
                print("正在删除重复的时间索引（保留第一个）...")
                df = df[~df.index.duplicated(keep='first')]

            # 仅当数据是规则时间序列时才使用asfreq
            # 对于高频率或不规则数据，跳过asfreq以避免创建大量NaN
            try:
                if len(df) > 10:  # 只有足够数据时才尝试
                    # 检查时间间隔是否大致规则
                    time_diffs = df.index.to_series().diff().dropna()
                    if time_diffs.nunique() <= 3:  # 时间间隔相对规则
                        df = df.asfreq(self.freq)
                        print(f"应用频率: {self.freq}")
                    else:
                        print(f"警告: 时间序列不规则，跳过asfreq()")
                else:
                    print("数据量较少，跳过asfreq()")
            except Exception as e:
                print(f"asfreq()失败: {e}，跳过频率调整")

        # 识别协变量
        self._identify_covariates(df)

        # 数据质量检查
        self._data_quality_check(df)

        self.data = df
        return df

    def get_variable_label(self, variable_name: str = None) -> str:
        """
        获取变量的中文标签

        参数:
            variable_name: 变量名，如果为None则使用self.demand_col

        返回:
            中文标签
        """
        if variable_name is None:
            variable_name = self.demand_col

        # 转换为小写进行比较（但保留原始大小写用于显示）
        var_lower = variable_name.lower()

        # 检查映射字典
        for key, label in self.variable_labels.items():
            if var_lower == key.lower():
                return label

        # 如果没找到映射，使用变量名本身
        return variable_name

    def get_variable_label_with_unit(self, variable_name: str = None, unit: str = None) -> str:
        """
        获取带单位的变量中文标签

        参数:
            variable_name: 变量名
            unit: 单位，如'元'、'件'、'千瓦时'等

        返回:
            带单位的中文标签
        """
        label = self.get_variable_label(variable_name)

        if unit:
            return f"{label}({unit})"
        return label

    def _identify_covariates(self, df: pd.DataFrame):
        """识别协变量"""
        all_cols = set(df.columns)
        core_cols = {self.demand_col, self.time_col if isinstance(self.time_col, str) else ''}
        self.covariates = list(all_cols - core_cols - {''})

        if self.covariates:
            print(f"识别到 {len(self.covariates)} 个协变量: {self.covariates}")
        else:
            print("未识别到协变量")

    def _data_quality_check(self, df: pd.DataFrame):
        """数据质量检查"""
        print("\n" + "=" * 40)
        print("数据质量检查报告")
        print("=" * 40)

        # 检查缺失值
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)

        missing_report = pd.DataFrame({
            '缺失值数量': missing,
            '缺失百分比%': missing_pct
        })

        missing_significant = missing_report[missing_report['缺失值数量'] > 0]

        if len(missing_significant) > 0:
            print("发现缺失值:")
            print(missing_significant.to_string())

            # 建议处理方法
            print("\n建议处理方法:")
            for col in missing_significant.index:
                if missing_pct[col] < 5:
                    print(f"  {col}: 缺失率{missing_pct[col]}% < 5%, 建议使用前向填充")
                elif missing_pct[col] < 30:
                    print(f"  {col}: 缺失率{missing_pct[col]}% < 30%, 建议使用插值法")
                else:
                    print(f"  {col}: 缺失率{missing_pct[col]}% >= 30%, 建议删除该列或使用模型预测")
        else:
            print("✓ 无缺失值")

        # 检查异常值（使用IQR方法）
        if self.demand_col in df.columns:
            Q1 = df[self.demand_col].quantile(0.25)
            Q3 = df[self.demand_col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = df[(df[self.demand_col] < lower_bound) | (df[self.demand_col] > upper_bound)]

            if len(outliers) > 0:
                print(f"\n发现 {len(outliers)} 个异常值 (IQR方法)")
                print(f"异常值范围: {lower_bound:.2f} ~ {upper_bound:.2f}")
                print(f"异常值占比: {len(outliers)/len(df)*100:.2f}%")
            else:
                print("✓ 无异常值")

        # 检查时间间隔
        if hasattr(df.index, 'freq') and df.index.freq is None:
            # 检查是否真的是不规则时间序列（不是所有NaN的情况）
            if len(df) > 1:
                time_diffs = df.index.to_series().diff().dropna()
                if len(time_diffs) > 0:
                    unique_diffs = time_diffs.nunique()
                    if unique_diffs > 3:  # 多种不同的时间间隔
                        print(f"\n⚠ 警告: 时间序列不规则，检测到 {unique_diffs} 种不同时间间隔")
                        print(f"  最常见间隔: {time_diffs.mode().iloc[0] if len(time_diffs.mode()) > 0 else 'N/A'}")
                        print("  对于不规则时间序列，某些模型可能需要数据重采样")
                    else:
                        print("\nℹ 时间序列相对规则，适合时间序列建模")
                else:
                    print("\nℹ 时间序列数据可用")

        # 时间序列分析
        self._time_series_analysis(df)

        print("=" * 40)

    def _time_series_analysis(self, df):
        """
        执行时间序列分析

        包括:
        1. 季节性分解 (趋势、季节性、残差)
        2. 平稳性检验 (ADF检验)
        3. 自相关和偏自相关分析
        4. 季节性检测
        """
        print("\n" + "=" * 40)
        print("时间序列分析")
        print("=" * 40)

        if self.demand_col not in df.columns:
            print(f"需求列 '{self.demand_col}' 不存在，跳过时间序列分析")
            return

        y = df[self.demand_col]

        # 1. 季节性分解
        decomposition = self._seasonal_decomposition(y)

        # 2. 平稳性检验
        stationarity_result = self._stationarity_test(y)

        # 3. ACF/PACF分析
        self._acf_pacf_analysis(y)

        # 4. 季节性检测
        seasonality_result = self._seasonality_detection(y)

        # 存储分析结果
        self.ts_analysis_results = {
            'decomposition': decomposition,
            'stationarity': stationarity_result,
            'seasonality': seasonality_result
        }

        print("时间序列分析完成")
        print("=" * 40)

    def _seasonal_decomposition(self, y):
        """执行季节性分解"""
        print("\n1. 季节性分解:")

        try:
            # 确定季节性周期
            if self.freq == 'D':
                period = 7  # 周季节性
            elif self.freq == 'W':
                period = 4  # 月季节性 (近似)
            elif self.freq == 'M':
                period = 12  # 年季节性
            elif self.freq == 'Q':
                period = 4  # 年季节性
            elif self.freq == 'Y':
                period = 1  # 无季节性
            else:
                period = None

            if period is not None and len(y) > period * 2:
                result = seasonal_decompose(y, model='additive', period=period)

                # 计算各个成分的贡献度
                trend_ratio = np.abs(result.trend).sum() / np.abs(y).sum() if result.trend is not None else 0
                seasonal_ratio = np.abs(result.seasonal).sum() / np.abs(y).sum() if result.seasonal is not None else 0
                residual_ratio = np.abs(result.resid).sum() / np.abs(y).sum() if result.resid is not None else 0

                print(f"  分解周期: {period}")
                print(f"  趋势成分占比: {trend_ratio:.2%}")
                print(f"  季节性成分占比: {seasonal_ratio:.2%}")
                print(f"  残差成分占比: {residual_ratio:.2%}")

                # 判断主要成分
                if seasonal_ratio > 0.3:
                    print("  ✅ 数据呈现强季节性")
                elif seasonal_ratio > 0.1:
                    print("  ⚠ 数据呈现中等季节性")
                else:
                    print("  ℹ 数据季节性较弱")

                if residual_ratio > 0.5:
                    print("  ⚠ 残差占比较高，表明存在未捕捉的模式或噪声")

                return {
                    'period': period,
                    'trend_ratio': trend_ratio,
                    'seasonal_ratio': seasonal_ratio,
                    'residual_ratio': residual_ratio,
                    'has_strong_seasonality': seasonal_ratio > 0.3
                }
            else:
                print("  数据长度不足，无法进行季节性分解")
                return None

        except Exception as e:
            print(f"  季节性分解失败: {e}")
            return None

    def _stationarity_test(self, y):
        """执行平稳性检验 (ADF检验)"""
        print("\n2. 平稳性检验 (ADF检验):")

        try:
            result = adfuller(y.dropna())

            adf_statistic = result[0]
            p_value = result[1]
            critical_values = result[4]

            print(f"  ADF统计量: {adf_statistic:.4f}")
            print(f"  P值: {p_value:.4f}")

            # 判断平稳性
            is_stationary = p_value < 0.05
            if is_stationary:
                print("  ✅ 数据是平稳的 (p < 0.05)")
            else:
                print("  ⚠ 数据是非平稳的 (p ≥ 0.05)")
                print("  建议进行差分处理")

            # 与临界值比较
            print("  临界值:")
            for key, value in critical_values.items():
                print(f"    {key}: {value:.4f}")

            return {
                'adf_statistic': adf_statistic,
                'p_value': p_value,
                'is_stationary': is_stationary,
                'critical_values': critical_values
            }

        except Exception as e:
            print(f"  平稳性检验失败: {e}")
            return None

    def _acf_pacf_analysis(self, y):
        """执行自相关和偏自相关分析"""
        print("\n3. 自相关(ACF)和偏自相关(PACF)分析:")

        try:
            # 计算ACF和PACF
            nlags = min(40, len(y) // 2)
            acf_values = acf(y.dropna(), nlags=nlags)
            pacf_values = pacf(y.dropna(), nlags=nlags)

            # 寻找显著滞后
            significant_acf = np.where(np.abs(acf_values) > 1.96 / np.sqrt(len(y)))[0]
            significant_pacf = np.where(np.abs(pacf_values) > 1.96 / np.sqrt(len(y)))[0]

            # 排除滞后0
            significant_acf = significant_acf[significant_acf > 0]
            significant_pacf = significant_pacf[significant_pacf > 0]

            print(f"  分析滞后数: {nlags}")

            if len(significant_acf) > 0:
                print(f"  显著ACF滞后: {significant_acf[:10].tolist()}" +
                      (f" (共{len(significant_acf)}个)" if len(significant_acf) > 10 else ""))

                # 检查季节性滞后
                if self.freq == 'D' and 7 in significant_acf:
                    print("  ✅ 检测到周度季节性 (滞后7)")
                elif self.freq == 'M' and 12 in significant_acf:
                    print("  ✅ 检测到年度季节性 (滞后12)")
            else:
                print("  无显著ACF滞后")

            if len(significant_pacf) > 0:
                print(f"  显著PACF滞后: {significant_pacf[:10].tolist()}" +
                      (f" (共{len(significant_pacf)}个)" if len(significant_pacf) > 10 else ""))
            else:
                print("  无显著PACF滞后")

            # ARIMA阶数建议
            if len(significant_acf) > 0 and len(significant_pacf) > 0:
                p = min(3, len(significant_pacf))
                q = min(3, len(significant_acf))
                print(f"  ARIMA阶数建议: p={p}, q={q} (基于显著滞后)")

            return {
                'significant_acf_lags': significant_acf.tolist(),
                'significant_pacf_lags': significant_pacf.tolist()
            }

        except Exception as e:
            print(f"  ACF/PACF分析失败: {e}")
            return None

    def _seasonality_detection(self, y):
        """检测季节性模式"""
        print("\n4. 季节性检测:")

        try:
            # 简单的季节性检测方法
            if len(y) < 14:
                print("  数据长度不足，无法进行季节性检测")
                return {'has_seasonality': False}

            # 方法1: 基于周期自相关
            if self.freq == 'D':
                test_periods = [7, 14, 30]  # 周、双周、月
            elif self.freq == 'W':
                test_periods = [4, 8, 13]  # 月、双月、季
            elif self.freq == 'M':
                test_periods = [3, 6, 12]  # 季、半年、年
            else:
                test_periods = []

            seasonality_found = False
            for period in test_periods:
                if len(y) > period * 2:
                    # 计算周期性自相关
                    autocorr = y.autocorr(lag=period)
                    if abs(autocorr) > 0.5:
                        print(f"  ✅ 检测到{period}期季节性 (自相关: {autocorr:.3f})")
                        seasonality_found = True

            if not seasonality_found:
                print("  ℹ 未检测到明显季节性模式")

            # 方法2: 基于方差分析 (简化)
            if self.freq == 'D' and len(y) >= 28:
                # 检查周内模式
                day_of_week = y.index.dayofweek if hasattr(y.index, 'dayofweek') else None
                if day_of_week is not None:
                    weekday_means = y.groupby(day_of_week).mean()
                    if weekday_means.std() / weekday_means.mean() > 0.2:
                        print("  ✅ 检测到周内模式变化")
                        seasonality_found = True

            return {
                'has_seasonality': seasonality_found,
                'suggested_periods': test_periods
            }

        except Exception as e:
            print(f"  季节性检测失败: {e}")
            return {'has_seasonality': False}

    def prepare_data(self):
        """准备训练和测试数据"""
        if self.data is None:
            raise ValueError("请先加载数据")

        if self.covariates:
            self.X = self.data[self.covariates]
            self.y = self.data[self.demand_col]
        else:
            # 如果没有协变量，创建滞后特征
            self._create_lag_features()
            # 滞后特征创建后，重新设置X和y
            self.X = self.data[self.covariates] if self.covariates else None
            self.y = self.data[self.demand_col]

        # 划分训练集和测试集 (80-20)
        split_idx = int(len(self.y) * (1 - self.test_size))

        self.train_data = {
            'X': self.X.iloc[:split_idx] if self.X is not None else None,
            'y': self.y.iloc[:split_idx]
        }

        self.test_data = {
            'X': self.X.iloc[split_idx:] if self.X is not None else None,
            'y': self.y.iloc[split_idx:]
        }

        print(f"数据划分完成:")
        print(f"  训练集: {len(self.train_data['y'])} 个观测值")
        print(f"  测试集: {len(self.test_data['y'])} 个观测值 ({self.test_size*100}%)")

    def _create_lag_features(self, n_lags: int = 5):
        """创建滞后特征"""
        print("创建滞后特征...")

        lag_features = []
        for i in range(1, n_lags + 1):
            lag_name = f'lag_{i}'
            self.data[lag_name] = self.data[self.demand_col].shift(i)
            lag_features.append(lag_name)

        # 删除包含NaN的行
        self.data = self.data.dropna()

        self.covariates = lag_features
        self.X = self.data[lag_features]

        print(f"创建了 {n_lags} 个滞后特征")

    def run_baseline_models(self):
        """
        Step 2: 建立时间序列基准

        拟合以下经典模型作为分析基础:
        1. Naïve / Seasonal Naïve: 作为最简单的基准
        2. 移动平均线 (MA): 自动选择最佳窗口长度
        3. 指数平滑 (ETS): 根据数据特征自动选择 Simple, Holt 或 Holt-Winters
        4. ARIMA / SARIMA: 通过 AIC 自动确定阶数
        """
        print("\n" + "=" * 60)
        print("Step 2: 建立时间序列基准模型")
        print("=" * 60)

        if self.train_data is None:
            self.prepare_data()

        y_train = self.train_data['y']
        y_test = self.test_data['y']

        def _safe_mape(y_true, y_pred):
            y_true = np.asarray(y_true, dtype=float)
            y_pred = np.asarray(y_pred, dtype=float)
            with np.errstate(divide='ignore', invalid='ignore'):
                m = np.abs((y_true - y_pred) / y_true)
                m = np.where(np.isfinite(m), m, 1.0)
            return float(np.mean(m) * 100)

        def _metrics(y_true, y_pred):
            return {
                "MAE": float(mean_absolute_error(y_true, y_pred)),
                "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "MAPE": _safe_mape(y_true, y_pred),
            }

        # 1. Naïve 模型
        print("\n1. 拟合Naïve模型...")
        naive_forecast = y_train.iloc[-1]
        naive_predictions = np.full(len(y_test), naive_forecast)

        self.baseline_models['naive'] = {
            'predictions': naive_predictions,
            'fitted': y_train.shift(1).values,
            'model': None,
            'params': {'strategy': 'last_observation'}
        }

        # 2. Seasonal Naïve (如果数据有季节性)
        print("2. 拟合Seasonal Naïve模型...")
        # 简单实现: 使用上一个周期的同期值
        if len(y_train) >= 7:  # 假设至少一周数据用于日频数据
            seasonal_period = 7 if self.freq == 'D' else 4 if self.freq == 'W' else 12 if self.freq == 'M' else 1
            seasonal_naive_forecast = y_train.iloc[-seasonal_period]
            seasonal_naive_predictions = np.full(len(y_test), seasonal_naive_forecast)

            self.baseline_models['seasonal_naive'] = {
                'predictions': seasonal_naive_predictions,
                'fitted': y_train.shift(seasonal_period).values,
                'model': None,
                'params': {'seasonal_period': int(seasonal_period)}
            }

        # 3. 移动平均 (MA)
        print("3. 拟合移动平均(MA)模型...")
        self._fit_moving_average(y_train, y_test)

        # 4. 指数平滑 (ETS)
        print("4. 拟合指数平滑(ETS)模型...")
        self._fit_exponential_smoothing(y_train, y_test)

        # 5. ARIMA/SARIMA
        print("5. 拟合ARIMA/SARIMA模型...")
        self._fit_arima(y_train, y_test)

        # 评估基准模型
        print("\n基准模型评估:")
        print("-" * 40)
        for model_name, result in self.baseline_models.items():
            predictions = result['predictions']
            if len(predictions) == len(y_test):
                result['holdout_metrics'] = _metrics(y_test, predictions)
                hm = result['holdout_metrics']
                print(f"{model_name:20s} MAE: {hm['MAE']:.2f}, RMSE: {hm['RMSE']:.2f}, MAPE: {hm['MAPE']:.2f}%")

        print("=" * 60)

    def _fit_moving_average(self, y_train, y_test):
        """拟合移动平均模型"""
        windows = [3, 5, 7, 10, 14, 21, 28]
        windows = [w for w in windows if w < len(y_train)]
        if not windows:
            return

        # 简单“AI选择”：在训练集尾部做一段验证，选择验证误差最小的窗口
        val_len = max(1, int(len(y_train) * 0.2))
        sub_train = y_train.iloc[:-val_len] if len(y_train) > val_len + 5 else y_train.iloc[:-1]
        val_y = y_train.iloc[len(sub_train):]

        def _ma_forecast(series, window, horizon):
            ma = series.rolling(window=window).mean()
            last = ma.iloc[-1]
            return np.full(horizon, float(last))

        best = (None, float("inf"))
        for window in windows:
            if window >= len(sub_train):
                continue
            preds_val = _ma_forecast(sub_train, window, len(val_y))
            mae = float(mean_absolute_error(val_y, preds_val))
            if mae < best[1]:
                best = (window, mae)

        best_window = best[0] or windows[0]
        preds_test = _ma_forecast(y_train, best_window, len(y_test))

        # in-sample fitted：用前window的均值预测当前点（shift避免泄露）
        fitted = y_train.rolling(window=best_window).mean().shift(1).values

        self.baseline_models['moving_average'] = {
            'predictions': preds_test,
            'fitted': fitted,
            'model': None,
            'params': {'window': int(best_window), 'selection': 'validation_mae'},
        }

    def _fit_exponential_smoothing(self, y_train, y_test):
        """拟合指数平滑模型"""
        try:
            def _seasonal_periods_default():
                if self.freq == "D":
                    return 7
                if self.freq == "W":
                    return 52
                if self.freq == "M":
                    return 12
                if self.freq == "Q":
                    return 4
                return 7

            sp = _seasonal_periods_default()
            candidates = [
                ("ets_simple", {"trend": None, "seasonal": None, "seasonal_periods": None}),
                ("ets_holt", {"trend": "add", "seasonal": None, "seasonal_periods": None}),
            ]
            if sp and len(y_train) >= 2 * sp:
                candidates.append(("ets_holt_winters", {"trend": "add", "seasonal": "add", "seasonal_periods": int(sp)}))

            for model_key, params in candidates:
                try:
                    model = ExponentialSmoothing(
                        y_train,
                        trend=params["trend"],
                        seasonal=params["seasonal"],
                        seasonal_periods=params["seasonal_periods"],
                    )
                    fitted_model = model.fit(optimized=True)
                    predictions = fitted_model.forecast(len(y_test))
                    fitted_vals = np.asarray(getattr(fitted_model, "fittedvalues", np.full(len(y_train), np.nan)), dtype=float)
                    fitted_params = dict(getattr(fitted_model, "params", {}) or {})
                    self.baseline_models[model_key] = {
                        "predictions": np.asarray(predictions, dtype=float),
                        "fitted": fitted_vals,
                        "model": None,
                        "params": {
                            "config": params,
                            "fitted_params": fitted_params,
                            "aic": float(getattr(fitted_model, "aic", np.nan)) if getattr(fitted_model, "aic", None) is not None else None,
                        },
                    }
                except Exception:
                    continue

        except Exception as e:
            print(f"指数平滑模型拟合失败: {e}")

    def _fit_arima(self, y_train, y_test):
        """拟合ARIMA模型"""
        try:
            if str(os.getenv("FORECASTPRO_SKIP_ARIMA", "")).strip().lower() in {"1", "true", "yes"}:
                return

            def _seasonal_periods_default():
                if self.freq == "D":
                    return 7
                if self.freq == "W":
                    return 52
                if self.freq == "M":
                    return 12
                if self.freq == "Q":
                    return 4
                return None

            fast = str(os.getenv("FORECASTPRO_FAST", "")).strip().lower() in {"1", "true", "yes"}
            maxiter_env = os.getenv("FORECASTPRO_ARIMA_MAXITER")
            try:
                maxiter = int(maxiter_env) if maxiter_env is not None else (25 if fast else 80)
            except Exception:
                maxiter = 25 if fast else 80

            if fast:
                orders = [(0, 1, 1), (1, 1, 0), (1, 1, 1)]
            else:
                max_p = 2
                max_q = 2
                max_d = 1
                orders = [(p, d, q) for p in range(max_p + 1) for d in range(max_d + 1) for q in range(max_q + 1)]

            sp = _seasonal_periods_default()
            use_seasonal = (not fast) and bool(sp) and len(y_train) >= 2 * int(sp)
            seasonal_orders = [(0, 0, 0, 0)]
            if use_seasonal:
                seasonal_orders = [(P, D, Q, int(sp)) for P in (0, 1) for D in (0, 1) for Q in (0, 1)]

            best = {"aic": float("inf"), "order": None, "seasonal_order": None, "fitted": None, "pred": None, "params": None}
            max_seconds = os.getenv("FORECASTPRO_ARIMA_MAX_SECONDS")
            try:
                max_seconds = float(max_seconds) if max_seconds is not None else (8.0 if fast else 25.0)
            except Exception:
                max_seconds = 8.0 if fast else 25.0
            t0 = time.monotonic()

            for order in orders:
                for sorder in seasonal_orders:
                    if (time.monotonic() - t0) > max_seconds:
                        break
                    try:
                        if fast:
                            model = ARIMA(y_train, order=order)
                            fitted_model = model.fit(method_kwargs={"maxiter": maxiter})
                        else:
                            model = SARIMAX(
                                y_train,
                                order=order,
                                seasonal_order=sorder if use_seasonal else (0, 0, 0, 0),
                                enforce_stationarity=False,
                                enforce_invertibility=False,
                            )
                            fitted_model = model.fit(disp=False, maxiter=maxiter)
                        aic = float(getattr(fitted_model, "aic", np.inf))
                        if not np.isfinite(aic) or aic >= best["aic"]:
                            continue
                        pred = fitted_model.forecast(len(y_test))
                        fitted_vals = np.asarray(getattr(fitted_model, "fittedvalues", np.full(len(y_train), np.nan)), dtype=float)
                        best = {
                            "aic": aic,
                            "order": order,
                            "seasonal_order": sorder if use_seasonal else (0, 0, 0, 0),
                            "fitted": fitted_vals,
                            "pred": np.asarray(pred, dtype=float),
                            "params": dict(getattr(fitted_model, "params", {}) or {}),
                        }
                    except Exception:
                        continue
                if (time.monotonic() - t0) > max_seconds:
                    break

            if best["pred"] is not None:
                self.baseline_models['arima'] = {
                    'predictions': best["pred"],
                    'fitted': best["fitted"],
                    'model': None,
                    'params': {
                        'order': best["order"],
                        'seasonal_order': best["seasonal_order"] if use_seasonal else None,
                        'aic': best["aic"],
                        'fitted_params': best["params"],
                        'selection': 'aic_grid_search',
                    }
                }

        except Exception as e:
            print(f"ARIMA模型拟合失败: {e}")

    def run_advanced_models(self):
        """
        Step 3: 高级模型探索与选择

        在基准之上，利用AI推理选择并运行以下至少一类高级模型:
        1. 回归分析: 基于滞后需求和协变量的OLS或Lasso/Ridge回归
        2. 机器学习: XGBoost、Random Forest，需配合时间感知交叉验证
        3. 深度学习/基础模型: LSTM、Temporal Fusion Transformer或TimeGPT
        4. 混合模型: 如使用ML修正ARIMA的残差
        """
        print("\n" + "=" * 60)
        print("Step 3: 高级模型探索与选择")
        print("=" * 60)

        if self.train_data is None:
            self.prepare_data()

        X_train = self.train_data['X']
        y_train = self.train_data['y']
        X_test = self.test_data['X']
        y_test = self.test_data['y']

        # 1. 回归分析
        print("\n1. 拟合回归模型...")
        self._fit_regression_models(X_train, y_train, X_test, y_test)

        # 2. 机器学习模型
        print("\n2. 拟合机器学习模型...")
        self._fit_ml_models(X_train, y_train, X_test, y_test)

        # 3. 深度学习模型 (如果可用)
        if HAS_TENSORFLOW and len(y_train) > 100:
            print("\n3. 尝试深度学习模型...")
            self._fit_dl_models(X_train, y_train, X_test, y_test)

        # 4. 混合模型
        print("\n4. 尝试混合模型...")
        self._fit_hybrid_models(X_train, y_train, X_test, y_test)

        # 评估高级模型
        print("\n高级模型评估:")
        print("-" * 40)
        for model_name, result in self.advanced_models.items():
            predictions = result['predictions']

            if len(predictions) == len(y_test):
                mae = mean_absolute_error(y_test, predictions)
                rmse = np.sqrt(mean_squared_error(y_test, predictions))
                # 避免除零错误：当y_test为0时，使用一个小值或跳过
                with np.errstate(divide='ignore', invalid='ignore'):
                    mape_array = np.abs((y_test - predictions) / y_test)
                    # 将无限大和NaN值替换为0或一个大的惩罚值
                    mape_array = np.where(np.isfinite(mape_array), mape_array, 1.0)  # 100%误差
                    mape = np.mean(mape_array) * 100

                result['metrics'] = {
                    'MAE': mae,
                    'RMSE': rmse,
                    'MAPE': mape
                }

                # 检查过拟合风险
                if 'train_score' in result and 'test_score' in result:
                    train_score = result['train_score']
                    test_score = result['test_score']

                    if train_score - test_score > 0.1:  # 训练集性能明显优于测试集
                        result['overfitting_risk'] = '高'
                    elif train_score - test_score > 0.05:
                        result['overfitting_risk'] = '中'
                    else:
                        result['overfitting_risk'] = '低'

                print(f"{model_name:20s} MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")

        print("=" * 60)

    def _fit_regression_models(self, X_train, y_train, X_test, y_test):
        """拟合回归模型"""
        models = {
            'linear_regression': LinearRegression(),
            'ridge_regression': Ridge(alpha=1.0),
            'lasso_regression': Lasso(alpha=0.1)
        }

        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)

                self.advanced_models[name] = {
                    'predictions': predictions,
                    'fitted': model.predict(X_train),
                    'model': model,
                    'params': model.get_params(),
                    'train_score': model.score(X_train, y_train),
                    'test_score': model.score(X_test, y_test)
                }

            except Exception as e:
                print(f"  {name} 拟合失败: {e}")

    def _fit_ml_models(self, X_train, y_train, X_test, y_test):
        """拟合机器学习模型"""
        # 时间序列交叉验证
        tscv = TimeSeriesSplit(n_splits=3)

        # Random Forest
        try:
            rf_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_seed
            )

            # 使用时间序列交叉验证
            cv_scores = []
            for train_idx, val_idx in tscv.split(X_train):
                X_train_cv, X_val_cv = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_train_cv, y_val_cv = y_train.iloc[train_idx], y_train.iloc[val_idx]

                rf_model.fit(X_train_cv, y_train_cv)
                cv_score = rf_model.score(X_val_cv, y_val_cv)
                cv_scores.append(cv_score)

            # 在整个训练集上训练
            rf_model.fit(X_train, y_train)
            predictions = rf_model.predict(X_test)

            self.advanced_models['random_forest'] = {
                'predictions': predictions,
                'fitted': rf_model.predict(X_train),
                'model': rf_model,
                'params': rf_model.get_params(),
                'train_score': rf_model.score(X_train, y_train),
                'test_score': rf_model.score(X_test, y_test),
                'cv_mean_score': np.mean(cv_scores)
            }

        except Exception as e:
            print(f"  Random Forest 拟合失败: {e}")

        # XGBoost
        try:
            xgb_model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_seed
            )

            xgb_model.fit(X_train, y_train)
            predictions = xgb_model.predict(X_test)

            self.advanced_models['xgboost'] = {
                'predictions': predictions,
                'fitted': xgb_model.predict(X_train),
                'model': xgb_model,
                'params': xgb_model.get_params(),
                'train_score': xgb_model.score(X_train, y_train),
                'test_score': xgb_model.score(X_test, y_test)
            }

        except Exception as e:
            print(f"  XGBoost 拟合失败: {e}")

    def _fit_dl_models(self, X_train, y_train, X_test, y_test):
        """拟合深度学习模型"""
        # 简化实现 - 实际应用中需要更复杂的网络架构
        try:
            # 简单LSTM模型
            if len(X_train.shape) == 2:
                # 重塑数据为LSTM输入格式 [samples, timesteps, features]
                n_features = X_train.shape[1]

                # 创建简单的神经网络
                model = tf.keras.Sequential([
                    tf.keras.layers.Dense(64, activation='relu', input_shape=(n_features,)),
                    tf.keras.layers.Dropout(0.2),
                    tf.keras.layers.Dense(32, activation='relu'),
                    tf.keras.layers.Dense(1)
                ])

                model.compile(
                    optimizer='adam',
                    loss='mse',
                    metrics=['mae']
                )

                # 训练模型
                history = model.fit(
                    X_train, y_train,
                    epochs=50,
                    batch_size=32,
                    validation_split=0.2,
                    verbose=0
                )

                predictions = model.predict(X_test, verbose=0).flatten()

                self.advanced_models['neural_network'] = {
                    'predictions': predictions,
                    'model': model,
                    'train_score': model.evaluate(X_train, y_train, verbose=0)[1],
                    'test_score': model.evaluate(X_test, y_test, verbose=0)[1],
                    'history': history.history
                }

        except Exception as e:
            print(f"  深度学习模型拟合失败: {e}")

    def _fit_hybrid_models(self, X_train, y_train, X_test, y_test):
        """拟合混合模型"""
        # 混合模型: ARIMA + 机器学习残差修正
        try:
            # 使用ARIMA作为基础模型
            if 'arima' in self.baseline_models:
                arima_predictions = self.baseline_models['arima']['predictions']

                # 计算ARIMA残差
                if len(arima_predictions) == len(y_test):
                    # 这里简化实现 - 实际需要更复杂的残差建模
                    residuals = y_test.values - arima_predictions

                    # 使用随机森林修正残差
                    rf_residual = RandomForestRegressor(
                        n_estimators=50,
                        random_state=self.random_seed
                    )

                    # 需要为残差模型准备特征
                    # 这里简化: 使用原始特征
                    rf_residual.fit(X_train, y_train - y_train.mean())
                    residual_predictions = rf_residual.predict(X_test)

                    # 混合预测 = ARIMA预测 + 残差修正
                    hybrid_predictions = arima_predictions + residual_predictions

                    self.advanced_models['arima_rf_hybrid'] = {
                        'predictions': hybrid_predictions,
                        'fitted': None,
                        'model': (self.baseline_models['arima'], rf_residual),
                        'params': {
                            'base': 'arima',
                            'residual_model': rf_residual.get_params(),
                        },
                        'type': 'hybrid'
                    }

        except Exception as e:
            print(f"  混合模型拟合失败: {e}")

    def evaluate_models(self):
        """
        Step 4: 模型评估与诊断

        留出测试 (Hold-out): 使用最后20%的观测值作为测试集
        统一指标: 报告所有模型的MAE, RMSE和MAPE
        警示标识: 必须明确指出高级模型中是否存在过拟合或数据泄露的风险
        """
        print("\n" + "=" * 60)
        print("Step 4: 模型评估与诊断")
        print("=" * 60)

        if not self.baseline_models and not self.advanced_models:
            print("警告: 没有已训练的模型，请先运行基准和高级模型")
            return

        y_test = self.test_data['y']
        y_train = self.train_data['y']

        # 收集所有模型结果
        all_models = {}
        all_models.update(self.baseline_models)
        all_models.update(self.advanced_models)

        # 计算指标
        evaluation_results = []

        model_details = {}
        for model_name, result in all_models.items():
            predictions = result['predictions']

            if len(predictions) != len(y_test):
                print(f"警告: {model_name} 预测长度不匹配")
                continue

            # 计算指标
            mae = mean_absolute_error(y_test, predictions)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            with np.errstate(divide='ignore', invalid='ignore'):
                mape_array = np.abs((y_test - predictions) / y_test)
                mape_array = np.where(np.isfinite(mape_array), mape_array, 1.0)
                mape = np.mean(mape_array) * 100

            in_sample_metrics = None
            fitted = result.get('fitted')
            if fitted is not None:
                try:
                    fitted_arr = np.asarray(fitted, dtype=float)
                    y_arr = np.asarray(y_train.values, dtype=float)
                    n = min(len(fitted_arr), len(y_arr))
                    yt = y_arr[:n]
                    yp = fitted_arr[:n]
                    mask = np.isfinite(yt) & np.isfinite(yp)
                    if int(mask.sum()) >= 5:
                        in_sample_metrics = {
                            "MAE": float(mean_absolute_error(yt[mask], yp[mask])),
                            "RMSE": float(np.sqrt(mean_squared_error(yt[mask], yp[mask]))),
                            "MAPE": float(np.mean(np.where(np.isfinite(np.abs((yt[mask] - yp[mask]) / yt[mask])), np.abs((yt[mask] - yp[mask]) / yt[mask]), 1.0)) * 100),
                        }
                except Exception:
                    in_sample_metrics = None

            # 检查过拟合风险
            overfitting_risk = '低'
            if model_name in self.advanced_models:
                if 'overfitting_risk' in self.advanced_models[model_name]:
                    overfitting_risk = self.advanced_models[model_name]['overfitting_risk']
                elif 'train_score' in result and 'test_score' in result:
                    train_score = result['train_score']
                    test_score = result['test_score']

                    if train_score - test_score > 0.1:
                        overfitting_risk = '高'
                    elif train_score - test_score > 0.05:
                        overfitting_risk = '中'

            evaluation_results.append({
                'model': model_name,
                'MAE': mae,
                'RMSE': rmse,
                'MAPE': mape,
                'in_sample_MAE': in_sample_metrics["MAE"] if in_sample_metrics else None,
                'in_sample_RMSE': in_sample_metrics["RMSE"] if in_sample_metrics else None,
                'in_sample_MAPE': in_sample_metrics["MAPE"] if in_sample_metrics else None,
                'overfitting_risk': overfitting_risk,
                'type': 'baseline' if model_name in self.baseline_models else 'advanced'
            })

            model_details[model_name] = {
                "type": 'baseline' if model_name in self.baseline_models else 'advanced',
                "params": result.get("params"),
                "in_sample": in_sample_metrics,
                "holdout": {"MAE": float(mae), "RMSE": float(rmse), "MAPE": float(mape)},
                "overfitting_risk": overfitting_risk,
            }

        # 按MAPE排序
        evaluation_df = pd.DataFrame(evaluation_results)
        evaluation_df = evaluation_df.sort_values('MAPE')

        # 保存评估结果
        self.evaluation_results = evaluation_df
        self.model_results = model_details

        # 选择最佳模型
        best_model_row = evaluation_df.iloc[0]
        self.best_model = best_model_row['model']

        # 打印评估报告
        print("\n模型性能排名 (按MAPE升序):")
        print("-" * 80)
        print(evaluation_df.to_string(index=False))

        print(f"\n最佳模型: {self.best_model}")
        print(f"测试集MAPE: {best_model_row['MAPE']:.2f}%")

        # 过拟合风险报告
        high_risk_models = evaluation_df[evaluation_df['overfitting_risk'] == '高']
        if len(high_risk_models) > 0:
            print("\n⚠ 过拟合高风险模型:")
            for _, row in high_risk_models.iterrows():
                print(f"  {row['model']}: MAPE={row['MAPE']:.2f}%")

        print("=" * 60)

        return evaluation_df

    def generate_forecast(
        self,
        periods: int = 4,
        forecast_method: str = "auto",
        seasonal_periods: Optional[int] = None,
        seasonal: Optional[bool] = None,
    ):
        """
        生成未来预测

        参数:
            periods: 预测周期数
            forecast_method: 预测方法，可选 {"auto", "trend", "ets", "seasonal_ets"}
            seasonal_periods: 季节周期（如日频7=周季节性，月频12=年季节性）
            seasonal: 是否启用季节性（当 forecast_method="ets" 时可用于启用/关闭季节项）
        """
        if self.data is None:
            raise ValueError("请先加载数据")

        if periods < 1:
            raise ValueError("periods 必须 >= 1")

        y = self.data[self.demand_col].astype(float)
        if len(y) < 2:
            raise ValueError("数据量过少，无法生成预测")

        def _future_dates():
            last_date = self.data.index[-1]
            try:
                idx = pd.date_range(start=last_date, periods=periods + 1, freq=self.freq)
                return idx[1:]
            except Exception:
                return pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods, freq='D')

        def _default_seasonal_periods():
            if self.freq == "D":
                return 7
            if self.freq == "W":
                return 52
            if self.freq == "M":
                return 12
            if self.freq == "Q":
                return 4
            return None

        method = (forecast_method or "auto").lower().strip()
        if method in {"auto", "best"}:
            if self.best_model in {"seasonal_ets", "ets_holt_winters"}:
                method = "seasonal_ets"
            elif self.best_model in {"ets", "exponential_smoothing", "ets_simple", "ets_holt"} or (
                isinstance(self.best_model, str) and self.best_model.startswith("ets_")
            ):
                method = "ets"
            elif self.best_model in {"arima"}:
                method = "arima"
            else:
                method = "trend"

        if method == "seasonal_ets":
            seasonal = True if seasonal is None else bool(seasonal)
        elif method == "ets":
            seasonal = False if seasonal is None else bool(seasonal)
        else:
            seasonal = False

        if seasonal_periods is None and seasonal:
            seasonal_periods = _default_seasonal_periods()

        if method in (self.advanced_models or {}):
            if self.train_data is None or self.train_data.get("X") is None:
                raise ValueError("高级模型未来预测需要特征矩阵，请先准备数据并训练高级模型")

            model_obj = self.advanced_models.get(method, {}).get("model")
            if model_obj is None:
                raise ValueError(f"高级模型 {method} 不可用")

            X_cols = list(self.train_data["X"].columns)
            lag_cols = [c for c in X_cols if isinstance(c, str) and c.startswith("lag_")]
            last_known_features = {}
            for c in X_cols:
                if isinstance(c, str) and c.startswith("lag_"):
                    continue
                try:
                    last_known_features[c] = float(self.data[c].iloc[-1])
                except Exception:
                    last_known_features[c] = 0.0

            n_lags = 0
            if lag_cols:
                try:
                    n_lags = max(int(str(c).split("_", 1)[1]) for c in lag_cols)
                except Exception:
                    n_lags = 0

            preds = []
            if n_lags <= 0:
                try:
                    X_last = self.data[X_cols].iloc[[-1]]
                    y_next = float(model_obj.predict(X_last)[0])
                    preds = [y_next for _ in range(int(periods))]
                except Exception:
                    raise ValueError("高级模型未来预测失败：缺少可用的滞后特征，且无法使用最后一行特征直接预测")
            else:
                if len(y) < n_lags + 1:
                    raise ValueError("数据量不足以进行高级模型未来预测")

                history_vals = [float(v) for v in y.iloc[-n_lags:].values]
                for _ in range(int(periods)):
                    row = {f"lag_{i}": history_vals[-i] for i in range(1, n_lags + 1)}
                    row.update(last_known_features)
                    X_row = pd.DataFrame([row])[X_cols]
                    y_next = float(model_obj.predict(X_row)[0])
                    preds.append(y_next)
                    history_vals.append(y_next)

            fitted = self.advanced_models.get(method, {}).get("fitted")
            sigma = float(y.std())
            if fitted is not None:
                try:
                    yt = np.asarray(self.train_data["y"].values, dtype=float)
                    yp = np.asarray(fitted, dtype=float)
                    n = min(len(yt), len(yp))
                    resid = yt[:n] - yp[:n]
                    resid = resid[np.isfinite(resid)]
                    if resid.size > 1:
                        sigma = float(np.std(resid))
                except Exception:
                    pass

            z = 1.96
            lower = [p - z * sigma for p in preds]
            upper = [p + z * sigma for p in preds]

            self.forecast_results = {
                "dates": [d.to_pydatetime() for d in _future_dates()],
                "forecast": preds,
                "lower_bound": lower,
                "upper_bound": upper,
                "model": method,
                "params": {"type": "advanced", "model_params": self.advanced_models.get(method, {}).get("params")},
            }
            return self.forecast_results

        if method in {"naive", "seasonal_naive", "moving_average"}:
            last_date = self.data.index[-1]
            future_dates = [d.to_pydatetime() for d in _future_dates()]
            if method == "naive":
                last_value = float(y.iloc[-1])
                fc = [last_value] * periods
                sigma = float(y.std())
                lower = [v - 1.96 * sigma for v in fc]
                upper = [v + 1.96 * sigma for v in fc]
                self.forecast_results = {
                    "dates": future_dates,
                    "forecast": fc,
                    "lower_bound": lower,
                    "upper_bound": upper,
                    "model": "naive",
                    "params": {"strategy": "last_observation"},
                }
                return self.forecast_results

            if method == "seasonal_naive":
                sp = seasonal_periods or _default_seasonal_periods() or 1
                sp = int(max(sp, 1))
                vals = []
                for i in range(periods):
                    vals.append(float(y.iloc[-sp + (i % sp)]))
                sigma = float(y.std())
                lower = [v - 1.96 * sigma for v in vals]
                upper = [v + 1.96 * sigma for v in vals]
                self.forecast_results = {
                    "dates": future_dates,
                    "forecast": vals,
                    "lower_bound": lower,
                    "upper_bound": upper,
                    "model": "seasonal_naive",
                    "params": {"seasonal_period": sp},
                }
                return self.forecast_results

            if method == "moving_average":
                window = None
                try:
                    window = int(self.baseline_models.get("moving_average", {}).get("params", {}).get("window"))
                except Exception:
                    window = None
                if not window:
                    window = int(min(14, max(3, len(y) // 10)))
                last_ma = float(y.rolling(window=window).mean().iloc[-1])
                fc = [last_ma] * periods
                sigma = float(y.std())
                lower = [v - 1.96 * sigma for v in fc]
                upper = [v + 1.96 * sigma for v in fc]
                self.forecast_results = {
                    "dates": future_dates,
                    "forecast": fc,
                    "lower_bound": lower,
                    "upper_bound": upper,
                    "model": "moving_average",
                    "params": {"window": window},
                }
                return self.forecast_results

        if method in {"arima"}:
            order = None
            seasonal_order = None
            try:
                arima_params = self.baseline_models.get("arima", {}).get("params", {}) or {}
                order = tuple(arima_params.get("order")) if arima_params.get("order") is not None else None
                seasonal_order = tuple(arima_params.get("seasonal_order")) if arima_params.get("seasonal_order") is not None else None
            except Exception:
                order = None
                seasonal_order = None
            if order is None:
                order = (1, 1, 1)
            if seasonal_order is None:
                seasonal_order = (0, 0, 0, 0)
            model = SARIMAX(
                y,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fitted = model.fit(disp=False)
            forecast = fitted.forecast(periods).astype(float)
            resid = (y - fitted.fittedvalues).dropna()
            sigma = float(resid.std()) if len(resid) > 1 else float(y.std())
            z = 1.96
            lower = forecast - z * sigma
            upper = forecast + z * sigma
            self.forecast_results = {
                "dates": [d.to_pydatetime() for d in _future_dates()],
                "forecast": forecast.tolist(),
                "lower_bound": lower.tolist(),
                "upper_bound": upper.tolist(),
                "model": "arima",
                "params": {"order": order, "seasonal_order": seasonal_order, "aic": float(getattr(fitted, "aic", np.nan))},
            }
            return self.forecast_results

        if method in {"ets", "seasonal_ets"}:
            if seasonal and (seasonal_periods is None or seasonal_periods < 2):
                seasonal = False
                seasonal_periods = None

            if seasonal and len(y) < 2 * seasonal_periods:
                seasonal = False
                seasonal_periods = None

            if seasonal:
                print(f"\n使用ETS(季节性)生成 {periods} 期预测，seasonal_periods={seasonal_periods}")
                model = ExponentialSmoothing(y, trend="add", seasonal="add", seasonal_periods=seasonal_periods)
                model_name = "seasonal_ets"
            else:
                print(f"\n使用ETS(非季节性)生成 {periods} 期预测")
                model = ExponentialSmoothing(y, trend="add", seasonal=None)
                model_name = "ets"

            fitted = model.fit()
            forecast = fitted.forecast(periods).astype(float)
            resid = (y - fitted.fittedvalues).dropna()
            sigma = float(resid.std()) if len(resid) > 1 else float(y.std())
            z = 1.96
            lower = forecast - z * sigma
            upper = forecast + z * sigma

            self.forecast_results = {
                "dates": [d.to_pydatetime() for d in _future_dates()],
                "forecast": forecast.tolist(),
                "lower_bound": lower.tolist(),
                "upper_bound": upper.tolist(),
                "model": model_name,
                "params": {"seasonal": seasonal, "seasonal_periods": seasonal_periods, "trend": "add"},
            }
            print(f"未来 {periods} 期预测生成完成")
            return self.forecast_results

        if self.best_model is None:
            print("警告: 没有最佳模型，请先运行模型评估")

        if method not in {"trend"}:
            method = "trend"

        model_label = self.best_model if self.best_model is not None else "trend"
        print(f"\n使用趋势外推生成 {periods} 期预测 (参考模型: {model_label})")

        trend = (y.iloc[-1] - y.iloc[0]) / len(y)
        last_value = y.iloc[-1]
        future_forecast = [float(last_value + trend * i) for i in range(1, periods + 1)]

        std_dev = float(y.std())
        lower_bound = [f - 1.96 * std_dev for f in future_forecast]
        upper_bound = [f + 1.96 * std_dev for f in future_forecast]

        self.forecast_results = {
            "dates": [d.to_pydatetime() for d in _future_dates()],
            "forecast": future_forecast,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "model": model_label,
            "params": {"method": "trend"},
        }

        print(f"未来 {periods} 期预测生成完成")
        return self.forecast_results

    def generate_report(self, save_to_disk: bool = True):
        """
        Step 5: 输出管理报告

        生成一份非技术经理也能读懂的报告，包含:
        1. 预测图表: 包含历史需求、拟合值及未来至少4个周期的预测值（含95%置信区间）
        2. 对比表: 按测试集MAPE对模型进行排名，高亮推荐模型
        3. 通俗总结: 解释胜出模型的原因，并描述未来的需求高峰、趋势和季节性
        4. 行动建议: 给出两项关于库存目标、人员配备或采购计划的具体建议
        """
        print("\n" + "=" * 60)
        print("Step 5: 生成管理报告")
        print("=" * 60)

        if self.evaluation_results is None:
            print("警告: 请先运行模型评估")
            return

        if self.forecast_results is None:
            self.generate_forecast(periods=4)

        # 生成报告内容
        report = {
            'report_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_summary': {
                'total_observations': len(self.data) if self.data is not None else 0,
                'training_period': len(self.train_data['y']) if self.train_data else 0,
                'testing_period': len(self.test_data['y']) if self.test_data else 0,
                'frequency': self.freq,
                'demand_variable': self.demand_col
            },
            'best_model': {
                'name': self.best_model,
                'metrics': self.evaluation_results[self.evaluation_results['model'] == self.best_model].to_dict('records')[0]
            },
            'model_comparison': self.evaluation_results.to_dict('records'),
            'model_details': self.model_results if hasattr(self, 'model_results') else None,
            'forecast_summary': self.forecast_results if hasattr(self, 'forecast_results') else None,
            'ts_analysis': self.ts_analysis_results if hasattr(self, 'ts_analysis_results') else None,
            'insights': self._generate_insights(),
            'recommendations': self._generate_recommendations()
        }

        self.report = report

        # 打印报告
        self._print_report(report)

        if save_to_disk:
            self._save_report(report)

        print("\n管理报告生成完成!")
        print("=" * 60)

        # 返回清理后的报告以确保JSON可序列化
        return self._clean_for_json(report)

    def _generate_insights(self):
        """生成业务洞察"""
        insights = []

        # 获取变量中文标签
        label = self.get_variable_label()

        # 分析趋势
        if self.data is not None:
            y = self.data[self.demand_col]

            # 趋势分析
            if len(y) > 1:
                growth_rate = ((y.iloc[-1] - y.iloc[0]) / y.iloc[0]) * 100
                if growth_rate > 0:
                    insights.append(f"{label}呈现增长趋势，整体增长率为 {growth_rate:.1f}%")
                elif growth_rate < 0:
                    insights.append(f"{label}呈现下降趋势，整体下降率为 {abs(growth_rate):.1f}%")
                else:
                    insights.append(f"{label}保持稳定，无明显增长或下降趋势")

            # 季节性分析 (简化)
            if len(y) >= 14 and self.freq == 'D':  # 至少两周数据
                weekly_pattern = y.rolling(window=7).mean()
                if weekly_pattern.std() > y.std() * 0.1:
                    insights.append("数据呈现明显的周度季节性模式")

            # 波动性分析
            volatility = y.pct_change().std() * 100
            if volatility > 20:
                insights.append(f"{label}波动性较高 ({volatility:.1f}%)，建议增加安全库存")
            elif volatility < 5:
                insights.append(f"{label}波动性较低 ({volatility:.1f}%)，库存管理相对稳定")

        # 时间序列分析洞察
        if hasattr(self, 'ts_analysis_results') and self.ts_analysis_results:
            ts_results = self.ts_analysis_results

            # 平稳性洞察
            if ts_results.get('stationarity'):
                stationarity = ts_results['stationarity']
                if not stationarity['is_stationary']:
                    insights.append("数据非平稳，建议进行差分处理以提高模型稳定性")
                else:
                    insights.append("数据平稳，适合直接建模")

            # 季节性洞察
            if ts_results.get('decomposition'):
                decomposition = ts_results['decomposition']
                if decomposition and decomposition.get('has_strong_seasonality'):
                    insights.append("数据呈现强季节性模式，建议使用季节性模型")
                elif decomposition and decomposition.get('seasonal_ratio', 0) > 0.1:
                    insights.append("数据呈现中等季节性，可考虑季节性调整")

            # 季节性检测洞察
            if ts_results.get('seasonality'):
                seasonality = ts_results['seasonality']
                if seasonality.get('has_seasonality'):
                    insights.append("检测到明显的季节性模式")

        # 模型洞察
        if self.best_model:
            insights.append(f"最佳预测模型为 {self.best_model}，在测试集上表现最优")

            if 'arima' in self.best_model:
                insights.append("ARIMA模型对时间序列的自相关结构捕捉较好")
            elif 'random_forest' in self.best_model or 'xgboost' in self.best_model:
                insights.append("树模型能够捕捉复杂的非线性关系")
            elif 'naive' in self.best_model or 'moving_average' in self.best_model:
                insights.append(f"简单模型表现最佳，表明{label}模式相对简单稳定")

        # 预测洞察
        if hasattr(self, 'forecast_results'):
            forecast_values = self.forecast_results['forecast']
            if len(forecast_values) >= 2:
                forecast_growth = ((forecast_values[-1] - forecast_values[0]) / forecast_values[0]) * 100
                if forecast_growth > 5:
                    insights.append(f"未来{label}预计增长 {forecast_growth:.1f}%，建议提前准备")
                elif forecast_growth < -5:
                    insights.append(f"未来{label}预计下降 {abs(forecast_growth):.1f}%，建议调整生产计划")

        return insights

    def _generate_recommendations(self):
        """生成行动建议"""
        recommendations = []

        # 获取变量中文标签
        label = self.get_variable_label()

        # 基于预测的建议
        if hasattr(self, 'forecast_results'):
            avg_forecast = np.mean(self.forecast_results['forecast'])
            current_level = self.data[self.demand_col].iloc[-1] if self.data is not None else avg_forecast

            # 库存建议
            if avg_forecast > current_level * 1.1:
                recommendations.append(
                    f"预计未来{label}将增加约 {(avg_forecast/current_level-1)*100:.1f}%，"
                    f"建议将安全库存水平提高至当前水平的110%-120%"
                )
            elif avg_forecast < current_level * 0.9:
                recommendations.append(
                    f"预计未来{label}将下降约 {(1-avg_forecast/current_level)*100:.1f}%，"
                    f"建议逐步减少采购量，避免库存积压"
                )
            else:
                recommendations.append(
                    f"{label}保持相对稳定，建议维持当前策略，"
                    "但保持对市场变化的敏感度"
                )

        # 基于模型性能的建议
        if self.evaluation_results is not None:
            best_mape = self.evaluation_results.iloc[0]['MAPE']

            if best_mape < 10:
                recommendations.append(
                    f"模型预测准确度高 (MAPE={best_mape:.1f}%)，"
                    "可以依赖模型预测进行精准的库存管理和生产计划"
                )
            elif best_mape < 20:
                recommendations.append(
                    f"模型预测准确度中等 (MAPE={best_mape:.1f}%)，"
                    "建议结合业务经验和模型预测进行决策，并保持一定缓冲"
                )
            else:
                recommendations.append(
                    f"模型预测误差较大 (MAPE={best_mape:.1f}%)，"
                    "建议谨慎使用模型预测，更多依赖历史经验和市场分析"
                )

        # 确保至少有2条建议
        if len(recommendations) < 2:
            recommendations.append(
                "建议建立定期(如每周)的预测复核机制，"
                "根据实际销售数据持续优化预测模型"
            )
            recommendations.append(
                "考虑引入外部数据源(如天气、节假日、促销活动)，"
                "进一步提升预测准确性"
            )

        return recommendations[:2]  # 返回前2条最重要的建议

    def _print_report(self, report):
        """打印报告到控制台"""
        print("\n" + "=" * 80)
        print("FORECASTPRO 管理报告")
        print("=" * 80)

        print(f"\n报告日期: {report['report_date']}")
        print(f"数据概览: {report['data_summary']['total_observations']} 个观测值，"
              f"频率: {report['data_summary']['frequency']}")

        print(f"\n📊 最佳模型: {report['best_model']['name']}")
        best_metrics = report['best_model']['metrics']
        print(f"   - 平均绝对百分比误差 (MAPE): {best_metrics['MAPE']:.2f}%")
        print(f"   - 平均绝对误差 (MAE): {best_metrics['MAE']:.2f}")
        print(f"   - 均方根误差 (RMSE): {best_metrics['RMSE']:.2f}")
        print(f"   - 过拟合风险: {best_metrics['overfitting_risk']}")

        print("\n🏆 模型性能排名:")
        print("-" * 80)
        comparison_df = pd.DataFrame(report['model_comparison'])
        print(comparison_df[['model', 'MAPE', 'MAE', 'RMSE', 'overfitting_risk', 'type']].to_string(index=False))

        # 时间序列分析结果
        if report.get('ts_analysis'):
            print("\n📈 时间序列分析:")
            print("-" * 40)
            ts_results = report['ts_analysis']

            if ts_results.get('stationarity'):
                stat = ts_results['stationarity']
                if stat:
                    stationarity_status = "平稳" if stat.get('is_stationary') else "非平稳"
                    print(f"  平稳性: {stationarity_status} (p值: {stat.get('p_value', 0):.4f})")

            if ts_results.get('decomposition'):
                decomp = ts_results['decomposition']
                if decomp:
                    print(f"  季节性分解:")
                    print(f"    - 趋势成分占比: {decomp.get('trend_ratio', 0):.2%}")
                    print(f"    - 季节性成分占比: {decomp.get('seasonal_ratio', 0):.2%}")
                    print(f"    - 残差成分占比: {decomp.get('residual_ratio', 0):.2%}")

            if ts_results.get('seasonality'):
                season = ts_results['seasonality']
                if season:
                    seasonality_status = "有季节性" if season.get('has_seasonality') else "无显著季节性"
                    print(f"  季节性检测: {seasonality_status}")

        print("\n🔮 业务洞察:")
        print("-" * 40)
        for i, insight in enumerate(report['insights'], 1):
            print(f"{i}. {insight}")

        print("\n💡 行动建议:")
        print("-" * 40)
        for i, recommendation in enumerate(report['recommendations'], 1):
            print(f"{i}. {recommendation}")

        print("\n" + "=" * 80)
        print("报告结束 - ForecastPro AI需求预测系统")
        print("=" * 80)

    def _clean_for_json(self, obj):
        """递归清理对象以便JSON序列化"""
        import pandas as pd
        import numpy as np
        from datetime import datetime as dt
        import math

        if isinstance(obj, (pd.Timestamp, dt)):
            return obj.isoformat()
        elif isinstance(obj, pd.DatetimeIndex):
            return [d.isoformat() for d in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            v = float(obj)
            return v if math.isfinite(v) else None
        elif isinstance(obj, float):
            return obj if math.isfinite(obj) else None
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, dict):
            return {k: self._clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._clean_for_json(item) for item in obj]
        elif isinstance(obj, (bool, np.bool_)):
            return bool(obj)
        else:
            return obj

    def _save_report(self, report, output_dir: str = "./reports"):
        """保存报告到文件"""
        import os

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 清理报告数据，确保JSON可序列化
        clean_report = self._clean_for_json(report)

        # 保存JSON报告
        json_path = os.path.join(output_dir, f"forecast_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(clean_report, f, ensure_ascii=False, indent=2)

        # 保存文本报告
        txt_path = os.path.join(output_dir, f"forecast_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

        report_text = f"""FORECASTPRO 管理报告
生成时间: {report['report_date']}

数据概览:
- 总观测值: {report['data_summary']['total_observations']}
- 训练期: {report['data_summary']['training_period']}
- 测试期: {report['data_summary']['testing_period']}
- 数据频率: {report['data_summary']['frequency']}
- 需求变量: {report['data_summary']['demand_variable']}

最佳模型: {report['best_model']['name']}
- MAPE: {report['best_model']['metrics']['MAPE']:.2f}%
- MAE: {report['best_model']['metrics']['MAE']:.2f}
- RMSE: {report['best_model']['metrics']['RMSE']:.2f}
- 过拟合风险: {report['best_model']['metrics']['overfitting_risk']}

模型性能排名:
{self.evaluation_results.to_string(index=False)}

业务洞察:
{chr(10).join(f'{i}. {insight}' for i, insight in enumerate(report['insights'], 1))}

行动建议:
{chr(10).join(f'{i}. {recommendation}' for i, recommendation in enumerate(report['recommendations'], 1))}

---
ForecastPro AI需求预测系统
减少预测误差，优化运营效率
"""

        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(report_text)

        print(f"\n报告已保存到:")
        print(f"  JSON: {json_path}")
        print(f"  TXT: {txt_path}")

    def run_full_pipeline(self, data_path: str):
        """
        运行完整预测管道

        参数:
            data_path: 数据文件路径
        """
        print("=" * 80)
        print("FORECASTPRO AI需求预测管道启动")
        print("=" * 80)

        # Step 1: 数据摄取与画像
        print("\n📊 Step 1: 数据摄取与画像")
        self.load_data(data_path)

        # Step 2: 基准模型
        print("\n📈 Step 2: 建立时间序列基准")
        self.prepare_data()
        self.run_baseline_models()

        # Step 3: 高级模型
        print("\n🤖 Step 3: 高级模型探索与选择")
        self.run_advanced_models()

        # Step 4: 模型评估
        print("\n📊 Step 4: 模型评估与诊断")
        self.evaluate_models()

        # Step 5: 生成预测
        print("\n🔮 Step 5: 生成未来预测")
        self.generate_forecast(periods=4)

        # Step 6: 管理报告
        print("\n📋 Step 6: 生成管理报告")
        report = self.generate_report()

        print("\n" + "=" * 80)
        print("FORECASTPRO 预测管道执行完成!")
        print("=" * 80)

        return report
