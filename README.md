# ForecastPro AI需求预测Agent系统

🤖 **ForecastPro** 是一个高级运营管理AI Agent，专门负责为企业设计和执行AI驱动的需求预测任务。核心目标是减少因预测不准导致的缺货、库存积压或产能闲置，从而提高运营效率。

## 🎯 核心功能

按照系统指令，ForecastPro实现完整的5步预测管道：

### Step 1: 数据摄取与画像 (Data Profiling)
- **数据读取**: 接受CSV、Excel等格式的历史数据
- **特征识别**: 自动识别时间索引、需求变量以及协变量（如促销、节假日、价格）
- **质量检查**: 检查并报告缺失值、异常值和不规则时间步长

### Step 2: 建立时间序列基准 (Baseline Models)
拟合以下经典模型作为分析基础：
- **Naïve / Seasonal Naïve**: 作为最简单的基准
- **移动平均线 (MA)**: 自动选择最佳窗口长度
- **指数平滑 (ETS)**: 根据数据特征自动选择Simple, Holt或Holt-Winters
- **ARIMA / SARIMA**: 通过AIC自动确定阶数

### Step 3: 高级模型探索与选择 (Advanced Models)
在基准之上，利用AI推理选择并运行以下至少一类高级模型：
- **回归分析**: 基于滞后需求和协变量的OLS或Lasso/Ridge回归
- **机器学习**: XGBoost、Random Forest，配合时间感知交叉验证
- **深度学习**: LSTM、Temporal Fusion Transformer（视数据量而定）
- **混合模型**: 如使用ML修正ARIMA的残差

### Step 4: 模型评估与诊断
- **留出测试**: 使用最后20%的观测值作为测试集
- **统一指标**: 报告所有模型的MAE, RMSE和MAPE
- **警示标识**: 明确指出高级模型中是否存在过拟合或数据泄露的风险

### Step 5: 输出管理报告 (Managerial Report)
生成非技术经理也能读懂的报告，包含：
- **预测图表**: 历史需求、拟合值及未来至少4个周期的预测值（含95%置信区间）
- **对比表**: 按测试集MAPE对模型进行排名，高亮推荐模型
- **通俗总结**: 解释胜出模型的原因，描述未来的需求高峰、趋势和季节性
- **行动建议**: 给出两项关于库存目标、人员配备或采购计划的具体建议

## 📁 项目结构

```
forecastpro_agent/
├── __init__.py              # 包初始化文件
├── forecastpro.py           # 核心Agent类
├── example_data.csv         # 示例数据
├── run_example.py           # 示例运行脚本
├── requirements.txt         # 依赖包列表
└── README.md               # 项目说明
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行示例

```bash
python run_example.py
```

### 3. 在自己的项目中使用

```python
from forecastpro import ForecastProAgent

# 创建Agent实例
agent = ForecastProAgent(
    time_col='date',      # 时间列名
    demand_col='demand',  # 需求列名
    freq='D',             # 数据频率 (D=日, W=周, M=月)
    random_seed=42
)

# 加载数据
agent.load_data('your_data.csv')

# 运行完整预测管道
report = agent.run_full_pipeline('your_data.csv')

# 或分步运行
agent.prepare_data()
agent.run_baseline_models()
agent.run_advanced_models()
agent.evaluate_models()
agent.generate_forecast(periods=4)
report = agent.generate_report()
```

## 📊 输入数据格式

### 基本要求
- 时间序列数据，包含至少一个时间列和一个需求列
- 支持CSV或Excel格式

### 示例数据格式

```csv
date,demand,promotion,price,holiday
2023-01-01,100,0,10.0,1
2023-01-02,120,1,9.5,0
2023-01-03,115,0,10.0,0
...
```

### 列说明
- **时间列**: 日期时间格式，自动识别列名如`date`, `time`, `timestamp`
- **需求列**: 数值型需求数据，自动识别列名如`demand`, `sales`, `quantity`
- **协变量**: 可选，如促销活动、价格、节假日标记等

## 📈 输出报告

ForecastPro生成两种格式的报告：

### 1. JSON报告 (`forecast_report_YYYYMMDD_HHMMSS.json`)
包含完整的预测结果、模型评估和业务洞察。

### 2. 文本报告 (`forecast_report_YYYYMMDD_HHMMSS.txt`)
适合非技术经理阅读的管理报告，包含：
- 数据概览
- 最佳模型选择及性能指标
- 模型性能排名
- 业务洞察
- 具体行动建议

## 🔧 配置选项

创建Agent时可配置的参数：

```python
agent = ForecastProAgent(
    data_path='data.csv',     # 数据文件路径（可选）
    time_col='date',          # 时间列名称
    demand_col='demand',      # 需求列名称
    freq='D',                 # 时间频率 ('D'=日, 'W'=周, 'M'=月)
    random_seed=42            # 随机种子
)
```

## 🧪 支持的模型

### 基准模型
- Naïve / Seasonal Naïve
- 移动平均 (Moving Average)
- 指数平滑 (Exponential Smoothing)
- ARIMA / SARIMA

### 高级模型
- 线性回归 (Linear Regression)
- Ridge/Lasso回归
- 随机森林 (Random Forest)
- XGBoost
- 神经网络 (Neural Network，需要TensorFlow)
- 混合模型 (ARIMA + ML)

## ⚠️ 注意事项

1. **数据量要求**: 建议至少50个观测值以获得可靠预测
2. **缺失值处理**: Agent会检测并报告缺失值，但不会自动填充（需用户决策）
3. **季节性数据**: 对于有明显季节性的数据，建议提供足够的历史数据（至少2-3个完整周期）
4. **计算资源**: 高级模型（特别是深度学习）可能需要较多计算资源

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进ForecastPro Agent！

### 开发环境设置

```bash
# 克隆仓库
git clone <repository-url>

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 安装开发依赖
pip install -r requirements.txt
```

### 运行测试

```bash
python run_example.py
```

## 📄 许可证

本项目采用MIT许可证。详见LICENSE文件。

## 🙏 致谢

ForecastPro Agent基于以下开源库构建：
- pandas / numpy - 数据处理
- scikit-learn - 机器学习
- statsmodels - 时间序列分析
- xgboost - 梯度提升树
- matplotlib - 数据可视化

---

**ForecastPro AI需求预测系统** - 减少预测误差，优化运营效率 🚀