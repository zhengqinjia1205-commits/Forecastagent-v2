#!/usr/bin/env python3
"""
ForecastPro AI需求预测可视化仪表板

基于Streamlit构建的交互式预测可视化工具，提供：
1. 预测结果图表
2. 模型性能比较
3. 业务洞察仪表板
4. 完整预测报告
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
import os
import sys
import warnings
from pathlib import Path

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入ForecastPro Agent
try:
    from forecastpro import ForecastProAgent
    HAS_FORECASTPRO = True
except ImportError as e:
    st.error(f"无法导入ForecastPro Agent: {e}")
    HAS_FORECASTPRO = False

# 忽略警告
warnings.filterwarnings('ignore')

def navigate_to(target_page: str):
    st.session_state.current_page = target_page
    st.rerun()

def reset_pipeline_state(clear_data_split: bool = False):
    st.session_state.report = None
    st.session_state.evaluation_results = None
    st.session_state.forecast_results = None
    st.session_state.ai_analysis_report = None
    st.session_state.chat_history = []
    st.session_state.pipeline_running = False

    agent = st.session_state.agent
    if agent is None:
        return

    if clear_data_split:
        agent.train_data = None
        agent.test_data = None
    agent.baseline_models = {}
    agent.advanced_models = {}
    agent.model_results = {}
    agent.best_model = None
    agent.report = {}
    if hasattr(agent, "evaluation_results"):
        agent.evaluation_results = None
    if hasattr(agent, "forecast_results"):
        agent.forecast_results = None

def horizon_to_periods(count: int, unit: str, freq: str) -> int:
    unit_days = {"day": 1, "week": 7, "month": 30, "quarter": 91, "year": 365}
    base_days = {"D": 1, "W": 7, "M": 30, "Q": 91, "Y": 365}
    u = unit_days.get(unit, 1)
    b = base_days.get(freq, 1)
    periods = int(np.ceil((max(int(count), 1) * u) / b))
    return max(periods, 1)

def clean_for_json(obj):
    """递归清理对象以便JSON序列化"""
    import pandas as pd
    import numpy as np
    from datetime import datetime as dt

    if isinstance(obj, (pd.Timestamp, dt)):
        return obj.isoformat()
    elif isinstance(obj, pd.DatetimeIndex):
        return [d.isoformat() for d in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    elif isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(item) for item in obj]
    else:
        return obj

def analyze_question(question, agent, report, evaluation_results, forecast_results):
    """
    分析用户问题并生成回答

    参数:
        question: 用户问题
        agent: ForecastProAgent实例
        report: 预测报告
        evaluation_results: 模型评估结果
        forecast_results: 预测结果

    返回:
        answer: 回答文本
        report_type: 报告类型（如果有的话）
        visualization_data: 可视化数据（可选）
    """
    question_lower = question.lower()

    # 初始化回答
    answer = ""
    report_type = None
    visualization_data = None

    # 检查数据是否已加载
    if agent is None or report is None:
        return "请先加载数据并运行预测管道。", None, None

    try:
        # 获取变量中文标签
        label = agent.get_variable_label() if hasattr(agent, 'get_variable_label') else '需求'

        # 1. 关于数据的问题
        if any(keyword in question_lower for keyword in ['数据', 'data', '样本', '记录', '观测']):
            data_summary = report.get('data_summary', {})
            answer = f"**数据概览:**\n"
            answer += f"- 总观测值: {data_summary.get('total_observations', 'N/A')}\n"
            answer += f"- 训练期: {data_summary.get('training_period', 'N/A')}\n"
            answer += f"- 测试期: {data_summary.get('testing_period', 'N/A')}\n"
            answer += f"- 数据频率: {data_summary.get('frequency', 'N/A')}\n"
            answer += f"- {label}变量: {data_summary.get('demand_variable', 'N/A')} ({label})\n"

            if hasattr(agent, 'covariates') and agent.covariates:
                answer += f"- 协变量: {', '.join(agent.covariates)}\n"

            # 添加统计信息
            if agent.data is not None:
                demand_data = agent.data[agent.demand_col]
                answer += f"\n**{label}统计:**\n"
                answer += f"- 平均值: {demand_data.mean():.2f}\n"
                answer += f"- 标准差: {demand_data.std():.2f}\n"
                answer += f"- 最小值: {demand_data.min():.2f}\n"
                answer += f"- 最大值: {demand_data.max():.2f}\n"
                answer += f"- 中位数: {demand_data.median():.2f}\n"

                # 增长趋势
                if len(demand_data) > 1:
                    growth_rate = ((demand_data.iloc[-1] - demand_data.iloc[0]) / demand_data.iloc[0]) * 100
                    trend = "增长" if growth_rate > 0 else "下降" if growth_rate < 0 else "稳定"
                    answer += f"- 整体趋势: {trend} ({abs(growth_rate):.1f}%)\n"

        # 2. 关于模型性能的问题
        elif any(keyword in question_lower for keyword in ['模型', 'model', '性能', '表现', 'mape', 'mae', 'rmse', '哪个好', '最佳']):
            if evaluation_results is not None:
                best_model = evaluation_results.iloc[0]
                answer = f"**模型性能分析:**\n"
                answer += f"🏆 **最佳模型**: {best_model['model']}\n"
                answer += f"- MAPE: {best_model['MAPE']:.2f}%\n"
                answer += f"- MAE: {best_model['MAE']:.2f}\n"
                answer += f"- RMSE: {best_model['RMSE']:.2f}\n"
                answer += f"- 过拟合风险: {best_model['overfitting_risk']}\n"

                # 显示前3个模型
                answer += f"\n**TOP 3模型排名:**\n"
                for i in range(min(3, len(evaluation_results))):
                    model = evaluation_results.iloc[i]
                    answer += f"{i+1}. {model['model']}: MAPE={model['MAPE']:.2f}%, 类型={model['type']}\n"

                report_type = "model_performance"
            else:
                answer = "尚未运行模型评估，请先运行预测管道。"

        # 3. 关于预测结果的问题
        elif any(keyword in question_lower for keyword in ['预测', 'forecast', '未来', '将来', '趋势']):
            if forecast_results is not None:
                answer = f"**未来预测结果:**\n"
                answer += f"📅 **预测模型**: {forecast_results.get('model', 'N/A')}\n"
                answer += f"📊 **预测期数**: {len(forecast_results.get('forecast', []))}\n"

                forecasts = forecast_results.get('forecast', [])
                if forecasts:
                    answer += f"\n**详细预测:**\n"
                    dates = forecast_results.get('dates', [])
                    for i, (date, value) in enumerate(zip(dates, forecasts)):
                        if i < 5:  # 只显示前5个
                            answer += f"- {date}: {value:.2f}\n"

                    if len(forecasts) > 1:
                        growth = ((forecasts[-1] - forecasts[0]) / forecasts[0]) * 100
                        answer += f"\n**预测趋势**: "
                        if growth > 5:
                            answer += f"预计增长 {growth:.1f}%"
                        elif growth < -5:
                            answer += f"预计下降 {abs(growth):.1f}%"
                        else:
                            answer += "保持相对稳定"

                report_type = "forecast_summary"
            else:
                answer = "尚未生成预测结果，请先运行预测管道。"

        # 4. 关于时间序列分析的问题
        elif any(keyword in question_lower for keyword in ['时间序列', '季节性', '趋势', '平稳性', 'acf', 'pacf', '自相关']):
            if hasattr(agent, 'ts_analysis_results') and agent.ts_analysis_results:
                ts_results = agent.ts_analysis_results
                answer = f"**时间序列分析结果:**\n"

                # 平稳性分析
                if ts_results.get('stationarity'):
                    stat = ts_results['stationarity']
                    if stat:
                        answer += f"📊 **平稳性检验**:\n"
                        answer += f"- ADF统计量: {stat.get('adf_statistic', 'N/A'):.4f}\n"
                        answer += f"- P值: {stat.get('p_value', 'N/A'):.4f}\n"
                        answer += f"- 是否平稳: {'是' if stat.get('is_stationary') else '否'}\n"
                        if not stat.get('is_stationary'):
                            answer += f"- 建议: 进行差分处理以提高模型稳定性\n"

                # 季节性分解
                if ts_results.get('decomposition'):
                    decomp = ts_results['decomposition']
                    if decomp:
                        answer += f"\n🔍 **季节性分解**:\n"
                        answer += f"- 趋势成分占比: {decomp.get('trend_ratio', 0):.2%}\n"
                        answer += f"- 季节性成分占比: {decomp.get('seasonal_ratio', 0):.2%}\n"
                        answer += f"- 残差成分占比: {decomp.get('residual_ratio', 0):.2%}\n"

                        if decomp.get('has_strong_seasonality'):
                            answer += f"- 结论: 数据呈现强季节性模式\n"
                        elif decomp.get('seasonal_ratio', 0) > 0.1:
                            answer += f"- 结论: 数据呈现中等季节性\n"
                        else:
                            answer += f"- 结论: 数据季节性较弱\n"

                # 季节性检测
                if ts_results.get('seasonality'):
                    season = ts_results['seasonality']
                    if season and season.get('has_seasonality'):
                        answer += f"\n📅 **季节性检测**: 检测到明显的季节性模式\n"

                report_type = "time_series_analysis"
            else:
                answer = "尚未进行时间序列分析，请先加载数据。"

        # 5. 关于业务洞察和建议的问题
        elif any(keyword in question_lower for keyword in ['洞察', '建议', 'action', 'recommendation', '业务', '库存', '生产', '采购']):
            insights = report.get('insights', [])
            recommendations = report.get('recommendations', [])

            answer = f"**业务洞察与建议:**\n"

            if insights:
                answer += f"\n🔮 **业务洞察**:\n"
                for i, insight in enumerate(insights[:5], 1):  # 最多显示5个
                    answer += f"{i}. {insight}\n"

            if recommendations:
                answer += f"\n💡 **行动建议**:\n"
                for i, recommendation in enumerate(recommendations[:3], 1):  # 最多显示3个
                    answer += f"{i}. {recommendation}\n"

            report_type = "business_insights"

        # 6. 请求生成特定报告
        elif any(keyword in question_lower for keyword in ['报告', 'report', '生成', '创建', '导出']):
            if '库存' in question_lower or 'inventory' in question_lower:
                report_type = "inventory_report"
                answer = "正在生成库存管理报告...\n\n"
                answer += generate_inventory_report(agent, report, forecast_results)
            elif '季节性' in question_lower or 'seasonal' in question_lower:
                report_type = "seasonal_report"
                answer = "正在生成季节性分析报告...\n\n"
                answer += generate_seasonal_report(agent, report)
            elif '模型' in question_lower or 'model' in question_lower:
                report_type = "model_report"
                answer = "正在生成模型性能报告...\n\n"
                answer += generate_model_report(evaluation_results, report)
            elif '预测' in question_lower or 'forecast' in question_lower:
                report_type = "forecast_report"
                answer = "正在生成预测结果报告...\n\n"
                answer += generate_forecast_report(forecast_results, report)
            else:
                report_type = "summary_report"
                answer = "正在生成综合报告...\n\n"
                answer += generate_summary_report(agent, report, evaluation_results, forecast_results)

        # 7. 默认回答
        else:
            answer = f"我分析了您的问题: \"{question}\"\n\n"
            answer += "我可以帮助您分析以下内容:\n"
            answer += "1. 📊 数据概览和统计信息\n"
            answer += "2. 🤖 模型性能和比较\n"
            answer += "3. 🔮 未来预测结果\n"
            answer += "4. 📈 时间序列分析（季节性、平稳性等）\n"
            answer += "5. 💡 业务洞察和行动建议\n"
            answer += "6. 📋 生成特定报告（库存、季节性、模型等）\n\n"
            answer += "请尝试更具体的问题，或点击右侧的预定义问题模板。"

        return answer, report_type, visualization_data

    except Exception as e:
        return f"分析问题时出现错误: {str(e)}", None, None

def generate_inventory_report(agent, report, forecast_results):
    """生成库存管理报告"""
    report_text = "**库存管理报告**\n\n"

    if agent is None or report is None:
        return report_text + "数据未加载，无法生成报告。"

    # 获取变量中文标签
    label = agent.get_variable_label() if hasattr(agent, 'get_variable_label') else '需求'

    # 数据概览
    data_summary = report.get('data_summary', {})
    report_text += f"📊 **数据概览**\n"
    report_text += f"- 总观测值: {data_summary.get('total_observations', 'N/A')}\n"
    report_text += f"- 数据频率: {data_summary.get('frequency', 'N/A')}\n"

    # 需求统计
    if agent.data is not None:
        demand_data = agent.data[agent.demand_col]
        report_text += f"\n📈 **{label}统计**\n"
        report_text += f"- 平均{label}: {demand_data.mean():.2f}\n"
        report_text += f"- {label}标准差: {demand_data.std():.2f}\n"
        report_text += f"- 变异系数: {(demand_data.std()/demand_data.mean()*100):.1f}%\n"

        # 计算安全库存建议
        cv = demand_data.std() / demand_data.mean()
        if cv < 0.2:
            safety_stock_percent = 10
        elif cv < 0.4:
            safety_stock_percent = 20
        else:
            safety_stock_percent = 30

        report_text += f"\n🛡️ **安全库存建议**\n"
        report_text += f"- {label}波动性: {'低' if cv < 0.2 else '中' if cv < 0.4 else '高'}\n"
        report_text += f"- 建议安全库存: {safety_stock_percent}% 的平均{label}\n"
        report_text += f"- 计算值: {demand_data.mean() * safety_stock_percent/100:.2f}\n"

    # 预测信息
    if forecast_results is not None:
        forecasts = forecast_results.get('forecast', [])
        if forecasts:
            avg_forecast = np.mean(forecasts)
            report_text += f"\n🔮 **未来预测**\n"
            report_text += f"- 平均预测值: {avg_forecast:.2f}\n"
            report_text += f"- 预测范围: {forecasts[0]:.2f} 到 {forecasts[-1]:.2f}\n"

            # 库存建议
            if agent.data is not None:
                current_level = agent.data[agent.demand_col].iloc[-1]
                if avg_forecast > current_level * 1.1:
                    increase_pct = (avg_forecast/current_level - 1) * 100
                    report_text += f"\n💡 **库存调整建议**\n"
                    report_text += f"- 预计{label}增加: {increase_pct:.1f}%\n"
                    report_text += f"- 建议: 增加库存 {increase_pct:.0f}%\n"
                elif avg_forecast < current_level * 0.9:
                    decrease_pct = (1 - avg_forecast/current_level) * 100
                    report_text += f"\n💡 **库存调整建议**\n"
                    report_text += f"- 预计{label}减少: {decrease_pct:.1f}%\n"
                    report_text += f"- 建议: 减少库存 {decrease_pct:.0f}%\n"
                else:
                    report_text += f"\n💡 **库存调整建议**\n"
                    report_text += f"- {label}保持稳定\n"
                    report_text += f"- 建议: 维持当前库存水平\n"

    # 最佳模型信息
    best_model = report.get('best_model', {})
    if best_model:
        report_text += f"\n🤖 **预测模型**\n"
        report_text += f"- 最佳模型: {best_model.get('name', 'N/A')}\n"
        report_text += f"- 预测准确度 (MAPE): {best_model.get('metrics', {}).get('MAPE', 'N/A'):.2f}%\n"

    report_text += f"\n---\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    return report_text

def generate_seasonal_report(agent, report):
    """生成季节性分析报告"""
    report_text = "**季节性分析报告**\n\n"

    if agent is None:
        return report_text + "数据未加载，无法生成报告。"

    # 时间序列分析结果
    if hasattr(agent, 'ts_analysis_results'):
        ts_results = agent.ts_analysis_results

        report_text += f"📈 **时间序列特性**\n"

        # 平稳性
        if ts_results.get('stationarity'):
            stat = ts_results['stationarity']
            if stat:
                is_stationary = stat.get('is_stationary', False)
                report_text += f"- 平稳性: {'平稳' if is_stationary else '非平稳'}\n"
                if not is_stationary:
                    report_text += f"- 建议: 考虑进行差分处理\n"

        # 季节性分解
        if ts_results.get('decomposition'):
            decomp = ts_results['decomposition']
            if decomp:
                report_text += f"- 趋势成分占比: {decomp.get('trend_ratio', 0):.2%}\n"
                report_text += f"- 季节性成分占比: {decomp.get('seasonal_ratio', 0):.2%}\n"
                report_text += f"- 残差成分占比: {decomp.get('residual_ratio', 0):.2%}\n"

                seasonal_strength = "强" if decomp.get('has_strong_seasonality') else "中等" if decomp.get('seasonal_ratio', 0) > 0.1 else "弱"
                report_text += f"- 季节性强度: {seasonal_strength}\n"

    # 季节性模式分析
    if agent.data is not None and agent.demand_col in agent.data.columns:
        y = agent.data[agent.demand_col]
        freq = report.get('data_summary', {}).get('frequency', 'D')

        report_text += f"\n📅 **季节性模式**\n"

        if freq == 'D' and len(y) >= 7:
            # 周度季节性
            day_of_week = y.groupby(y.index.dayofweek).mean()
            days = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']

            report_text += f"- 分析周期: 周度\n"
            max_day = days[day_of_week.idxmax()]
            min_day = days[day_of_week.idxmin()]
            variation = (day_of_week.max() - day_of_week.min()) / day_of_week.mean() * 100

            report_text += f"- 最高需求日: {max_day} ({day_of_week.max():.2f})\n"
            report_text += f"- 最低需求日: {min_day} ({day_of_week.min():.2f})\n"
            report_text += f"- 周内变异: {variation:.1f}%\n"

            if variation > 30:
                report_text += f"- 结论: 强周度季节性模式\n"
            elif variation > 15:
                report_text += f"- 结论: 中等周度季节性模式\n"
            else:
                report_text += f"- 结论: 弱周度季节性模式\n"

        elif freq == 'M' and len(y) >= 12:
            # 月度季节性
            month_of_year = y.groupby(y.index.month).mean()
            months = ['1月', '2月', '3月', '4月', '5月', '6月', '7月', '8月', '9月', '10月', '11月', '12月']

            report_text += f"- 分析周期: 月度\n"
            max_month = months[month_of_year.idxmax() - 1]
            min_month = months[month_of_year.idxmin() - 1]

            report_text += f"- 最高需求月: {max_month}\n"
            report_text += f"- 最低需求月: {min_month}\n"

    # 业务建议
    report_text += f"\n💡 **季节性管理建议**\n"

    if hasattr(agent, 'ts_analysis_results'):
        ts_results = agent.ts_analysis_results
        decomp = ts_results.get('decomposition', {})
        if decomp and decomp.get('seasonal_ratio', 0) > 0.2:
            report_text += f"1. 数据呈现强季节性，建议使用季节性预测模型\n"
            report_text += f"2. 针对季节性高峰提前准备库存\n"
            report_text += f"3. 考虑季节性促销或定价策略\n"
        elif decomp and decomp.get('seasonal_ratio', 0) > 0.1:
            report_text += f"1. 数据呈现中等季节性，可进行季节性调整\n"
            report_text += f"2. 监控季节性模式变化\n"
            report_text += f"3. 灵活调整生产和库存计划\n"
        else:
            report_text += f"1. 季节性较弱，可按稳定需求规划\n"
            report_text += f"2. 关注长期趋势而非季节性波动\n"
            report_text += f"3. 定期检查季节性模式变化\n"

    report_text += f"\n---\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    return report_text

def generate_model_report(evaluation_results, report):
    """生成模型性能报告"""
    report_text = "**模型性能报告**\n\n"

    if evaluation_results is None:
        return report_text + "尚未运行模型评估，无法生成报告。"

    # 最佳模型
    best_model = evaluation_results.iloc[0]
    report_text += f"🏆 **最佳模型**\n"
    report_text += f"- 模型名称: {best_model['model']}\n"
    report_text += f"- MAPE: {best_model['MAPE']:.2f}%\n"
    report_text += f"- MAE: {best_model['MAE']:.2f}\n"
    report_text += f"- RMSE: {best_model['RMSE']:.2f}\n"
    report_text += f"- 类型: {best_model['type']}\n"
    report_text += f"- 过拟合风险: {best_model['overfitting_risk']}\n"

    # 模型排名
    report_text += f"\n📊 **模型性能排名**\n"
    for i in range(min(5, len(evaluation_results))):
        model = evaluation_results.iloc[i]
        report_text += f"{i+1}. {model['model']}: MAPE={model['MAPE']:.2f}%, 类型={model['type']}\n"

    # 模型类型统计
    baseline_count = len(evaluation_results[evaluation_results['type'] == 'baseline'])
    advanced_count = len(evaluation_results[evaluation_results['type'] == 'advanced'])

    report_text += f"\n🔧 **模型类型分布**\n"
    report_text += f"- 基准模型: {baseline_count} 个\n"
    report_text += f"- 高级模型: {advanced_count} 个\n"
    report_text += f"- 总计: {len(evaluation_results)} 个\n"

    # 性能分析
    avg_mape = evaluation_results['MAPE'].mean()
    min_mape = evaluation_results['MAPE'].min()
    max_mape = evaluation_results['MAPE'].max()

    report_text += f"\n📈 **性能分析**\n"
    report_text += f"- 平均MAPE: {avg_mape:.2f}%\n"
    report_text += f"- 最佳MAPE: {min_mape:.2f}%\n"
    report_text += f"- 最差MAPE: {max_mape:.2f}%\n"
    report_text += f"- 性能范围: {(max_mape - min_mape):.2f}%\n"

    # 过拟合分析
    high_risk = len(evaluation_results[evaluation_results['overfitting_risk'] == '高'])
    medium_risk = len(evaluation_results[evaluation_results['overfitting_risk'] == '中'])
    low_risk = len(evaluation_results[evaluation_results['overfitting_risk'] == '低'])

    report_text += f"\n⚠️ **过拟合风险分析**\n"
    report_text += f"- 高风险: {high_risk} 个模型\n"
    report_text += f"- 中风险: {medium_risk} 个模型\n"
    report_text += f"- 低风险: {low_risk} 个模型\n"

    if high_risk > 0:
        report_text += f"- 建议: 关注高风险模型的泛化能力\n"

    # 模型选择建议
    report_text += f"\n💡 **模型选择建议**\n"
    if best_model['MAPE'] < 10:
        report_text += f"1. 预测准确度高，可依赖模型进行决策\n"
        report_text += f"2. 最佳模型 {best_model['model']} 表现优异\n"
    elif best_model['MAPE'] < 20:
        report_text += f"1. 预测准确度中等，建议结合业务经验\n"
        report_text += f"2. 最佳模型 {best_model['model']} 可作为参考\n"
    else:
        report_text += f"1. 预测误差较大，需谨慎使用\n"
        report_text += f"2. 建议改进数据质量或尝试其他方法\n"

    report_text += f"\n---\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    return report_text

def generate_forecast_report(forecast_results, report):
    """生成预测结果报告"""
    report_text = "**预测结果报告**\n\n"

    if forecast_results is None:
        return report_text + "尚未生成预测结果，无法生成报告。"

    # 预测概览
    report_text += f"📊 **预测概览**\n"
    report_text += f"- 预测模型: {forecast_results.get('model', 'N/A')}\n"
    report_text += f"- 预测期数: {len(forecast_results.get('forecast', []))}\n"
    report_text += f"- 置信水平: 95%\n"

    # 详细预测
    forecasts = forecast_results.get('forecast', [])
    lower_bounds = forecast_results.get('lower_bound', [])
    upper_bounds = forecast_results.get('upper_bound', [])
    dates = forecast_results.get('dates', [])

    if forecasts and dates:
        report_text += f"\n📅 **详细预测值**\n"
        for i, (date, forecast, lower, upper) in enumerate(zip(dates, forecasts, lower_bounds, upper_bounds)):
            report_text += f"- {date}: {forecast:.2f} (范围: {lower:.2f} - {upper:.2f})\n"
            if i >= 9:  # 最多显示10个
                remaining = len(forecasts) - 10
                if remaining > 0:
                    report_text += f"- ... 还有{remaining}期预测\n"
                break

    # 预测趋势分析
    if len(forecasts) >= 2:
        report_text += f"\n📈 **预测趋势分析**\n"

        # 计算增长率
        growth_rate = ((forecasts[-1] - forecasts[0]) / forecasts[0]) * 100

        report_text += f"- 起始值: {forecasts[0]:.2f}\n"
        report_text += f"- 结束值: {forecasts[-1]:.2f}\n"
        report_text += f"- 总体变化: {growth_rate:+.1f}%\n"

        if growth_rate > 5:
            report_text += f"- 趋势: 明显增长\n"
        elif growth_rate < -5:
            report_text += f"- 趋势: 明显下降\n"
        else:
            report_text += f"- 趋势: 相对稳定\n"

        # 计算平均预测值
        avg_forecast = np.mean(forecasts)
        std_forecast = np.std(forecasts)
        cv_forecast = (std_forecast / avg_forecast * 100) if avg_forecast != 0 else 0

        report_text += f"- 平均值: {avg_forecast:.2f}\n"
        report_text += f"- 标准差: {std_forecast:.2f}\n"
        report_text += f"- 变异系数: {cv_forecast:.1f}%\n"

    # 不确定性分析
    if lower_bounds and upper_bounds:
        avg_range = np.mean([upper - lower for upper, lower in zip(upper_bounds, lower_bounds)])
        avg_forecast_value = np.mean(forecasts) if forecasts else 0
        range_percent = (avg_range / avg_forecast_value * 100) if avg_forecast_value != 0 else 0

        report_text += f"\n🎯 **不确定性分析**\n"
        report_text += f"- 平均置信区间宽度: {avg_range:.2f}\n"
        report_text += f"- 相对区间宽度: {range_percent:.1f}%\n"

        if range_percent < 20:
            report_text += f"- 不确定性: 低\n"
        elif range_percent < 40:
            report_text += f"- 不确定性: 中等\n"
        else:
            report_text += f"- 不确定性: 高\n"

    # 业务建议
    report_text += f"\n💡 **业务建议**\n"

    if forecasts:
        avg_forecast = np.mean(forecasts)
        if len(forecasts) >= 2:
            growth_rate = ((forecasts[-1] - forecasts[0]) / forecasts[0]) * 100

            if growth_rate > 10:
                report_text += f"1. 预计需求显著增长 ({growth_rate:.1f}%)，建议增加生产和库存\n"
                report_text += f"2. 提前规划供应链，避免缺货\n"
                report_text += f"3. 考虑扩大产能或增加采购\n"
            elif growth_rate < -10:
                report_text += f"1. 预计需求显著下降 ({abs(growth_rate):.1f}%)，建议减少库存\n"
                report_text += f"2. 调整生产计划，避免积压\n"
                report_text += f"3. 考虑促销或市场拓展\n"
            else:
                report_text += f"1. 需求保持相对稳定，维持当前运营策略\n"
                report_text += f"2. 监控市场变化，及时调整\n"
                report_text += f"3. 优化库存水平，平衡成本和服务\n"

    report_text += f"\n---\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    return report_text

def generate_summary_report(agent, report, evaluation_results, forecast_results):
    """生成综合报告"""
    report_text = "**综合分析报告**\n\n"

    # 数据概览
    if report and 'data_summary' in report:
        data_summary = report['data_summary']
        report_text += f"📊 **数据概览**\n"
        report_text += f"- 总观测值: {data_summary.get('total_observations', 'N/A')}\n"
        report_text += f"- 训练期: {data_summary.get('training_period', 'N/A')}\n"
        report_text += f"- 测试期: {data_summary.get('testing_period', 'N/A')}\n"
        report_text += f"- 数据频率: {data_summary.get('frequency', 'N/A')}\n"
        report_text += f"- 需求变量: {data_summary.get('demand_variable', 'N/A')}\n"

    # 最佳模型
    if report and 'best_model' in report:
        best_model = report['best_model']
        report_text += f"\n🤖 **最佳模型**\n"
        report_text += f"- 模型名称: {best_model.get('name', 'N/A')}\n"
        report_text += f"- MAPE: {best_model.get('metrics', {}).get('MAPE', 'N/A'):.2f}%\n"
        report_text += f"- 过拟合风险: {best_model.get('metrics', {}).get('overfitting_risk', 'N/A')}\n"

    # 时间序列分析
    if hasattr(agent, 'ts_analysis_results'):
        ts_results = agent.ts_analysis_results
        report_text += f"\n📈 **时间序列分析**\n"

        if ts_results.get('stationarity'):
            stat = ts_results['stationarity']
            if stat:
                report_text += f"- 平稳性: {'平稳' if stat.get('is_stationary') else '非平稳'}\n"

        if ts_results.get('decomposition'):
            decomp = ts_results['decomposition']
            if decomp:
                report_text += f"- 季节性强度: {'强' if decomp.get('has_strong_seasonality') else '中等' if decomp.get('seasonal_ratio', 0) > 0.1 else '弱'}\n"

    # 预测摘要
    if forecast_results:
        report_text += f"\n🔮 **预测摘要**\n"
        report_text += f"- 预测期数: {len(forecast_results.get('forecast', []))}\n"

        forecasts = forecast_results.get('forecast', [])
        if len(forecasts) >= 2:
            growth_rate = ((forecasts[-1] - forecasts[0]) / forecasts[0]) * 100
            report_text += f"- 预测趋势: {'增长' if growth_rate > 0 else '下降' if growth_rate < 0 else '稳定'} ({abs(growth_rate):.1f}%)\n"

    # 业务洞察
    if report and 'insights' in report and report['insights']:
        report_text += f"\n💡 **关键洞察**\n"
        for i, insight in enumerate(report['insights'][:3], 1):
            report_text += f"{i}. {insight}\n"

    # 行动建议
    if report and 'recommendations' in report and report['recommendations']:
        report_text += f"\n🚀 **行动建议**\n"
        for i, recommendation in enumerate(report['recommendations'][:2], 1):
            report_text += f"{i}. {recommendation}\n"

    report_text += f"\n---\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    return report_text

# 设置页面配置
st.set_page_config(
    page_title="ForecastPro AI需求预测仪表板",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
    <style>
    :root{
        --fp-bg-image: url("https://images.unsplash.com/photo-1500375592092-40eb2168fd21?auto=format&fit=crop&w=2400&q=80");
        --fp-bg-position: center 52%;
        --fp-mask-top: rgba(7, 18, 26, 0.06);
        --fp-mask-mid: rgba(7, 18, 26, 0.14);
        --fp-mask-bottom: rgba(246, 250, 252, 0.98);
    }
    [data-testid="stAppViewContainer"]{
        background: transparent;
        position: relative;
    }
    [data-testid="stAppViewContainer"]::before{
        content:"";
        position: fixed;
        inset: 0;
        z-index: -2;
        background-image: var(--fp-bg-image);
        background-size: cover;
        background-repeat: no-repeat;
        background-position: var(--fp-bg-position);
        filter: saturate(1.22) contrast(1.16) brightness(0.96);
        transform: scale(1.02);
    }
    [data-testid="stAppViewContainer"]::after{
        content:"";
        position: fixed;
        inset: 0;
        z-index: -1;
        background: linear-gradient(180deg, var(--fp-mask-top) 0%, var(--fp-mask-mid) 46%, var(--fp-mask-bottom) 100%);
        pointer-events: none;
    }
    [data-testid="stHeader"]{
        background: rgba(0,0,0,0);
    }
    [data-testid="stSidebar"]{
        background: rgba(246, 250, 252, 0.82);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(16, 50, 66, 0.10);
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"]{
        color: #15222A;
    }
    .main{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    .block-container{
        padding-top: 3rem;
        padding-bottom: 3rem;
        background: linear-gradient(180deg, rgba(246,250,252,0.0) 0%, rgba(246,250,252,0.86) 260px, rgba(246,250,252,0.86) 100%);
        border-radius: 22px;
        box-shadow: 0 18px 50px rgba(11,31,42,0.14);
    }
    .stMetric {
        background-color: #ffffff;
        padding: 12px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .card {
        background-color: rgba(255,255,255,0.92);
        backdrop-filter: blur(10px);
        padding: 20px;
        border-radius: 10px;
        border-left: 6px solid #2c3e50;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .insight-card {
        background-color: #fff9f0;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #ffe4b5;
        margin-bottom: 10px;
    }
    .model-card {
        background-color: #f0f7ff;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #c2d9ff;
        margin-bottom: 10px;
    }
    .header-text {
        color: #2c3e50;
        font-weight: 600;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# 初始化session state
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'report' not in st.session_state:
    st.session_state.report = None
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = None
if 'forecast_results' not in st.session_state:
    st.session_state.forecast_results = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'ai_analysis_report' not in st.session_state:
    st.session_state.ai_analysis_report = None
if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0
if 'last_config' not in st.session_state:
    st.session_state.last_config = None
if 'pipeline_running' not in st.session_state:
    st.session_state.pipeline_running = False
if 'current_page' not in st.session_state:
    st.session_state.current_page = "🏠 首页"
if 'forecast_periods' not in st.session_state:
    st.session_state.forecast_periods = 10
if 'forecast_horizon_unit' not in st.session_state:
    st.session_state.forecast_horizon_unit = "auto"
if 'forecast_horizon_count' not in st.session_state:
    st.session_state.forecast_horizon_count = 10
if 'auto_run_on_load' not in st.session_state:
    st.session_state.auto_run_on_load = True
if 'last_upload_signature' not in st.session_state:
    st.session_state.last_upload_signature = None
if 'force_reprocess_upload' not in st.session_state:
    st.session_state.force_reprocess_upload = False

# 页面标题
st.title("📈 ForecastPro AI需求预测仪表板")
st.markdown("---")

# 侧边栏导航
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/forecast.png", width=80)
    st.title("导航")

    pages = ["🏠 首页", "📊 数据管理", "🔮 预测结果", "📈 模型比较", "💡 业务洞察", "📋 完整报告", "🤖 AI问答与分析"]
    page_index = pages.index(st.session_state.current_page) if st.session_state.current_page in pages else 0

    selected_page = st.radio(
        "选择页面",
        pages,
        index=page_index,
        key="page_selector"
    )
    if selected_page != st.session_state.current_page:
        st.session_state.current_page = selected_page
    page = st.session_state.current_page

    st.markdown("---")
    st.markdown("### 配置设置")

    # 数据频率选择
    freq_options = {'日': 'D', '周': 'W', '月': 'M', '季': 'Q', '年': 'Y'}
    selected_freq_label = st.selectbox("数据频率", list(freq_options.keys()), index=0)
    freq = freq_options[selected_freq_label]

    # 随机种子
    random_seed = st.number_input("随机种子", min_value=0, max_value=100, value=42)

    # 测试集比例
    test_size = st.slider("测试集比例 (%)", min_value=10, max_value=40, value=20) / 100
    st.checkbox("上传后自动生成报告", value=bool(st.session_state.auto_run_on_load), key="auto_run_on_load")

    current_config = {"freq": freq, "random_seed": random_seed, "test_size": test_size}
    if st.session_state.last_config is None:
        st.session_state.last_config = current_config
    elif st.session_state.last_config != current_config:
        if st.session_state.agent is not None:
            st.session_state.agent.freq = freq
            st.session_state.agent.random_seed = random_seed
            np.random.seed(random_seed)
            st.session_state.agent.test_size = test_size
        reset_pipeline_state(clear_data_split=True)
        st.session_state.last_config = current_config

    st.markdown("---")

    if st.button("🔄 重置会话", type="secondary"):
        st.session_state.clear()
        st.rerun()

    st.markdown("---")
    st.caption("ForecastPro AI需求预测系统")
    st.caption("减少预测误差，优化运营效率")

# 首页
if page == "🏠 首页":
    st.header("欢迎使用ForecastPro AI需求预测仪表板")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="card">
            <h3 class="header-text">📊 数据管理</h3>
            <p>上传您的需求数据，支持CSV和Excel格式。系统自动识别时间列和需求列。</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("进入数据管理", use_container_width=True):
            navigate_to("📊 数据管理")

    with col2:
        st.markdown("""
        <div class="card">
            <h3 class="header-text">🔮 预测结果</h3>
            <p>查看历史需求、模型拟合值和未来预测的交互式图表，包含置信区间。</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("进入预测结果", use_container_width=True):
            navigate_to("🔮 预测结果")

    with col3:
        st.markdown("""
        <div class="card">
            <h3 class="header-text">📈 模型比较</h3>
            <p>对比不同预测模型的性能指标（MAE、RMSE、MAPE），选择最佳模型。</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("进入模型比较", use_container_width=True):
            navigate_to("📈 模型比较")

    col4, col5, col6 = st.columns(3)

    with col4:
        st.markdown("""
        <div class="card">
            <h3 class="header-text">💡 业务洞察</h3>
            <p>获取数据趋势、季节性分析和具体的业务行动建议。</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("进入业务洞察", use_container_width=True):
            navigate_to("💡 业务洞察")

    with col5:
        st.markdown("""
        <div class="card">
            <h3 class="header-text">📋 完整报告</h3>
            <p>生成并下载完整的管理报告，包含所有分析结果和建议。</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("进入完整报告", use_container_width=True):
            navigate_to("📋 完整报告")

    with col6:
        st.markdown("""
        <div class="card">
            <h3 class="header-text">⚡ 快速开始</h3>
            <p>1. 在侧边栏设置数据频率</p>
            <p>2. 前往"数据管理"上传数据</p>
            <p>3. 运行预测并查看结果</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("开始上传数据", use_container_width=True):
            navigate_to("📊 数据管理")

    st.markdown("---")

    # 快速使用示例数据
    st.subheader("快速开始使用示例数据")

    if st.button("🚀 使用示例数据运行完整预测管道", type="primary"):
        with st.spinner("正在加载示例数据并运行预测管道..."):
            try:
                # 创建Agent实例
                agent = ForecastProAgent(
                    time_col='date',
                    demand_col='demand',
                    freq=freq,
                    random_seed=random_seed
                )

                # 加载示例数据
                data_path = os.path.join(os.path.dirname(__file__), 'example_data.csv')
                agent.load_data(data_path)
                agent.prepare_data()
                agent.run_baseline_models()
                agent.run_advanced_models()
                agent.evaluate_models()
                agent.generate_forecast(periods=int(st.session_state.forecast_periods), forecast_method="auto")
                report = agent.generate_report()

                # 保存到session state
                st.session_state.agent = agent
                st.session_state.data_loaded = True
                st.session_state.report = report
                st.session_state.evaluation_results = agent.evaluation_results
                st.session_state.forecast_results = agent.forecast_results

                st.success("✅ 示例数据预测管道运行完成！")
                st.info("请使用上方导航查看预测结果。")

            except Exception as e:
                st.error(f"运行失败: {e}")
                st.exception(e)

# 数据管理页面
elif page == "📊 数据管理":
    st.header("数据管理")

    if not HAS_FORECASTPRO:
        st.error("ForecastPro Agent未正确导入，请检查依赖安装。")
        st.stop()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("上传数据")

        uploaded_file = st.file_uploader(
            "选择数据文件 (CSV或Excel)",
            type=['csv', 'xlsx', 'xls'],
            help="支持CSV和Excel格式，需要包含时间列和需求列",
            key=f"uploader_{st.session_state.uploader_key}"
        )

        if st.button("🧹 清除已选文件", type="secondary"):
            st.session_state.uploader_key += 1
            st.session_state.last_upload_signature = None
            st.session_state.force_reprocess_upload = False
            st.session_state.agent = None
            st.session_state.data_loaded = False
            reset_pipeline_state(clear_data_split=False)
            st.rerun()

        if uploaded_file is not None:
            # 保存上传的文件
            file_ext = Path(uploaded_file.name).suffix.lower()
            temp_path = f"./temp_upload{file_ext}"

            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # 预览数据
            try:
                if file_ext == '.csv':
                    df_preview = pd.read_csv(temp_path, nrows=10)
                else:
                    df_preview = pd.read_excel(temp_path, nrows=10)

                st.subheader("数据预览")
                st.dataframe(df_preview, use_container_width=True)

                # 显示数据信息
                st.subheader("数据信息")
                col_info1, col_info2, col_info3 = st.columns(3)
                with col_info1:
                    st.metric("行数", df_preview.shape[0])
                with col_info2:
                    st.metric("列数", df_preview.shape[1])
                with col_info3:
                    st.metric("文件类型", file_ext)

                col_actions_left, col_actions_right = st.columns([1, 1])
                with col_actions_left:
                    if st.button("🔁 重新识别并运行", type="secondary", use_container_width=True):
                        st.session_state.force_reprocess_upload = True
                        st.rerun()

                should_process = False
                try:
                    import hashlib
                    md5 = hashlib.md5(uploaded_file.getbuffer()).hexdigest()
                    signature = (md5, freq, float(test_size), int(random_seed))
                    should_process = st.session_state.force_reprocess_upload or (st.session_state.last_upload_signature != signature)
                except Exception:
                    signature = None
                    should_process = True

                if should_process:
                    with st.spinner("正在自动识别列并加载数据..."):
                        try:
                            agent = ForecastProAgent(
                                data_path=temp_path,
                                time_col="date",
                                demand_col="demand",
                                freq=freq,
                                random_seed=random_seed
                            )
                            agent.test_size = test_size

                            df = agent.load_data(temp_path)
                            agent.prepare_data()

                            st.session_state.agent = agent
                            st.session_state.data_loaded = True
                            reset_pipeline_state(clear_data_split=False)
                            st.session_state.last_upload_signature = signature
                            st.session_state.force_reprocess_upload = False

                            st.success("✅ 数据已自动识别并加载成功！")
                            st.info(f"加载了 {len(df)} 行数据，时间范围: {df.index[0]} 至 {df.index[-1]}")
                            st.caption(f"识别结果：时间列={agent.time_col}，目标列={agent.demand_col}")

                            with st.expander("列识别结果（可选调整）", expanded=False):
                                columns = df_preview.columns.tolist()
                                detected_time = agent.time_col if agent.time_col in columns else columns[0]
                                detected_demand = agent.demand_col if agent.demand_col in columns else columns[min(1, len(columns)-1)]
                                col_sel1, col_sel2 = st.columns(2)
                                with col_sel1:
                                    override_time_col = st.selectbox("时间列", columns, index=columns.index(detected_time))
                                with col_sel2:
                                    override_demand_col = st.selectbox("需求列", columns, index=columns.index(detected_demand))

                                if st.button("使用选择重新运行", type="primary"):
                                    with st.spinner("正在按选择重新加载并运行..."):
                                        agent2 = ForecastProAgent(
                                            data_path=temp_path,
                                            time_col=override_time_col,
                                            demand_col=override_demand_col,
                                            freq=freq,
                                            random_seed=random_seed
                                        )
                                        agent2.test_size = test_size
                                        df2 = agent2.load_data(temp_path)
                                        agent2.prepare_data()
                                        st.session_state.agent = agent2
                                        st.session_state.data_loaded = True
                                        reset_pipeline_state(clear_data_split=False)
                                        if st.session_state.auto_run_on_load:
                                            agent2.run_baseline_models()
                                            agent2.run_advanced_models()
                                            agent2.evaluate_models()
                                            agent2.generate_forecast(periods=int(st.session_state.forecast_periods), forecast_method="auto")
                                            report2 = agent2.generate_report()
                                            st.session_state.report = report2
                                            st.session_state.evaluation_results = agent2.evaluation_results
                                            st.session_state.forecast_results = agent2.forecast_results
                                            st.session_state.current_page = "💡 业务洞察"
                                        st.rerun()

                            if st.session_state.auto_run_on_load:
                                with st.spinner("正在自动运行预测并生成洞察报告..."):
                                    agent.run_baseline_models()
                                    agent.run_advanced_models()
                                    agent.evaluate_models()
                                    agent.generate_forecast(periods=int(st.session_state.forecast_periods), forecast_method="auto")
                                    report = agent.generate_report()
                                    st.session_state.report = report
                                    st.session_state.evaluation_results = agent.evaluation_results
                                    st.session_state.forecast_results = agent.forecast_results
                                st.session_state.current_page = "💡 业务洞察"
                                st.rerun()

                        except Exception as e:
                            st.error(f"自动加载失败: {e}")
                            st.exception(e)

                # 清理临时文件
                if os.path.exists(temp_path):
                    os.remove(temp_path)

            except Exception as e:
                st.error(f"文件读取失败: {e}")
                st.exception(e)

    with col2:
        st.subheader("数据状态")

        if st.session_state.data_loaded and st.session_state.agent is not None:
            st.success("✅ 数据已加载")

            agent = st.session_state.agent
            df = agent.data

            st.metric("总观测值", len(df))
            train_size_value = "暂无" if not agent.train_data or 'y' not in agent.train_data else len(agent.train_data['y'])
            test_size_value = "暂无" if not agent.test_data or 'y' not in agent.test_data else len(agent.test_data['y'])
            st.metric("训练集大小", train_size_value)
            st.metric("测试集大小", test_size_value)

            # 数据显示
            with st.expander("查看数据摘要"):
                st.write(f"时间范围: {df.index[0]} 至 {df.index[-1]}")
                st.write(f"数据频率: {freq}")
                st.write(f"需求列: {agent.demand_col}")

                if agent.covariates:
                    st.write(f"协变量: {', '.join(agent.covariates)}")

                # 统计摘要
                st.subheader("统计摘要")
                st.dataframe(df[agent.demand_col].describe(), use_container_width=True)

        else:
            st.warning("⚠️ 尚未加载数据")
            st.info("请上传数据文件或使用示例数据")

# 预测结果页面
elif page == "🔮 预测结果":
    st.header("预测结果可视化")

    if not st.session_state.data_loaded or st.session_state.agent is None:
        st.warning("请先加载数据")
        st.info("前往'数据管理'页面上传数据或使用示例数据")
        st.stop()

    agent = st.session_state.agent

    col_controls_left, col_controls_right = st.columns([1, 1])
    with col_controls_left:
        if st.button("🔄 重新运行预测", type="secondary", use_container_width=True):
            reset_pipeline_state(clear_data_split=False)
            st.rerun()
    with col_controls_right:
        with st.expander("⚙️ 预测设置", expanded=False):
            freq_label_map = {"D": "天", "W": "周", "M": "月", "Q": "季", "Y": "年"}
            freq_label = freq_label_map.get(freq, freq)
            st.caption(f"当前数据频率: {freq_label}（预测会按该频率生成日期）")

            unit_display = st.selectbox(
                "预测单位",
                ["按数据频率", "天", "周", "月", "季", "年"],
                index=0,
                key="forecast_horizon_unit_display",
            )
            unit_key = {
                "按数据频率": "auto",
                "天": "day",
                "周": "week",
                "月": "month",
                "季": "quarter",
                "年": "year",
            }[unit_display]
            st.session_state.forecast_horizon_unit = unit_key

            st.number_input(
                f"预测数量（{freq_label if unit_key == 'auto' else unit_display}）",
                min_value=1,
                max_value=3650,
                value=int(st.session_state.forecast_horizon_count),
                key="forecast_horizon_count",
            )

            if st.session_state.forecast_horizon_unit == "auto":
                st.session_state.forecast_periods = int(st.session_state.forecast_horizon_count)
            else:
                st.session_state.forecast_periods = horizon_to_periods(
                    int(st.session_state.forecast_horizon_count),
                    st.session_state.forecast_horizon_unit,
                    freq,
                )
            st.caption(f"将折算为 {int(st.session_state.forecast_periods)} 个预测周期（频率: {freq_label}）")

            if st.button("生成未来预测", type="primary", use_container_width=True):
                try:
                    agent.generate_forecast(periods=int(st.session_state.forecast_periods), forecast_method="auto")
                    st.session_state.forecast_results = agent.forecast_results
                    if st.session_state.report is not None:
                        try:
                            st.session_state.report["forecast_summary"] = clean_for_json(agent.forecast_results)
                        except Exception:
                            pass
                    st.success("✅ 未来预测已更新")
                    st.rerun()
                except Exception as e:
                    st.error(f"生成预测失败: {e}")
                    st.exception(e)

    # 运行预测管道（如果尚未运行）
    if st.session_state.report is None:
        if st.session_state.pipeline_running:
            st.info("预测管道正在运行中，请稍候...")
            st.stop()

        st.info("尚未运行预测管道，正在运行...")

        with st.spinner("正在运行预测管道..."):
            st.session_state.pipeline_running = True
            try:
                if agent.train_data is None or agent.test_data is None:
                    agent.prepare_data()
                agent.run_baseline_models()
                agent.run_advanced_models()
                agent.evaluate_models()
                agent.generate_forecast(periods=int(st.session_state.forecast_periods), forecast_method="auto")
                report = agent.generate_report()

                st.session_state.report = report
                st.session_state.evaluation_results = agent.evaluation_results
                st.session_state.forecast_results = agent.forecast_results

                st.success("✅ 预测管道运行完成！")
            except Exception as e:
                st.error(f"预测管道运行失败: {e}")
                st.exception(e)
                st.stop()
            finally:
                st.session_state.pipeline_running = False

    # 获取数据
    y_train = agent.train_data['y']
    y_test = agent.test_data['y']

    # 创建预测结果图表
    st.subheader("历史需求与预测")

    # 选择要显示的模型
    all_models = {}
    if hasattr(agent, 'baseline_models'):
        all_models.update(agent.baseline_models)
    if hasattr(agent, 'advanced_models'):
        all_models.update(agent.advanced_models)

    if all_models:
        model_names = list(all_models.keys())
        display_name_map = {
            "naive": "Naive",
            "seasonal_naive": "季节性Naive",
            "moving_average": "移动平均(MA)",
            "exponential_smoothing": "指数平滑(ETS)",
            "ets": "指数平滑(ETS)",
            "seasonal_ets": "季节性指数平滑(ETS)",
            "arima": "ARIMA",
            "linear_regression": "线性回归",
            "ridge_regression": "岭回归",
            "lasso_regression": "Lasso回归",
            "random_forest": "随机森林",
            "xgboost": "XGBoost",
            "arima_rf_hybrid": "ARIMA+RF(混合)",
        }
        display_to_key = {}
        for key in model_names:
            display = display_name_map.get(key, key)
            if display in display_to_key and display_to_key[display] != key:
                display = f"{display} ({key})"
            display_to_key[display] = key

        default_display = []
        if hasattr(agent, "best_model") and agent.best_model in model_names:
            best_display = display_name_map.get(agent.best_model, agent.best_model)
            if best_display not in display_to_key:
                best_display = next((d for d, k in display_to_key.items() if k == agent.best_model), best_display)
            default_display.append(best_display)
        if "exponential_smoothing" in model_names:
            ets_display = display_name_map["exponential_smoothing"]
            if ets_display in display_to_key and ets_display not in default_display:
                default_display.append(ets_display)
        if not default_display:
            default_display = list(display_to_key.keys())[:3]

        selected_models = st.multiselect(
            "选择要显示的模型",
            list(display_to_key.keys()),
            default=default_display
        )

        # 创建Plotly图表
        fig = go.Figure()

        # 添加历史数据
        fig.add_trace(go.Scatter(
            x=y_train.index,
            y=y_train.values,
            mode='lines',
            name='历史需求（训练集）',
            line=dict(color='blue', width=2),
            opacity=0.7
        ))

        fig.add_trace(go.Scatter(
            x=y_test.index,
            y=y_test.values,
            mode='lines',
            name='历史需求（测试集）',
            line=dict(color='green', width=2),
            opacity=0.7
        ))

        # 添加模型预测
        colors = px.colors.qualitative.Set3
        for i, model_display in enumerate(selected_models):
            model_key = display_to_key.get(model_display)
            if model_key in all_models:
                predictions = all_models[model_key]['predictions']
                if len(predictions) == len(y_test):
                    color = colors[i % len(colors)]
                    fig.add_trace(go.Scatter(
                        x=y_test.index,
                        y=predictions,
                        mode='lines',
                        name=f'{model_display} 预测',
                        line=dict(color=color, width=2, dash='dash'),
                        opacity=0.8
                    ))

        # 添加未来预测（如果有）
        if st.session_state.forecast_results:
            forecast = st.session_state.forecast_results
            fig.add_trace(go.Scatter(
                x=forecast['dates'],
                y=forecast['forecast'],
                mode='lines+markers',
                name='未来预测',
                line=dict(color='red', width=3),
                marker=dict(size=8)
            ))

            # 添加置信区间
            fig.add_trace(go.Scatter(
                x=list(forecast['dates']) + list(forecast['dates'])[::-1],
                y=list(forecast['upper_bound']) + list(forecast['lower_bound'])[::-1],
                fill='toself',
                fillcolor='rgba(255,0,0,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='95% 置信区间',
                showlegend=True
            ))

        # 更新布局
        fig.update_layout(
            title='需求预测结果',
            xaxis_title='时间',
            yaxis_title='需求量',
            hovermode='x unified',
            height=600,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )

        st.plotly_chart(fig, use_container_width=True)

        # 预测结果表格
        with st.expander("查看详细预测数据"):
            if st.session_state.forecast_results:
                forecast_df = pd.DataFrame({
                    '日期': st.session_state.forecast_results['dates'],
                    '预测值': st.session_state.forecast_results['forecast'],
                    '下限': st.session_state.forecast_results['lower_bound'],
                    '上限': st.session_state.forecast_results['upper_bound']
                })
                st.dataframe(forecast_df, use_container_width=True)

    else:
        st.warning("没有可用的模型预测结果")

# 模型比较页面
elif page == "📈 模型比较":
    st.header("模型性能比较")

    if st.session_state.evaluation_results is None:
        st.warning("请先运行预测管道")
        st.info("前往'预测结果'页面运行预测")
        st.stop()

    evaluation_df = st.session_state.evaluation_results

    st.subheader("模型性能排名")

    # 性能指标选择
    metric_options = ['MAPE', 'MAE', 'RMSE']
    selected_metric = st.selectbox("选择排序指标", metric_options, index=0)

    # 排序
    sorted_df = evaluation_df.sort_values(selected_metric)

    # 显示表格
    st.dataframe(sorted_df, use_container_width=True)

    # 可视化比较
    st.subheader("模型性能可视化")

    col1, col2 = st.columns(2)

    with col1:
        # MAPE条形图
        fig_mape = px.bar(
            sorted_df,
            x='model',
            y='MAPE',
            color='type',
            title='模型MAPE比较',
            labels={'MAPE': '平均绝对百分比误差 (%)', 'model': '模型'},
            color_discrete_map={'baseline': '#3498db', 'advanced': '#2ecc71'}
        )
        fig_mape.update_layout(height=400)
        st.plotly_chart(fig_mape, use_container_width=True)

    with col2:
        # 雷达图（如果模型数量合适）
        if len(sorted_df) <= 8:
            # 标准化指标用于雷达图
            radar_df = sorted_df.copy()
            for col in ['MAE', 'RMSE', 'MAPE']:
                radar_df[col] = (radar_df[col] - radar_df[col].min()) / (radar_df[col].max() - radar_df[col].min() + 1e-8)

            fig_radar = go.Figure()

            for i, row in radar_df.iterrows():
                fig_radar.add_trace(go.Scatterpolar(
                    r=[row['MAE'], row['RMSE'], row['MAPE'], row['MAE']],
                    theta=['MAE', 'RMSE', 'MAPE', 'MAE'],
                    name=row['model'],
                    fill='toself',
                    opacity=0.5
                ))

            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                showlegend=True,
                height=400,
                title="模型性能雷达图（越小越好）"
            )

            st.plotly_chart(fig_radar, use_container_width=True)
        else:
            # MAE和RMSE散点图
            fig_scatter = px.scatter(
                sorted_df,
                x='MAE',
                y='RMSE',
                color='type',
                size='MAPE',
                hover_name='model',
                title='模型MAE vs RMSE',
                labels={'MAE': '平均绝对误差', 'RMSE': '均方根误差'},
                color_discrete_map={'baseline': '#3498db', 'advanced': '#2ecc71'}
            )
            fig_scatter.update_layout(height=400)
            st.plotly_chart(fig_scatter, use_container_width=True)

    # 最佳模型信息
    best_model_row = sorted_df.iloc[0]
    st.subheader(f"🏆 最佳模型: {best_model_row['model']}")

    col_best1, col_best2, col_best3, col_best4 = st.columns(4)
    with col_best1:
        st.metric("MAPE", f"{best_model_row['MAPE']:.2f}%")
    with col_best2:
        st.metric("MAE", f"{best_model_row['MAE']:.2f}")
    with col_best3:
        st.metric("RMSE", f"{best_model_row['RMSE']:.2f}")
    with col_best4:
        st.metric("过拟合风险", best_model_row['overfitting_risk'])

# 业务洞察页面
elif page == "💡 业务洞察":
    st.header("业务洞察与建议")

    if st.session_state.report is None:
        st.warning("请先运行预测管道")
        st.info("前往'预测结果'页面运行预测")
        st.stop()

    report = st.session_state.report

    # 数据概览
    st.subheader("📊 数据概览")

    col_insight1, col_insight2, col_insight3, col_insight4 = st.columns(4)
    with col_insight1:
        st.metric("总观测值", report['data_summary']['total_observations'])
    with col_insight2:
        st.metric("训练期", report['data_summary']['training_period'])
    with col_insight3:
        st.metric("测试期", report['data_summary']['testing_period'])
    with col_insight4:
        st.metric("数据频率", report['data_summary']['frequency'])

    # 业务洞察
    st.subheader("🔮 业务洞察")

    if 'insights' in report and report['insights']:
        for i, insight in enumerate(report['insights'], 1):
            st.markdown(f"""
            <div class="insight-card">
                <strong>{i}. {insight}</strong>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("暂无业务洞察")

    # 行动建议
    st.subheader("💡 行动建议")

    if 'recommendations' in report and report['recommendations']:
        for i, recommendation in enumerate(report['recommendations'], 1):
            st.markdown(f"""
            <div class="insight-card" style="background-color: #f0f7ff; border-color: #c2d9ff;">
                <strong>{i}. {recommendation}</strong>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("暂无行动建议")

    # 趋势分析（如果可能）
    if st.session_state.agent is not None:
        st.subheader("📈 趋势分析")

        agent = st.session_state.agent
        if agent.data is not None:
            y = agent.data[agent.demand_col]

            # 计算移动平均
            window_size = min(7, len(y) // 10)
            if window_size > 1:
                moving_avg = y.rolling(window=window_size).mean()

                # 趋势图表
                fig_trend = go.Figure()
                fig_trend.add_trace(go.Scatter(
                    x=y.index,
                    y=y.values,
                    mode='lines',
                    name='原始需求',
                    line=dict(color='blue', width=1),
                    opacity=0.5
                ))
                fig_trend.add_trace(go.Scatter(
                    x=moving_avg.index,
                    y=moving_avg.values,
                    mode='lines',
                    name=f'{window_size}期移动平均',
                    line=dict(color='red', width=2)
                ))

                fig_trend.update_layout(
                    title='需求趋势分析',
                    xaxis_title='时间',
                    yaxis_title='需求量',
                    height=400
                )

                st.plotly_chart(fig_trend, use_container_width=True)

            # 季节性分析（如果数据足够）
            if len(y) >= 14 and freq == 'D':
                st.subheader("📅 季节性分析")

                # 简单的周度季节性
                y_dayofweek = y.groupby(y.index.dayofweek).mean()
                days = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']

                fig_seasonal = px.bar(
                    x=days,
                    y=y_dayofweek.values,
                    title='周度季节性模式',
                    labels={'x': '星期', 'y': '平均需求量'}
                )
                fig_seasonal.update_layout(height=300)
                st.plotly_chart(fig_seasonal, use_container_width=True)

# 完整报告页面
elif page == "📋 完整报告":
    st.header("完整预测报告")

    if st.session_state.report is None:
        st.warning("请先运行预测管道")
        st.info("前往'预测结果'页面运行预测")
        st.stop()

    report = st.session_state.report

    # 报告标题
    st.markdown(f"""
    <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white; margin-bottom: 30px;">
        <h1>FORECASTPRO 管理报告</h1>
        <p>生成时间: {report['report_date']}</p>
    </div>
    """, unsafe_allow_html=True)

    # 数据概览
    st.subheader("📊 数据概览")
    data_summary = report['data_summary']

    col_report1, col_report2 = st.columns(2)

    with col_report1:
        st.markdown(f"""
        - **总观测值**: {data_summary['total_observations']}
        - **训练期**: {data_summary['training_period']}
        - **测试期**: {data_summary['testing_period']}
        - **数据频率**: {data_summary['frequency']}
        """)

    with col_report2:
        st.markdown(f"""
        - **需求变量**: {data_summary['demand_variable']}
        - **最佳模型**: {report['best_model']['name']}
        - **测试集MAPE**: {report['best_model']['metrics']['MAPE']:.2f}%
        - **过拟合风险**: {report['best_model']['metrics']['overfitting_risk']}
        """)

    # 模型性能排名
    st.subheader("🏆 模型性能排名")

    if st.session_state.evaluation_results is not None:
        evaluation_df = st.session_state.evaluation_results
        st.dataframe(evaluation_df, use_container_width=True)

    # 业务洞察
    st.subheader("🔮 业务洞察")

    if 'insights' in report and report['insights']:
        for i, insight in enumerate(report['insights'], 1):
            st.markdown(f"- **{insight}**")

    # 行动建议
    st.subheader("💡 行动建议")

    if 'recommendations' in report and report['recommendations']:
        for i, recommendation in enumerate(report['recommendations'], 1):
            st.markdown(f"{i}. {recommendation}")

    # 预测摘要
    if st.session_state.forecast_results:
        st.subheader("🔮 未来预测摘要")

        forecast = st.session_state.forecast_results
        forecast_df = pd.DataFrame({
            '日期': forecast['dates'],
            '预测值': forecast['forecast'],
            '下限': forecast['lower_bound'],
            '上限': forecast['upper_bound']
        })

        st.dataframe(forecast_df, use_container_width=True)

    # 导出报告
    st.subheader("📤 导出报告")

    col_export1, col_export2 = st.columns(2)

    with col_export1:
        # JSON导出
        report_json = json.dumps(clean_for_json(report), ensure_ascii=False, indent=2)
        st.download_button(
            label="📄 下载JSON报告",
            data=report_json,
            file_name=f"forecast_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

    with col_export2:
        # 文本报告
        report_text = f"""FORECASTPRO 管理报告
生成时间: {report['report_date']}

数据概览:
- 总观测值: {data_summary['total_observations']}
- 训练期: {data_summary['training_period']}
- 测试期: {data_summary['testing_period']}
- 数据频率: {data_summary['frequency']}
- 需求变量: {data_summary['demand_variable']}

最佳模型: {report['best_model']['name']}
- MAPE: {report['best_model']['metrics']['MAPE']:.2f}%
- MAE: {report['best_model']['metrics']['MAE']:.2f}
- RMSE: {report['best_model']['metrics']['RMSE']:.2f}
- 过拟合风险: {report['best_model']['metrics']['overfitting_risk']}

业务洞察:
{chr(10).join(f'{i}. {insight}' for i, insight in enumerate(report.get('insights', []), 1))}

行动建议:
{chr(10).join(f'{i}. {recommendation}' for i, recommendation in enumerate(report.get('recommendations', []), 1))}

---
ForecastPro AI需求预测系统
减少预测误差，优化运营效率
"""

        st.download_button(
            label="📝 下载文本报告",
            data=report_text,
            file_name=f"forecast_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

# AI问答与分析页面
elif page == "🤖 AI问答与分析":
    st.header("🤖 AI问答与分析助手")

    if not st.session_state.data_loaded or st.session_state.agent is None:
        st.warning("请先加载数据")
        st.info("前往'数据管理'页面上传数据或使用示例数据")
        st.stop()

    agent = st.session_state.agent
    report = st.session_state.report
    evaluation_results = st.session_state.evaluation_results
    forecast_results = st.session_state.forecast_results

    # 确保已运行预测管道
    if report is None:
        st.info("正在运行预测管道以获取分析数据...")
        with st.spinner("正在运行预测管道..."):
            try:
                agent.run_baseline_models()
                agent.run_advanced_models()
                agent.evaluate_models()
                agent.generate_forecast(periods=int(st.session_state.forecast_periods), forecast_method="auto")
                report = agent.generate_report()

                st.session_state.report = report
                st.session_state.evaluation_results = agent.evaluation_results
                st.session_state.forecast_results = agent.forecast_results

                st.success("✅ 预测管道运行完成！")
                st.rerun()
            except Exception as e:
                st.error(f"预测管道运行失败: {e}")
                st.exception(e)
                st.stop()

    # 创建两列布局
    col_chat, col_templates = st.columns([2, 1])

    with col_chat:
        st.subheader("💬 智能问答")

        # 聊天历史显示
        chat_container = st.container()
        with chat_container:
            if st.session_state.chat_history:
                for chat in st.session_state.chat_history[-10:]:  # 显示最近10条消息
                    if chat['role'] == 'user':
                        st.markdown(f"""
                        <div style="background-color: #e3f2fd; padding: 10px; border-radius: 10px; margin-bottom: 10px; text-align: right;">
                            <strong>您:</strong> {chat['content']}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="background-color: #f5f5f5; padding: 15px; border-radius: 10px; margin-bottom: 15px;">
                            <strong>🤖 AI助手:</strong><br/>
                            {chat['content']}
                        </div>
                        """, unsafe_allow_html=True)

                        # 如果有报告类型，显示下载按钮
                        if chat.get('report_type'):
                            report_text = chat.get('report_text', '')
                            if report_text:
                                st.download_button(
                                    label="📥 下载此报告",
                                    data=report_text,
                                    file_name=f"{chat['report_type']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                    mime="text/plain",
                                    key=f"download_{len(st.session_state.chat_history)}"
                                )
            else:
                st.info("👋 您好！我是ForecastPro AI助手。我可以帮您分析数据、解释预测结果，并生成特定报告。")

        # 输入框
        st.markdown("---")
        question = st.text_area("输入您的问题:", placeholder="例如：哪个模型表现最好？数据有什么季节性特征？生成库存报告...", height=100)

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🚀 发送问题", type="primary", use_container_width=True):
                if question.strip():
                    with st.spinner("正在分析您的问题..."):
                        # 分析问题
                        answer, report_type, visualization_data = analyze_question(
                            question, agent, report, evaluation_results, forecast_results
                        )

                        # 保存到聊天历史
                        st.session_state.chat_history.append({
                            'role': 'user',
                            'content': question,
                            'timestamp': datetime.now().strftime('%H:%M:%S')
                        })

                        # 生成报告文本（如果需要）
                        report_text = ""
                        if report_type:
                            if report_type == "inventory_report":
                                report_text = generate_inventory_report(agent, report, forecast_results)
                            elif report_type == "seasonal_report":
                                report_text = generate_seasonal_report(agent, report)
                            elif report_type == "model_report":
                                report_text = generate_model_report(evaluation_results, report)
                            elif report_type == "forecast_report":
                                report_text = generate_forecast_report(forecast_results, report)
                            elif report_type == "summary_report":
                                report_text = generate_summary_report(agent, report, evaluation_results, forecast_results)

                        st.session_state.chat_history.append({
                            'role': 'assistant',
                            'content': answer,
                            'report_type': report_type,
                            'report_text': report_text,
                            'timestamp': datetime.now().strftime('%H:%M:%S')
                        })

                        st.rerun()
                else:
                    st.warning("请输入问题")

        with col_btn2:
            if st.button("🗑️ 清除对话", type="secondary", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

    with col_templates:
        st.subheader("📋 快速提问模板")

        st.markdown("""
        <div class="card" style="padding: 15px; margin-bottom: 15px;">
            <strong>点击问题快速提问</strong>
        </div>
        """, unsafe_allow_html=True)

        # 预定义问题
        predefined_questions = [
            ("📊 数据概览", "数据的基本情况和统计信息是什么？"),
            ("🤖 模型比较", "哪个预测模型表现最好？"),
            ("🔮 未来预测", "未来的需求预测结果是什么？"),
            ("📈 时间序列分析", "数据有什么时间序列特征？"),
            ("💡 业务洞察", "有什么业务洞察和建议？"),
            ("🛒 库存报告", "生成库存管理报告"),
            ("📅 季节性报告", "生成季节性分析报告"),
            ("📋 模型报告", "生成模型性能报告"),
            ("🔮 预测报告", "生成预测结果报告"),
            ("🌟 综合报告", "生成综合分析报告")
        ]

        for icon, question_text in predefined_questions:
            if st.button(f"{icon} {question_text}", key=f"predefined_{question_text}", use_container_width=True):
                # 直接触发分析
                with st.spinner("正在分析..."):
                    answer, report_type, visualization_data = analyze_question(
                        question_text, agent, report, evaluation_results, forecast_results
                    )

                    # 保存到聊天历史
                    st.session_state.chat_history.append({
                        'role': 'user',
                        'content': question_text,
                        'timestamp': datetime.now().strftime('%H:%M:%S')
                    })

                    # 生成报告文本（如果需要）
                    report_text = ""
                    if report_type:
                        if report_type == "inventory_report":
                            report_text = generate_inventory_report(agent, report, forecast_results)
                        elif report_type == "seasonal_report":
                            report_text = generate_seasonal_report(agent, report)
                        elif report_type == "model_report":
                            report_text = generate_model_report(evaluation_results, report)
                        elif report_type == "forecast_report":
                            report_text = generate_forecast_report(forecast_results, report)
                        elif report_type == "summary_report":
                            report_text = generate_summary_report(agent, report, evaluation_results, forecast_results)

                    st.session_state.chat_history.append({
                        'role': 'assistant',
                        'content': answer,
                        'report_type': report_type,
                        'report_text': report_text,
                        'timestamp': datetime.now().strftime('%H:%M:%S')
                    })

                    st.rerun()

        st.markdown("---")
        st.subheader("💡 使用提示")

        st.markdown("""
        <div class="insight-card">
            <strong>📌 您可以问：</strong><br/>
            1. 关于数据的任何问题<br/>
            2. 模型性能比较<br/>
            3. 预测结果解释<br/>
            4. 业务建议<br/>
            5. 生成特定报告
        </div>
        """, unsafe_allow_html=True)

    # 如果有可视化数据，显示图表
    # 这里可以扩展以支持图表显示

# 页面底部
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9em;">
    <p>ForecastPro AI需求预测系统 | 基于运筹学驱动 | © 2026</p>
    <p>减少预测误差，优化运营效率 📈</p>
</div>
""", unsafe_allow_html=True)
