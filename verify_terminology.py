#!/usr/bin/env python3
"""
快速验证术语适配功能
只测试标签映射和洞察生成，不运行完整模型
"""

import sys
import os
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from forecastpro import ForecastProAgent

def test_label_mapping():
    """测试变量标签映射"""
    print("测试变量标签映射...")
    print("=" * 60)

    test_cases = [
        ('demand', '需求'),
        ('sales', '销售额'),
        ('sales_revenue', '销售收入'),
        ('revenue', '收入'),
        ('quantity', '数量'),
        ('units_sold', '销售数量'),
        ('volume', '成交量'),
        ('consumption', '消耗量'),
        ('cost', '成本'),
        ('price', '价格'),
        ('y', '目标变量'),
        ('target', '目标变量'),
        ('unknown_column', 'unknown_column'),
    ]

    all_pass = True
    for var_name, expected_label in test_cases:
        agent = ForecastProAgent(demand_col=var_name)
        label = agent.get_variable_label()
        pass_status = label == expected_label
        all_pass = all_pass and pass_status

        status = "✓" if pass_status else "✗"
        print(f"{status} {var_name:15s} -> {label:10s} (期望: {expected_label})")

    print(f"\n标签映射测试: {'全部通过 ✓' if all_pass else '有失败 ✗'}")

def test_insight_generation():
    """测试洞察生成（使用模拟数据）"""
    print(f"\n测试洞察生成...")
    print("=" * 60)

    # 测试不同的变量类型
    test_vars = ['demand', 'sales_revenue', 'cost', 'units_sold']

    for var_name in test_vars:
        print(f"\n测试变量: {var_name}")

        # 创建Agent
        agent = ForecastProAgent(demand_col=var_name)

        # 模拟一些数据
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        values = list(range(100, 200))

        # 创建模拟的data属性
        agent.data = pd.DataFrame({var_name: values}, index=dates)
        agent.demand_col = var_name

        # 模拟一些分析结果
        agent.ts_analysis_results = {
            'stationarity': {'is_stationary': True, 'p_value': 0.01},
            'decomposition': {'seasonal_ratio': 0.15, 'has_strong_seasonality': False}
        }

        agent.best_model = 'exponential_smoothing'
        agent.forecast_results = {'forecast': [195, 196, 197, 198]}

        # 生成洞察
        insights = agent._generate_insights()

        # 获取标签
        label = agent.get_variable_label()

        print(f"  变量标签: {label}")
        print(f"  生成的洞察 ({len(insights)} 个):")

        # 检查洞察中是否使用了正确的标签
        uses_correct_label = any(label in insight for insight in insights)
        uses_hardcoded_demand = any('需求' in insight and label != '需求' for insight in insights)

        for i, insight in enumerate(insights[:3], 1):
            print(f"    {i}. {insight}")

        if uses_correct_label:
            print(f"  ✓ 洞察中使用了正确的标签 '{label}'")
        else:
            print(f"  ✗ 洞察中没有使用正确的标签 '{label}'")

        if uses_hardcoded_demand and label != '需求':
            print(f"  ⚠️  警告: 洞察中检测到硬编码的'需求'")

def test_recommendation_generation():
    """测试建议生成（使用模拟数据）"""
    print(f"\n测试建议生成...")
    print("=" * 60)

    test_vars = ['demand', 'sales_revenue', 'cost']

    for var_name in test_vars:
        print(f"\n测试变量: {var_name}")

        # 创建Agent
        agent = ForecastProAgent(demand_col=var_name)

        # 模拟数据
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        values = [100 + i*2 for i in range(50)]

        agent.data = pd.DataFrame({var_name: values}, index=dates)
        agent.demand_col = var_name
        agent.evaluation_results = pd.DataFrame({
            'model': ['model1', 'model2'],
            'MAPE': [5.0, 10.0],
            'MAE': [10.0, 20.0],
            'RMSE': [15.0, 25.0],
            'overfitting_risk': ['低', '中'],
            'type': ['baseline', 'advanced']
        })

        agent.forecast_results = {
            'forecast': [200, 205, 210, 215],
            'dates': pd.date_range('2023-02-20', periods=4, freq='D')
        }

        # 生成建议
        recommendations = agent._generate_recommendations()

        # 获取标签
        label = agent.get_variable_label()

        print(f"  变量标签: {label}")
        print(f"  生成的建议 ({len(recommendations)} 个):")

        # 检查建议中是否使用了正确的标签
        uses_correct_label = any(label in rec for rec in recommendations)
        uses_hardcoded_demand = any('需求' in rec and label != '需求' for rec in recommendations)

        for i, rec in enumerate(recommendations, 1):
            print(f"    {i}. {rec}")

        if uses_correct_label:
            print(f"  ✓ 建议中使用了正确的标签 '{label}'")
        else:
            print(f"  ✗ 建议中没有使用正确的标签 '{label}'")

        if uses_hardcoded_demand and label != '需求':
            print(f"  ⚠️  警告: 建议中检测到硬编码的'需求'")

def check_dashboard_dynamic_terms():
    """检查dashboard.py中的动态术语"""
    print(f"\n检查dashboard.py中的动态术语...")
    print("=" * 60)

    dashboard_file = 'dashboard.py'
    if not os.path.exists(dashboard_file):
        print(f"文件不存在: {dashboard_file}")
        return

    with open(dashboard_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 需要动态更新的术语模式
    dynamic_patterns = [
        '需求变量',
        '需求统计',
        '需求波动性',
        '预计需求',
        '需求保持',
        '需求量',
        '历史需求',
        '未来需求',
        '平均需求'
    ]

    print("查找可能需要动态更新的术语:")
    for pattern in dynamic_patterns:
        count = content.count(pattern)
        if count > 0:
            print(f"  '{pattern}' 出现 {count} 次")

    # 检查我们是否已经更新了一些位置
    updated_patterns = [
        'label统计',
        'label变量',
        'label波动性',
        '预计label',
        'label保持'
    ]

    print(f"\n检查已更新的术语:")
    for pattern in updated_patterns:
        count = content.count(pattern.replace('label', '{}'))
        if count > 0:
            print(f"  找到已更新的模式: {pattern}")

    print(f"\n注意:")
    print("  1. 系统名称如'ForecastPro AI需求预测仪表板'不需要更新")
    print("  2. 通用描述如'上传您的需求数据'可以更新为更通用的表述")
    print("  3. 动态内容如'需求量'、'需求统计'应该使用动态术语")

if __name__ == "__main__":
    print("ForecastPro 术语适配功能验证")
    print("=" * 60)

    test_label_mapping()
    test_insight_generation()
    test_recommendation_generation()
    check_dashboard_dynamic_terms()

    print(f"\n{'='*60}")
    print("验证完成!")
    print("=" * 60)