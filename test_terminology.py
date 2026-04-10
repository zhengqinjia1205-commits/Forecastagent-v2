#!/usr/bin/env python3
"""
测试变量术语适配功能
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from forecastpro import ForecastProAgent

def test_variable_labels():
    """测试变量标签映射"""
    print("测试变量标签映射...")

    # 测试不同的变量名
    test_cases = [
        ('demand', '需求'),
        ('sales', '销售额'),
        ('quantity', '数量'),
        ('revenue', '收入'),
        ('consumption', '消耗量'),
        ('cost', '成本'),
        ('price', '价格'),
        ('y', '目标变量'),
        ('target', '目标变量'),
        ('unknown_var', 'unknown_var'),  # 未知变量，应返回原变量名
    ]

    for var_name, expected_label in test_cases:
        agent = ForecastProAgent(demand_col=var_name)
        label = agent.get_variable_label(var_name)
        status = "✓" if label == expected_label else "✗"
        print(f"{status} {var_name:15s} -> {label:10s} (期望: {expected_label})")

    print("\n测试完成!")

def test_with_example_data():
    """使用示例数据测试"""
    print("\n使用示例数据测试...")

    # 使用demand列
    agent1 = ForecastProAgent(
        data_path='example_data.csv',
        time_col='date',
        demand_col='demand',
        freq='D'
    )

    df1 = agent1.load_data()
    print(f"变量名: demand -> 标签: {agent1.get_variable_label()}")

    # 生成洞察
    agent1.prepare_data()
    agent1.run_baseline_models()
    agent1.run_advanced_models()
    agent1.evaluate_models()
    insights = agent1._generate_insights()

    print("\n生成的洞察（demand变量）:")
    for i, insight in enumerate(insights[:3], 1):
        print(f"{i}. {insight}")

    # 测试推荐
    recommendations = agent1._generate_recommendations()
    print("\n生成的建议（demand变量）:")
    for i, rec in enumerate(recommendations[:2], 1):
        print(f"{i}. {rec}")

if __name__ == "__main__":
    test_variable_labels()
    test_with_example_data()