#!/usr/bin/env python3
"""
全面测试变量术语适配功能
测试不同变量类型：需求、销售、成本
"""

import sys
import os
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from forecastpro import ForecastProAgent

def run_complete_pipeline(data_path, time_col, demand_col, freq='D', test_name=""):
    """运行完整预测管道并返回洞察和建议"""
    print(f"\n{'='*60}")
    print(f"测试: {test_name}")
    print(f"数据文件: {data_path}")
    print(f"时间列: {time_col}, 目标列: {demand_col}")
    print(f"{'='*60}")

    # 创建Agent
    agent = ForecastProAgent(
        data_path=data_path,
        time_col=time_col,
        demand_col=demand_col,
        freq=freq,
        random_seed=42
    )

    # 加载数据
    try:
        df = agent.load_data()
        print(f"数据加载成功: {len(df)} 行")
        print(f"变量标签: '{demand_col}' -> '{agent.get_variable_label()}'")
    except Exception as e:
        print(f"数据加载失败: {e}")
        return None, None

    # 准备数据
    agent.prepare_data()

    # 运行基准模型
    agent.run_baseline_models()

    # 运行高级模型
    agent.run_advanced_models()

    # 评估模型
    agent.evaluate_models()

    # 生成洞察和建议
    insights = agent._generate_insights()
    recommendations = agent._generate_recommendations()

    return insights, recommendations

def test_all_data_types():
    """测试所有数据类型"""
    test_cases = [
        {
            'file': 'example_data.csv',
            'time_col': 'date',
            'demand_col': 'demand',
            'test_name': '需求数据 (demand)',
            'expected_label': '需求'
        },
        {
            'file': 'example_sales_data.csv',
            'time_col': 'date',
            'demand_col': 'sales_revenue',
            'test_name': '销售数据 (sales_revenue)',
            'expected_label': '销售收入'
        },
        {
            'file': 'example_sales_data.csv',
            'time_col': 'date',
            'demand_col': 'units_sold',
            'test_name': '销售数量数据 (units_sold)',
            'expected_label': '销售数量'
        },
        {
            'file': 'example_cost_data.csv',
            'time_col': 'date',
            'demand_col': 'cost',
            'test_name': '成本数据 (cost)',
            'expected_label': '成本'
        }
    ]

    results = []

    for test_case in test_cases:
        # 检查文件是否存在
        if not os.path.exists(test_case['file']):
            print(f"文件不存在: {test_case['file']}")
            continue

        insights, recommendations = run_complete_pipeline(
            data_path=test_case['file'],
            time_col=test_case['time_col'],
            demand_col=test_case['demand_col'],
            test_name=test_case['test_name']
        )

        if insights and recommendations:
            # 创建Agent验证标签
            agent = ForecastProAgent(demand_col=test_case['demand_col'])
            actual_label = agent.get_variable_label()

            results.append({
                'test_name': test_case['test_name'],
                'expected_label': test_case['expected_label'],
                'actual_label': actual_label,
                'insights': insights[:3],  # 取前3个洞察
                'recommendations': recommendations[:2]  # 取前2个建议
            })

            print(f"\n📊 测试结果:")
            print(f"  变量标签: '{test_case['demand_col']}' -> '{actual_label}'")
            print(f"  标签匹配: {'✓' if actual_label == test_case['expected_label'] else '✗'}")

            print(f"\n🔮 生成的洞察:")
            for i, insight in enumerate(insights[:3], 1):
                print(f"  {i}. {insight}")

            print(f"\n💡 生成的建议:")
            for i, rec in enumerate(recommendations[:2], 1):
                print(f"  {i}. {rec}")

            print(f"\n{'='*60}")

    return results

def analyze_results(results):
    """分析测试结果"""
    print(f"\n{'='*60}")
    print("测试结果分析")
    print(f"{'='*60}")

    total_tests = len(results)
    passed_tests = 0

    for result in results:
        label_match = result['actual_label'] == result['expected_label']
        insights_use_label = any(result['actual_label'] in insight for insight in result['insights'])

        print(f"\n📋 {result['test_name']}:")
        print(f"  期望标签: {result['expected_label']}")
        print(f"  实际标签: {result['actual_label']}")
        print(f"  标签匹配: {'✓' if label_match else '✗'}")
        print(f"  洞察中使用标签: {'✓' if insights_use_label else '✗'}")

        # 检查硬编码的"需求"是否出现在洞察中
        hardcoded_demand = any('需求' in insight and result['expected_label'] != '需求' for insight in result['insights'])
        if hardcoded_demand:
            print(f"  ⚠️  警告: 洞察中检测到硬编码的'需求'")

        if label_match and insights_use_label and not hardcoded_demand:
            passed_tests += 1

    print(f"\n{'='*60}")
    print(f"总结:")
    print(f"  总测试数: {total_tests}")
    print(f"  通过测试: {passed_tests}")
    print(f"  通过率: {passed_tests/total_tests*100:.1f}%")

    if passed_tests == total_tests:
        print(f"\n🎉 所有测试通过! 术语适配功能正常工作。")
    else:
        print(f"\n⚠️  有 {total_tests - passed_tests} 个测试失败，请检查代码。")

def check_dashboard_hardcoded_terms():
    """检查dashboard.py中是否还有硬编码的术语"""
    print(f"\n{'='*60}")
    print("检查dashboard.py中的硬编码术语")
    print(f"{'='*60}")

    dashboard_file = 'dashboard.py'
    if not os.path.exists(dashboard_file):
        print(f"文件不存在: {dashboard_file}")
        return

    with open(dashboard_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 查找可能硬编码的术语
    hardcoded_terms = ['需求', '销售', '成本', '库存', '生产']

    print("扫描硬编码术语...")
    for term in hardcoded_terms:
        count = content.count(term)
        if count > 0:
            print(f"  发现 '{term}' 出现 {count} 次")

            # 查找上下文（可选）
            if term == '需求':
                # 检查是否是通用描述还是硬编码
                lines = [line.strip() for line in content.split('\n') if term in line]
                print(f"  示例行:")
                for i, line in enumerate(lines[:3], 1):
                    print(f"    {i}. {line[:100]}...")

    print("\n注意: 某些术语可能是通用描述（如'需求预测系统'），")
    print("      需要区分系统名称和动态变量术语。")

if __name__ == "__main__":
    print("ForecastPro 变量术语适配功能测试")
    print("测试不同变量类型的标签映射和报告生成")
    print(f"{'='*60}")

    # 运行所有测试
    results = test_all_data_types()

    # 分析结果
    if results:
        analyze_results(results)

    # 检查dashboard.py中的硬编码术语
    check_dashboard_hardcoded_terms()

    print(f"\n{'='*60}")
    print("测试完成!")
    print(f"{'='*60}")