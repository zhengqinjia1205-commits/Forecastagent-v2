#!/usr/bin/env python3
"""
测试成本数据预测
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from forecastpro import ForecastProAgent

def test_cost_data():
    """测试成本数据预测"""
    print("=" * 60)
    print("测试成本数据预测")
    print("=" * 60)

    # 创建Agent实例，指定cost为目标列
    agent = ForecastProAgent(
        time_col='date',
        demand_col='cost',  # 注意：这里还是叫demand_col，但实际上预测的是cost
        freq='D',
        random_seed=42
    )

    # 加载成本数据
    print("1. 加载成本数据...")
    agent.load_data('example_cost_data.csv')

    # 准备数据
    print("2. 准备数据...")
    agent.prepare_data()

    # 运行基准模型
    print("3. 运行基准模型...")
    agent.run_baseline_models()

    # 运行高级模型
    print("4. 运行高级模型...")
    agent.run_advanced_models()

    # 模型评估
    print("5. 评估模型...")
    agent.evaluate_models()

    # 生成预测
    print("6. 生成未来4期成本预测...")
    forecast = agent.generate_forecast(periods=4)

    if forecast:
        print(f"\n未来4期成本预测:")
        for i in range(len(forecast['dates'])):
            date = forecast['dates'][i]
            pred = forecast['forecast'][i]
            lower = forecast['lower_bound'][i]
            upper = forecast['upper_bound'][i]
            print(f"  {date}: {pred:.2f} (95%置信区间: {lower:.2f} - {upper:.2f})")

    # 生成报告
    print("\n7. 生成管理报告...")
    report = agent.generate_report()

    if report:
        print(f"\n最佳模型: {report['best_model']['name']}")
        print(f"测试集MAPE: {report['best_model']['metrics']['MAPE']:.2f}%")
        print(f"MAE: {report['best_model']['metrics']['MAE']:.2f}")
        print(f"RMSE: {report['best_model']['metrics']['RMSE']:.2f}")

    print("\n✅ 成本数据测试完成!")
    return True

def test_sales_data():
    """测试销售数据预测"""
    print("\n" + "=" * 60)
    print("测试销售数据预测")
    print("=" * 60)

    # 创建Agent实例，指定sales_revenue为目标列
    agent = ForecastProAgent(
        time_col='date',
        demand_col='sales_revenue',  # 预测销售收入
        freq='D',
        random_seed=42
    )

    # 加载销售数据
    print("1. 加载销售数据...")
    agent.load_data('example_sales_data.csv')

    # 准备数据
    print("2. 准备数据...")
    agent.prepare_data()

    # 运行基准模型
    print("3. 运行基准模型...")
    agent.run_baseline_models()

    # 运行高级模型
    print("4. 运行高级模型...")
    agent.run_advanced_models()

    # 模型评估
    print("5. 评估模型...")
    agent.evaluate_models()

    # 生成预测
    print("6. 生成未来4期销售预测...")
    forecast = agent.generate_forecast(periods=4)

    if forecast:
        print(f"\n未来4期销售预测:")
        for i in range(len(forecast['dates'])):
            date = forecast['dates'][i]
            pred = forecast['forecast'][i]
            lower = forecast['lower_bound'][i]
            upper = forecast['upper_bound'][i]
            print(f"  {date}: {pred:.2f} (95%置信区间: {lower:.2f} - {upper:.2f})")

    # 生成报告
    print("\n7. 生成管理报告...")
    report = agent.generate_report()

    if report:
        print(f"\n最佳模型: {report['best_model']['name']}")
        print(f"测试集MAPE: {report['best_model']['metrics']['MAPE']:.2f}%")
        print(f"MAE: {report['best_model']['metrics']['MAE']:.2f}")
        print(f"RMSE: {report['best_model']['metrics']['RMSE']:.2f}")

    print("\n✅ 销售数据测试完成!")
    return True

if __name__ == "__main__":
    try:
        test_cost_data()
        test_sales_data()
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)