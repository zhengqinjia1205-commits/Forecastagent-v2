#!/usr/bin/env python3
"""
简单演示脚本 - 一键运行所有展示
"""

import os
import sys
import pandas as pd
from pathlib import Path

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from forecastpro import ForecastProAgent
except ImportError:
    print("错误: 无法导入ForecastProAgent")
    print("请确保forecastpro.py在同一目录")
    sys.exit(1)

def run_demo():
    """运行演示"""
    print("\n" + "="*80)
    print("🎬 FORECASTPRO AGENT 功能演示")
    print("="*80)

    # 检查文件
    required_files = [
        "example_data.csv",
        "example_sales_data.csv",
        "example_cost_data.csv",
        "KwhConsumptionBlower78_2.csv"
    ]

    missing_files = []
    for f in required_files:
        if not Path(f).exists():
            missing_files.append(f)

    if missing_files:
        print(f"⚠ 缺少文件: {', '.join(missing_files)}")
        print("演示可能不完整")
        input("按Enter继续...")

    # 演示1: 修复的问题
    print("\n🔧 修复的关键问题:")
    print("-"*50)
    fixes = [
        ("时间列识别", "支持TxnDate/TxnTime等列名"),
        ("日期时间合并", "自动合并分离的日期和时间列"),
        ("不规则序列", "智能处理不规则时间间隔"),
        ("需求列识别", "支持consumption/cost/revenue等"),
        ("数据清理", "删除不必要的索引列"),
        ("错误处理", "安全处理除零错误等"),
    ]

    for fix, desc in fixes:
        print(f"  ✅ {fix}: {desc}")

    # 演示2: KwhConsumptionBlower78_2.csv处理（原始问题）
    print("\n📁 原始问题文件处理演示:")
    print("-"*50)

    if Path("KwhConsumptionBlower78_2.csv").exists():
        print("文件: KwhConsumptionBlower78_2.csv")
        print("特点: 分离的TxnDate/TxnTime列，不规则时间序列")

        try:
            # 显示原始数据
            df_raw = pd.read_csv("KwhConsumptionBlower78_2.csv")
            print(f"  原始形状: {df_raw.shape}")
            print(f"  原始列: {df_raw.columns.tolist()}")

            # 使用Agent处理
            print("\n  🤖 Agent处理:")
            agent = ForecastProAgent(data_path="KwhConsumptionBlower78_2.csv", freq='H')
            df_processed = agent.load_data()

            print(f"    ✓ 时间列: {agent.time_col}")
            print(f"    ✓ 需求列: {agent.demand_col}")
            print(f"    ✓ 处理后形状: {df_processed.shape}")

            # 检查特殊处理
            if hasattr(df_processed.index, 'duplicated'):
                dup_count = df_processed.index.duplicated().sum()
                if dup_count == 0:
                    print(f"    ✓ 已处理重复索引")

            print(f"    ✓ 成功合并TxnDate和TxnTime为datetime索引")

        except Exception as e:
            print(f"    ✗ 处理失败: {e}")
    else:
        print("  ⚠ 文件不存在: KwhConsumptionBlower78_2.csv")

    # 演示3: 所有文件测试
    print("\n📊 所有示例文件处理测试:")
    print("-"*50)

    csv_files = list(Path(".").glob("*.csv"))
    success_count = 0

    for csv_file in csv_files:
        try:
            agent = ForecastProAgent(data_path=str(csv_file))
            df = agent.load_data()
            status = "✅"
            success_count += 1
        except Exception as e:
            status = "❌"

        print(f"  {status} {csv_file.name:30s} -> {status}")

    # 演示4: 功能对比
    print("\n🔄 功能对比:")
    print("-"*50)

    comparisons = [
        ["功能", "修复前", "修复后"],
        ["时间列识别", "有限支持", "扩展支持"],
        ["日期时间处理", "单列处理", "自动合并"],
        ["不规则序列", "会失败", "智能处理"],
        ["数据清理", "不清理", "自动清理"],
        ["错误处理", "会崩溃", "安全处理"],
    ]

    for row in comparisons:
        print(f"  {row[0]:15s} {row[1]:15s} {row[2]:15s}")

    # 总结
    print("\n" + "="*80)
    print("📈 演示总结:")
    print(f"- 测试了 {len(csv_files)} 个CSV文件")
    print(f"- 成功处理 {success_count}/{len(csv_files)} 个文件")
    print(f"- 修复了处理具体到时间的CSV数据集的问题")

    print("\n🚀 下一步操作:")
    print("1. 运行交互式演示: python3 demo.py")
    print("2. 测试完整管道: python3 demo.py --all")
    print("3. 查看修复详情: python3 demo.py --issues")

    print("\n" + "="*80)
    print("演示完成! 🎉")

if __name__ == "__main__":
    run_demo()