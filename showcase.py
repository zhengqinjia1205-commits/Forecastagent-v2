#!/usr/bin/env python3
"""
ForecastPro Agent 功能展示
快速本地演示修复后的功能
"""

import sys
import pandas as pd
from pathlib import Path
from forecastpro import ForecastProAgent

def print_header(text):
    """打印标题"""
    print("\n" + "="*80)
    print(f"📊 {text}")
    print("="*80)

def showcase_fixed_issues():
    """展示修复的问题"""
    print_header("修复的问题展示")

    issues = {
        "时间列识别": "扩展支持TxnDate/TxnTime等列名",
        "日期时间合并": "自动合并分离的日期和时间列",
        "重复索引处理": "检测并处理重复的时间点",
        "不规则时间序列": "智能频率调整",
        "需求列识别": "支持多种列名格式",
        "数据清理": "自动删除不必要的列"
    }

    for issue, description in issues.items():
        print(f"✅ {issue}: {description}")

def showcase_kwh_file():
    """展示KwhConsumptionBlower78_2.csv的处理"""
    print_header("原始问题文件处理展示")

    file_path = "KwhConsumptionBlower78_2.csv"
    if not Path(file_path).exists():
        print(f"⚠ 文件不存在: {file_path}")
        return

    print(f"文件: {file_path}")
    print("特点: 分离的TxnDate和TxnTime列，不规则时间序列")

    # 先显示原始数据
    print("\n📄 原始数据查看:")
    df_original = pd.read_csv(file_path)
    print(f"  形状: {df_original.shape}")
    print(f"  列名: {df_original.columns.tolist()}")
    print(f"  前3行:")
    print(df_original.head(3).to_string())

    # 使用Agent处理
    print("\n🤖 Agent处理过程:")
    try:
        agent = ForecastProAgent(data_path=file_path, freq='H')
        df_processed = agent.load_data()

        print(f"  ✓ 时间列识别: {agent.time_col}")
        print(f"  ✓ 需求列识别: {agent.demand_col}")
        print(f"  ✓ 处理后的形状: {df_processed.shape}")

        # 显示处理后的索引
        if hasattr(df_processed, 'index'):
            print(f"  ✓ 索引类型: {type(df_processed.index)}")
            print(f"  ✓ 前3个时间点:")
            for i, ts in enumerate(df_processed.index[:3]):
                print(f"      {i+1}. {ts}")

        # 检查是否有重复索引被处理
        if hasattr(df_processed.index, 'duplicated'):
            duplicates = df_processed.index.duplicated().sum()
            if duplicates > 0:
                print(f"  ⚠ 发现并处理了 {duplicates} 个重复时间索引")

    except Exception as e:
        print(f"  ✗ 处理失败: {e}")
        import traceback
        traceback.print_exc()

def showcase_all_files():
    """展示所有文件处理能力"""
    print_header("所有示例文件处理测试")

    csv_files = list(Path(".").glob("*.csv"))
    if not csv_files:
        print("未找到CSV文件")
        return

    print(f"找到 {len(csv_files)} 个CSV文件:\n")

    results = []
    for csv_file in csv_files:
        print(f"🔍 处理: {csv_file.name}")

        try:
            # 快速测试，不运行完整管道
            agent = ForecastProAgent(data_path=str(csv_file))
            df = agent.load_data()

            # 获取文件特点
            characteristics = []
            if "TxnDate" in str(csv_file):
                characteristics.append("分离日期时间")
            if df.index.duplicated().sum() > 0:
                characteristics.append("有重复索引")
            if hasattr(df.index, 'freq') and df.index.freq is None:
                time_diffs = df.index.to_series().diff().dropna()
                if time_diffs.nunique() > 3:
                    characteristics.append("不规则时间序列")

            status = "✓ 成功"
            details = f"形状: {df.shape}, 时间列: {agent.time_col}, 需求列: {agent.demand_col}"
            if characteristics:
                details += f", 特点: {', '.join(characteristics)}"

        except Exception as e:
            status = "✗ 失败"
            details = f"错误: {str(e)[:50]}..."

        results.append((csv_file.name, status, details))

    # 打印结果表
    print("\n" + "-"*100)
    print(f"{'文件':30s} {'状态':8s} {'详情'}")
    print("-"*100)
    for filename, status, details in results:
        print(f"{filename:30s} {status:8s} {details}")

def showcase_feature_comparison():
    """展示功能对比"""
    print_header("功能对比: 修复前 vs 修复后")

    comparison_data = [
        ("时间列识别", "仅支持date,time等", "支持TxnDate,TxnTime,Date,TIME等10+种格式"),
        ("日期时间处理", "只能处理单列", "自动合并分离的日期和时间列"),
        ("重复索引", "会导致asfreq()失败", "自动检测并处理重复索引"),
        ("不规则序列", "asfreq()创建大量NaN", "智能判断，不规则时跳过asfreq()"),
        ("需求列识别", "仅demand,sales等", "支持consumption,cost,revenue等10+种格式"),
        ("数据清理", "不清理索引列", "自动删除Unnamed列和原始日期时间列"),
        ("错误处理", "MAPE计算会崩溃", "安全处理除零错误"),
    ]

    print(f"{'功能':15s} {'修复前':30s} {'修复后'}")
    print("-"*85)
    for feature, before, after in comparison_data:
        print(f"{feature:15s} {before:30s} {after}")

def quick_demo():
    """快速演示"""
    print_header("ForecastPro Agent 快速演示")
    print("修复了处理具体到时间的CSV数据集的问题")

    # 展示修复的问题
    showcase_fixed_issues()

    # 展示原始问题文件处理
    showcase_kwh_file()

    # 展示所有文件处理能力
    showcase_all_files()

    # 展示功能对比
    showcase_feature_comparison()

    print_header("演示完成")
    print("✅ ForecastPro Agent 现在可以处理:")
    print("   • 标准日期格式的CSV")
    print("   • 分离日期和时间的CSV")
    print("   • 不规则时间序列")
    print("   • 各种列名格式")
    print("   • 重复时间索引")

    print("\n🚀 下一步:")
    print("  1. 运行完整测试: python3 demo.py --all")
    print("  2. 交互式演示: python3 demo.py")
    print("  3. 测试特定文件: python3 demo.py --file 文件名.csv")

def main():
    """主函数"""
    # 检查是否有CSV文件
    csv_files = list(Path(".").glob("*.csv"))
    if not csv_files:
        print("错误: 当前目录没有CSV文件")
        print("请确保有以下示例文件:")
        print("  - example_data.csv")
        print("  - example_sales_data.csv")
        print("  - example_cost_data.csv")
        print("  - KwhConsumptionBlower78_2.csv (原始问题文件)")
        sys.exit(1)

    # 运行快速演示
    quick_demo()

if __name__ == "__main__":
    main()