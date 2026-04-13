#!/usr/bin/env python3
"""
ForecastPro Agent 演示脚本
本地展示修复后的功能

使用说明:
  python3 demo.py [选项]

选项:
  --all          测试所有示例文件
  --file <路径>  测试特定文件
  --list         列出所有示例文件
  --help         显示此帮助信息
"""

import sys
import os
import argparse
from pathlib import Path
from forecastpro import ForecastProAgent

def print_banner():
    """打印欢迎横幅"""
    print("\n" + "="*80)
    print("FORECASTPRO AGENT 演示")
    print("修复问题: 处理具体到时间的CSV数据集")
    print("="*80)

def list_example_files():
    """列出所有示例文件"""
    print("\n📁 可用的示例文件:")
    print("-"*40)

    csv_files = list(Path(".").glob("*.csv"))
    if not csv_files:
        print("未找到CSV文件")
        return []

    for i, csv_file in enumerate(csv_files, 1):
        file_size = csv_file.stat().st_size
        print(f"{i}. {csv_file.name} ({file_size:,} bytes)")

    print("-"*40)
    return csv_files

def test_file(file_path, full_pipeline=True):
    """测试单个文件"""
    print(f"\n🔍 测试文件: {file_path}")
    print("-"*60)

    try:
        # 让agent自动识别列
        print("初始化ForecastPro Agent...")
        agent = ForecastProAgent(data_path=str(file_path))

        # 加载数据
        df = agent.load_data()
        print(f"✓ 数据加载成功")
        print(f"  形状: {df.shape}")
        print(f"  时间列: {agent.time_col}")
        print(f"  需求列: {agent.demand_col}")

        if full_pipeline:
            print("\n运行完整预测管道...")
            agent.prepare_data()
            agent.run_baseline_models()
            agent.run_advanced_models()
            agent.evaluate_models()
            agent.generate_forecast(periods=4)
            agent.generate_report()
            print("✓ 完整管道执行完成")
        else:
            print("\n运行快速测试...")
            agent.prepare_data()
            print(f"✓ 数据准备完成")
            print(f"  训练集: {len(agent.train_data['y'])} 个观测值")
            print(f"  测试集: {len(agent.test_data['y'])} 个观测值")

        return True

    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_all_files(full_pipeline=False):
    """测试所有CSV文件"""
    csv_files = list(Path(".").glob("*.csv"))

    if not csv_files:
        print("未找到CSV文件")
        return

    print(f"\n📊 测试 {len(csv_files)} 个文件:")

    results = []
    for csv_file in csv_files:
        success = test_file(csv_file, full_pipeline)
        results.append((csv_file.name, success))

    # 打印总结
    print("\n" + "="*80)
    print("测试总结:")
    print("-"*40)
    for file_name, success in results:
        status = "✓ 成功" if success else "✗ 失败"
        print(f"{file_name:30s} {status}")

def show_fixed_issues():
    """显示修复的问题"""
    print("\n🔧 修复的问题:")
    print("-"*60)

    issues = [
        "1. 时间列识别失败 - 扩展支持TxnDate/TxnTime等列名",
        "2. 日期时间分离 - 自动合并分离的日期和时间列",
        "3. 重复时间索引 - 检测并处理重复的时间点",
        "4. 不规则时间序列 - 智能频率调整，避免asfreq()问题",
        "5. 需求列识别 - 支持Consumption、sales_revenue、cost等列名",
        "6. 数据形状不一致 - 修复滞后特征创建后的训练/测试集匹配",
        "7. 除零错误 - MAPE计算中安全处理零值",
        "8. 数据清理 - 删除不必要的索引列和原始日期时间列"
    ]

    for issue in issues:
        print(issue)

    print("\n📈 支持的数据格式:")
    print("-"*40)
    print("标准日期格式:     2023-01-01")
    print("分离日期时间:     TxnDate='21 Jan 2022', TxnTime='01:57:23'")
    print("不规则时间序列:   时间间隔不固定的数据")
    print("各种列名格式:     date, time, timestamp, TxnDate, sales_revenue等")

def interactive_mode():
    """交互模式"""
    while True:
        print("\n" + "="*80)
        print("ForecastPro Agent 演示菜单")
        print("="*80)
        print("1. 列出所有示例文件")
        print("2. 测试所有文件（快速）")
        print("3. 测试所有文件（完整管道）")
        print("4. 测试特定文件")
        print("5. 查看修复的问题")
        print("6. 运行示例演示")
        print("0. 退出")
        print("-"*80)

        choice = input("请选择 (0-6): ").strip()

        if choice == "0":
            print("\n再见!")
            break

        elif choice == "1":
            list_example_files()

        elif choice == "2":
            test_all_files(full_pipeline=False)

        elif choice == "3":
            test_all_files(full_pipeline=True)

        elif choice == "4":
            csv_files = list(Path(".").glob("*.csv"))
            if not csv_files:
                print("未找到CSV文件")
                continue

            print("\n选择要测试的文件:")
            for i, csv_file in enumerate(csv_files, 1):
                print(f"{i}. {csv_file.name}")

            try:
                file_num = int(input("输入文件编号: "))
                if 1 <= file_num <= len(csv_files):
                    pipeline_choice = input("运行完整管道? (y/N): ").lower()
                    full_pipeline = pipeline_choice == 'y'
                    test_file(csv_files[file_num-1], full_pipeline)
                else:
                    print("无效的编号")
            except ValueError:
                print("请输入有效的数字")

        elif choice == "5":
            show_fixed_issues()

        elif choice == "6":
            run_example_demo()

        else:
            print("无效的选择，请重试")

def run_example_demo():
    """运行示例演示"""
    print("\n🎬 运行示例演示")
    print("="*60)

    examples = [
        ("example_data.csv", "标准日期格式CSV", "D"),
        ("example_sales_data.csv", "销售数据CSV", "D"),
        ("example_cost_data.csv", "成本数据CSV", "D"),
        ("KwhConsumptionBlower78_2.csv", "时间戳CSV（原始问题文件）", "H")
    ]

    for file_name, description, freq in examples:
        if not Path(file_name).exists():
            print(f"⚠ 文件不存在: {file_name}")
            continue

        print(f"\n▶ 演示: {description}")
        print(f"   文件: {file_name}")
        print(f"   频率: {freq}")
        print("-"*40)

        try:
            # 只做快速测试，不运行完整管道以节省时间
            agent = ForecastProAgent(data_path=file_name, freq=freq)
            df = agent.load_data()
            print(f"✓ 成功加载")
            print(f"  形状: {df.shape}")
            print(f"  时间列: {agent.time_col}")
            print(f"  需求列: {agent.demand_col}")

            # 如果是KwhConsumptionBlower78_2.csv，显示特殊处理
            if file_name == "KwhConsumptionBlower78_2.csv":
                print(f"  处理: 自动合并TxnDate和TxnTime为datetime索引")

        except Exception as e:
            print(f"✗ 失败: {e}")

    print("\n" + "="*60)
    print("示例演示完成!")
    print("使用 --all 选项运行完整的预测管道")

def main():
    """主函数"""
    print_banner()

    # 解析命令行参数
    parser = argparse.ArgumentParser(description="ForecastPro Agent演示脚本")
    parser.add_argument("--all", action="store_true", help="测试所有示例文件")
    parser.add_argument("--file", type=str, help="测试特定文件")
    parser.add_argument("--list", action="store_true", help="列出所有示例文件")
    parser.add_argument("--quick", action="store_true", help="快速测试（不运行完整管道）")
    parser.add_argument("--demo", action="store_true", help="运行示例演示")
    parser.add_argument("--issues", action="store_true", help="显示修复的问题")

    args = parser.parse_args()

    if args.list:
        list_example_files()

    elif args.issues:
        show_fixed_issues()

    elif args.demo:
        run_example_demo()

    elif args.file:
        if Path(args.file).exists():
            test_file(args.file, full_pipeline=not args.quick)
        else:
            print(f"错误: 文件不存在: {args.file}")

    elif args.all:
        test_all_files(full_pipeline=not args.quick)

    else:
        # 进入交互模式
        interactive_mode()

if __name__ == "__main__":
    main()