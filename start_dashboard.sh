#!/bin/bash
# ForecastPro Dashboard 启动脚本

echo "🚀 启动 ForecastPro AI需求预测仪表板..."
echo "========================================"

# 检查当前目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 检查streamlit是否安装
if ! command -v streamlit &> /dev/null; then
    echo "❌ 未找到streamlit，请先安装: pip install streamlit"
    exit 1
fi

# 检查Python依赖
echo "🔍 检查Python依赖..."
python3 -c "import pandas, numpy, streamlit, plotly" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  部分依赖可能缺失，尝试安装..."
    pip3 install -r requirements_dashboard.txt 2>/dev/null || echo "请手动安装依赖"
fi

# 启动仪表板
echo "🌐 启动Streamlit仪表板..."
echo "📊 仪表板将在浏览器中打开..."
echo "🔗 如果未自动打开，请访问: http://localhost:8501"
echo "🛑 按 Ctrl+C 停止服务"
echo "========================================"

streamlit run dashboard.py