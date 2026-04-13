# 🚀 ForecastPro Dashboard 快速启动指南

## 一键启动方式

### macOS/Linux 用户
```bash
./start_dashboard.sh
```

### Windows 用户
双击 `start_dashboard.bat` 文件

## 手动启动方式

### 1. 安装依赖（如果尚未安装）
```bash
pip install -r requirements_dashboard.txt
```

### 2. 启动仪表板
```bash
streamlit run dashboard.py
```

## 访问仪表板

启动后，浏览器会自动打开仪表板界面：
- **地址**: http://localhost:8501
- **默认端口**: 8501

如果未自动打开，请手动在浏览器中输入以上地址。

## 快速开始

1. **使用示例数据**（推荐新手）：
   - 在仪表板首页点击"🚀 使用示例数据运行完整预测管道"按钮
   - 系统自动加载示例数据并运行预测
   - 使用左侧导航栏查看各页面分析结果

2. **使用自定义数据**：
   - 前往"📊 数据管理"页面
   - 上传CSV或Excel数据文件
   - 配置时间列和需求列
   - 点击"🔧 初始化ForecastPro Agent"
   - 前往"🔮 预测结果"页面运行预测

## 仪表板功能概览

- **🏠 首页**: 系统介绍和快速开始
- **📊 数据管理**: 上传和管理数据
- **🔮 预测结果**: 可视化预测图表
- **📈 模型比较**: 比较不同模型性能
- **💡 业务洞察**: 获取业务建议
- **📋 完整报告**: 生成和导出报告
- **🤖 AI问答与分析**: 智能问答和报告生成

## 系统要求

- Python 3.8+
- 4GB+ RAM
- 现代浏览器（Chrome/Firefox/Safari/Edge）

## 故障排除

### 1. 无法启动
- 检查Python和pip是否正确安装
- 确保所有依赖已安装：`pip install -r requirements_dashboard.txt`

### 2. 浏览器未自动打开
- 手动访问 http://localhost:8501
- 如果端口被占用，streamlit会自动使用其他端口（查看终端输出）

### 3. 导入错误
- 确保在 `forecastpro_agent` 目录下运行
- 检查 `forecastpro.py` 文件是否存在

## 获取帮助

- 查看完整文档: [DASHBOARD_README.md](DASHBOARD_README.md)
- 查看核心模块文档: [README.md](README.md)

---

**📞 需要帮助？** 检查终端错误信息或重新安装依赖。

**🎉 现在开始使用 ForecastPro AI需求预测仪表板！**