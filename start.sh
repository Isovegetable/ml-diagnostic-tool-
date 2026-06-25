#!/bin/bash
# 材料机器学习自动诊断工具 - Mac/Linux 启动脚本

echo "============================================"
echo " 材料机器学习自动诊断工具 - 启动脚本"
echo "============================================"
echo ""

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "[错误] 未检测到 Python3，请先安装 Python 3.9+"
    echo "下载地址：https://www.python.org/downloads/"
    exit 1
fi
echo "[OK] 检测到 Python3"

# 检查是否为首次运行
if [ ! -f ".venv/bin/python3" ]; then
    echo ""
    echo "============ 首次启动需完成以下步骤（仅一次）============"
    echo ""
    echo " 步骤 1/2：正在创建虚拟环境..."
    python3 -m venv .venv
    if [ $? -ne 0 ]; then
        echo "[错误] 虚拟环境创建失败"
        exit 1
    fi
    echo "[OK] 虚拟环境创建完成"
    echo ""
    echo " 步骤 2/2：正在安装依赖（约 2-5 分钟，取决于网络速度）"
    echo " 正在下载 shap/xgboost/lightgbm/catboost 等包..."
    echo ""
    .venv/bin/pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "[错误] 依赖安装失败"
        exit 1
    fi
    echo "[OK] 依赖安装完成"
    echo "============================================================"
else
    echo "[..] 检查依赖..."
    .venv/bin/pip install -r requirements.txt -q
fi

echo ""
echo "[OK] 正在启动..."
echo ""
echo "提示："
echo "  - 等待 Streamlit 启动后，浏览器将自动打开"
echo "  - 上传 Excel/CSV 数据即可开始分析"
echo "  - 终端中按 Ctrl+C 即可退出程序"
echo ""

# 后台启动 Streamlit，等几秒后打开浏览器
.venv/bin/streamlit run app.py &
sleep 5
open http://localhost:8501

echo "============================================"
echo " 浏览器已打开，如果显示"无法访问"，刷新一下即可"
echo "============================================"

wait
