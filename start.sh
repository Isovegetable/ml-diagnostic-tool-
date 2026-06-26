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
    echo ""
    echo "按 Enter 键退出..."
    read -r
    exit 1
fi
echo "[OK] 检测到 Python3 ($(python3 --version))"

# 切换到脚本所在目录（解决双击运行时工作目录不对的问题）
cd "$(dirname "$0")" || {
    echo "[错误] 无法切换到脚本所在目录"
    echo ""
    echo "按 Enter 键退出..."
    read -r
    exit 1
}

# 清理上次残留的 Streamlit 进程
echo "[..] 清理残留进程..."
pkill -f "streamlit run" 2>/dev/null
sleep 1

# 首次启动：创建虚拟环境并安装依赖
if [ ! -f ".venv/bin/python3" ]; then
    echo ""
    echo "============ 首次启动需完成以下步骤（仅一次）============"
    echo ""
    echo " 步骤 1/2：正在创建虚拟环境..."
    python3 -m venv .venv
    if [ $? -ne 0 ]; then
        echo "[错误] 虚拟环境创建失败"
        echo ""
        echo "按 Enter 键退出..."
        read -r
        exit 1
    fi
    echo "[OK] 虚拟环境创建完成"
    echo ""
    echo " 步骤 2/2：正在安装依赖..."
    echo ""

    # 优先从本地离线包安装（速度快，无需网络）
    if [ -d "packages" ] && [ -f "packages/index.html" ]; then
        echo " 检测到离线依赖包，正在从本地安装..."
        .venv/bin/pip install -r requirements.txt --no-index --find-links=packages
        if [ $? -ne 0 ]; then
            echo " 本地安装失败，尝试从网络安装..."
            .venv/bin/pip install -r requirements.txt
        fi
    else
        echo " 正在从网络下载依赖（约 2-5 分钟，取决于网络速度）..."
        .venv/bin/pip install -r requirements.txt
    fi

    if [ $? -ne 0 ]; then
        echo "[错误] 依赖安装失败"
        echo ""
        echo "按 Enter 键退出..."
        read -r
        exit 1
    fi
    echo "[OK] 依赖安装完成"
    echo "============================================================"
fi

echo ""
echo "[OK] 正在启动 Streamlit 服务..."
echo ""

# 后台启动 Streamlit（headless 模式，不自动打开浏览器）
.venv/bin/streamlit run app.py --server.headless true > streamlit_log.txt 2>&1 &
STREAMLIT_PID=$!

# 等待 Streamlit 就绪（每 2 秒检测一次，最多等 30 秒）
echo "[..] 等待 Streamlit 启动..."
WAIT_COUNT=0
while [ $WAIT_COUNT -lt 15 ]; do
    WAIT_COUNT=$((WAIT_COUNT + 1))
    python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8501')" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "[OK] Streamlit 已就绪！"
        break
    fi
    echo " 启动中... ($WAIT_COUNT/15)"
    sleep 2
done

echo ""
echo "[OK] 正在打开浏览器..."
open http://localhost:8501

echo ""
echo "============================================"
echo " 如果浏览器未自动打开，请手动访问："
echo "  http://localhost:8501"
echo " 关闭此终端窗口即可退出"
echo "============================================"
echo ""

# 保持终端窗口打开（类似 Windows pause）
while true; do
    sleep 10
done
