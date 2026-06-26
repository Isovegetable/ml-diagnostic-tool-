#!/bin/bash
# 材料机器学习自动诊断工具 - Mac/Linux 启动脚本

# 切换到脚本所在目录（解决双击运行时工作目录不对的问题）
cd "$(dirname "$0")" || { echo "[错误] 无法切换到脚本所在目录"; exit 1; }

echo "============================================"
echo " 材料机器学习自动诊断工具 - 启动脚本"
echo "============================================"
echo ""
echo "工作目录: $(pwd)"
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

# 检查是否为首次运行
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
    echo " 步骤 2/2：正在安装依赖（约 2-5 分钟，取决于网络速度）"
    echo " 正在下载 shap/xgboost/lightgbm/catboost 等包..."
    echo ""
    .venv/bin/pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "[错误] 依赖安装失败"
        echo ""
        echo "按 Enter 键退出..."
        read -r
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

# 后台启动 Streamlit（输出重定向到日志文件，避免终端杂乱）
.venv/bin/streamlit run app.py > streamlit_log.txt 2>&1 &
STREAMLIT_PID=$!

# 等待 Streamlit 就绪（最多等 30 秒）
echo "[..] 等待 Streamlit 启动..."
for i in $(seq 1 30); do
    sleep 1
    if curl -s http://localhost:8501 > /dev/null 2>&1; then
        echo "[OK] Streamlit 已就绪（约 ${i} 秒）"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "[警告] Streamlit 启动较慢，仍在等待中..."
    fi
done

# 打开浏览器
echo "[OK] 正在打开浏览器..."
open http://localhost:8501

echo ""
echo "============================================"
echo " 浏览器已打开，如果显示"无法访问"请稍后刷新"
echo " 关闭此终端窗口即可停止服务"
echo "============================================"

# 捕获 Ctrl+C 优雅退出
trap 'echo ""; echo "[OK] 正在关闭服务..."; kill $STREAMLIT_PID 2>/dev/null; echo "已退出"; exit 0' SIGINT SIGTERM

# 等待 Streamlit 进程（保持终端打开）
wait $STREAMLIT_PID
