@echo off
chcp 65001 >nul
title 材料机器学习诊断工具

echo ============================================
echo  材料机器学习自动诊断工具 - 启动脚本
echo ============================================
echo.

:: 检查 Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未检测到 Python，请先安装 Python 3.9+
    echo 下载地址：https://www.python.org/downloads/
    echo.
    echo 安装时务必勾选 "Add Python to PATH"
    pause
    exit /b 1
)
echo [OK] 检测到 Python

:: 检查是否为首次运行（虚拟环境是否存在）
if not exist ".venv\Scripts\python.exe" (
    echo.
    echo ============ 首次启动需完成以下步骤（仅一次）============
    echo.
    echo  步骤 1/2：正在创建虚拟环境...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo [错误] 虚拟环境创建失败
        pause
        exit /b 1
    )
    echo [OK] 虚拟环境创建完成
    echo.
    echo  步骤 2/2：正在安装依赖（约 2-5 分钟，取决于网络速度）
    echo  正在下载 shap/xgboost/lightgbm/catboost 等包...
    echo.
    call .venv\Scripts\pip.exe install -r requirements.txt
    if %errorlevel% neq 0 (
        echo [错误] 依赖安装失败
        pause
        exit /b 1
    )
    echo [OK] 依赖安装完成
    echo ============================================================
) else (
    :: 检查依赖是否有更新
    echo [..] 检查依赖...
    call .venv\Scripts\pip.exe install -r requirements.txt -q
)

echo.
echo [OK] 正在启动...
echo.
echo 提示：
echo   - 等待 Streamlit 启动后，浏览器将自动打开
echo   - 上传 Excel/CSV 数据即可开始分析
echo   - 关闭本窗口即可退出程序
echo.

:: 先启动 Streamlit，等几秒后再打开浏览器
start /B "" ".venv\Scripts\streamlit.exe" run app.py
timeout /t 5 /nobreak >nul
start http://localhost:8501

echo ============================================
echo  浏览器已打开，如果显示"无法访问"，刷新一下即可
echo ============================================

:: 等待用户关闭
pause >nul
