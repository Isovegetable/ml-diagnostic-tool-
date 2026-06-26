"""
材料机器学习诊断工具 - 启动器
双击此 exe 即可启动，浏览器自动打开
"""
import os
import sys
import webbrowser
import threading
import time

# 设置路径：打包后数据文件在 _internal 目录中
if getattr(sys, 'frozen', False):
    base_path = os.path.join(sys._MEIPASS)
else:
    base_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(base_path)

from streamlit.web import bootstrap

# 启动 Streamlit
real_argv = ["streamlit", "run", "app.py"]
sys.argv = real_argv

# 延迟打开浏览器
def _open_browser():
    time.sleep(3)
    webbrowser.open("http://localhost:8501")

threading.Thread(target=_open_browser, daemon=True).start()

# 启动
bootstrap.run("app.py", "", [], {})
