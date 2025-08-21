import os

# pkgs = ['ultralytics','PyQt5==5.15.2','pyqt5-tools==5.15.2.3.1']
pkgs = ['ultralytics']

for each in pkgs:
    cmd_line = f"pip install {each} -i https://pypi.tuna.tsinghua.edu.cn/simple"
    os.system(cmd_line)
