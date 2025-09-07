# 马铃薯芽眼识别程序（Windows）

本项目基于 PyQt5 + OpenCV + Ultralytics YOLO。程序启动后可打开摄像头、进行目标检测，并通过 TCP/Modbus 与机械臂/PLC 通信。

## 1. 环境要求

- Windows 10/11
- Python 3.9
- 摄像头厂商华睿相机 SDK（需要提供 `open_cv_show1.py` 中的 `retrun_frame`，路径见代码）
- 网络可达（PLC 与机械臂的 IP/端口需联通）

## 2. 目录与关键文件

- 入口脚本：`testUI8_dual_PLC_8.py`
- YOLO 模型：请将 `models/best.pt` 放在上述目录下的 `models/` 目录
- 相机 SDK 示例路径（代码已硬编码）：
  `F:\YOLOV8\PotatoDetection-main\MV Viewer\Development\Samples\Python\IMV\opencv_byGetFrame\open_cv_show1.py`

> 如路径不同，请修改 `testUI8_dual_PLC_8.py` 顶部的 `sys.path.append(...)` 或把该文件复制到项目内并修正导入。

## 3. 创建虚拟环境并安装依赖

# 切换到项目目录

# 建立并激活虚拟环境
# 建议使用 Anaconda Prompt（或在 PowerShell 中先 `conda init powershell` 后重启）
conda create -n yolo python=3.9 -y
conda activate yolo

# 安装 Qt
conda install -c conda-forge pyqt=5.15 -y
python -m pip install --upgrade pip

# 基础依赖
pip install -r requirements.txt

# 安装 PyTorch（如果连接失效请自行安装）

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 安装  CUDA 12.8：
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128


# 自检（应显示 cuda.is_available=True）
直接运行envCheck.py
[PASS] 所有必须项通过即可


## 4. 摄像头 SDK 配置

- 确认厂商 SDK 已安装，可通过 SDK 自带示例采集到图像。
- 确认 `open_cv_show1.py` 存在，且包含 `retrun_frame()` 生成器函数。
- 如 SDK 需要 DLL（驱动/运行库），请确保其所在目录已加入系统 PATH 或与 Python 进程可见。

## 5. 模型权重

- 将您的 YOLO 权重文件放置在：
  `f:\YOLOV8\potato_detection\potato_detaction\models\best.pt`
- 程序会以 `YOLO('models/best.pt')` 方式加载。

## 6. 网络与设备参数

在 `testUI8_dual_PLC_8.py` 中可调整：

- 机械臂服务器：`robot_ip = "192.168.31.xxx"`, `robot_port = xxx`
- PLC：`PLC_IP = "192.xxx"`, `PLC_PORT = 5xxx02`，寄存器 `%MW3`（地址 3）
- 修改后请确保 Windows 防火墙和交换机策略允许通信。

## 7. 运行

```powershell
# 激活虚拟环境（若未激活）
.\.venv\Scripts\Activate

# 运行
python .\testUI8_dual_PLC_8.py
```

操作步骤：
- 开启摄像头 → 预览画面
- 开始检测 → 程序先发送 50 条空坐标，同时预热 YOLO
- 切换为真实坐标发送 → 与机械臂/PLC 通信

## 8. 常见问题

- 无法导入 open_cv_show1/retrun_frame：
  - 检查 `sys.path.append(...)` 路径是否存在；
  - 或将 `open_cv_show1.py` 放入项目并直接相对导入。
- Torch 安装失败：
  - 使用上文给出的官方源 `--index-url` 指向 CPU 或对应 CUDA 版本；
  - 确保显卡驱动和 CUDA 版本匹配（查看 PyTorch 官网安装表）。
- 画面卡顿/闪烁：
  - 已在代码中仅保留 YOLO 线程驱动 GUI 刷新，并启用快速缩放；
  - 仍卡顿时可降低 UI 更新频率或减小画面尺寸。
- 机器人/PLC 不通：
  - 检查 IP/端口、交换机、网段、防火墙；
  - 用 `Test-NetConnection <ip> -Port <port>` 测试端口连通。

## 9. 开发小贴士

- VS Code 建议安装扩展：Python、Pylance
- 激活虚拟环境后运行/调试，避免与系统 Python 混用
- 若需要 GPU 加速，优先保证 PyTorch/CUDA 版本匹配