from __future__ import annotations

import sys
import time
import platform
from pathlib import Path
from typing import Tuple

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "best.pt"
# README 中提到的相机 SDK 示例路径（如与实际不符，请修改为你的实际路径）
SDK_FILE = Path(r"F:\YOLOV8\PotatoDetection-main\MV Viewer\Development\Samples\Python\IMV\opencv_byGetFrame\open_cv_show1.py")

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"

def log(ok: bool, msg: str) -> bool:
    print(f"{PASS if ok else FAIL} {msg}")
    return ok

def warn(msg: str) -> None:
    print(f"{WARN} {msg}")

def check_import(mod: str, attr: str | None = None) -> Tuple[bool, object | None]:
    try:
        m = __import__(mod, fromlist=[attr] if attr else [])
        if attr:
            getattr(m, attr)
        return True, m
    except Exception as e:
        print(f"{FAIL} 导入失败: {mod}{('.' + attr) if attr else ''} -> {e}")
        return False, None

def main() -> int:
    print("=== 环境自检开始 ===")
    print(f"- Python: {sys.version.split()[0]} on {platform.system()} {platform.release()}")
    print(f"- 项目目录: {PROJECT_DIR}")

    failures = 0

    # 1) 基础包导入
    ok_cv2, cv2 = check_import("cv2")
    ok_np, np = check_import("numpy")
    ok_qt, _ = check_import("PyQt5", "QtWidgets")
    ok_modbus, _ = check_import("pymodbus")
    ok_ultra, ultra = check_import("ultralytics")
    ok_torch, torch = check_import("torch")

    failures += 0 if ok_cv2 else 1
    failures += 0 if ok_np else 1
    failures += 0 if ok_qt else 1
    failures += 0 if ok_modbus else 1
    failures += 0 if ok_ultra else 1
    failures += 0 if ok_torch else 1

    # 2) Torch/CUDA 检查（GPU 环境为必需）
    if ok_torch:
        cuda_avail = torch.cuda.is_available()
        cudnn_avail = torch.backends.cudnn.is_available() if hasattr(torch.backends, "cudnn") else False
        torch_ver = getattr(torch, "__version__", "unknown")
        torch_cuda = getattr(torch.version, "cuda", None)

        print(f"{PASS} torch 版本: {torch_ver}")
        print(f"{PASS if cuda_avail else FAIL} CUDA 可用: {cuda_avail}")
        print(f"{PASS if cudnn_avail else FAIL} cuDNN 可用: {cudnn_avail}")
        print(f"{PASS if torch_cuda else WARN} 编译绑定的 CUDA: {torch_cuda}")

        if cuda_avail:
            try:
                device_count = torch.cuda.device_count()
                print(f"{PASS} GPU 数量: {device_count}")
                for i in range(device_count):
                    print(f"  - [{i}] {torch.cuda.get_device_name(i)}")
                # 简单张量上 GPU 自检
                t0 = time.time()
                x = torch.randn(1024, 1024, device="cuda")
                y = torch.mm(x, x)
                torch.cuda.synchronize()
                dt = (time.time() - t0) * 1000
                print(f"{PASS} CUDA 简单计算完成，耗时约 {dt:.1f} ms")
            except Exception as e:
                print(f"{FAIL} CUDA 运行测试失败: {e}")
                failures += 1
        else:
            print(f"{FAIL} 检测到 GPU 版本需求，但当前 CUDA 不可用。请检查 pytorch-cuda 版本与驱动匹配。")
            failures += 1

    # 3) YOLO/权重与推理自检（可选：存在权重时执行）
    if ok_ultra:
        from ultralytics import YOLO
        if MODEL_PATH.exists():
            try:
                device = 0 if ok_torch and torch.cuda.is_available() else "cpu"
                print(f"- 加载模型: {MODEL_PATH} (device={device})")
                model = YOLO(str(MODEL_PATH))
                dummy = np.zeros((320, 320, 3), dtype=np.uint8)
                t0 = time.time()
                _ = model.predict(source=dummy, imgsz=320, device=device, verbose=False)
                dt = (time.time() - t0) * 1000
                print(f"{PASS} YOLO 预热推理成功，耗时约 {dt:.1f} ms")
            except Exception as e:
                print(f"{FAIL} 加载/推理失败（models/best.pt）: {e}")
                failures += 1
        else:
            warn(f"未找到模型权重: {MODEL_PATH}（跳过 YOLO 推理自检）")

    # 4) 相机 SDK 路径与接口检查（存在则检查 retrun_frame）
    if SDK_FILE.exists():
        try:
            sys.path.insert(0, str(SDK_FILE.parent))
            mod_name = SDK_FILE.stem  # open_cv_show1
            m = __import__(mod_name)
            has_func = hasattr(m, "retrun_frame")
            print(f"{PASS if has_func else FAIL} 相机 SDK: {SDK_FILE.name} 中 retrun_frame 存在: {has_func}")
            if not has_func:
                failures += 1
        except Exception as e:
            print(f"{FAIL} 相机 SDK 导入失败: {e}")
            failures += 1
        finally:
            if str(SDK_FILE.parent) in sys.path:
                sys.path.remove(str(SDK_FILE.parent))
    else:
        warn(f"未找到相机 SDK 文件: {SDK_FILE}（若路径不同，请在脚本顶部 SDK_FILE 中修改）")

    # 5) OpenCV 基础功能自检（简易）
    if ok_cv2:
        try:
            _ = cv2.getBuildInformation()
            print(f"{PASS} OpenCV 可用，版本: {cv2.__version__}")
        except Exception as e:
            print(f"{FAIL} OpenCV 自检失败: {e}")
            failures += 1

    # 汇总
    print("=== 自检结束 ===")
    if failures == 0:
        print(f"{PASS} 所有必须项通过。")
        return 0
    else:
        print(f"{FAIL} 有 {failures} 项失败，请根据上方提示修复后重试。")
        return 1

if __name__ == "__main__":
    sys.exit(main())