# 标准库
import os
import sys
import time
import math
import json
import socket
import random
import threading
import collections

# 第三方库
import cv2
import numpy as np
from pymodbus.client import ModbusTcpClient
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QMutex, QMetaObject
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QTextEdit, QGridLayout, QComboBox
from ultralytics import YOLO

# 项目内模块
from test_yolo6 import detect_objects

# 相机 SDK 示例路径（请确认路径存在）
sys.path.append(r"F:\YOLOV8\PotatoDetection\MV Viewer\Development\Samples\Python\IMV\opencv_byGetFrame")
from open_cv_show1 import retrun_frame



plc_mw3_flag = 0  # 默认发0
plc_enabled = True# plc_enabled = False


def send_mw3_value(value, plc_enabled):
    print(f"🛰 writing PLC: %MW3 = {value}")
    if not plc_enabled:
        print(" plc_enabled = False")
        return

    PLC_IP = "192.168.1.88"
    PLC_PORT = 502
    REGISTER_ADDR = 3  # 对应 %MW3

    client = ModbusTcpClient(PLC_IP, port=PLC_PORT)
    if client.connect():
        result = client.write_register(REGISTER_ADDR, value)
        if not result.isError():
            print(f" writing PLC success: %MW3 = {value}")
        else:
            print(" writing PLC failed :", result)
        client.close()
    else:
        print("connect PLC failed")

def mw3_loop_thread():
    global plc_mw3_flag
    sequence = [1, 2]
    idx = 0

    while True:
        if not plc_enabled:
            # time.sleep(0.5)
            continue

        # 根据标志决定是否发1/2 或 0
        value_to_send = plc_mw3_flag if plc_mw3_flag in [0] else sequence[idx % len(sequence)]
        idx += 1 if plc_mw3_flag != 0 else 0

        # 发送值
        send_mw3_value(value_to_send, plc_enabled=True)

        # time.sleep(0.5)


class CameraThread(QThread):
    """摄像头线程"""
    frame_signal = pyqtSignal(np.ndarray)
    timestamp_signal = pyqtSignal(int)  # 发出相机时间戳（毫秒或 SDK 单位）

    def __init__(self):
        super().__init__()
        self.running = False
        self.cap = None
        self.current_frame = None
        self.current_timestamp = None  # 毫秒时间戳（线程安全通过 mutex 访问）
        self.mutex = QMutex()
        self.emit_frame = True  # 控制是否发出信号的标志

        self.frame_max_len = 10  # 保存最近若干帧，避免阻塞时丢帧
        self.frame_queue = collections.deque(maxlen=self.frame_max_len)

    def run(self):
        """打开摄像头并持续读取画面（根据 retrun_frame 返回值兼容处理）"""
        self.running = True
        while self.running:
            try:
                # retrun_frame() 可能返回 generator，支持 (ret, frame) 或 (ret, frame, timestamp)
                for item in retrun_frame():
                    if not self.running:
                        break
                    if isinstance(item, (tuple, list)):#判断是否返回了相机时间戳
                        if len(item) == 3:
                            ret, frame, ts = item
                        elif len(item) == 2:
                            ret, frame = item
                            ts = None
                        else:
                            # 非预期结构，跳过
                            continue
                    else:
                        # 非 tuple 返回，跳过
                        continue

                    if not ret or frame is None:
                        continue

                    self.mutex.lock()
                    try:
                        self.current_frame = frame
                        # 若相机提供 ts 使用之，否则使用系统毫秒时间作为备选
                        try:
                            self.current_timestamp = int(ts) if ts is not None else int(time.time() * 1000)
                        except Exception:
                            self.current_timestamp = int(time.time() * 1000)
                    finally:
                        self.mutex.unlock()

                    if self.emit_frame:# 发送帧和时间戳信号
                        try:
                            self.frame_signal.emit(self.current_frame)
                        except Exception:
                            pass
                        try:
                            self.timestamp_signal.emit(int(self.current_timestamp))
                        except Exception:
                            pass

                    # 继续读取下一个帧
                    if not self.running:
                        break
            except Exception as e:
                print("CameraThread.run 异常:", e)
                time.sleep(0.1)

        # 退出前清理（如有需要）
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        print("---CameraThread end---")

    def get_current_frame(self):
        """线程安全获取当前帧"""
        self.mutex.lock()
        try:
            frame = self.current_frame
        finally:
            self.mutex.unlock()
        return frame

    def get_current_timestamp(self):
        """线程安全读取当前帧时间戳（毫秒或 SDK 单位）"""
        self.mutex.lock()
        try:
            ts = self.current_timestamp
        finally:
            self.mutex.unlock()
        return ts

    def set_emit_frame(self, enable: bool):
        """允许/禁止发出 frame_signal"""
        self.mutex.lock()
        try:
            self.emit_frame = bool(enable)
        finally:
            self.mutex.unlock()

    def stop(self):
        """关闭摄像头"""
        self.running = False
        if self.cap:
            self.cap.release()
        self.quit()
        self.wait()


class YoloThread(QThread):
    """YOLO 目标检测线程"""
    detection_signal = pyqtSignal(list, list, int)
    frame_signal = pyqtSignal(np.ndarray)

    def __init__(self, camera_thread):
        super().__init__()
        self.running = False
        self.camera_thread = camera_thread
        self.model = None  # 延迟在 run 中加载模型，避免在主线程阻塞
        self.last_detections = collections.deque(maxlen=30)
        self.alpha = 0.6  # 平滑因子

        self.frame_max_len = 5
        self.frame_queue = collections.deque(maxlen=self.frame_max_len)

    def run(self):
        try:
            if self.model is None:
                t0 = time.perf_counter()
                _append_send_log(f"yolo_load_start,{t0}")
                self.model = YOLO('models/best.pt')
                t1 = time.perf_counter()
                _append_send_log(f"yolo_load_end,{t1},cost={t1-t0:.4f}")
        except Exception as e:
            print("YOLO 模型加载失败：", e)
            _append_send_log(f"yolo_load_fail,{time.perf_counter()},{repr(e)}")
            return

        self.running = True
        while self.running:
            frame = self.camera_thread.get_current_frame()
            if frame is not None:
                det_begin = time.perf_counter()
                res = detect_objects(frame, self.model)
                det_end = time.perf_counter()
                if isinstance(res, tuple) or isinstance(res, list):
                    if len(res) == 4:#看是否返回时间戳
                        frame, detections, centroids, det_time = res
                    else:
                        frame, detections, centroids = res
                        det_time = None
                else:
                    continue
                ts = self.camera_thread.get_current_timestamp()
                if ts is None:
                    ts = det_time if det_time is not None else int(time.time() * 1000)
                detections = self._smooth_detections(detections)
                try:
                    self.detection_signal.emit(detections, centroids, int(ts))
                    self.frame_signal.emit(frame)
                finally:
                    _append_send_log(
                        f"yolo_detect_emit,{time.perf_counter():.6f},det_cost={det_end-det_begin:.4f},objs={len(centroids) if centroids else 0}"
                    )
            time.sleep(0.01)

    def _smooth_detections(self, new_detections):
        """指数加权移动平均（EWMA）平滑检测框"""
        smoothed_detections = []

        for det in new_detections:
            best_match = None
            # 遍历所有历史检测列表中的每个检测项
            for prev_dets in self.last_detections:
                for prev_det in prev_dets:
                    if self._is_same_target(det, prev_det):
                        best_match = prev_det
                        break  # 找到匹配后跳出内层循环
                if best_match:
                    break  # 找到匹配后跳出外层循环

            if best_match:
                # 应用平滑
                det["x1"] = int(self.alpha * det["x1"] + (1 - self.alpha) * best_match["x1"])
                det["y1"] = int(self.alpha * det["y1"] + (1 - self.alpha) * best_match["y1"])
                det["x2"] = int(self.alpha * det["x2"] + (1 - self.alpha) * best_match["x2"])
                det["y2"] = int(self.alpha * det["y2"] + (1 - self.alpha) * best_match["y2"])

            smoothed_detections.append(det)

        # 将当前帧的平滑结果保存到历史记录中
        self.last_detections.append(smoothed_detections)
        return smoothed_detections

    def _is_same_target(self, det1, det2):
        """判断两个目标框是否为同一目标"""
        iou_threshold = 0.5
        x1, y1, x2, y2 = det1["x1"], det1["y1"], det1["x2"], det1["y2"]
        x1_, y1_, x2_, y2_ = det2["x1"], det2["y1"], det2["x2"], det2["y2"]

        inter_x1 = max(x1, x1_)
        inter_y1 = max(y1, y1_)
        inter_x2 = min(x2, x2_)
        inter_y2 = min(y2, y2_)

        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2_ - x1_) * (y2_ - y1_)

        iou = inter_area / (box1_area + box2_area - inter_area)
        return iou > iou_threshold

    def stop(self):
        self.running = False
        self.quit()
        self.wait()


class MyWindow(QWidget):
    empty_frames_sent = pyqtSignal()  # 声明信号
    
    def __init__(self):
        super().__init__()
        self.initUI()
        self.camera_thread = CameraThread()
        self.camera_thread.frame_signal.connect(self.update_camera_frame)
        self.priority_direction = "上"
        self.yolo_thread = None
        self.send_data_enabled = True
        self.socket_client = None
        self.robot_ip = "192.168.31.139"  # 机械臂服务器IP
        self.robot_port = 3366  # 机械臂服务器端口
        self.frame_num = 0  # 帧计数
        self.heartbeat_count = 1  # 心跳计数，避免 AttributeError
        self.last_heartbeat_time = 0  # 上次心跳时间（如需）
        self.last_camera_timestamp = None  # 如需使用相机时间戳也可初始化
        self._frame_num_lock = threading.Lock()
        self._sending_initial = False  # 防止重复触发初始发送
        self._initial_empty_phase = False            # 初始 50 次空帧阶段标志
        self._first_real_det_cached = None  
        # 确保信号已连接（将发送完成信号绑定到继续启动函数）
        try:
            self.empty_frames_sent.connect(self._continue_start_detection)#转换为真实坐标发送
        except Exception as e:
            print("empty_frames_sent connect failed:", e)
            pass

        # 发送阶段调试辅助属性（避免 AttributeError）
        self._last_data_send_mono = None          # 上一条实际数据帧发送完成时间
        self._last_detect_emit_mono = None        # 上一次检测结果发出时间（yolo_detect_emit 对应）
        self._last_any_send_mono = None           # 任意一次发送/心跳完成时间

    def _connect_robot(self):
        """连接机械臂服务器"""
        try:
            if self.socket_client is None:
                self.socket_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket_client.settimeout(3)
                self.socket_client.connect((self.robot_ip, self.robot_port))
                # 禁用 Nagle 算法，减少小包发送延迟
                try:
                    self.socket_client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                except Exception:
                    pass
                print("机械臂服务器连接成功")
        except Exception as e:
            print(f"连接失败: {e}")
            self.socket_client = None

    # def _send_data_frame(self, centroids,current_time):
    #     """发送数据帧，刀具角度在第七位"""
    #     if not self.send_data_enabled:  # 如果关闭数据发送，直接返回
    #         return
    #     if not self.socket_client:
    #         self._connect_robot()
    #         if not self.socket_client:
    #             return
    #     try:
    #         obj_number = len(centroids)
    #         trigger_time = current_time  # 时间戳
    #         data_header = f"Data;{self.frame_num};{trigger_time};{obj_number};"
    #         data_body = []
    #         # angle=random.randint(10,70)
    #         for idx, (cx, cy, angle) in enumerate(centroids):
    #             # dx = 640 - cx  # 第一机械臂的x轴
    #             dx=cx           #第二个机械臂的X轴
    #             dy = -cy
    #             angle=-angle
    #             # cy-=40
    #             # dx+=5
    #             # if angle<=-180 or angle>=180:
    #             #     angle=0;
    #             # dx = -415
    #             # cy = 263
    #             data_body.append(
    #                 f"{idx},{dx},{dy},0,0,0,{angle},0,0,0,0,0,no"
    #                 # f"{idx},{dx},{dy},0,0,0,30,0,0,0,0,0,no"
    #             )
    #         # if idx>0:
    #         #     full_data = "STX" + data_header + "|".join(data_body) + "ETX"
    #         # else:
    #         #     full_data = "STX" + data_header + "ETX"
    #         if(len(data_body)!=0):
    #              full_data = "STX" + data_header+"|" + "|".join(data_body) + "ETX"
    #         else:
    #              full_data = "STX" + data_header + "ETX"
    #         self.socket_client.sendall(full_data.encode("ascii"))
    #         self.frame_num += 1
    #     except Exception as e:
    #         print(f"发送数据帧失败: {e}")
    #         self.socket_client.close()
    #         self.socket_client = None

    # def _send_heartbeat(self):
    #     """发送心跳帧"""
    #     if not self.send_data_enabled:  # 如果关闭数据发送，直接返回
    #         return
    #     if not self.socket_client:
    #         return
    #     try:
    #         current_time = time.time()
    #         if current_time - self.last_heartbeat_time > 60:  # 每分钟发送一次
    #             heartbeat_msg = f"Heart;{self.heartbeat_count};"
    #             self.socket_client.sendall(heartbeat_msg.encode("ascii"))
    #             self.heartbeat_count += 1
    #             self.last_heartbeat_time = current_time
    #     except Exception as e:
    #         print(f"心跳发送失败: {e}")
    #         self.socket_client.close()
    #         self.socket_client = None

    def initUI(self):
        self.setWindowTitle("马铃薯芽眼识别程序")
        # self.setGeometry(100, 100, 900, 600)
        # self.resize(1200, 800)
        self.resize(1000, 720)#触摸屏1024*768
        self.center_window()
        self.setStyleSheet("background-color: #2E2E2E; color: white;")

        # 摄像头画面
        self.video_label = QLabel(self)
        # self.video_label.setFixedSize(1280, 1024)
        self.video_label.setFixedSize(640, 480)
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.setScaledContents(True)

        # 结果显示区域
        self.result_text = QTextEdit(self)
        self.result_text.setReadOnly(True)
        self.result_text.setFont(QFont("Arial", 12))
        self.result_text.setStyleSheet("background-color: #1E1E1E; color: #FFD700;")

        # 结果信息布局
        self.info_grid = QGridLayout()
        self.info_labels = {}
        info_titles = ["类别", "置信度", "X1", "Y1", "X2", "Y2", "中心坐标X", "中心坐标Y", "优先目标中心X",
                       "优先目标中心Y", "刀具旋转角度"]
        
        for i, title in enumerate(info_titles):
            label = QLabel(f"{title}：")
            label.setFont(QFont("Arial", 14, QFont.Bold))
            label.setStyleSheet("color: #00FFFF;")
            value = QLabel("0")
            value.setFont(QFont("Arial", 14))
            value.setStyleSheet("color: #FFD700;")
            self.info_labels[title] = value
            self.info_grid.addWidget(label, i, 0)
            self.info_grid.addWidget(value, i, 1)

        # 优先目标信息
        self.priority_label = QLabel("优先目标信息：")
        self.priority_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.priority_label.setStyleSheet("color: #FF4500;")
        self.priority_info = QTextEdit()
        self.priority_info.setReadOnly(True)
        self.priority_info.setFont(QFont("Arial", 12))
        self.priority_info.setStyleSheet("background-color: #1E1E1E; color: #FFD700;")

        # 方向选择按钮
        self.direction_selector = QComboBox(self)
        self.direction_selector.addItems(["上", "下", "左", "右"])
        self.direction_selector.currentIndexChanged.connect(self.change_priority_direction)
        self.direction_label = QLabel("当前优先方向: 上")
        self.direction_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.direction_label.setStyleSheet("color: #FFFFFF;")
        # 按钮
        self.btn_open_camera = QPushButton("开启摄像头", self)
        self.btn_open_camera.setStyleSheet("background-color: #4CAF50; color: white;")
        self.btn_open_camera.clicked.connect(self.start_camera)

        self.btn_close_camera = QPushButton("关闭摄像头", self)
        self.btn_close_camera.setStyleSheet("background-color: #F44336; color: white;")
        self.btn_close_camera.clicked.connect(self.stop_camera)

        self.btn_start_detection = QPushButton("开始检测", self)
        self.btn_start_detection.setStyleSheet("background-color: #2196F3; color: white;")
        self.btn_start_detection.clicked.connect(self.start_detection)

        self.btn_stop_detection = QPushButton("关闭检测", self)
        self.btn_stop_detection.setStyleSheet("background-color: #FF9800; color: white;")
        self.btn_stop_detection.clicked.connect(self.stop_detection)

        # 添加数据发送开关按钮
        self.btn_toggle_send_data = QPushButton("关闭数据发送", self)
        self.btn_toggle_send_data.setStyleSheet("background-color: #FF5733; color: white;")
        self.btn_toggle_send_data.clicked.connect(self.toggle_send_data)

        # 按钮布局
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.btn_open_camera)
        btn_layout.addWidget(self.btn_close_camera)
        btn_layout.addWidget(self.btn_start_detection)
        btn_layout.addWidget(self.btn_stop_detection)
        btn_layout.addWidget(self.btn_toggle_send_data)

        priority_layout = QVBoxLayout()
        priority_layout.addWidget(self.direction_label)
        priority_layout.addWidget(self.direction_selector)
        priority_layout.addWidget(self.priority_label)
        priority_layout.addWidget(self.priority_info)
        # 主布局
        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.video_label)
        left_layout.addLayout(btn_layout)
        main_layout.addLayout(left_layout)
        right_layout = QVBoxLayout()
        right_layout.addLayout(self.info_grid)
        main_layout.addLayout(right_layout)

        self.setLayout(main_layout)

    def center_window(self):
        """将窗口居中显示"""
        # 获取屏幕的几何信息
        screen = QApplication.primaryScreen().availableGeometry()
        # 获取窗口的几何信息
        size = self.geometry()
        # 计算居中位置
        x = (screen.width() - size.width()) // 2
        y = (screen.height() - size.height()) // 2
        # 移动窗口到计算的位置
        self.move(x, y)


    def toggle_send_data(self):
        """切换数据发送状态"""
        self.send_data_enabled = not self.send_data_enabled
        if self.send_data_enabled:
            self.btn_toggle_send_data.setText("关闭数据发送")
            self.btn_toggle_send_data.setStyleSheet("background-color: #FF5733; color: white;")
            print("数据发送已启用")
        else:
            self.btn_toggle_send_data.setText("开启数据发送")
            self.btn_toggle_send_data.setStyleSheet("background-color: #4CAF50; color: white;")
            print("数据发送已关闭")

    def change_priority_direction(self):
        """更改优先方向"""
        self.priority_direction = self.direction_selector.currentText()
        self.direction_label.setText(f"当前优先方向: {self.priority_direction}")

    def start_camera(self):
        """启动摄像头"""
        if not self.camera_thread.isRunning():
            self.camera_thread.start()

    def stop_camera(self):
        """关闭摄像头"""
        if self.camera_thread.isRunning():
            self.camera_thread.stop()
        self.video_label.clear()
        self.video_label.setStyleSheet("background-color: black;")

    def start_detection(self):
        # 防止重复点击/并发
        if self._sending_initial:
            return
        self._sending_initial = True
        self._initial_empty_phase = True
        _append_send_log(f"phase_start_detection,{time.perf_counter():.6f}")
        try:
            if not self.yolo_thread or not self.yolo_thread.isRunning():#保证先不开检测
                _append_send_log(f"warm_yolo_thread_start,{time.perf_counter():.6f}")
                self.yolo_thread = YoloThread(self.camera_thread)
                self.yolo_thread.detection_signal.connect(self.update_detections)
                self.yolo_thread.frame_signal.connect(self.update_camera_frame)
                self.yolo_thread.start()
                try:
                    self.camera_thread.set_emit_frame(False)#此时把camera_thread的发帧关掉
                    _append_send_log(f"cam_gui_emit_off,{time.perf_counter():.6f}")
                except Exception:
                    pass
        except Exception as e:
            _append_send_log(f"warm_yolo_thread_start_fail,{time.perf_counter():.6f},{repr(e)}")


        try:
            self._connect_robot()# 优先尝试建立并复用 socket 连接，降低发送失败和重连延迟
        except Exception:
            pass
        # 启动异步发送空坐标（50 次）
        self._send_initial_empty_frames(count=50, interval=0.12)
        print("start_detection: 已触发初始空坐标发送线程（50 次），完成后继续启动检测）")
        _append_send_log(f"phase_initial_empty_frames_begin,{time.perf_counter():.6f}")

    def _send_initial_empty_frames(self, count=50, interval=0.12):
        def worker():
            client = None
            created_client_for_worker = False
            prev_mono = None
            start_all = time.perf_counter()
            _append_send_log(f"init_begin,{start_all},count={count},interval={interval}")
            try:
                if getattr(self, "socket_client", None):
                    client = self.socket_client
                else:
                    try:
                        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        client.settimeout(1.0)
                        client.connect((self.robot_ip, self.robot_port))
                        try:
                            client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                        except Exception:
                            pass
                        created_client_for_worker = True
                        _append_send_log(f"init_conn_ok,{time.perf_counter():.6f}")
                    except Exception as e:
                        _append_send_log(f"init_conn_fail,{time.perf_counter():.6f},{repr(e)}")
                        try:
                            client.close()
                        except:
                            pass
                        client = None

                for i in range(count):
                    loop_begin = time.perf_counter()
                    try:
                        with self._frame_num_lock:
                            cur_frame_num = self.frame_num
                            self.frame_num += 1
                        ts_ms = int(time.time() * 1000)
                        data_header = f"Data;{cur_frame_num};{ts_ms};0;"
                        full_data = "STX" + data_header + "ETX"

                        dt_prev = None if prev_mono is None else (loop_begin - prev_mono)
                        _append_send_log(
                            f"init_send_try,{loop_begin:.6f},idx={i},seq={cur_frame_num},dt_prev={dt_prev if dt_prev is not None else 'None'}"
                        )

                        send_ok = False
                        send_start = time.perf_counter()
                        if client:
                            try:
                                client.sendall(full_data.encode("ascii"))
                                send_ok = True
                            except OSError as oe:
                                _append_send_log(f"init_send_fail_persist,{time.perf_counter():.6f},idx={i},seq={cur_frame_num},{repr(oe)}")
                                try:
                                    client.close()
                                except:
                                    pass
                                client = None
                                try:
                                    tmp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                                    tmp.settimeout(1.0)
                                    tmp.connect((self.robot_ip, self.robot_port))
                                    try:
                                        tmp.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                                    except Exception:
                                        pass
                                    tmp.sendall(full_data.encode("ascii"))
                                    send_ok = True
                                except Exception as e2:
                                    _append_send_log(f"init_send_fail_retry,{time.perf_counter():.6f},idx={i},seq={cur_frame_num},{repr(e2)}")
                                finally:
                                    try:
                                        tmp.close()
                                    except:
                                        pass
                        else:
                            try:
                                tmp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                                tmp.settimeout(1.0)
                                tmp.connect((self.robot_ip, self.robot_port))
                                try:
                                    tmp.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                                except Exception:
                                    pass
                                tmp.sendall(full_data.encode("ascii"))
                                send_ok = True
                            except Exception as e3:
                                _append_send_log(f"init_send_fail_short,{time.perf_counter():.6f},idx={i},seq={cur_frame_num},{repr(e3)}")
                            finally:
                                try:
                                    tmp.close()
                                except:
                                    pass

                        send_end = time.perf_counter()
                        if send_ok:
                            _append_send_log(
                                f"init_send_ok,{send_end:.6f},idx={i},seq={cur_frame_num},send_cost={send_end - send_start:.4f}"
                            )
                            self._last_initial_send_mono = send_end
                            self._last_any_send_mono = send_end
                        prev_mono = loop_begin
                    except Exception as e:
                        _append_send_log(f"init_loop_exception,{time.perf_counter():.6f},idx={i},{repr(e)}")
                    time.sleep(interval)
            finally:
                end_all = time.perf_counter()
                _append_send_log(f"init_done,{end_all:.6f},elapsed={end_all - start_all:.4f}")
                if created_client_for_worker and client:
                    try:
                        client.close()
                    except:
                        pass
                try:
                    self.empty_frames_sent.emit()
                except Exception:
                    pass
                self._sending_initial = False

        threading.Thread(target=worker, daemon=True).start()

    def _continue_start_detection(self):
        _append_send_log(f"cont_phase_switch,{time.perf_counter():.6f}")
        self._initial_empty_phase = False
        # self.camera_thread.set_emit_frame(False)
        # self.yolo_thread = YoloThread(self.camera_thread)
        # self.yolo_thread.detection_signal.connect(self.update_detections)
        # self.yolo_thread.frame_signal.connect(self.update_camera_frame)
        # self.yolo_thread.start()

        # global plc_mw3_flag
        # plc_mw3_flag = 1
        # print("开始检测：初始空帧已发送完毕，plc_mw3_flag = 1")
        # _append_send_log(f"cont_after_yolo_start,{time.perf_counter():.6f}")

        if self._first_real_det_cached:
            centroids, ts_ms = self._first_real_det_cached
            _append_send_log(f"first_cached_det_send_try,{time.perf_counter():.6f}")
            try:
                self._send_data_frame(centroids, ts_ms)
                _append_send_log(f"first_cached_det_send_done,{time.perf_counter():.6f}")
            except Exception as e:
                _append_send_log(f"first_cached_det_send_fail,{time.perf_counter():.6f},{repr(e)}")
            self._first_real_det_cached = None
        _append_send_log(f"phase_real_send_active,{time.perf_counter():.6f}")

    def _send_data_frame(self, centroids, current_time):
        if not self.send_data_enabled:
            return
        if not self.socket_client:
            self._connect_robot()
            if not self.socket_client:
                _append_send_log(f"data_send_skip_no_conn,{time.perf_counter():.6f}")
                return
        obj_number = len(centroids)
        with self._frame_num_lock:
            cur_frame_num = self.frame_num
            self.frame_num += 1
        t_try = time.perf_counter()
        dt_prev_send = ('None' if self._last_data_send_mono is None else f"{t_try - self._last_data_send_mono:.4f}")
        dt_detect_to_try = ('None' if self._last_detect_emit_mono is None else f"{t_try - self._last_detect_emit_mono:.4f}")
        _append_send_log(f"data_send_try,{t_try:.6f},seq={cur_frame_num},objs={obj_number},dt_prev_send={dt_prev_send},dt_detect_to_try={dt_detect_to_try}")
        # 组包
        data_header = f"Data;{cur_frame_num};{current_time};{obj_number};"
        body_parts = []
        for idx,(cx,cy,angle,size) in enumerate(centroids):
            body_parts.append(f"{idx},{cx},{-cy},0,0,0,{-angle},0,0,0,0,{size},no")
        if body_parts:
            full_data = "STX" + data_header + "|" + "|".join(body_parts) + "ETX"
        else:
            full_data = "STX" + data_header + "ETX"
        # 发送
        try:
            t0 = time.perf_counter()
            self.socket_client.sendall(full_data.encode("ascii"))
            t1 = time.perf_counter()
            self._last_data_send_mono = t1
            self._last_any_send_mono = t1
            _append_send_log(f"data_send_ok,{t1:.6f},seq={cur_frame_num},objs={obj_number},send_cost={t1 - t0:.4f},latency_detect_to_ok={( 'None' if self._last_detect_emit_mono is None else f'{t1 - self._last_detect_emit_mono:.4f}' )}")
        except Exception as e:
            _append_send_log(f"data_send_fail,{time.perf_counter():.6f},seq={cur_frame_num},{repr(e)}")
            try:
                self.socket_client.close()
            except:
                pass
            self.socket_client = None

    def _send_heartbeat(self):
        if not self.send_data_enabled:
            return
        if not self.socket_client:
            return
        try:
            current_time = time.time()
            if current_time - self.last_heartbeat_time > 60:
                t0 = time.perf_counter()
                dt_prev = None
                if self._last_any_send_mono is not None:
                    dt_prev = t0 - self._last_any_send_mono
                heartbeat_msg = f"Heart;{self.heartbeat_count};"
                self.socket_client.sendall(heartbeat_msg.encode("ascii"))
                self.heartbeat_count += 1
                self.last_heartbeat_time = current_time
                t1 = time.perf_counter()
                _append_send_log(
                    f"heartbeat_send,{t1:.6f},seq=H{self.heartbeat_count},dt_prev={dt_prev if dt_prev is not None else 'None'},cost={t1 - t0:.4f}"
                )
        except Exception as e:
            _append_send_log(f"heartbeat_fail,{time.perf_counter():.6f},{repr(e)}")
            try:
                self.socket_client.close()
            except:
                pass
            self.socket_client = None

    def stop_detection(self):
        """停止检测并恢复摄像头帧直出"""
        _append_send_log(f"stop_detection_call,{time.perf_counter():.6f}")
        try:
            if self.yolo_thread:
                self.yolo_thread.stop()
                self.yolo_thread = None
                _append_send_log(f"stop_detection_yolo_stopped,{time.perf_counter():.6f}")
        except Exception as e:
            _append_send_log(f"stop_detection_yolo_err,{time.perf_counter():.6f},{repr(e)}")
        try:
            if self.camera_thread and self.camera_thread.isRunning():
                self.camera_thread.set_emit_frame(True)
        except Exception:
            pass
        try:
            global plc_mw3_flag
            plc_mw3_flag = 0
            _append_send_log(f"stop_detection_plc_reset,{time.perf_counter():.6f}")
        except Exception:
            pass
        print("检测已停止")

    def update_camera_frame(self, frame):
        try:
            # 深拷贝，避免下一帧覆写底层缓冲导致撕裂
            # 同时转 RGB，替代 rgbSwapped()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = frame_rgb.copy()  # 强制副本，确保 QImage 不引用共享内存

            height, width, channel = frame_rgb.shape
            bytes_per_line = channel * width
            q_image = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888).copy()  # 再次拷贝，确保完全独立
            pixmap = QPixmap.fromImage(q_image)

            # 使用快速缩放，减少每帧开销；或将 video_label 设为与输入一致大小则可直接 setPixmap
            scaled_pixmap = pixmap.scaled(
                self.video_label.width(),
                self.video_label.height(),
                Qt.KeepAspectRatio,
                Qt.FastTransformation  # 替换 Smooth → Fast，提升流畅度
            )
            self.video_label.setPixmap(scaled_pixmap)
        except Exception as e:
            _append_send_log(f"update_frame_err,{time.perf_counter():.6f},{repr(e)}")



    def update_detections(self, detections, centroids, ts_ms):
        """更新检测结果"""
        import time
        now = time.perf_counter()
        try:
            _append_send_log(f"detect_slot_enter,{now:.6f},objs={len(centroids) if centroids else 0},dt_prev_detect={('None' if self._last_detect_emit_mono is None else f'{now - self._last_detect_emit_mono:.4f}')}")
            # self._last_detect_emit_mono = now
        except Exception:
            pass
        
        if self._initial_empty_phase:#预热阶段，发空坐标
            self._first_real_det_cached = (centroids, ts_ms)
            return

        bug_centers = []
        bugs = [c for c in detections if c['cls'] == 1]

        # 在这里加上非空判断
        if bugs:  # 检查 bugs 列表是否为空
            for bug in bugs:
                x1, y1, x2, y2 = bug['x1'], bug['y1'], bug['x2'], bug['y2']
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                # 将中心坐标添加到列表中
                bug_centers.append((int(center_x), int(center_y)))
        else:
            print("没有检测到类别为1的芽眼。")  # 如果列表为空，输出提示信息
        potatoes = [d for d in detections if d['cls'] == 0]
        if potatoes:
            self.priority_info.setPlainText("\n".join([str(p) for p in potatoes]))
            # 计算优先目标
            priority_potato = min(
                potatoes,
                key=lambda p: p['y1'] if self.priority_direction == "上" else
                (p['y2'] if self.priority_direction == "下" else
                 (p['x1'] if self.priority_direction == "左" else p['x2']))
            )

            # 计算优先目标的中心坐标
            priority_center_x = (priority_potato["x1"] + priority_potato["x2"]) // 2
            priority_center_y = (priority_potato["y1"] + priority_potato["y2"]) // 2

            self.info_labels["优先目标中心X"].setText(str(priority_center_x))
            self.info_labels["优先目标中心Y"].setText(str(priority_center_y))

            self.info_labels["类别"].setText(str(priority_potato['cls']))
            self.info_labels["置信度"].setText(f"{priority_potato['conf']:.2f}")
            self.info_labels["X1"].setText(str(priority_potato['x1']))
            self.info_labels["Y1"].setText(str(priority_potato['y1']))
            self.info_labels["X2"].setText(str(priority_potato['x2']))
            self.info_labels["Y2"].setText(str(priority_potato['y2']))
            self.info_labels["中心坐标X"].setText(str((priority_potato['x1'] + priority_potato['x2']) // 2))
            self.info_labels["中心坐标Y"].setText(str((priority_potato['y1'] + priority_potato['y2']) // 2))

            # self.info_labels["刀具旋转角度"].setText(f"{centroids[2]}°")
            # print("target_angle:", centroids[2])

            if centroids:
                centroid_text = "\\n".join([f"形心: ({cx}, {cy})" for cx, cy, angle,size in centroids])
                self.result_text.setPlainText(centroid_text)  # 更新 UI 显示
            else:
                self.result_text.clear()
        else:
            self.priority_info.clear()
            for key in self.info_labels:
                self.info_labels[key].setText("0")
        self._send_data_frame(centroids, ts_ms)  # 发送数据帧

        # self._send_heartbeat()  # 发送心跳帧
        # self._send_to_robot(robot_data)

# 在文件中任意较前位置（例如 imports 之后）加入一个简单的日志工具：
def _append_send_log(line: str):
    """追加调试日志到 send_timestamps.log（忽略所有异常）"""
    try:
        with open("send_timestamps.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass
# 可选：启动时写一个开头
_append_send_log(f"--- run_start,{time.perf_counter()} ---")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyWindow()

    window.show()

    threading.Thread(target=mw3_loop_thread, daemon=True).start()
    sys.exit(app.exec_())
