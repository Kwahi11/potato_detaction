import random
import sys
import cv2
import numpy as np
import time
import collections
import socket
import json
import threading
import psutil
from PyQt5.QtCore import QThread, pyqtSignal, QMutex, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QPushButton, QVBoxLayout,
                             QHBoxLayout, QTextEdit, QGridLayout, QComboBox, QMessageBox)
from test_yolo6 import detect_objects
from ultralytics import YOLO
from fengzhuangwutu import daoju

import numpy as np
import math
import time

sys.path.append(r"F:\YOLOV8\PotatoDetection-main\MV Viewer\Development\Samples\Python\IMV\opencv_byGetFrame")
from open_cv_show3 import *

import time
from collections import deque


class CameraThread(QThread):
    """摄像头线程 - 支持多设备"""
    frame_signal = pyqtSignal(np.ndarray)
    error_signal = pyqtSignal(str)

    def __init__(self, camera_index):
        super().__init__()
        self.camera_index = camera_index
        self.running = False
        self.cap = None
        self.current_frame = None
        self.mutex = QMutex()
        self.emit_frame = True
        self.frame_max_len = 10
        self.frame_queue = collections.deque(maxlen=self.frame_max_len)
        self.frame_counter = 0
        self.last_log_time = time.time()
        self.resolution = (640, 480)  # 默认分辨率
        self.fps = 15  # 默认帧率
        self.frame_generator = None  # 存储帧生成器

    def set_resolution(self, width, height):
        """设置分辨率"""
        self.resolution = (width, height)

    def set_fps(self, fps):
        """设置帧率"""
        self.fps = fps

    def run(self):
        """打开摄像头并持续读取画面"""
        try:
            # 根据摄像头索引设置
            # self.cap = cv2.VideoCapture(self.camera_index)
            # if not self.cap.isOpened():
            #     self.error_signal.emit(f"无法打开摄像头 {self.camera_index}")
            #     return

            # # 设置分辨率
            # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            # self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            self.frame_generator = retrun_frame(self.camera_index)
            if not self.frame_generator:
                self.error_signal.emit(f"无法打开相机 {self.camera_index}")
                return

            self.running = True
            target_frame_time = 1.0 / self.fps

            while self.running:
                start_time = time.time()

                # ret, frame = self.cap.read()
                for ret, frame in retrun_frame():
                    if not ret:
                        current_time = time.time()
                        if current_time - self.last_log_time > 5:  # 每5秒记录一次错误
                            self.error_signal.emit(f"摄像头 {self.camera_index} 读取失败")
                            self.last_log_time = current_time
                        continue

                    # 缩放帧以降低分辨率
                    if self.resolution != (frame.shape[1], frame.shape[0]):
                        frame = cv2.resize(frame, self.resolution)

                    self.mutex.lock()
                    self.current_frame = frame
                    self.frame_counter += 1
                    self.mutex.unlock()

                    if self.emit_frame:
                        self.frame_signal.emit(self.current_frame)

                elapsed_time = time.time() - start_time
                sleep_time = max(0, target_frame_time - elapsed_time)
                time.sleep(sleep_time)

                # 每5秒记录一次帧率
                current_time = time.time()
                if current_time - self.last_log_time > 5:
                    fps = self.frame_counter / 5.0
                    print(f"摄像头 {self.camera_index} 当前帧率: {fps:.1f} FPS")
                    self.frame_counter = 0
                    self.last_log_time = current_time

            # if self.cap:
            #     self.cap.release()
            cv2.destroyAllWindows()
            print(f"---摄像头 {self.camera_index} 结束---")

        except Exception as e:
            self.error_signal.emit(f"摄像头 {self.camera_index} 错误: {str(e)}")
        finally:
            self.running = False
            if self.frame_generator:
                # 安全释放相机资源
                try:
                    next(self.frame_generator)  # 触发生成器清理
                except StopIteration:
                    pass
                self.frame_generator = None
            print(f"---相机 {self.camera_index} 结束---")

    def get_current_frame(self):
        """获取当前帧"""
        self.mutex.lock()
        frame = self.current_frame
        self.mutex.unlock()
        return frame

    def set_emit_frame(self, emit_frame):
        """设置是否发出信号"""
        self.mutex.lock()
        self.emit_frame = emit_frame
        self.mutex.unlock()

    def stop(self):
        """关闭摄像头"""
        self.running = False
        if self.cap:
            self.cap.release()
        self.quit()
        self.wait(2000)  # 等待2秒线程退出


class YoloThread(QThread):
    """YOLO 目标检测线程 - 支持模型缓存"""
    detection_signal = pyqtSignal(list, list, int)
    frame_signal = pyqtSignal(np.ndarray)
    error_signal = pyqtSignal(str)

    # 模型缓存 - 避免重复加载
    model_cache = {}

    def __init__(self, camera_thread, model_path='models/best.pt'):
        super().__init__()
        self.running = False
        self.camera_thread = camera_thread
        self.model_path = model_path

        # 从缓存获取模型或加载新模型
        if model_path in YoloThread.model_cache:
            self.model = YoloThread.model_cache[model_path]
            print(f"使用缓存的模型: {model_path}")
        else:
            try:
                self.model = YOLO(model_path)
                YoloThread.model_cache[model_path] = self.model
                print(f"加载新模型: {model_path}")
            except Exception as e:
                self.error_signal.emit(f"模型加载失败: {str(e)}")
                self.model = None

        self.last_detections = collections.deque(maxlen=30)
        self.alpha = 0.6  # 平滑因子
        self.frame_max_len = 5
        self.frame_queue = collections.deque(maxlen=self.frame_max_len)
        self.skip_frames = 0  # 跳帧计数器
        self.skip_interval = 1  # 每2帧处理1帧 (0=不跳帧)
        self.last_process_time = time.time()
        self.process_counter = 0

    def set_skip_interval(self, interval):
        """设置跳帧间隔"""
        self.skip_interval = max(0, interval)

    def run(self):
        """运行 YOLO 检测"""
        if not self.model:
            self.error_signal.emit("YOLO模型未加载，无法启动检测")
            return

        self.running = True
        while self.running:
            try:
                start_time = time.time()
                frame = self.camera_thread.get_current_frame()

                if frame is None:
                    time.sleep(0.02)
                    continue

                # 跳帧处理 - 减轻CPU负担
                self.skip_frames += 1
                if self.skip_frames <= self.skip_interval:
                    time.sleep(0.02)
                    continue
                self.skip_frames = 0

                # 处理帧
                frame, detections, centroids, current_time = detect_objects(frame, self.model)
                detections = self._smooth_detections(detections)

                self.detection_signal.emit(detections, centroids, current_time)
                self.frame_signal.emit(frame)

                # 性能监控
                self.process_counter += 1
                current_time = time.time()
                if current_time - self.last_process_time > 5:
                    fps = self.process_counter / (current_time - self.last_process_time)
                    print(f"YOLO处理帧率: {fps:.1f} FPS")
                    self.process_counter = 0
                    self.last_process_time = current_time

                elapsed_time = time.time() - start_time
                sleep_time = max(0, 0.1 - elapsed_time)
                time.sleep(sleep_time)

            except Exception as e:
                self.error_signal.emit(f"YOLO处理错误: {str(e)}")
                time.sleep(1)  # 出错后暂停1秒

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
        self.wait(2000)  # 等待2秒线程退出


class SendThread(QThread):
    """独立的网络发送线程 - 避免阻塞主线程"""
    error_signal = pyqtSignal(str)
    status_signal = pyqtSignal(str)

    def __init__(self, robot_ip, robot_port):
        super().__init__()
        self.robot_ip = robot_ip
        self.robot_port = robot_port
        self.running = False
        self.socket_client = None
        self.queue = collections.deque(maxlen=100)  # 发送队列
        self.mutex = QMutex()
        self.send_interval = 0.05  # 20FPS
        self.frame_num = 0
        self.heartbeat_count = 1
        self.last_heartbeat_time = 0
        self.connected = False
        self.connect_attempts = 0
        self.last_connect_attempt = 0

    def add_to_queue(self, centroids, current_time):
        """添加数据到发送队列"""
        if not self.running:
            return

        self.mutex.lock()
        self.queue.append((centroids, current_time))
        self.mutex.unlock()

    def run(self):
        """运行发送线程"""
        self.running = True
        self.status_signal.emit("发送线程启动")

        while self.running:
            try:
                # 检查连接状态
                if not self.connected:
                    current_time = time.time()
                    if current_time - self.last_connect_attempt > 5:  # 每5秒尝试一次
                        self._connect_robot()
                        self.last_connect_attempt = current_time
                    else:
                        time.sleep(0.5)
                        continue

                # 处理队列中的消息
                if self.queue:
                    self.mutex.lock()
                    centroids, current_time = self.queue.popleft()
                    self.mutex.unlock()

                    self._send_data_frame(centroids, current_time)
                else:
                    self._send_heartbeat()

                # 控制发送速率 - 防止网络拥塞
                time.sleep(self.send_interval)

            except Exception as e:
                self.error_signal.emit(f"发送线程错误: {str(e)}")
                self.connected = False
                if self.socket_client:
                    self.socket_client.close()
                    self.socket_client = None
                time.sleep(1)

    def _connect_robot(self):
        """连接机械臂服务器"""
        try:
            self.socket_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket_client.settimeout(3)
            self.socket_client.connect((self.robot_ip, self.robot_port))
            self.connected = True
            self.connect_attempts = 0
            self.status_signal.emit(f"已连接到 {self.robot_ip}:{self.robot_port}")
        except Exception as e:
            self.connect_attempts += 1
            self.connected = False
            if self.socket_client:
                self.socket_client.close()
                self.socket_client = None
            self.error_signal.emit(f"连接失败({self.connect_attempts}): {str(e)}")

    def _send_data_frame(self, centroids, current_time):
        """发送数据帧"""
        if not self.connected or not self.socket_client:
            return

        try:
            obj_number = len(centroids)
            trigger_time = current_time

            data_header = f"Data;{self.frame_num};{trigger_time};{obj_number};"
            data_body = []

            for idx, (cx, cy, angle) in enumerate(centroids):
                dx = cx
                dy = -cy
                data_body.append(
                    f"{idx},{dx},{dy},0,0,0,{angle},0,0,0,0,0,no"
                )

            if data_body:
                full_data = "STX" + data_header + "|" + "|".join(data_body) + "ETX"
            else:
                full_data = "STX" + data_header + "ETX"

            self.socket_client.sendall(full_data.encode("ascii"))
            self.frame_num += 1

        except Exception as e:
            self.error_signal.emit(f"发送数据失败: {str(e)}")
            self.connected = False
            if self.socket_client:
                self.socket_client.close()
                self.socket_client = None

    def _send_heartbeat(self):
        """发送心跳帧"""
        if not self.connected or not self.socket_client:
            return

        try:
            current_time = time.time()
            if current_time - self.last_heartbeat_time > 60:  # 每分钟发送一次
                heartbeat_msg = f"Heart;{self.heartbeat_count};"
                self.socket_client.sendall(heartbeat_msg.encode("ascii"))
                self.heartbeat_count += 1
                self.last_heartbeat_time = current_time
        except Exception as e:
            self.error_signal.emit(f"心跳发送失败: {str(e)}")
            self.connected = False
            if self.socket_client:
                self.socket_client.close()
                self.socket_client = None

    def stop(self):
        """停止发送线程"""
        self.running = False
        if self.socket_client:
            self.socket_client.close()
        self.quit()
        self.wait(2000)


class PotatoDetectionApp(QWidget):
    """马铃薯检测应用 - 支持多实例"""

    def __init__(self, camera_index, robot_ip, robot_port, model_path='models/best.pt', title="马铃薯芽眼识别程序"):
        super().__init__()
        self.camera_index = camera_index
        self.robot_ip = robot_ip
        self.robot_port = robot_port
        self.model_path = model_path
        self.title = f"{title} - 摄像头{camera_index}"

        # 先初始化所有属性
        self.fps_label = QLabel("FPS: 0.0")
        self.fps_label.setStyleSheet("color: #00FF00; font-weight: bold;")


        self.initUI()

        # 性能监控
        self.last_update_time = time.time()
        self.fps_counter = 0
        self.fps_label = QLabel("FPS: 0.0")
        self.fps_label.setStyleSheet("color: #00FF00; font-weight: bold;")

        # 资源监控定时器
        self.resource_timer = QTimer(self)
        self.resource_timer.timeout.connect(self.update_resource_usage)
        self.resource_timer.start(2000)  # 每2秒更新一次

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(100, 100, 900, 600)
        self.setStyleSheet("background-color: #2E2E2E; color: white;")

        # 摄像头画面
        self.video_label = QLabel(self)
        self.video_label.setFixedSize(640, 480)  # 减小显示尺寸
        self.video_label.setStyleSheet("background-color: black; border: 1px solid #444;")

        # 分辨率选择
        self.resolution_combo = QComboBox(self)
        self.resolution_combo.addItems(["640x480", "800x600", "1024x768", "1280x720"])
        self.resolution_combo.setCurrentText("640x480")
        self.resolution_combo.currentIndexChanged.connect(self.change_resolution)

        # 帧率选择
        self.fps_combo = QComboBox(self)
        self.fps_combo.addItems(["10 FPS", "15 FPS", "20 FPS", "30 FPS"])
        self.fps_combo.setCurrentText("15 FPS")
        self.fps_combo.currentIndexChanged.connect(self.change_fps)

        # 结果显示区域
        self.result_text = QTextEdit(self)
        self.result_text.setReadOnly(True)
        self.result_text.setFont(QFont("Arial", 10))
        self.result_text.setFixedHeight(100)
        self.result_text.setStyleSheet("background-color: #1E1E1E; color: #FFD700; border: 1px solid #444;")

        # 结果信息布局
        self.info_grid = QGridLayout()
        self.info_labels = {}
        info_titles = ["类别", "置信度", "X1", "Y1", "X2", "Y2", "中心坐标X", "中心坐标Y", "刀具旋转角度"]

        for i, title in enumerate(info_titles):
            label = QLabel(f"{title}：")
            label.setFont(QFont("Arial", 10))
            label.setStyleSheet("color: #00FFFF;")
            value = QLabel("0")
            value.setFont(QFont("Arial", 10))
            value.setStyleSheet("color: #FFD700;")
            self.info_labels[title] = value
            self.info_grid.addWidget(label, i, 0)
            self.info_grid.addWidget(value, i, 1)

        # 优先目标信息
        self.priority_label = QLabel("优先目标信息：")
        self.priority_label.setFont(QFont("Arial", 10, QFont.Bold))
        self.priority_label.setStyleSheet("color: #FF4500;")
        self.priority_info = QTextEdit()
        self.priority_info.setReadOnly(True)
        self.priority_info.setFont(QFont("Arial", 10))
        self.priority_info.setFixedHeight(80)
        self.priority_info.setStyleSheet("background-color: #1E1E1E; color: #FFD700; border: 1px solid #444;")

        # 方向选择按钮
        self.direction_selector = QComboBox(self)
        self.direction_selector.addItems(["上", "下", "左", "右"])
        self.direction_selector.currentIndexChanged.connect(self.change_priority_direction)
        self.direction_label = QLabel("当前优先方向: 上")
        self.direction_label.setFont(QFont("Arial", 10))
        self.direction_label.setStyleSheet("color: #FFFFFF;")

        # 状态栏
        status_layout = QHBoxLayout()
        self.status_label = QLabel("状态: 就绪")
        self.status_label.setFont(QFont("Arial", 10))
        self.status_label.setStyleSheet("color: #00FF00;")
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        status_layout.addWidget(self.fps_label)

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

        # 数据发送开关按钮
        self.btn_toggle_send_data = QPushButton("关闭数据发送", self)
        self.btn_toggle_send_data.setStyleSheet("background-color: #FF5733; color: white;")
        self.btn_toggle_send_data.clicked.connect(self.toggle_send_data)

        # 跳帧控制
        self.skip_frame_combo = QComboBox(self)
        self.skip_frame_combo.addItems(["不跳帧", "每2帧处理1帧", "每3帧处理1帧"])
        self.skip_frame_combo.setCurrentIndex(0)
        self.skip_frame_combo.currentIndexChanged.connect(self.change_skip_interval)
        skip_label = QLabel("处理策略:")
        skip_label.setStyleSheet("color: white;")

        # 按钮布局
        btn_layout = QGridLayout()
        btn_layout.addWidget(self.btn_open_camera, 0, 0)
        btn_layout.addWidget(self.btn_close_camera, 0, 1)
        btn_layout.addWidget(self.btn_start_detection, 1, 0)
        btn_layout.addWidget(self.btn_stop_detection, 1, 1)
        btn_layout.addWidget(self.btn_toggle_send_data, 2, 0, 1, 2)
        btn_layout.addWidget(skip_label, 3, 0)
        btn_layout.addWidget(self.skip_frame_combo, 3, 1)

        # 摄像头控制布局
        cam_control_layout = QHBoxLayout()
        cam_control_layout.addWidget(QLabel("分辨率:"))
        cam_control_layout.addWidget(self.resolution_combo)
        cam_control_layout.addSpacing(10)
        cam_control_layout.addWidget(QLabel("帧率:"))
        cam_control_layout.addWidget(self.fps_combo)
        cam_control_layout.addStretch()

        # 主布局
        main_layout = QVBoxLayout()

        # 顶部布局 - 视频和结果
        top_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        left_layout.addLayout(cam_control_layout)
        left_layout.addWidget(self.video_label)
        left_layout.addLayout(btn_layout)

        right_layout = QVBoxLayout()
        right_layout.addLayout(self.info_grid)
        right_layout.addWidget(self.direction_label)
        right_layout.addWidget(self.direction_selector)
        right_layout.addWidget(self.priority_label)
        right_layout.addWidget(self.priority_info)

        top_layout.addLayout(left_layout, 60)
        top_layout.addLayout(right_layout, 40)

        main_layout.addLayout(top_layout)
        main_layout.addWidget(self.result_text)
        main_layout.addLayout(status_layout)

        self.setLayout(main_layout)

        # 初始化摄像头线程
        self.camera_thread = CameraThread(self.camera_index)
        self.camera_thread.frame_signal.connect(self.update_camera_frame)
        self.camera_thread.error_signal.connect(self.show_error)

        # 初始化YOLO线程
        self.yolo_thread = None

        # 初始化发送线程 - 解决网络不稳定问题
        self.send_thread = SendThread(self.robot_ip, self.robot_port)
        self.send_thread.error_signal.connect(self.show_error)
        self.send_thread.status_signal.connect(self.update_status)
        self.send_thread.start()

    def update_resource_usage(self):
        """更新资源使用情况"""
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        memory_percent = memory.percent

        # 更新状态标签
        self.status_label.setText(f"状态: CPU: {cpu_percent}% | 内存: {memory_percent}%")

        # 根据资源使用情况动态调整
        if cpu_percent > 80:
            self.change_skip_interval(2)  # 高负载时跳更多帧
        elif cpu_percent > 60:
            self.change_skip_interval(1)
        else:
            self.change_skip_interval(0)

    def update_status(self, message):
        """更新状态信息"""
        self.status_label.setText(f"状态: {message}")

    def show_error(self, message):
        """显示错误信息"""
        QMessageBox.warning(self, "错误", message)
        self.status_label.setText(f"错误: {message}")
        self.status_label.setStyleSheet("color: #FF0000;")

    def change_resolution(self):
        """更改分辨率"""
        res_text = self.resolution_combo.currentText()
        width, height = map(int, res_text.split('x'))
        if self.camera_thread.isRunning():
            self.camera_thread.stop()
            time.sleep(0.5)
            self.camera_thread.set_resolution(width, height)
            self.camera_thread.start()
        else:
            self.camera_thread.set_resolution(width, height)

    def change_fps(self):
        """更改帧率"""
        fps_text = self.fps_combo.currentText()
        fps = int(fps_text.split()[0])
        if self.camera_thread.isRunning():
            self.camera_thread.stop()
            time.sleep(0.5)
            self.camera_thread.set_fps(fps)
            self.camera_thread.start()
        else:
            self.camera_thread.set_fps(fps)

    def change_skip_interval(self, index=None):
        """更改跳帧间隔"""
        if index is None:
            index = self.skip_frame_combo.currentIndex()
        else:
            self.skip_frame_combo.setCurrentIndex(index)

        intervals = [0, 1, 2]
        interval = intervals[index] if index < len(intervals) else 0

        if self.yolo_thread and self.yolo_thread.isRunning():
            self.yolo_thread.set_skip_interval(interval)

    def toggle_send_data(self):
        """切换数据发送状态"""
        self.send_thread.running = not self.send_thread.running
        if self.send_thread.running:
            self.btn_toggle_send_data.setText("关闭数据发送")
            self.btn_toggle_send_data.setStyleSheet("background-color: #FF5733; color: white;")
            self.status_label.setText("状态: 数据发送已启用")
        else:
            self.btn_toggle_send_data.setText("开启数据发送")
            self.btn_toggle_send_data.setStyleSheet("background-color: #4CAF50; color: white;")
            self.status_label.setText("状态: 数据发送已关闭")

    def change_priority_direction(self):
        """更改优先方向"""
        self.priority_direction = self.direction_selector.currentText()
        self.direction_label.setText(f"当前优先方向: {self.priority_direction}")

    def start_camera(self):
        """启动摄像头"""
        if not self.camera_thread.isRunning():
            self.camera_thread.start()
            self.status_label.setText("状态: 摄像头已启动")

    def stop_camera(self):
        """关闭摄像头"""
        if self.camera_thread.isRunning():
            self.camera_thread.stop()
            self.status_label.setText("状态: 摄像头已停止")
        self.video_label.clear()
        self.video_label.setStyleSheet("background-color: black; border: 1px solid #444;")

    def start_detection(self):
        """启动 YOLO 检测"""
        if self.yolo_thread is None or not self.yolo_thread.isRunning():
            self.yolo_thread = YoloThread(self.camera_thread, self.model_path)
            self.camera_thread.set_emit_frame(False)
            self.yolo_thread.detection_signal.connect(self.update_detections)
            self.yolo_thread.frame_signal.connect(self.update_camera_frame)
            self.yolo_thread.error_signal.connect(self.show_error)
            self.change_skip_interval()  # 设置当前跳帧间隔
            self.yolo_thread.start()
            self.status_label.setText("状态: 检测已启动")

    def stop_detection(self):
        """停止 YOLO 检测"""
        if self.yolo_thread is not None and self.yolo_thread.isRunning():
            self.yolo_thread.stop()
            self.yolo_thread = None
            self.camera_thread.set_emit_frame(True)
            self.status_label.setText("状态: 检测已停止")

    def update_camera_frame(self, frame):
        """更新摄像头画面"""
        # 计算FPS
        current_time = time.time()
        self.fps_counter += 1
        if current_time - self.last_update_time >= 1.0:
            fps = self.fps_counter / (current_time - self.last_update_time)
            self.fps_label.setText(f"FPS: {fps:.1f}")
            self.fps_counter = 0
            self.last_update_time = current_time

        # 更新画面
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        self.video_label.setPixmap(QPixmap.fromImage(q_image).scaled(
            self.video_label.width(),
            self.video_label.height(),
            aspectRatioMode=1  # 保持宽高比
        ))

    def update_detections(self, detections, centroids, current_time):
        """更新检测结果"""
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

            if centroids:
                centroid_text = "\\n".join([f"形心: ({cx}, {cy})" for cx, cy, angle in centroids])
                self.result_text.setPlainText(centroid_text)  # 更新 UI 显示
            else:
                self.result_text.clear()
        else:
            self.priority_info.clear()
            for key in self.info_labels:
                self.info_labels[key].setText("0")

        # 将数据添加到发送队列 - 避免直接发送导致阻塞
        if self.send_thread.running:
            self.send_thread.add_to_queue(centroids, current_time)

    def closeEvent(self, event):
        """窗口关闭事件处理"""
        self.stop_camera()
        if self.yolo_thread and self.yolo_thread.isRunning():
            self.yolo_thread.stop()
        if self.send_thread and self.send_thread.isRunning():
            self.send_thread.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # 设置进程优先级 (Windows)
    if sys.platform == "win32":
        import win32api, win32process, win32con

        pid = win32api.GetCurrentProcessId()
        handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, True, pid)
        win32process.SetPriorityClass(handle, win32process.BELOW_NORMAL_PRIORITY_CLASS)

    # 实例1 - 生产线A
    app1 = PotatoDetectionApp(
        camera_index=0,
        robot_ip="192.168.32.78",
        robot_port=3367,
        model_path='models/best.pt',
        title="马铃薯芽眼识别系统 - 生产线A"
    )
    app1.move(50, 50)

    # 实例2 - 生产线B
    app2 = PotatoDetectionApp(
        camera_index=1,
        robot_ip="192.168.31.139",
        robot_port=3366,
        model_path='models/best.pt',
        title="马铃薯芽眼识别系统 - 生产线B"
    )
    app2.move(1000, 50)

    app1.show()
    app2.show()

    sys.exit(app.exec_())