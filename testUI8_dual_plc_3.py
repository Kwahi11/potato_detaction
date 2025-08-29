import random
import sys
import cv2
import numpy as np
import time
import collections
import socket
import json
from PyQt5.QtCore import QThread, pyqtSignal, QMutex
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import Qt  # 添加到现有的导入语句中
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QTextEdit, \
    QGridLayout, QComboBox
from test_yolo6 import detect_objects
from ultralytics import YOLO
from fengzhuangwutu import daoju

import numpy as np
import math
import time

sys.path.append(r"F:\YOLOV8\PotatoDetection-main\MV Viewer\Development\Samples\Python\IMV\opencv_byGetFrame")
from open_cv_show1 import *

import time
from collections import deque

import threading
import time
from pymodbus.client import ModbusTcpClient

plc_mw3_flag = 0  # 默认发0
plc_enabled = True
# plc_enabled = False

def send_mw3_value(value, plc_enabled):
    print(f"🛰️ 写入 PLC: %MW3 = {value}")
    if not plc_enabled:
        print(" 模拟模式：未发送，仅打印")
        return

    PLC_IP = "192.168.1.88"
    PLC_PORT = 502
    REGISTER_ADDR = 3  # 对应 %MW3

    client = ModbusTcpClient(PLC_IP, port=PLC_PORT)
    if client.connect():
        result = client.write_register(REGISTER_ADDR, value)
        if not result.isError():
            print(f" 成功写入: %MW3 = {value}")
        else:
            print(" 写入失败:", result)
        client.close()
    else:
        print("无法连接 PLC")

def mw3_loop_thread():
    global plc_mw3_flag
    sequence = [1, 2]
    idx = 0

    while True:
        if not plc_enabled:
            time.sleep(0.5)
            continue

        # 根据标志决定是否发1/2 或 0
        value_to_send = plc_mw3_flag if plc_mw3_flag in [0] else sequence[idx % len(sequence)]
        idx += 1 if plc_mw3_flag != 0 else 0

        # 发送值
        send_mw3_value(value_to_send, plc_enabled=True)

        time.sleep(0.5)


class CameraThread(QThread):
    """摄像头线程"""
    frame_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.running = False
        self.cap = None
        self.current_frame = None
        self.mutex = QMutex()
        self.emit_frame = True  # 控制是否发出信号的标志

        self.frame_max_len = 10
        self.frame_queue = collections.deque(maxlen=self.frame_max_len)

    def run(self):
        """打开摄像头并持续读取画面"""
        # self.cap = cv2.VideoCapture(1)
        # if not self.cap.isOpened():
        #     print("Error: 无法打开摄像头")
        #     return

        self.running = True
        while self.running:
            start_time = time.time()
            # ret, frame = self.cap.read()
            for ret, frame in retrun_frame():

                if ret:
                    # self.frame_queue.append(frame)
                    self.mutex.lock()
                    self.current_frame = frame
                    self.mutex.unlock()
                    # self.frame_signal.emit(fra                    self.current_frame = frameme)
                    if self.emit_frame:  # 如果允许发出信号
                        self.frame_signal.emit(self.current_frame)  # 发送画面

                # 睡眠限制取消
                elapsed_time = time.time() - start_time
                sleep_time = max(0, 0.033 - elapsed_time)  # 30 FPS 限制
                time.sleep(sleep_time)
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break

            cv2.destroyAllWindows()
            print("---Demo end---")

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
        self.wait()


class YoloThread(QThread):
    """YOLO 目标检测线程"""
    detection_signal = pyqtSignal(list, list,int)
    frame_signal = pyqtSignal(np.ndarray)

    def __init__(self, camera_thread):
        super().__init__()
        self.running = False
        self.camera_thread = camera_thread
        self.model = YOLO('models/best.pt')  # 加载 YOLO 模型
        self.last_detections = collections.deque(maxlen=30)
        self.alpha = 0.6  # 平滑因子

        self.frame_max_len = 5
        self.frame_queue = collections.deque(maxlen=self.frame_max_len)

    def run(self):
        """运行 YOLO 检测"""
        self.running = True
        while self.running:
            start_time = time.time()
            frame = self.camera_thread.get_current_frame()
            # self.frame_queue.append(frame)

            if frame is not None:
                # current_frame = self.frame_queue[-1]
                frame, detections, centroids,current_time = detect_objects(frame, self.model)
                detections = self._smooth_detections(detections)

                self.detection_signal.emit(detections, centroids,current_time)
                self.frame_signal.emit(frame)

            elapsed_time = time.time() - start_time
            sleep_time = max(0, 0.1 - elapsed_time)
            time.sleep(sleep_time)

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
    def __init__(self):
        super().__init__()
        self.initUI()
        self.camera_thread = CameraThread()
        self.camera_thread.frame_signal.connect(self.update_camera_frame)
        self.priority_direction = "上"  # 默认优先方向
        self.yolo_thread = None  # YOLO 检测线程
        self.send_data_enabled = True  # 默认启用数据发送

        # Socket客户端初始化
        self.socket_client = None
        # self.robot_ip = "192.168.32.78"  # 机械臂服务器IP
        # self.robot_port = 3367  # 机械臂服务器端口

        self.robot_ip = "192.168.31.139"  # 机械臂服务器IP
        self.robot_port = 3366  # 机械臂服务器端口
        self.frame_num = 0  # 帧计数
        self.heartbeat_count = 1  # 心跳计数
        self.last_heartbeat_time = 0  # 心跳计时



    def _connect_robot(self):
        """连接机械臂服务器"""
        try:
            if self.socket_client is None:
                self.socket_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket_client.settimeout(3)
                self.socket_client.connect((self.robot_ip, self.robot_port))
                print("机械臂服务器连接成功")
        except Exception as e:
            print(f"连接失败: {e}")
            self.socket_client = None

    def _send_data_frame(self, centroids,current_time):
        """发送数据帧，刀具角度在第七位"""
        if not self.send_data_enabled:  # 如果关闭数据发送，直接返回
            return
        if not self.socket_client:
            self._connect_robot()
            if not self.socket_client:
                return

        try:
            obj_number = len(centroids)
            trigger_time = current_time  # 时间戳

            data_header = f"Data;{self.frame_num};{trigger_time};{obj_number};"
            data_body = []
            # angle=random.randint(10,70)

            for idx, (cx, cy, angle) in enumerate(centroids):
                # dx = 640 - cx  # 第一机械臂的x轴
                dx=cx           #第二个机械臂的X轴
                dy = -cy
                angle=-angle
                # cy-=40
                # dx+=5
                # if angle<=-180 or angle>=180:
                #     angle=0;
                # dx = -415
                # cy = 263
                data_body.append(

                    f"{idx},{dx},{dy},0,0,0,{angle},0,0,0,0,0,no"
                    # f"{idx},{dx},{dy},0,0,0,30,0,0,0,0,0,no"
                )
            # if idx>0:
            #     full_data = "STX" + data_header + "|".join(data_body) + "ETX"
            # else:
            #     full_data = "STX" + data_header + "ETX"
            if(len(data_body)!=0):
                 full_data = "STX" + data_header+"|" + "|".join(data_body) + "ETX"
            else:
                 full_data = "STX" + data_header + "ETX"

            self.socket_client.sendall(full_data.encode("ascii"))
            self.frame_num += 1

        except Exception as e:
            print(f"发送数据帧失败: {e}")
            self.socket_client.close()
            self.socket_client = None

    def _send_heartbeat(self):
        """发送心跳帧"""
        if not self.send_data_enabled:  # 如果关闭数据发送，直接返回
            return
        if not self.socket_client:
            return

        try:
            current_time = time.time()
            if current_time - self.last_heartbeat_time > 60:  # 每分钟发送一次
                heartbeat_msg = f"Heart;{self.heartbeat_count};"
                self.socket_client.sendall(heartbeat_msg.encode("ascii"))
                self.heartbeat_count += 1
                self.last_heartbeat_time = current_time
        except Exception as e:
            print(f"心跳发送失败: {e}")
            self.socket_client.close()
            self.socket_client = None

    def initUI(self):
        self.setWindowTitle("马铃薯芽眼识别程序")
        # self.setGeometry(100, 100, 900, 600)
        self.resize(1200, 800)
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

        # info_titles = ["类别", "置信度", "X1", "Y1", "X2", "Y2", "中心坐标X", "中心坐标Y", "刀具旋转角度"]
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
        """启动 YOLO 检测"""
        if self.yolo_thread is None or not self.yolo_thread.isRunning():
            self.yolo_thread = YoloThread(self.camera_thread)
            # 修改信号与槽的连接，让 update_detections 能接收 centroids
            self.camera_thread.set_emit_frame(False)  # 禁用 CameraThread 的画面更新
            self.yolo_thread.detection_signal.connect(self.update_detections)
            self.yolo_thread.frame_signal.connect(self.update_camera_frame)
            self.yolo_thread.start()

        global plc_mw3_flag
        print("start_detection plc_mw3_flag = 1")

        # 通知后台线程开始交替发1、2
        plc_mw3_flag = 1

    def stop_detection(self):
        """停止 YOLO 检测"""
        if self.yolo_thread is not None and self.yolo_thread.isRunning():
            self.yolo_thread.stop()
            self.yolo_thread = None
            self.camera_thread.set_emit_frame(True)  # 重新启用 CameraThread 的画面更新

        global plc_mw3_flag
        plc_mw3_flag = 0  # 发送0

    def update_camera_frame(self, frame):
        """更新摄像头画面"""
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()

        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(
            self.video_label.width(),
            self.video_label.height(),
            Qt.KeepAspectRatio,  # 保持宽高比
            Qt.SmoothTransformation  # 平滑变换
        )
        self.video_label.setPixmap(QPixmap.fromImage(q_image))

    def update_detections(self, detections, centroids,current_time):
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

            # self.info_labels["刀具旋转角度"].setText(f"{centroids[2]}°")
            # print("target_angle:", centroids[2])

            if centroids:
                centroid_text = "\\n".join([f"形心: ({cx}, {cy})" for cx, cy, angle in centroids])
                self.result_text.setPlainText(centroid_text)  # 更新 UI 显示
            else:
                self.result_text.clear()
        else:
            self.priority_info.clear()
            for key in self.info_labels:
                self.info_labels[key].setText("0")
        self._send_data_frame(centroids,current_time)  # 发送数据帧

        self._send_heartbeat()  # 发送心跳帧
        # self._send_to_robot(robot_data)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyWindow()

    window.show()

    threading.Thread(target=mw3_loop_thread, daemon=True).start()
    sys.exit(app.exec_())
