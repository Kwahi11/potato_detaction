import cv2
import json
import numpy as np
from ultralytics import YOLO
import math
import time
from unet import Unet  # 确保正确导入Unet类
from PIL import Image

# 初始化Unet模型
unet_detector = Unet(
    model_path=r'F:\YOLOV8\PotatoDetection-main\ep010-loss0.015-val_loss0.013.pth',
    num_classes=2,
    cuda=True  # 使用GPU，确保GPU可用
)
m1 = (58-198) / (373-307)
theta1 = math.degrees(math.atan(m1))
def detect_objects(frame, model):
    """对输入画面进行目标检测，并返回检测后的画面和结果"""
    detections = []
    results = model(frame)  # 使用 YOLO 模型进行检测
    centroids_2 = []
    # 初始化一个空列表来存储芽眼的中心坐标


    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0]  # 获取目标框坐标
            conf = box.conf[0].item()  # 置信度
            cls = int(box.cls[0].item())  # 目标类别
            detections.append({"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2), "conf": conf, "cls": cls})

            # 在画面上绘制目标框
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{cls}: {conf:.2f}", (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 计算形心
    centroids = compute_centroids(frame)
    # centroids_2 = [[187, 99], [492, 401]]
    # 检查形心是否在马铃薯（类别为 0）的目标框内
    for centroid in centroids:
        cx, cy = centroid
        for detection in detections:
            if detection["cls"] == 0:  # 只考虑类别为 0（马铃薯）的检测框
                x1, y1, x2, y2 = detection["x1"], detection["y1"], detection["x2"], detection["y2"]
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    centroids_2.append(centroid)
                    break

    # 在画面上绘制形心
    for cx, cy in centroids_2:
        cv2.circle(frame, (cx, cy), 10, (0, 0, 255), -1)  # 画红色圆点
        cv2.putText(frame, f"({cx}, {cy})", (cx + 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return frame, detections, centroids_2



def compute_centroids(frame):
    """计算马铃薯区域的形心坐标"""
    """计算马铃薯区域的形心坐标"""
    # 将OpenCV的BGR图像转换为PIL的RGB图像
    centroids=[]
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # 使用Unet模型进行分割预测
    seg_result = unet_detector.detect_image(pil_image)

    # 将分割结果转换回OpenCV的BGR格式
    seg_frame = cv2.cvtColor(np.array(seg_result), cv2.COLOR_RGB2BGR)

    # 开始计算红色区域的形心
    # 1. 将分割结果转换为HSV格式（便于提取红色）
    hsv = cv2.cvtColor(seg_frame, cv2.COLOR_BGR2HSV)

    # 2. 定义红色的HSV范围（可能需要根据实际分割结果调整阈值）
    lower_red1 = np.array([0, 100, 100])  # 红色的第一部分（0-10度）
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])  # 红色的第二部分（170-180度）
    upper_red2 = np.array([180, 255, 255])

    # 3. 创建红色掩码
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # 4. 查找红色区域的轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 5. 遍历轮廓计算形心并绘制蓝色点
    for cnt in contours:
        # 过滤掉过小的轮廓（可选）
        if cv2.contourArea(cnt) < 2500:  # 根据需要调整最小面积阈值
            continue

        # 计算轮廓的矩
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            # 计算质心坐标
            # cx = int(M['m10'] / M['m00'])
            # cy = int(M['m01'] / M['m01'])  # 注意：这里应该是M['m01']/M['m00']
            # 修正错误：原代码中的分母应为 M['m00']
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            if 200 <= cy <= 280:
                centroids.append((cx, cy))


    return centroids


