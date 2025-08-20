import cv2
import json
import numpy as np
from ultralytics import YOLO
import math
import time
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
    # 转换为 HSV 色彩空间
    image_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    saturation = image_hsv[:, :, 1]

    # 设定饱和度阈值，提取马铃薯区域
    saturation_threshold = 60
    potato_mask = (saturation <= saturation_threshold).astype(np.uint8) * 255

    # 查找轮廓
    contours, _ = cv2.findContours(potato_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centroids = []
    min_area = 50 * 50  # 设定最小面积阈值

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            moments = cv2.moments(contour)
            if moments["m00"] != 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
                if 220<=cy<=260:
                 centroids.append((cx, cy))

    return centroids


