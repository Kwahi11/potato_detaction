import sys, os
sys.path.append(r"D:\workSpace\unet_pytorch_main")
sys.path.append(r"F:\YOLOV8\PotatoDetection-main\unet_pytorch_main")
import cv2
import json
import numpy as np
from ultralytics import YOLO
import math
import time
from unet import Unet  # 确保正确导入Unet类
from PIL import Image

# 面积阈值（像素），≥该值输出 15，否则输出 5；按分辨率与镜头调整
MIN_CONTOUR_AREA = 20000       # 去噪下限（你原先的过滤值）
POTATO_AREA_THRESHOLD = 36000 # 大小判定阈值（示例值，需按实际调参）
DEBOUNCE_N = 3
_size_code_stable = None
_switch_counter = 0

# 初始化Unet模型
unet_detector = Unet(
    model_path='ep010-loss0.025-val_loss0.019.pth',
    num_classes=2,
    cuda=True  # 使用GPU，确保GPU可用
)
# m1 = (58-198) / (373-307)
m1=(695-102)/(686-668)
# m1=(836-464)/(649-746)
theta1 = math.degrees(math.atan(m1))
def detect_objects(frame, model):
    """对输入画面进行目标检测，并返回检测后的画面和结果"""
    #
    # detections = []
    # results = model(frame)  # 使用 YOLO 模型进行检测
    # centroids_2 = []
    # # 初始化一个空列表来存储芽眼的中心坐标
    # # targets_angles=[]
    # current_time=int(time.time()*1000)
    #
    # for result in results:
    #     for box in result.boxes:
    #         x1, y1, x2, y2 = box.xyxy[0]  # 获取目标框坐标
    #         conf = box.conf[0].item()  # 置信度
    #         cls = int(box.cls[0].item())  # 目标类别
    #         detections.append({"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2), "conf": conf, "cls": cls})
    #
    #         # 在画面上绘制目标框
    #         cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    #         cv2.putText(frame, f"{cls}: {conf:.2f}", (int(x1), int(y1) - 5),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    #
    # # 计算形心
    # centroids = compute_centroids(frame)
    # # centroids_2 = [[187, 99], [492, 401]]
    # # 检查形心是否在马铃薯（类别为 0）的目标框内
    # for centroid in centroids:
    #     cx, cy,angle = centroid
    #     for detection in detections:
    #         if detection["cls"] == 0:  # 只考虑类别为 0（马铃薯）的检测框
    #             x1, y1, x2, y2 = detection["x1"], detection["y1"], detection["x2"], detection["y2"]
    #             if x1 <= cx <= x2 and y1 <= cy <= y2:
    #                 centroids_2.append(centroid)
    #                 break
    #
    # # 在画面上绘制形心
    # for cx, cy,angle in centroids:
    #     cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)  # 画红色圆点
    #     cv2.putText(frame, f"({cx}, {cy})", (cx + 10, cy - 10),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    #
    # return frame, detections, centroids_2,current_time

    detections = []
    # 设置阈值参数
    confidence_threshold = 0.5
    results = model(frame, conf=confidence_threshold)  # 使用YOLO模型进行检测，设置置信度阈值
    centroids_2 = []
    # 初始化一个空列表来存储芽眼的中心坐标
    current_time = int(time.time() * 1000)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0]  # 获取目标框坐标
            conf = box.conf[0].item()  # 置信度
            cls = int(box.cls[0].item())  # 目标类别

            # 记录所有检测结果，包括低于阈值的（用于可视化）
            detections.append({"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2), "conf": conf, "cls": cls})

            # 在画面上绘制目标框，只绘制高于阈值的土豆
            if cls == 0 and conf >= confidence_threshold:  # 类别0且置信度大于等于阈值
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"{cls}: {conf:.2f}", (int(x1), int(y1) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 计算形心
    centroids = compute_centroids(frame)

    # 检查形心是否在马铃薯（类别为0）的目标框内，只考虑置信度高于阈值的土豆
    valid_potatoes = [d for d in detections if d["cls"] == 0 and d["conf"] >= confidence_threshold]
    for centroid in centroids:
        cx, cy, angle,sizeCode = centroid
        for detection in valid_potatoes:
            x1, y1, x2, y2 = detection["x1"], detection["y1"], detection["x2"], detection["y2"]
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                centroids_2.append(centroid)
                break

    # 在画面上绘制形心
    for cx, cy, angle,sizeCode in centroids:
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)  # 画红色圆点
        cv2.putText(frame, f"({cx}, {cy})", (cx + 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return frame, detections, centroids_2, current_time



def compute_centroids(frame):

        # 将OpenCV的BGR图像转换为PIL的RGB图像
        centroids = []
        x_old=0
        x_new=0

        # angles = []
        min_rotation_angle=0
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # 使用Unet模型进行分割预测
        seg_result = unet_detector.detect_image(pil_image)
        if seg_result:
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
                area = cv2.contourArea(cnt)
                if cv2.contourArea(cnt) < 20000:  # 根据需要调整最小面积阈值
                    continue
                # if len(cnt) < 5:
                #
                #     continue

                # 计算轮廓的矩
                M = cv2.moments(cnt)

                def find_contour_edge(start_x, start_y, direction_angle):
                    theta = math.radians(direction_angle)
                    for d in range(1, 300):
                        x = int(start_x + d * math.cos(theta))
                        y = int(start_y + d * math.sin(theta))
                        if y < 0 or y >= mask.shape[0] or x < 0 or x >= mask.shape[1]:
                            break
                        if mask[y, x] == 0:
                            return (x, y)
                    for d in range(1, 300):
                        x = int(start_x - d * math.cos(theta))
                        y = int(start_y - d * math.sin(theta))
                        if mask[y, x] == 0:
                            return (x, y)
                    return (start_x, start_y)
                if M["m00"] != 0:
                    # 计算质心坐标
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    x_new=cx
                    if 350 <= cy <= 650:
                    # if True:




                            # # # 计算轮廓的长轴和方向
                            ellipse = cv2.fitEllipse(cnt)
                            (center, axes, angle) = ellipse
                            major_axis_length = max(axes)  # 最长轴的长度
                            half_major_axis_length = major_axis_length / 2
                            rad_angle = math.radians(angle + 90)  # 调整角度以匹配长轴方向

                            # 计算长轴两端点
                            x_end_1 = int(center[0] + half_major_axis_length * math.cos(rad_angle))
                            y_end_1 = int(center[1] + half_major_axis_length * math.sin(rad_angle))
                            x_end_2 = int(center[0] - half_major_axis_length * math.cos(rad_angle))
                            y_end_2 = int(center[1] - half_major_axis_length * math.sin(rad_angle))

                            if x_end_2 != x_end_1:
                                m2 = (y_end_2 - y_end_1) / (x_end_2 - x_end_1)
                                theta2 = math.degrees(math.atan(m2))
                                # 计算两个角度之间的差值
                                theta2 = (theta2 + 180) % 360 - 180
                                angle_diff = theta2 - theta1
                            else:
                                # 使用默认角度值
                                angle_diff = -64.7

                            min_rotation_angle = angle_diff
                            if min_rotation_angle < -90:
                                min_rotation_angle = 180 + min_rotation_angle
                            if min_rotation_angle>90:
                                min_rotation_angle=min_rotation_angle-180
                            # min_rotation_angle=-min_rotation_angle

                            # 确保每个形心都有对应的角度
                            # angles.append(min_rotation_angle)
                            # if min_rotation_angle and abs(x_new-x_old)>20:

                            ###################################################################################
                    #         ellipse = cv2.fitEllipse(cnt)
                    #         (_, axes, angle) = ellipse
                    #         major_axis_angle = angle + 90  # 实际长轴角度
                    # # 获取长轴端点
                    #         ptA = find_contour_edge(cx, cy, major_axis_angle)
                    #         ptB = find_contour_edge(cx, cy, major_axis_angle + 180)
                    #
                    # # 计算形心到端点的距离
                    #         distA = math.hypot(ptA[0] - cx, ptA[1] - cy)
                    #         distB = math.hypot(ptB[0] - cx, ptB[1] - cy)
                    #         print(distA)
                    #         print(distB)
                    #         # 确定基准方向（指向更远端点）
                    #         if distA > distB:
                    #
                    #             base_angle = math.degrees(math.atan2(ptB[1] - cy, ptB[0] - cx))
                    #             # cv2.line(frame, (cx, cy), ptB, (255, 0, 0), 3)
                    #         else:
                    #             base_angle = math.degrees(math.atan2(ptA[1] - cy, ptA[0] - cx))
                    #             # cv2.line(frame, (cx, cy), ptA, (255, 0, 0), 3)  # 蓝色基准线
                    #
                    #         # 计算最终角度
                    #         theta2 = base_angle
                    #         angle_diff = theta2 - theta1
                    #         min_rotation_angle =(angle_diff + 180) % 360 - 180
                            # final_angle = normalize_angle(rotation_angle)




                            if min_rotation_angle:
                                # 面积大小判定：≥阈值输出 15，否则 5
                                size_code = get_stable_size_code(area)
                                print(f"面积={int(area)} -> 输出={size_code}") 
                                centroids.append((cx, cy, min_rotation_angle, size_code))
                                x_old = x_new
                              
                            # else:
                            #     centroids.append(cx,cy,0)
                            # for i in stack:
                            #     last_angle = i[2]
                            #     if min_rotation_angle != last_angle:
                            #         stack.append((cx, cy, min_rotation_angle))
                            #         centroids.append((cx, cy, min_rotation_angle))
                            #     else:
                            #         continue




        return centroids





def get_stable_size_code(area: float) -> int:
    """面积→size_code(15/5) 的最简去抖"""
    global _size_code_stable, _switch_counter
    try:
        threshold = POTATO_AREA_THRESHOLD
    except NameError:
        threshold = 36000  # 若未定义，给个默认

    inst = 15 if area >= threshold else 5

    if _size_code_stable is None or inst == _size_code_stable:
        _size_code_stable = inst
        _switch_counter = 0
        return _size_code_stable

    _switch_counter += 1
    if _switch_counter >= DEBOUNCE_N:
        _size_code_stable = inst
        _switch_counter = 0
    return _size_code_stable
