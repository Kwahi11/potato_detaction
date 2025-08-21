import cv2
from PIL import Image
import numpy as np
import math

# 假设Unet类已经正确导入
from unet import Unet  # 确保正确导入Unet类
import math
# 初始化Unet模型
unet_detector = Unet(
    model_path=r'F:\YOLOV8\PotatoDetection-main\ep010-loss0.015-val_loss0.013.pth',
    num_classes=2,
    cuda=True  # 使用GPU，确保GPU可用
)


m1 = (58-198) / (373-307)
theta1 = math.degrees(math.atan(m1))

def compute_axes_and_draw(frame, mask):
    """计算最长轴并绘制相关信息"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) < 2500:  # 根据需要调整最小面积阈值
            continue

        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            # 绘制蓝色圆点表示形心
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

            # 计算轮廓的长轴和方向
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

            # 绘制绿色长轴线
            cv2.line(frame, (x_end_1, y_end_1), (x_end_2, y_end_2), (0, 255, 0), 2)
            cv2.putText(frame, f"({x_end_1}, {y_end_1})", (x_end_1 + 10, y_end_1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(frame, f"({x_end_2}, {y_end_2})", (x_end_2 + 10, y_end_2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            if x_end_2!=x_end_1:
               m2 = (y_end_2-y_end_1)/(x_end_2-x_end_1)
               theta2 = math.degrees(math.atan(m2))
            # 计算两个角度之间的差值
               theta2 = (theta2 + 180) % 360 - 180
               angle_diff = theta2 - theta1
            else:
                angle_diff=64.7

            # # 如果差值为负，则表示需要逆时针旋转，转换为顺时针旋转
            # if angle_diff < -90:
            #     angle_diff = 180+ angle_diff;

            # 最小旋转角度
            # min_rotation_angle = min(angle_diff, 360- angle_diff,180+angle_diff)
            min_rotation_angle=-angle_diff
            if min_rotation_angle<-90:
                min_rotation_angle=180+min_rotation_angle
            # if angle_diff<-90:
            #     min_rotation_angle=180+angle_diff
            # if angle_diff>90:
            #     min_rotation_angle=180-angle_diff
            # min_rotation_angle=-min_rotation_angle
            cv2.putText(frame, f"({min_rotation_angle})", (x_end_2 + 20, y_end_2 +20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 3)
    red_region = cv2.bitwise_and(frame, frame, mask=mask)
    combined_image = cv2.addWeighted(frame, 1, red_region, 0.5, 0)

    return combined_image

cap = cv2.VideoCapture(1)  # 使用摄像头输入，或者用视频文件路径替换1

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 将OpenCV的BGR图像转换为PIL的RGB图像
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # 使用Unet模型进行分割预测
    seg_result = unet_detector.detect_image(pil_image)

    # 将分割结果转换回OpenCV的BGR格式
    seg_frame = cv2.cvtColor(np.array(seg_result), cv2.COLOR_RGB2BGR)

    # 开始计算红色区域的形心
    hsv = cv2.cvtColor(seg_frame, cv2.COLOR_BGR2HSV)

    # 定义红色的HSV范围（可能需要根据实际分割结果调整阈值）
    lower_red1 = np.array([0, 100, 100])  # 红色的第一部分（0-10度）
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])  # 红色的第二部分（170-180度）
    upper_red2 = np.array([180, 255, 255])

    # 创建红色掩码
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # 调用函数计算形心和长轴，并绘制相关信息
    result_frame=compute_axes_and_draw(frame, mask)

    # 显示结果
    cv2.imshow("Unet Segmentation with Centroids and Major Axes", result_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()