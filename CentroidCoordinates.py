import time
import cv2
import numpy as np

# 视频输入路径和输出路径
video_input_path = r"E:\TUDOU\fanhuizhi\td\c\1.mp4"  # 输入视频路径
video_output_path = r"E:\TUDOU\fanhuizhi\td\c\output_video2.mp4"  # 输出视频路径

# 打开视频文件
cap = cv2.VideoCapture(video_input_path)

# 获取视频的帧率（FPS）和尺寸
fps = int(cap.get(cv2.CAP_PROP_FPS))  # 获取帧率
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取帧宽
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取帧高

# 定义输出视频编码和文件
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4v 编码

# 初始化 VideoWriter
out = cv2.VideoWriter(video_output_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # 转换为 HSV 色彩空间
    image_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 提取 S（饱和度）通道
    saturation = image_hsv[:, :, 1]

    # 设置一个饱和度阈值，假设饱和度低于该值的是黑色区域（马铃薯区域）
    saturation_threshold = 60

    # 创建掩膜，只保留饱和度低于阈值的区域（黑色部分）
    potato_mask = saturation <= saturation_threshold

    # 将掩膜转为 uint8 类型，供 OpenCV 使用
    potato_mask = potato_mask.astype(np.uint8) * 255

    # 使用 OpenCV 查找连通区域（所有区域）
    contours, _ = cv2.findContours(potato_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 在原图上绘制质心
    image_with_centroids = frame.copy()

    # 只保留面积大于 50×50 的连通区域
    min_area = 50 * 50  # 50×50 像素的最小面积阈值

    # 计算形心识别部分的消耗时间
    start_time = time.time()

    # 只保留面积大于阈值的连通区域
    for contour in contours:
        area = cv2.contourArea(contour)  # 计算当前轮廓的面积
        if area > min_area:  # 如果当前轮廓的面积大于最小阈值
            # 计算轮廓的矩
            moments = cv2.moments(contour)

            if moments["m00"] != 0:  # 防止除零错误
                # 计算质心坐标
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])

                print(f"Centroid coordinates: ({cx}, {cy})")

                # 在原图上绘制质心（红色圆点）
                cv2.circle(image_with_centroids, (cx, cy), 10, (0, 0, 255), -1)  # 红色圆点

                # 在原图上绘制质心坐标（红色字体）
                cv2.putText(image_with_centroids, f"({cx}, {cy})",
                            (cx + 10, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # 记录结束时间
    end_time = time.time()

    # 打印程序运行时间（仅计算形心识别部分）
    print(f"形心识别部分的运行时间: {end_time - start_time:.4f} 秒")

    # 将处理后的帧写入输出视频文件
    out.write(image_with_centroids)

# 释放视频资源
cap.release()
out.release()

# 关闭所有 OpenCV 窗口
cv2.destroyAllWindows()
