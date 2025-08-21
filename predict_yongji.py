from ultralytics import YOLO
import cv2
import numpy as np
import os
import time

# 加载模型
model = YOLO(
    'models/best.pt')

# 打开视频文件
video_path = r'C:\Users\谢林江\Documents\WeChat Files\wxid_8b1c2dfjkcj422\FileStorage\Video\2025-03\32c56a70d4f9aaba163586944e4015bf.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("无法打开视频文件")
    exit()

# 获取视频信息
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 创建输出目录
output_dir = "output_videos"
os.makedirs(output_dir, exist_ok=True)

# 创建带时间戳的输出文件名
timestamp = time.strftime("%Y%m%d_%H%M%S")
output_path = os.path.join(output_dir, f"y_range_count_{timestamp}.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# 初始化计数器
frame_count = 0
total_class0_count = 0

# 定义检测区域 (y范围)
detection_y_min = 30
detection_y_max = 550

# 处理开始时间
start_time = time.time()

print(f"开始处理视频: {video_path}")
print(f"视频尺寸: {width}x{height}, FPS: {fps:.2f}, 总帧数: {total_frames}")
print(f"检测区域: y={detection_y_min}-{detection_y_max}")
print(f"结果将保存至: {output_path}")
print("=" * 50)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # 每处理10帧显示一次进度
    if frame_count % 10 == 0:
        elapsed = time.time() - start_time
        fps_processed = frame_count / elapsed if elapsed > 0 else 0
        progress = (frame_count / total_frames) * 100
        print(
            f"处理中... 帧: {frame_count}/{total_frames} ({progress:.1f}%) | FPS: {fps_processed:.1f} | 当前计数: {total_class0_count}")

    # 使用YOLO进行预测
    results = model.predict(frame, conf=0.5, verbose=False)

    # 初始化当前帧计数
    class0_count = 0

    # 创建带范围线的帧（不显示检测框）
    annotated_frame = frame.copy()

    # 在画面中绘制检测范围线（水平线）
    cv2.line(annotated_frame, (0, detection_y_min), (width, detection_y_min), (0, 255, 0), 2)  # 上边界线
    cv2.line(annotated_frame, (0, detection_y_max), (width, detection_y_max), (0, 255, 0), 2)  # 下边界线

    # 添加范围文字说明（放在左侧）
    region_text = f"Detection Region: y={detection_y_min}-{detection_y_max}"
    cv2.putText(annotated_frame, region_text, (10, detection_y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    # 遍历检测结果（仅统计，不绘制检测框）
    for result in results:
        # 获取当前帧的所有检测框
        boxes = result.boxes

        # 统计类别0的数量（仅在指定区域内）
        for box in boxes:
            cls = int(box.cls[0])  # 获取类别ID
            if cls == 0:  # 如果类别是0
                # 获取边界框坐标
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # 计算中心点y坐标
                center_y = (y1 + y2) // 2

                # 检查是否在检测区域内
                if detection_y_min <= center_y <= detection_y_max:
                    class0_count += 1

    # 更新总计数
    total_class0_count += class0_count

    # 在画面正上方添加计数显示
    text = f"Class 0 Count: {class0_count} )"

    # 设置文本位置（画面顶部居中）
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
    text_x = (width - text_size[0]) // 2
    text_y = 50  # 顶部位置

    # 添加背景矩形（红色）
    cv2.rectangle(annotated_frame,
                  (text_x - 10, text_y - text_size[1] - 10),
                  (text_x + text_size[0] + 10, text_y + 10),
                  (0, 0, 255), -1)

    # 添加文字（白色）
    cv2.putText(annotated_frame, text, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)

    # 保存带计数的帧
    out.write(annotated_frame)

# 释放资源
cap.release()
out.release()

# 计算处理时间
end_time = time.time()
total_time = end_time - start_time
avg_fps = frame_count / total_time if total_time > 0 else 0

print("=" * 50)
print(f"处理完成!")
print(f"总帧数: {frame_count}")
print(f"处理时间: {total_time:.2f} 秒")
print(f"平均 FPS: {avg_fps:.2f}")
print(f"在 y{detection_y_min}-{detection_y_max} 范围内共检测到 {total_class0_count} 个类别0目标")
print(f"结果视频已保存至: {output_path}")