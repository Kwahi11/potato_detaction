import cv2
import numpy as np
from ultralytics import YOLO
from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat

# 加载 YOLOv8 模型
model = YOLO(r'C:\Users\10402\Desktop\potato_yolo_renew\potato_yolo_renew\runs\detect\yolov8 20251-31 200\weights\best.pt')

# 初始化奥比中光的管道和配置
pipeline = Pipeline()
config = Config()

# 获取设备支持的流配置
color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)

# 选择默认的彩色流和深度流配置
color_profile = color_profiles.get_default_video_stream_profile()
depth_profile = depth_profiles.get_default_video_stream_profile()

# 启用流
config.enable_stream(color_profile)  # 启用彩色流
config.enable_stream(depth_profile)  # 启用深度流

# 启动管道
pipeline.start(config)

try:
    while True:
        # 等待获取一帧数据，超时时间为 1000 毫秒
        frames = pipeline.wait_for_frames(1000)
        if frames is None:
            print("No frames received within the timeout period.")
            continue

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if color_frame is None or depth_frame is None:
            print("One of the frames is missing.")
            continue

        # 获取彩色帧数据
        color_data = color_frame.get_data()
        color_image = cv2.imdecode(np.frombuffer(color_data, dtype=np.uint8), cv2.IMREAD_COLOR)

        # 获取深度帧数据
        depth_data = depth_frame.get_data()
        depth_image = np.frombuffer(depth_data, dtype=np.uint16).reshape((depth_frame.get_height(), depth_frame.get_width()))

        # 使用 YOLOv8 模型进行目标检测
        results = model(color_image)

        # 在彩色图像和深度图像上绘制边界框
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].astype(int)

                # 在彩色图像上绘制边界框
                cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # 缩放边界框坐标到深度图像的分辨率
                scale_x = depth_frame.get_width() / color_frame.get_width()
                scale_y = depth_frame.get_height() / color_frame.get_height()
                center_x = int((x1 + x2) / 2 * scale_x)
                center_y = int((y1 + y2) / 2 * scale_y)

                # 确保坐标在深度图像范围内
                if 0 <= center_x < depth_frame.get_width() and 0 <= center_y < depth_frame.get_height():
                    depth_value = depth_image[center_y, center_x]
                    print(f"Object detected at (x, y, depth): ({center_x}, {center_y}, {depth_value})")
                else:
                    print(f"Object detected at (x, y) but out of depth image bounds: ({center_x}, {center_y})")

        # 显示图像
        cv2.imshow('Color Image', color_image)
        cv2.imshow('Depth Image with Marked Objects', depth_image.astype(np.uint8) * (255 / np.max(depth_image)))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()