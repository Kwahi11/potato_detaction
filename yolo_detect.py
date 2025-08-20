import sys
import cv2
import json
import numpy as np
from ultralytics import YOLO

# 【更换模型路径】如果你有自己的模型，请更改这里
MODEL_PATH = 'models/best.pt'

# 加载 YOLOv8 模型
model = YOLO(MODEL_PATH)

# 打开摄像头
cap = cv2.VideoCapture(2)

if not cap.isOpened():
    print("Error: 无法打开摄像头")
    sys.exit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: 读取摄像头画面失败")
        break

    # 运行 YOLO 目标检测
    results = model(frame)

    detections = []
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

    # 显示画面
    cv2.imshow("YOLO Detection", frame)

    # 将检测结果转换为 JSON 并输出
    print(json.dumps({"detections": detections}), flush=True)

    # 按 `q` 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头资源
cap.release()
cv2.destroyAllWindows()
