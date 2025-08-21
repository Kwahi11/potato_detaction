from ultralytics import YOLO
import cv2
import numpy as np

def detect_objects(frame, model):

    detections = []
    potato_sprout_counts = []  # 用于存储每个马铃薯的芽眼计数信息
    processed_frame = frame.copy()  # 复制原始画面以避免修改

    # 使用 YOLO 模型进行检测
    results = model(processed_frame)

    # 分离马铃薯和芽眼的检测框
    potato_boxes = []
    sprout_boxes = []
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()

        for box, cls, conf in zip(boxes, classes, confidences):
            x1, y1, x2, y2 = box.astype(int)
            detections.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "cls": int(cls), "conf": conf})

            # 绘制检测框
            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(processed_frame, f"{int(cls)}: {conf:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 分离检测框
            if int(cls) == 0:  # 马铃薯
                potato_boxes.append(box)
            elif int(cls) == 1:  # 芽眼
                sprout_boxes.append(box)

    # 计算每个马铃薯框内的芽眼数量
    for potato_box in potato_boxes:
        x1, y1, x2, y2 = potato_box.astype(int)
        sprout_count = 0

        # 检查芽眼是否在马铃薯框内
        for sprout_box in sprout_boxes:
            sx1, sy1, sx2, sy2 = sprout_box.astype(int)
            center_x = (sx1 + sx2) // 2
            center_y = (sy1 + sy2) // 2
            if x1 <= center_x <= x2 and y1 <= center_y <= y2:
                sprout_count += 1

        # 在马铃薯框上显示芽眼数量
        cv2.putText(processed_frame, f"Sprouts: {sprout_count}", (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        potato_sprout_counts.append((x1, y1, x2, y2, sprout_count))

    # 计算形心
    centroids = []
    for box in sprout_boxes:
        x1, y1, x2, y2 = box.astype(int)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        centroids.append((cx, cy))

    # 在画面上绘制形心
    for cx, cy in centroids:
        cv2.circle(processed_frame, (cx, cy), 10, (0, 0, 255), -1)  # 画红色圆点
        cv2.putText(processed_frame, f"({cx}, {cy})", (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        #return processed_frame, detections, potato_sprout_counts, centroids#芽眼数量不需要
    return processed_frame, detections,centroids#芽眼数量不需要