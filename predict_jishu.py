# from ultralytics import YOLO
# import cv2
#
# # 加载训练好的模型
# model = YOLO(r'F:\pycharm_xiangmu\Bifpn_potato\Bifpn_potato\ultralytics-main\demo1\runs\detect\new_potato_300_1280\weights\best.pt')  # 替换为你的 .pt 文件路径
#
# # 进行预测
# results = model.predict(r'F:\pycharm_xiangmu\Bifpn_potato\Bifpn_potato\ultralytics-main\demo1\32c56a70d4f9aaba163586944e4015bf.mp4', save=True, stream=True)
#
# # 遍历每一帧的预测结果
# for result in results:
#     # 获取当前帧图像
#     frame = result.orig_img
#
#     # 初始化马铃薯芽眼计数
#     sprout_count = 0
#
#     # 遍历每个检测到的目标
#     for box in result.boxes:
#         # 获取目标类别
#         cls = int(box.cls)
#         if cls == 1:  # 类别为 1 表示马铃薯芽眼
#             # 增加计数
#             sprout_count += 1
#
#             # 获取目标框的坐标
#             x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
#
#             # 在图像上绘制目标框
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#
#     # 在图像上显示马铃薯芽眼计数
#     cv2.putText(frame, f'Sprout Count: {sprout_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#
#     # 显示当前帧
#     cv2.imshow('YOLOv8 Detection', frame)
#
#     # 按 'q' 键退出循环
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # 释放窗口
# cv2.destroyAllWindows()




from ultralytics import YOLO
import cv2
import numpy as np

# 加载训练好的模型
model = YOLO(r'F:\pycharm_xiangmu\Bifpn_potato\Bifpn_potato\ultralytics-main\demo1\runs\detect\new_potato_300_1280\weights\best.pt')

# 视频路径
video_path = r'F:\pycharm_xiangmu\Bifpn_potato\Bifpn_potato\ultralytics-main\demo1\32c56a70d4f9aaba163586944e4015bf.mp4'
cap = cv2.VideoCapture(video_path)

# 获取视频的帧率、宽度和高度
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 创建视频写入对象
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 进行预测
    results = model(frame)

    for result in results:
        # 获取检测框的坐标和类别
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()

        # 分离马铃薯和芽眼的检测框
        potato_boxes = boxes[classes == 0]
        sprout_boxes = boxes[classes == 1]

        # 绘制芽眼检测框
        for sprout_box in sprout_boxes:
            sx1, sy1, sx2, sy2 = sprout_box.astype(int)
            cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), (255, 0, 0), 2)

        for i, potato_box in enumerate(potato_boxes):
            x1, y1, x2, y2 = potato_box.astype(int)
            sprout_count = 0

            # 检查每个芽眼检测框是否在马铃薯检测框内
            for sprout_box in sprout_boxes:
                sx1, sy1, sx2, sy2 = sprout_box.astype(int)
                center_x = (sx1 + sx2) // 2
                center_y = (sy1 + sy2) // 2
                if x1 <= center_x <= x2 and y1 <= center_y <= y2:
                    sprout_count += 1

            # 在马铃薯检测框上显示芽眼数量
            cv2.putText(frame, f'Sprouts: {sprout_count}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 显示结果
    cv2.imshow('YOLOv8 Inference', frame)

    # 保存结果
    out.write(frame)

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()