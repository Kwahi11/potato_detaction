import cv2
import numpy as np

# 定义标定板上圆点的行列数
pattern_size = (7, 7)  # 根据实际情况修改

# 打开摄像头
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法读取视频帧")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 检测圆点阵列
    found, corners = cv2.findCirclesGrid(gray, pattern_size, flags=cv2.CALIB_CB_SYMMETRIC_GRID)
    print(f"是否检测到圆点阵列: {found}")

    if found:
        # 亚像素级角点细化，提高圆心坐标精度
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                   (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))
        # 在图像上绘制检测到的圆心
        cv2.drawChessboardCorners(frame, pattern_size, corners, found)
        print("圆心坐标：", corners)

    cv2.imshow('Camera Feed', frame)

    # 按下 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()