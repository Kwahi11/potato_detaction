import cv2

# 初始化相机（0为设备索引，多个摄像头时需调整）
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("无法打开相机")
    exit()

# 读取并显示视频流
while True:
    ret, frame = cap.read()
    if not ret:
        print("无法获取帧")
        break

    cv2.imshow('华瑞相机', frame)

    # 按q键退出
    if cv2.waitKey(1) == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()