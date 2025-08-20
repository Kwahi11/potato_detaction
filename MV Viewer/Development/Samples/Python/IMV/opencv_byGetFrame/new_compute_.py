# -- coding: utf-8 --

import sys
from ctypes import *
import datetime
import numpy
import cv2
import numpy as np
import gc
from PIL import Image
from unet import Unet
import math
from pymodbus.client import ModbusTcpClient


sys.path.append(r"F:\YOLOV8\PotatoDetection-main\MV Viewer\Development\Samples\Python\IMV\MVSDK")
from IMVApi import *

# 初始化Unet模型
unet_detector = Unet(
    model_path=r'F:\YOLOV8\PotatoDetection-main\ep010-loss0.015-val_loss0.013.pth',
    num_classes=2,
    cuda=True
)

# 计算参考角度（根据实际机械臂基准方向设置）
m1 = (836 - 464) / (649 - 746)
theta1 = math.degrees(math.atan(m1))

# 新增：堆积状态判断阈值（像素）
LACK_THRESHOLD = 10000  # 缺料阈值
LIGHT_STACK_THRESHOLD = 150000  # 轻度堆积阈值
MODERATE_STACK_THRESHOLD = 200000  # 中度堆积阈值
SEVERE_STACK_THRESHOLD = 250000  # 重度堆积阈值


def displayDeviceInfo(deviceInfoList):
    print("Idx  Type   Vendor              Model           S/N                 DeviceUserID    IP Address")
    print("------------------------------------------------------------------------------------------------")
    for i in range(0, deviceInfoList.nDevNum):
        pDeviceInfo = deviceInfoList.pDevInfo[i]
        strType = ""
        strVendorName = pDeviceInfo.vendorName.decode("utf-8", errors="ignore")  # 厂商名称
        strModeName = pDeviceInfo.modelName.decode("utf-8", errors="ignore")  # 型号
        strSerialNumber = pDeviceInfo.serialNumber.decode("utf-8", errors="ignore")  # 序列号
        strCameraname = pDeviceInfo.cameraName.decode("utf-8", errors="ignore")  # 相机名称
        strIpAdress = pDeviceInfo.DeviceSpecificInfo.gigeDeviceInfo.ipAddress.decode("utf-8", errors="ignore")  # IP地址
        if pDeviceInfo.nCameraType == typeGigeCamera:
            strType = "Gige"
        elif pDeviceInfo.nCameraType == typeU3vCamera:
            strType = "U3V"
        print("[%d]  %s   %s    %s      %s     %s           %s" % (
            i + 1, strType, strVendorName, strModeName, strSerialNumber, strCameraname, strIpAdress))


def normalize_angle(angle):
    """将角度规范到[-90, 90]范围内"""
    angle %= 180  # 十字刀具有180度周期性
    if angle > 90:
        angle -= 180
    elif angle < -90:
        angle += 180
    return angle


def compute_rotation_angle(angle_diff):
    """计算最优旋转角度（考虑十字刀具特性）"""
    # 先规范到[-180, 180]
    angle_diff = (angle_diff + 180) % 360 - 180
    return angle_diff


def compute_axes_and_draw(frame, mask):
    """计算长轴并根据端点距离调整旋转方向，新增面积和堆积状态计算"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_angle = None
    total_area = 0  # 用于累计土豆总面积

    # 绘制原始边界
    cv2.drawContours(frame, contours, -1, (0, 0, 255), 2)

    for cnt in contours:
        # 计算轮廓面积（像素数）并累加
        contour_area = cv2.contourArea(cnt)
        if contour_area < 2500:  # 过滤小面积噪声
            continue
        total_area += contour_area  # 累计有效面积

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue

        # 计算形心
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # 通过椭圆拟合获取长轴方向
        ellipse = cv2.fitEllipse(cnt)
        (_, axes, angle) = ellipse
        major_axis_angle = angle + 90  # 实际长轴角度

        # 射线法寻找真实轮廓交点
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

        # 获取长轴端点
        ptA = find_contour_edge(cx, cy, major_axis_angle)
        ptB = find_contour_edge(cx, cy, major_axis_angle + 180)

        # 计算形心到端点的距离
        distA = math.hypot(ptA[0] - cx, ptA[1] - cy)
        distB = math.hypot(ptB[0] - cx, ptB[1] - cy)

        # 确定基准方向（指向更远端点）
        if distA > distB:
            base_angle = math.degrees(math.atan2(ptB[1] - cy, ptB[0] - cx))
            cv2.line(frame, (cx, cy), ptB, (255, 0, 0), 3)
        else:
            base_angle = math.degrees(math.atan2(ptA[1] - cy, ptA[0] - cx))
            cv2.line(frame, (cx, cy), ptA, (255, 0, 0), 3)  # 蓝色基准线

        # 计算最终角度
        theta2 = base_angle
        angle_diff = theta2 - theta1
        rotation_angle = compute_rotation_angle(angle_diff)
        final_angle = rotation_angle

        # 显示距离信息
        cv2.putText(frame, f"Dist: {max(distA, distB):.1f}", (cx - 50, cy - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)

        # 绘制特征
        cv2.line(frame, ptA, ptB, (0, 255, 0), 2)  # 绿色长轴
        cv2.circle(frame, (cx, cy), 5, (0, 165, 255), -1)  # 橙色形心

    # 叠加半透明掩码
    red_region = cv2.bitwise_and(frame, frame, mask=mask)
    result_frame = cv2.addWeighted(frame, 0.7, red_region, 0.3, 0)

    # 新增：根据面积判断堆积状态
    if total_area < LACK_THRESHOLD:
        stack_status = "lack of potato"
        status_color = (0, 255, 255)  # 黄色（缺料）
    elif total_area > SEVERE_STACK_THRESHOLD:
        stack_status = "third class warning"
        status_color = (0, 0, 255)  # 红色（重度堆积）
    elif total_area > MODERATE_STACK_THRESHOLD:
        stack_status = "second class warning"
        status_color = (0, 165, 255)  # 橙色（中度堆积）
    elif total_area > LIGHT_STACK_THRESHOLD:
        stack_status = "first class warning"
        status_color = (0, 255, 0)  # 绿色（轻度堆积）
    else:
        stack_status = "normal"
        status_color = (255, 0, 0)  # 蓝色（正常）

    # 在画面正上方绘制面积和堆积状态（文字带黑色描边，确保清晰）
    if total_area > 0:
        # 1. 面积文字（顶部第一行）
        area_text = f"Potato Area: {total_area:.1f} pixels"
        area_text_size = cv2.getTextSize(area_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        area_text_x = (result_frame.shape[1] - area_text_size[0]) // 2  # 水平居中
        area_text_y = 30  # 距离顶部30像素

        # 2. 堆积状态文字（顶部第二行，与第一行间距25像素）
        status_text = f"Status: {stack_status}"
        status_text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        status_text_x = (result_frame.shape[1] - status_text_size[0]) // 2  # 水平居中
        status_text_y = area_text_y + 25  # 位于面积文字下方

        # 绘制面积文字（黑色描边+白色文字）
        cv2.putText(result_frame, area_text, (area_text_x, area_text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)  # 黑色描边
        cv2.putText(result_frame, area_text, (area_text_x, area_text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)  # 白色文字

        # 绘制堆积状态文字（黑色描边+对应颜色文字）
        cv2.putText(result_frame, status_text, (status_text_x, status_text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)  # 黑色描边
        cv2.putText(result_frame, status_text, (status_text_x, status_text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)  # 状态对应颜色文字

    return result_frame, final_angle, total_area, stack_status  # 新增返回堆积状态


def retrun_frame():
    deviceList = IMV_DeviceList()
    interfaceType = IMV_EInterfaceType.interfaceTypeAll
    nWidth = c_uint()
    nHeight = c_uint()

    # 枚举设备
    nRet = MvCamera.IMV_EnumDevices(deviceList, interfaceType)
    if IMV_OK != nRet:
        print("Enumeration devices failed! ErrorCode", nRet)
        sys.exit()
    if deviceList.nDevNum == 0:
        print("find no device!")
        sys.exit()

    print("deviceList size is", deviceList.nDevNum)

    displayDeviceInfo(deviceList)

    nConnectionNum = 1

    cam = MvCamera()
    # 创建设备句柄
    nRet = cam.IMV_CreateHandle(IMV_ECreateHandleMode.modeByIndex, byref(c_void_p(int(nConnectionNum) - 1)))
    if IMV_OK != nRet:
        print("Create devHandle failed! ErrorCode", nRet)
        sys.exit()

    # 打开相机
    nRet = cam.IMV_Open()
    if IMV_OK != nRet:
        print("Open devHandle failed! ErrorCode", nRet)
        sys.exit()

    # 通用属性设置:设置触发模式为off
    nRet = IMV_OK
    nRet = cam.IMV_SetEnumFeatureSymbol("TriggerSource", "Software")
    if IMV_OK != nRet:
        print("Set triggerSource value failed! ErrorCode[%d]" % nRet)
        sys.exit()

    nRet = cam.IMV_SetEnumFeatureSymbol("TriggerSelector", "FrameStart")
    if IMV_OK != nRet:
        print("Set triggerSelector value failed! ErrorCode[%d]" % nRet)
        sys.exit()

    nRet = cam.IMV_SetEnumFeatureSymbol("TriggerMode", "Off")
    if IMV_OK != nRet:
        print("Set triggerMode value failed! ErrorCode[%d]" % nRet)
        sys.exit()

    # 开始拉流
    nRet = cam.IMV_StartGrabbing()
    if IMV_OK != nRet:
        print("Start grabbing failed! ErrorCode", nRet)
        sys.exit()

    isGrab = True
    while isGrab:
        # 主动取图
        frame = IMV_Frame()
        stPixelConvertParam = IMV_PixelConvertParam()

        nRet = cam.IMV_GetFrame(frame, 1000)

        if IMV_OK != nRet:
            print("getFrame fail! Timeout:[1000]ms")
            continue

        if None == byref(frame):
            print("pFrame is NULL!")
            continue

        if IMV_EPixelType.gvspPixelMono8 == frame.frameInfo.pixelFormat:
            nDstBufSize = frame.frameInfo.width * frame.frameInfo.height
        else:
            nDstBufSize = frame.frameInfo.width * frame.frameInfo.height * 3

        pDstBuf = (c_ubyte * nDstBufSize)()
        memset(byref(stPixelConvertParam), 0, sizeof(stPixelConvertParam))

        stPixelConvertParam.nWidth = frame.frameInfo.width
        stPixelConvertParam.nHeight = frame.frameInfo.height
        stPixelConvertParam.ePixelFormat = frame.frameInfo.pixelFormat
        stPixelConvertParam.pSrcData = frame.pData
        stPixelConvertParam.nSrcDataLen = frame.frameInfo.size
        stPixelConvertParam.nPaddingX = frame.frameInfo.paddingX
        stPixelConvertParam.nPaddingY = frame.frameInfo.paddingY
        stPixelConvertParam.eBayerDemosaic = IMV_EBayerDemosaic.demosaicNearestNeighbor
        stPixelConvertParam.eDstPixelFormat = frame.frameInfo.pixelFormat
        stPixelConvertParam.pDstBuf = pDstBuf
        stPixelConvertParam.nDstBufSize = nDstBufSize

        # 释放驱动图像缓存
        nRet = cam.IMV_ReleaseFrame(frame)
        if IMV_OK != nRet:
            print("Release frame failed! ErrorCode[%d]\n", nRet)
            sys.exit()

        # 如果图像格式是 Mono8 直接使用
        if stPixelConvertParam.ePixelFormat == IMV_EPixelType.gvspPixelMono8:
            imageBuff = stPixelConvertParam.pSrcData
            userBuff = c_buffer(b'\0', stPixelConvertParam.nDstBufSize)

            memmove(userBuff, imageBuff, stPixelConvertParam.nDstBufSize)
            grayByteArray = bytearray(userBuff)

            cvImage = numpy.array(grayByteArray).reshape(stPixelConvertParam.nHeight, stPixelConvertParam.nWidth)

        else:
            # 转码 => BGR24
            stPixelConvertParam.eDstPixelFormat = IMV_EPixelType.gvspPixelBGR8

            nRet = cam.IMV_PixelConvert(stPixelConvertParam)
            if IMV_OK != nRet:
                print("image convert to failed! ErrorCode[%d]" % nRet)
                del pDstBuf
                sys.exit()
            rgbBuff = c_buffer(b'\0', stPixelConvertParam.nDstBufSize)
            memmove(rgbBuff, stPixelConvertParam.pDstBuf, stPixelConvertParam.nDstBufSize)
            colorByteArray = bytearray(rgbBuff)
            cvImage = numpy.array(colorByteArray).reshape(stPixelConvertParam.nHeight, stPixelConvertParam.nWidth, 3)
            if None != pDstBuf:
                del pDstBuf

        # 处理键盘事件，按 'q' 键退出循环
        key = cv2.waitKey(1)
        if key == ord('q'):
            isGrab = False
            break

        yield 1, cvImage

    # 停止拉流
    nRet = cam.IMV_StopGrabbing()
    if IMV_OK != nRet:
        print("Stop grabbing failed! ErrorCode", nRet)
        sys.exit()

    # 关闭相机
    nRet = cam.IMV_Close()
    if IMV_OK != nRet:
        print("Close camera failed! ErrorCode", nRet)
        sys.exit()

    # 销毁句柄
    if cam.handle:
        nRet = cam.IMV_DestroyHandle()


def send_stack_status_to_plc(stack_status, plc_enabled):
    """
    将堆积状态转换为PLC控制值并发送，或仅打印结果（取决于开关）
    :param stack_status: str, 状态字符串，如 'normal', 'first class warning' 等
    :param plc_enabled: bool, 是否启用PLC通讯
    """
    # 状态映射表：将状态字符串映射为 PLC 所需数值
    status_to_value = {
        "normal": 0,
        "first class warning": 1,
        "second class warning": 2,
        "third class warning": 3,
        "lack of potato": 4
    }

    # 获取PLC数值
    plc_value = status_to_value.get(stack_status, 0)

    # 打印当前状态和数值
    print(f"📊 状态: {stack_status}, 发送PLC数值: {plc_value}")

    if not plc_enabled:
        print("🚫 模拟模式：未发送PLC，仅打印状态")
        return

    # PLC配置（可改为参数）
    PLC_IP = "192.168.1.88"  # 修改为你的实际IP
    PLC_PORT = 502
    target_register = 1  # 对应 %MW1，堆积状态地址

    # 建立连接并写入数据
    client = ModbusTcpClient(PLC_IP, port=PLC_PORT)
    if client.connect():
        result = client.write_register(target_register, plc_value)
        # result = client.write_register(target_register, 1)
        if not result.isError():
            print(f"已成功发送至PLC: 地址 {target_register}, 数值 {plc_value}")
        else:
            print("❌ 写入失败：", result)
        client.close()
    else:
        print("无法连接到 PLC，请检查网络或地址配置")


if __name__ == "__main__":
    for ret, sdk_frame in retrun_frame():
        if not ret:
            continue

        # 图像预处理
        pil_image = Image.fromarray(cv2.cvtColor(sdk_frame, cv2.COLOR_BGR2RGB))

        # UNet分割
        seg_result = unet_detector.detect_image(pil_image)
        seg_frame = cv2.cvtColor(np.array(seg_result), cv2.COLOR_RGB2BGR)

        # 生成红色掩码
        hsv = cv2.cvtColor(seg_frame, cv2.COLOR_BGR2HSV)

        # 定义红色的HSV范围
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        # 创建红色掩码
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        # 计算角度、面积、堆积状态并显示（接收新增的堆积状态参数）
        result_frame, cut_angle, potato_area, stack_status = compute_axes_and_draw(sdk_frame.copy(), mask)

        plc_enabled = True                                 # 设置 True 以启用PLC通讯
        # plc_enabled = False  # 设置 True 以启用PLC通讯
        send_stack_status_to_plc(stack_status, plc_enabled)


        # 机械臂控制逻辑（新增打印堆积状态）
        # if cut_angle is not None:
        #     print(f"发送给机械臂的旋转角度：{cut_angle:.2f}度")
        #     print(f"土豆总面积：{potato_area:.1f}像素 | 堆积状态：{stack_status}")  # 打印面积和状态

        cv2.imshow("Potato Cutting System", result_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()