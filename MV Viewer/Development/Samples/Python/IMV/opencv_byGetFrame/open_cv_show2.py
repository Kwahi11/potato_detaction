# -- coding: utf-8 --

import sys
from ctypes import *
import datetime
import numpy
import cv2
import gc
import  numpy as np

sys.path.append(r"F:\YOLOV8\PotatoDetection-main\MV Viewer\Development\Samples\Python\IMV\MVSDK")
from IMVApi import *




def displayDeviceInfo(deviceInfoList):
    print("Idx  Type   Vendor              Model           S/N                 DeviceUserID    IP Address")
    print("------------------------------------------------------------------------------------------------")
    for i in range(0, deviceInfoList.nDevNum):
        pDeviceInfo = deviceInfoList.pDevInfo[i]
        strType = ""
        strVendorName = pDeviceInfo.vendorName.decode("ascii")
        strModeName = pDeviceInfo.modelName.decode("ascii")
        strSerialNumber = pDeviceInfo.serialNumber.decode("ascii")
        strCameraname = pDeviceInfo.cameraName.decode("ascii")
        strIpAdress = pDeviceInfo.DeviceSpecificInfo.gigeDeviceInfo.ipAddress.decode("ascii")
        if pDeviceInfo.nCameraType == typeGigeCamera:
            strType = "Gige"
        elif pDeviceInfo.nCameraType == typeU3vCamera:
            strType = "U3V"
        print("[%d]  %s   %s    %s      %s     %s           %s" % (
            i + 1, strType, strVendorName, strModeName, strSerialNumber, strCameraname, strIpAdress))

#
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

    nConnectionNum = 2

    if int(nConnectionNum) > deviceList.nDevNum:
        print ("intput error!")
        sys.exit()

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
    # nRet = cam.IMV_SetEnumFeatureSymbol("TriggerSource", "Software")
    nRet = cam.IMV_SetEnumFeatureSymbol("TriggerSource", "Line1")
    if IMV_OK != nRet:
        print("Set triggerSource value failed! ErrorCode[%d]" % nRet)
        sys.exit()

    nRet = cam.IMV_SetEnumFeatureSymbol("TriggerSelector", "FrameStart")
    if IMV_OK != nRet:
        print("Set triggerSelector value failed! ErrorCode[%d]" % nRet)
        sys.exit()

    # nRet = cam.IMV_SetEnumFeatureSymbol("TriggerMode", "Off")
    nRet = cam.IMV_SetEnumFeatureSymbol("TriggerMode", "On")
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
        # else:
            # print("getFrame success BlockId = [" + str(frame.frameInfo.blockId) + "], get frame time: " + str(
            #     datetime.datetime.now()))


        if None == byref(frame):
            print("pFrame is NULL!")
            continue
        # 给转码所需的参数赋值

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
        # release frame resource at the end of use

        nRet = cam.IMV_ReleaseFrame(frame)
        if IMV_OK != nRet:
            print("Release frame failed! ErrorCode[%d]\n", nRet)
            sys.exit()

        # 如果图像格式是 Mono8 直接使用
        # no format conversion required for Mono8
        if stPixelConvertParam.ePixelFormat == IMV_EPixelType.gvspPixelMono8:
            imageBuff = stPixelConvertParam.pSrcData
            userBuff = c_buffer(b'\0', stPixelConvertParam.nDstBufSize)

            memmove(userBuff, imageBuff, stPixelConvertParam.nDstBufSize)
            grayByteArray = bytearray(userBuff)

            cvImage = numpy.array(grayByteArray).reshape(stPixelConvertParam.nHeight, stPixelConvertParam.nWidth)

        else:
            # 转码 => BGR24
            # convert to BGR24
            stPixelConvertParam.eDstPixelFormat = IMV_EPixelType.gvspPixelBGR8
            # stPixelConvertParam.nDstBufSize=nDstBufSize

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

        yield 1,cvImage







# def retrun_frame(camera_index=1):
#         """支持多摄像头并行的帧获取生成器"""
#         deviceList = IMV_DeviceList()
#         interfaceType = IMV_EInterfaceType.interfaceTypeAll
#
#         # 枚举设备
#         nRet = MvCamera.IMV_EnumDevices(deviceList, interfaceType)
#         if IMV_OK != nRet:
#             print(f"Camera {camera_index} enum failed! ErrorCode", nRet)
#             sys.exit()
#         if deviceList.nDevNum == 0:
#             print(f"Camera {camera_index} find no device!")
#             sys.exit()
#
#         # 创建指定摄像头的实例
#         cam = MvCamera()
#         nRet = cam.IMV_CreateHandle(
#             IMV_ECreateHandleMode.modeByIndex,
#             byref(c_void_p(camera_index))  # 直接使用传入的索引
#         )
#         if IMV_OK != nRet:
#             print(f"Camera {camera_index} create handle failed! ErrorCode", nRet)
#             sys.exit()
#
#         # 打开相机
#         nRet = cam.IMV_Open()
#         if IMV_OK != nRet:
#             print(f"Camera {camera_index} open failed! ErrorCode", nRet)
#             sys.exit()
#
#         # 配置触发模式（每个摄像头独立配置）
#         try:
#             # cam.IMV_SetEnumFeatureSymbol("TriggerSource", "Software")
#             cam.IMV_SetEnumFeatureSymbol("TriggerSource", "Line1")
#             cam.IMV_SetEnumFeatureSymbol("TriggerSelector", "FrameStart")
#             # cam.IMV_SetEnumFeatureSymbol("TriggerMode", "Off")
#             cam.IMV_SetEnumFeatureSymbol("TriggerMode", "On")
#         except Exception as e:
#             print(f"Camera {camera_index} config failed:", e)
#             sys.exit()
#
#         # 开始采集
#         nRet = cam.IMV_StartGrabbing()
#         if IMV_OK != nRet:
#             print(f"Camera {camera_index} start grabbing failed! ErrorCode", nRet)
#             sys.exit()
#
#         isGrab = True
#         while isGrab:
#             frame = IMV_Frame()
#             stPixelConvertParam = IMV_PixelConvertParam()
#
#             # 获取帧数据
#             nRet = cam.IMV_GetFrame(frame, 1000)
#             if IMV_OK != nRet:
#                 continue
#
#             # 图像处理逻辑
#             try:
#                 if frame.frameInfo.pixelFormat == IMV_EPixelType.gvspPixelMono8:
#                     nDstBufSize = frame.frameInfo.width * frame.frameInfo.height
#                 else:
#                     nDstBufSize = frame.frameInfo.width * frame.frameInfo.height * 3
#
#                 pDstBuf = (c_ubyte * nDstBufSize)()
#                 memset(byref(stPixelConvertParam), 0, sizeof(stPixelConvertParam))
#
#                 # 填充转换参数
#                 stPixelConvertParam.nWidth = frame.frameInfo.width
#                 stPixelConvertParam.nHeight = frame.frameInfo.height
#                 stPixelConvertParam.ePixelFormat = frame.frameInfo.pixelFormat
#                 stPixelConvertParam.pSrcData = frame.pData
#                 stPixelConvertParam.nSrcDataLen = frame.frameInfo.size
#                 stPixelConvertParam.eDstPixelFormat = IMV_EPixelType.gvspPixelBGR8
#                 stPixelConvertParam.pDstBuf = pDstBuf
#                 stPixelConvertParam.nDstBufSize = nDstBufSize
#
#                 # 执行像素转换
#                 nRet = cam.IMV_PixelConvert(stPixelConvertParam)
#                 if IMV_OK == nRet:
#                     # 构建numpy数组
#                     img_data = bytearray(pDstBuf)
#                     if frame.frameInfo.pixelFormat == IMV_EPixelType.gvspPixelMono8:
#                         cvImage = np.array(img_data).reshape(
#                             frame.frameInfo.height,
#                             frame.frameInfo.width
#                         )
#                     else:
#                         cvImage = np.array(img_data).reshape(
#                             frame.frameInfo.height,
#                             frame.frameInfo.width,
#                             3
#                         )
#                     yield True, cvImage
#                 else:
#                     yield False, None
#
#             finally:
#                 # 释放帧资源
#                 cam.IMV_ReleaseFrame(frame)
#                 del pDstBuf
#
#         # 停止采集并释放资源
#         cam.IMV_StopGrabbing()
#         cam.IMV_Close()
#         if cam.handle:
#             cam.IMV_DestroyHandle()
#
#
#     # 停止拉流
#         nRet = cam.IMV_StopGrabbing()
#         if IMV_OK != nRet:
#          print("Stop grabbing failed! ErrorCode", nRet)
#          sys.exit()
#
#     # 关闭相机
#         nRet = cam.IMV_Close()
#         if IMV_OK != nRet:
#          print("Close camera failed! ErrorCode", nRet)
#          sys.exit()
#
#     # 销毁句柄
#         if cam.handle:
#          nRet = cam.IMV_DestroyHandle()


if __name__ == "__main__":
    for ret,frame in retrun_frame():
        cv2.imshow('Camera Feed', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    print("---Demo end---")
