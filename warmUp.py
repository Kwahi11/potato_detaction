import socket
import time
import threading
import random

# 给机械臂发送数据帧来提高触发频率

def send_data_frame(robot_ip, robot_port, centroids, frame_num=0, timeout=3):
    """
    发送一次与 MyWindow._send_data_frame 相同格式的数据帧（同步发送后立即关闭连接）。
    centroids: 列表，元素为 (cx, cy, angle)
    frame_num: 帧编号
    """
    try:
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.settimeout(timeout)
        client.connect((robot_ip, robot_port))
    except Exception as e:
        print("连接机械臂失败：", e)
        return False
    try:
        obj_number = len(centroids)
        trigger_time = int(time.time() * 1000)  # 毫秒时间戳
        data_header = f"Data;{frame_num};{trigger_time};{obj_number};"
        data_body = []
        for idx, (cx, cy, angle) in enumerate(centroids):
            dx = cx
            dy = -cy
            ang = -angle
            data_body.append(f"{idx},{dx},{dy},0,0,0,{ang},0,0,0,0,0,no")
        if len(data_body) != 0:
            full_data = "STX" + data_header + "|" + "|".join(data_body) + "ETX"
        else:
            full_data = "STX" + data_header + "ETX"
        client.sendall(full_data.encode("ascii"))
        print("已发送：", full_data)
    except Exception as e:
        print("发送数据出错：", e)
        client.close()
        return False
    client.close()
    return True

def send_data_frame_continuous(robot_ip, robot_port, centroid_generator,
                               interval=0.1, repeat=None, reconnect=False, timeout=3):
    """
    连续发送：
    - centroid_generator: 一个可迭代/生成器，每次返回 centroids 列表
    - interval: 发送间隔（秒）
    - repeat: 发送次数，None 表示无限
    - reconnect: True 每次发送前重连（鲁棒）；False 保持一个长连接（高效）
    """
    client = None
    frame_num = 0
    sent = 0
    try:
        if not reconnect:
            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client.settimeout(timeout)
            client.connect((robot_ip, robot_port))
        while repeat is None or sent < repeat:
            try:
                centroids = next(centroid_generator)
            except StopIteration:
                break
            if reconnect:
                try:
                    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    client.settimeout(timeout)
                    client.connect((robot_ip, robot_port))
                except Exception as e:
                    print("重连失败：", e)
                    time.sleep(interval)
                    continue
            # 构造并发送（与 send_data_frame 相同格式）
            try:
                obj_number = len(centroids)
                trigger_time = int(time.time() * 1000)
                data_header = f"Data;{frame_num};{trigger_time};{obj_number};"
                data_body = []
                for idx, (cx, cy, angle) in enumerate(centroids):
                    dx = cx
                    dy = -cy
                    ang = -angle
                    data_body.append(f"{idx},{dx},{dy},0,0,0,{ang},0,0,0,0,0,no")
                if len(data_body) != 0:
                    full_data = "STX" + data_header + "|" + "|".join(data_body) + "ETX"
                else:
                    full_data = "STX" + data_header + "ETX"
                client.sendall(full_data.encode("ascii"))
                print(f"[{sent}] 已发送：", full_data)
                frame_num += 1
                sent += 1
            except Exception as e:
                print("发送失败：", e)
                if client:
                    try:
                        client.close()
                    except:
                        pass
                    client = None
                if not reconnect:
                    # 尝试重连后继续
                    try:
                        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        client.settimeout(timeout)
                        client.connect((robot_ip, robot_port))
                    except Exception as e2:
                        print("重连尝试失败：", e2)
                        time.sleep(interval)
            finally:
                if reconnect and client:
                    try:
                        client.close()
                    except:
                        pass
                    client = None
            time.sleep(interval)
    except KeyboardInterrupt:
        print("被用户中断(KeyboardInterrupt)，停止发送")
    finally:
        if client:
            try:
                client.close()
            except:
                pass
        print("发送结束，共发送:", sent)
    return sent

def generator_random_centroids(num_per_frame=2):
    """示例生成器：每次返回 num_per_frame 个随机 centroids"""
    while True:
        lst = []
        for i in range(num_per_frame):
            cx = random.randint(0, 640)
            cy = random.randint(0, 480)
            angle = random.randint(-45, 45)
            lst.append((cx, cy, angle))
        yield lst

if __name__ == "__main__":
    ROBOT_IP = "192.168.31.139"
    ROBOT_PORT = 3366

 
    gen = generator_random_centroids(2)
    sent_count = send_data_frame_continuous(ROBOT_IP, ROBOT_PORT, gen,
                                           interval=0.12, repeat=100, reconnect=False)
    print("总共发送帧数：", sent_count)