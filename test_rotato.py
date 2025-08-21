import numpy as np
import math
import time

# 配置参数
M = 2  # 选择模式 1, 2, 或 3
red_coords = [(700, 320), (395, 325),
    (700, 300), (150, 950)]  # 替换为实际芽眼坐标
X = (380, 320)  # 形心坐标

black_point = X  # 使用形心坐标

def distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

valid_points = []
for coord in red_coords[:20]:
    if not isinstance(coord, (tuple, list)) or len(coord) != 2:
        continue  # 跳过无效元素
    if coord == (0, 0):
        continue
    if distance(coord, black_point) <= 20:
        continue
    valid_points.append(coord)

def generate_rays(base_angle):
    if M == 3:
        return [base_angle + i * 90 for i in range(4)]
    elif M == 1:
        return [base_angle, base_angle + 180]
    elif M == 2:
        return [base_angle + i * 120 for i in range(3)]

def classify_point(angle, point):
    bx, by = black_point
    px, py = point
    dx = px - bx
    dy = by - py  # 图像坐标系y轴向下
    point_angle = math.degrees(math.atan2(dy, dx)) % 360
    relative_angle = (point_angle - angle) % 360

    if M == 3:
        return 1 + int(relative_angle // 90)
    elif M == 1:
        return 1 if relative_angle < 180 else 2
    elif M == 2:
        return 1 + int(relative_angle // 120)

def is_valid_angle(base_angle):
    bx, by = black_point
    rays = generate_rays(base_angle)
    for (px, py) in valid_points:
        for angle in rays:
            end_x = bx + 1000 * math.cos(math.radians(angle))
            end_y = by - 1000 * math.sin(math.radians(angle))
            numerator = abs((end_y - by) * px - (end_x - bx) * py + end_x * by - end_y * bx)
            denominator = math.sqrt((end_y - by) ** 2 + (end_x - bx) ** 2)
            if denominator != 0 and numerator / denominator <= 20:
                return False
    return True

start_time = time.perf_counter()
optimal_angle = None
min_empty = {3: 4, 1: 2, 2: 3}[M]

for theta in range(0, 360, 1):
    if not is_valid_angle(theta):
        continue

    regions = {i + 1: 0 for i in range({3: 4, 1: 2, 2: 3}[M])}
    for pt in valid_points:
        region = classify_point(theta, pt)
        regions[region] += 1

    empty_count = sum(1 for v in regions.values() if v == 0)
    if empty_count < min_empty:
        min_empty, optimal_angle = empty_count, theta
        if min_empty == 0:
            break

end_time = time.perf_counter()
elapsed_time = end_time - start_time

if optimal_angle is not None:
    print(f"模式{M}最优解：空区域数={min_empty}，旋转角度={optimal_angle}°，耗时{elapsed_time:.3f}秒")
else:
    optimal_angle = 15
    print(f"未找到最优角度，默认使用 {optimal_angle}°")
