from __future__ import division
import os
import cv2
import sys
import struct
import socket
import logging
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# ===================== 路径配置 =====================
# 添加必要的模块路径（适配你的工程目录）
sys.path.append(os.path.join(os.path.dirname(__file__), 'detect_wrapper'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'tracking_wrapper\\dronetracker'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'tracking_wrapper\\drtracker'))

# 导入核心检测/跟踪模块
try:
    from detect_wrapper.Detectoruav import DroneDetection
    from tracking_wrapper.dronetracker.trackinguav.evaluation.tracker import Tracker
except ImportError as e:
    print(f"模块导入失败: {e}")
    print("请确保detect_wrapper和tracking_wrapper文件夹存在且路径正确")
    sys.exit(1)

# ===================== 全局配置 =====================
# 基础参数（重点：替换为拆帧后的图片文件夹路径）
IMG_SEQ_DIR = r'C:\Users\kray0\PycharmProjects\PythonProject3\drone_system\test\n19'  # 拆帧后的图片文件夹
IR_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), 'detect_wrapper\\weights\\best.pt')  # 检测模型权重
TRACK_MAX_COUNT = 150  # 单次检测后跟踪最大帧数
MAGNIFICATION = 2  # 可视化放大倍数
VISUALIZATION = 1  # 是否可视化（1=开启，0=关闭）
SEND_LOCATION = 0  # 是否发送UDP坐标（1=开启，0=关闭）

# UDP配置（发送跟踪坐标用）
UDP_TARGET_IP = '192.168.0.171'
UDP_TARGET_PORT = 9921
UDP_SERVER_ADDR = '127.0.0.1'
UDP_SERVER_PORT = 9999
UDP_FRAME_SIZE = 8192

# 全局变量
g_init = False  # 初始化标记
g_detector = None  # 检测器实例
g_tracker = None  # 跟踪器实例
g_logger = None  # 日志实例
g_frame_counter = 0  # 跟踪帧数计数器
g_enable_log = True  # 是否开启日志
detect_first = True  # 是否首次检测
count = 0  # 帧计数
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # 发送UDP的socket


# ===================== 工具函数 =====================
def safe_log(msg):
    """安全日志输出（避免logger未初始化报错）"""
    if g_logger:
        g_logger.info(msg)


def mono_to_rgb(data):
    """单通道红外图转RGB图"""
    w, h = data.shape
    img = np.zeros((w, h, 3), dtype=np.uint8)
    img[:, :, 0] = data
    img[:, :, 1] = data
    img[:, :, 2] = data
    return img


def distance_check(bbx1, bbx2, thd=60):
    """检查两个检测框的中心距离是否小于阈值"""
    cx1 = bbx1[0] + bbx1[2] / 2
    cy1 = bbx1[1] + bbx1[3] / 2
    cx2 = bbx2[0] + bbx2[2] / 2
    cy2 = bbx2[1] + bbx2[3] / 2
    dist = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
    return dist < thd


def scale_coords(img1_shape, coords, img0_shape):
    """将检测框坐标从img1_shape缩放到img0_shape"""
    gainx = img1_shape[0] / img0_shape[0]
    gainy = img1_shape[1] / img0_shape[1]
    coords = [int(x / gain) for x, gain in zip(coords, [gainx, gainy, gainx, gainy])]
    return coords


def send_coord(coord):
    """通过UDP发送跟踪框坐标"""
    if not SEND_LOCATION or coord is None:
        return
    try:
        address = (UDP_TARGET_IP, UDP_TARGET_PORT)
        # 打包成C结构体格式：msgCode, nAimType, x, y, w, h, nTrackType, nState
        data = struct.pack("iiiiiiii", 1, 1, coord[0], coord[1], coord[2], coord[3], 1, 1)
        udp_socket.sendto(data, address)
        safe_log(f"发送坐标成功: {coord}")
    except Exception as e:
        safe_log(f"发送坐标失败: {e}")


def result_visualization(img, bbox):
    """可视化跟踪框（放大显示）"""
    if not VISUALIZATION or bbox is None:
        return
    oframe = img.copy()
    # 放大图像
    visuframe = cv2.resize(oframe,
                           (oframe.shape[1] * MAGNIFICATION,
                            oframe.shape[0] * MAGNIFICATION),
                           cv2.INTER_LINEAR)
    # 放大检测框坐标
    bbx = [i * MAGNIFICATION for i in bbox]
    # 绘制绿色矩形框
    cv2.rectangle(visuframe,
                  (bbx[0], bbx[1]),
                  (bbx[0] + bbx[2], bbx[1] + bbx[3]),
                  (0, 255, 0), 2)
    cv2.imshow("Drone Tracking", visuframe)
    cv2.waitKey(1)  # 刷新显示（1ms延迟，保证实时性）


# ===================== 核心初始化 =====================
def global_init():
    """初始化检测器、跟踪器、日志"""
    global g_init, g_detector, g_tracker, g_logger
    if g_init:
        return

    # 1. 初始化日志
    if g_enable_log:
        g_logger = logging.getLogger('DroneTracker')
        g_logger.setLevel(logging.INFO)
        # 日志文件保存路径
        log_path = os.path.join('c:/data', 'drone_track_log.txt')
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        fh = logging.FileHandler(log_path, mode='a', encoding='utf-8')
        fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        g_logger.addHandler(fh)

    # 2. 初始化检测器
    try:
        g_detector = DroneDetection(IRweights_path=IR_WEIGHTS_PATH,
                                    RGBweights_path=IR_WEIGHTS_PATH)
        safe_log("检测器初始化成功")
    except Exception as e:
        safe_log(f"检测器初始化失败: {e}")
        raise

    # 3. 初始化跟踪器
    try:
        g_tracker = Tracker()
        safe_log("跟踪器初始化成功")
    except Exception as e:
        safe_log(f"跟踪器初始化失败: {e}")
        raise

    g_init = True
    safe_log("全局初始化完成")


# ===================== 图像处理核心逻辑 =====================
def imgproc(frame):
    """处理单帧图像：检测+跟踪"""
    global g_frame_counter, count, detect_first

    count += 1
    safe_log(f"处理第 {count} 帧")
    bbx = None
    IMG_TYPE = 0  # 0=RGB, 1=IR

    # 处理图像类型（单通道IR转RGB）
    if len(frame.shape) == 2:
        IMG_TYPE = 1
        frame = mono_to_rgb(frame)

    # 核心逻辑：检测/跟踪切换
    if g_detector and g_tracker:
        # 情况1：跟踪帧数耗尽 → 重新检测
        if g_frame_counter <= 0:
            # 执行检测（IR/RGB分支）
            if IMG_TYPE == 0:
                init_box = g_detector.forward_RGB(frame)
                center_box = [320, 192, 0, 0]  # RGB图中心参考点
            else:
                init_box = g_detector.forward_IR(frame)
                center_box = [320, 256, 0, 0]  # IR图中心参考点

            # 首次检测逻辑
            if detect_first:
                if init_box is not None and distance_check(init_box, center_box, 60):
                    # 初始化跟踪器
                    g_tracker.init_track(init_box, frame)
                    g_frame_counter = TRACK_MAX_COUNT
                    detect_first = False
                    safe_log("首次检测成功，初始化跟踪器")
                    # 可视化+发送坐标
                    init_box = [int(x) for x in init_box]
                    if IMG_TYPE == 1 and count % 8 == 1:
                        send_coord(init_box)
                    elif IMG_TYPE == 0 and count % 8 == 1:
                        init_box = scale_coords([640, 384], init_box, [1920, 1080])
                        send_coord(init_box)
                    result_visualization(frame, init_box)
                else:
                    safe_log("首次检测未找到目标")
                    result_visualization(frame, None)

            # 非首次检测逻辑
            else:
                if init_box is not None and distance_check(init_box, center_box, 60):
                    # 更新跟踪器状态
                    g_tracker.change_state(init_box)
                    g_frame_counter = TRACK_MAX_COUNT
                    safe_log("重新检测到目标，更新跟踪器")
                    # 可视化+发送坐标
                    init_box = [int(x) for x in init_box]
                    if IMG_TYPE == 1 and count % 2 == 1:
                        send_coord(init_box)
                    elif IMG_TYPE == 0 and count % 2 == 1:
                        init_box = scale_coords([640, 384], init_box, [1920, 1080])
                        send_coord(init_box)
                    result_visualization(frame, init_box)
                else:
                    # 检测不到目标，继续跟踪
                    safe_log("未检测到新目标，继续跟踪")
                    g_frame_counter = TRACK_MAX_COUNT
                    bbx = g_tracker.on_track(frame)
                    g_frame_counter -= 1
                    # 可视化+发送坐标
                    if IMG_TYPE == 1 and count % 2 == 1:
                        send_coord(bbx)
                    elif IMG_TYPE == 0 and count % 2 == 1:
                        bbx = scale_coords([640, 384], bbx, [1920, 1080])
                        send_coord(bbx)
                    result_visualization(frame, bbx)

        # 情况2：跟踪帧数未耗尽 → 继续跟踪
        else:
            bbx = g_tracker.on_track(frame)
            g_frame_counter -= 1
            safe_log(f"跟踪中，剩余帧数: {g_frame_counter}")
            # 可视化+发送坐标
            if IMG_TYPE == 1 and count % 2 == 1:
                send_coord(bbx)
            elif IMG_TYPE == 0 and count % 2 == 1:
                bbx = scale_coords([640, 384], bbx, [1920, 1080])
                send_coord(bbx)
            result_visualization(frame, bbx)


# ===================== 主函数 =====================
def main():
    # 1. 全局初始化（检测器+跟踪器+日志）
    global_init()

    # 2. 校验拆帧图片文件夹
    if not os.path.exists(IMG_SEQ_DIR):
        print(f"错误：拆帧图片文件夹不存在 → {IMG_SEQ_DIR}")
        sys.exit(1)

    # 3. 读取并排序所有帧图片（保证按顺序处理）
    frame_files = sorted([
        f for f in os.listdir(IMG_SEQ_DIR)
        if f.endswith(('.jpg', '.png', '.jpeg'))  # 支持常见图片格式
    ])

    if len(frame_files) == 0:
        print(f"错误：{IMG_SEQ_DIR} 中未找到任何图片文件（支持jpg/png/jpeg）")
        sys.exit(1)
    print(f"✅ 成功加载 {len(frame_files)} 帧图片，开始检测跟踪...")

    # 4. 逐帧处理拆帧后的图片
    for idx, frame_file in enumerate(frame_files):
        frame_path = os.path.join(IMG_SEQ_DIR, frame_file)
        # 读取单帧图片
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"⚠️  跳过无效图片：{frame_file}")
            continue

        # 调用检测跟踪核心逻辑
        imgproc(frame)

        # 可选：按ESC键提前退出
        if cv2.waitKey(1) & 0xFF == 27:
            print("⚠️  用户按下ESC键，提前退出")
            break

    # 5. 资源清理
    cv2.destroyAllWindows()
    udp_socket.close()
    print("\n✅ 拆帧图片检测跟踪完成！")
    print(f"📊 处理总帧数：{count} | 日志文件：c:/data/drone_track_log.txt")


if __name__ == "__main__":
    main()