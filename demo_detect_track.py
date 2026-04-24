from __future__ import division
import sys
import os
import cv2
import torch
import warnings
import struct
import socket

# ===================== 路径配置 =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "detect_wrapper"))

# ===================== 基础配置 =====================
warnings.filterwarnings("ignore")
device = torch.device('cpu')

# ===================== UDP发送配置 =====================
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
IP = '192.168.0.171'
Port = 9921


def send_coord(coord):
    if coord is None:
        return
    address = (IP, Port)
    msgCode = 1
    nAimType = 1
    nTrackType = 1
    nState = 1
    nAimX, nAimY, nAimW, nAimH = coord
    data = struct.pack("iiiiiiii", msgCode, nAimType, nAimX, nAimY, nAimW, nAimH, nTrackType, nState)
    udp_socket.sendto(data, address)

# ===================== 工具函数 =====================
def filter_invalid_box(box, frame_shape):
    if box is None:
        return None
    x, y, w, h = box
    h_frame, w_frame = frame_shape[:2]
    if w <= 0 or h <= 0:
        return None
    x = max(0, min(x, w_frame - 1))
    y = max(0, min(y, h_frame - 1))
    w = max(1, min(w, w_frame - x))
    h = max(1, min(h, h_frame - y))
    return [int(x), int(y), int(w), int(h)]

# ===================== 主程序：无间隔 持续实时跟踪 =====================
def test():
    from detect_wrapper.Detectoruav import DroneDetection
    from tracking_wrapper.dronetracker.trackinguav.evaluation.tracker import Tracker

    # 权重路径
    weights_path = os.path.join(BASE_DIR, "detect_wrapper", "weights", "weights", "best.pt")
    print("🔍 加载权重：", weights_path)

    if not os.path.exists(weights_path):
        print("❌ 权重文件不存在！")
        return

    # 视频/摄像头配置
    video_path =r"C:\Users\kray0\PycharmProjects\PythonProject3\drone_system\test\20190925_124612_1_4\visible.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("⚠️ 视频打开失败，切换为摄像头")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ 摄像头打开失败！")
            return

    ret, frame = cap.read()
    if not ret:
        print("❌ 画面读取失败！")
        return
    print(f"✅ 视频加载成功：{frame.shape}")

    # 初始化检测+跟踪
    drone_det = DroneDetection(IRweights_path=weights_path, RGBweights_path=weights_path)
    drone_tracker = Tracker()
    first_track = True  # 仅第一次初始化跟踪

    print("🚀 启动【无间隔 持续实时跟踪】，实时输出无人机位置...")

    # ===================== 核心：每帧都跟踪，无任何间隔 =====================
    while ret:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. 每帧都执行检测（实时修正跟踪，无间隔）
        init_box = drone_det.forward_IR(frame)
        init_box = filter_invalid_box(init_box, frame.shape)

        # 2. 首次检测到目标 → 初始化跟踪（仅执行1次）
        if init_box is not None and first_track:
            try:
                drone_tracker.init_track(init_box, frame)
                first_track = False
                print(f"\n✅ 持续跟踪启动！初始位置：X={init_box[0]}, Y={init_box[1]}")
            except Exception as e:
                print(f"❌ 跟踪初始化失败：{e}")
                continue

        # 3. 【全程持续跟踪】每帧都运行跟踪器，无间隔、无暂停
        if not first_track:
            try:
                # 实时跟踪
                track_box = drone_tracker.on_track(frame)
                track_box = filter_invalid_box(track_box, frame.shape)

                if track_box is not None:
                    x, y, w, h = track_box
                    # ==============================================
                    # 🔥 实时输出位置（每帧都打印，无延迟）
                    # ==============================================
                    print(f"📍 持续跟踪位置 → X:{x}  Y:{y}  W:{w}  H:{h}")
                    # send_coord(track_box)  # 需要UDP发送就打开注释

                    # 绘制跟踪框
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
                    cv2.putText(frame, "TRACKING", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            except Exception as e:
                print(f"⚠️ 跟踪异常，重新锁定：{e}")
                first_track = True

        # 实时显示画面
        cv2.imshow('【无间隔 持续跟踪】DRONE SYSTEM', cv2.resize(frame, (1280, 720)))
        # ESC退出
        if cv2.waitKey(1) == 27:
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    udp_socket.close()
    print("✅ 程序正常退出")

if __name__ == "__main__":
    test()

