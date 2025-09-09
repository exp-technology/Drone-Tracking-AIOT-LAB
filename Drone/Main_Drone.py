import os
# 完全無頭（禁用 X/Wayland / GTK 需求）
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("DISPLAY", "")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")
# 只影響影像抓取後端，不會觸發 GUI，但留著無妨
os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_GSTREAMER", "0")
# 立即關閉 stdout/stderr 緩衝（避免看不到 print）
os.environ.setdefault("PYTHONUNBUFFERED", "1")

from ultralytics.utils import SETTINGS
from ultralytics import YOLO
SETTINGS.update({'emoji': False, 'show': False, 'save': False, 'plots': False})

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="ignore", line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="ignore", line_buffering=True)

import cv2
# import vart
import time
import socket
import pickle
import asyncio
import numpy as np
import json
import csv
import textwrap
from pymavlink import mavutil
from mav_function import DroneCommand
import struct
# from Detection import YoloByteTracker
from Mavlink_UDP_hub import MavUdpHub
from drone_autotrack_controller import AutoTrackController
import config

# 固定 PC 端 IP 與視訊埠
VIDEO_IP_   = config.PC_IP

VIDEO_PORT_ = 5000
INFO_PORT_ = 5001
KEYBOARD_PORT_ = 5002

JPEG_QUALITY  = 70                
TARGET_FPS    = 30

# TCP 傳輸的訊息類型
MSG_JPEG = 1    # JPEG 圖片(串流video)
MSG_META = 2    # 追蹤框元資料(JSON)
MSG_EVENT = 3   # 事件通知

SHOW_FPS_OVERLAY = True

PERSON_CLASS_ID = 0  # person

TRT_IMGSZ_ = config.TRT_IMGSZ
MODEL_NAME_ = config.MODEL


def draw_tracks(frame_bgr, tracks, color=(0, 255, 0), thickness=2):
    """
    tracks: [{'track_id': int, 'tlwh': [x,y,w,h]}, ...]
    """
    img = frame_bgr.copy()
    for tr in tracks:
        x, y, w, h = tr['tlwh']
        cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
        cv2.putText(img, f"ID:{tr['track_id']}", (x, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img


def draw_dashed_rect(img, pt1, pt2, color=(0, 0, 255), thickness=2, dash=10, gap=6):
    """
    以短線段畫出虛線矩形
    pt1=(x1,y1), pt2=(x2,y2) 皆為整數座標
    color: BGR，預設紅色
    thickness: 線寬
    dash: 每段實線長度（像素）
    gap: 段與段之間的空白長度（像素）
    """
    x1, y1 = int(pt1[0]), int(pt1[1])
    x2, y2 = int(pt2[0]), int(pt2[1])

    # 上、下邊
    for x in range(x1, x2, dash + gap):
        x_end = min(x + dash, x2)
        cv2.line(img, (x, y1), (x_end, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x, y2), (x_end, y2), color, thickness, cv2.LINE_AA)

    # 左、右邊
    for y in range(y1, y2, dash + gap):
        y_end = min(y + dash, y2)
        cv2.line(img, (x1, y), (x1, y_end), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y), (x2, y_end), color, thickness, cv2.LINE_AA)


def ensure_botsort_reid_yaml(path="./model/botsort_reid.yaml"):
    """
    確保有一份含 ReID 與必要鍵 (track_buffer, model) 的 BoT-SORT 設定檔。
    若檔案不存在，會建立；存在就直接使用。
    回傳實際檔案路徑。
    """
    if not os.path.exists(path):
        yaml_text = textwrap.dedent("""
            # botsort_reid.yaml
            tracker_type: botsort

            # ---- thresholds & buffers ----
            track_high_thresh: 0.5
            track_low_thresh: 0.1
            new_track_thresh: 0.6
            match_thresh: 0.8
            track_buffer: 30
            max_age: 70
            min_box_area: 10
            fuse_score: true

            # ---- camera motion compensation ----
            gmc_method: sparseOptFlow

            # ---- ReID ----
            with_reid: true
            model: auto              # 或填 ReID 權重檔路徑（如 osnet_x0_25_msmt17.pt）
            proximity_thresh: 0.5
            appearance_thresh: 0.25
        """).strip() + "\n"
        with open(path, "w", encoding="utf-8") as f:
            f.write(yaml_text)
    return os.path.abspath(path)


class MainServer:
    detection_flag = True
    tracking_flag = False
    AutoTracking_flag = False
    tracking_box = [208-50, 208-50, 208+50, 208+50]
    target_id = None
    video_port = 5000
    info_port = 5001
    keyboard_port = 5002

    
    def __init__(self):
        # Video Socket
        # Set up the socket
        self.buffer_size = 64*1024 # 64 KB
        host_name = socket.gethostname()
        # host_ip = socket.gethostbyname(host_name)
        host_ip = config.HOST_IP

        # === FPS / 延遲度量 ===
        self._fps_counter = 0
        self._fps_last_t = time.perf_counter()
        self._fps_value = 0.0
        self._last_trk_ms = 0.0
        self._last_enc_ms = 0.0

        self.target_id = None          # 鎖定的追蹤 ID
        self._last_seen_ts = None      # 上次看到該 ID 的時間戳
        self._lost_timeout = 1.0      # 追丟容許秒數（可調 0.5~1.0）

        self.tracker_yaml_path = ensure_botsort_reid_yaml()  # 產生/取得 botsort_reid.yaml
        self.yolo = None                                     # 之後 warmup_tracker 會載入
        self.tracker_ready = asyncio.Event()                 # 沿用你的 ready 機制
        

        # Command Socket
        info_address = (host_ip, self.info_port)
        self.info_server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.info_server.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.buffer_size)
        self.info_server.bind(info_address)
        
        keyboard_address = (host_ip, self.keyboard_port)
        self.keyboard_server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.keyboard_server.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.buffer_size)
        self.keyboard_server.bind(keyboard_address)

        print('Initial Server OK')
        print(f'Video (TCP) Destination: {VIDEO_IP_}:{VIDEO_PORT_}')
        print('Info Address:', info_address)
        print('Keyboard Address:', keyboard_address)
        print()

        self.ultra_kwargs = dict(
            persist=True,
            tracker=self.tracker_yaml_path,
            conf=0.25,
            iou=0.5,
            classes=[0],     # 只抓 person，少很多後處理
            max_det=30,      # 限制每幀最多 30 個框（視場景調）
            imgsz=TRT_IMGSZ_,       # ↓ 減少輸入長邊，有感加速：512/448/416 自行取捨
            device=0,        # 指定 GPU
            verbose=False
        )

        # Camera and DPU
        self.init_camera()
        
        # Drone Command
        self.init_drone_master()


        # === Auto-Tracking Controller ===
        self.ctrl = AutoTrackController(
            bias_y=0.12,
            lock_tol_x=0.05, lock_tol_y=0.05,
            min_box=80, lock_box=100, max_box=120,
            
            pid_vx =(0.06, 0.0, 0.02, 0.3),        
            pid_z  =(0.35, 0.0, 0.06, 0.5),
            pid_yaw=(0.50, 0.0, 0.10, 0.5),
            
            
            clip_vx=0.5, clip_vz=0.5, clip_yaw=0.35,
            min_vx=0.05, min_vz=0.05, min_yaw=0.05,
           
           pid_sample_time=None,  
        )
        self._last_control_ts = None  # for dt

        # === CSV logger for control commands ===
        self.csv_path = "autotrack_cmds.csv"
        new_file = not os.path.exists(self.csv_path)
        self._csv_f = open(self.csv_path, "w", newline="", encoding="utf-8")
        self._csv_w = csv.writer(self._csv_f)
        if new_file:
            self._csv_w.writerow(["t","target_id","vx","vy","vz","yaw_rate"])
            self._csv_f.flush()
       

    def init_camera(self):
        print('Connecting camera ...', end='')

        # 先把舊的相機資源釋放（避免重複初始化時卡死）
        try:
            if hasattr(self, "camera") and self.camera is not None:
                self.camera.release()
        except Exception:
            pass

        self.camera = None

        cam = cv2.VideoCapture(2, cv2.CAP_ANY)
        if cam.isOpened():
            
            # 試讀一幀確認真的有畫面
            ok, frame = cam.read()
            if ok and frame is not None:
                self.camera = cam
                print('OK (device index 2)')
                return
            else:
                cam.release()

    def init_drone_master(self):
        print('Connecting drone ...', end='')
        # self.drone_master = mavutil.mavlink_connection('/dev/fcu', baud=115200)
        self.drone_master = mavutil.mavlink_connection('udpin:0.0.0.0:14551', autoreconnect=True)
        # self.drone_master = mavutil.mavlink_connection('/dev/ttyUSB0', baud=115200)
        # self.drone_master = mavutil.mavlink_connection('COM3', baud=57600)
        self.drone_command = DroneCommand(self.drone_master)
        recv_command = {0: 'HEARTBEAT', 24: 'GPS_RAW_INT', 27: 'RAW_IMU', 30: 'ATTITUDE', 33: 'GLOBAL_POSITION_INT'}
        for command in recv_command.keys():
            send_msg = self.drone_master.mav.command_long_encode(self.drone_master.target_system,
                                                                 self.drone_master.target_component,
                                                                 mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                                                                 0, command, 500000, 0, 0, 0, 0, 0)
            self.drone_master.mav.send(send_msg)
        print('OK')
    
    def tracking_control(self, frame, tracker):
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                    continue
            bbox = track.to_tlbr()
            w = int((int(bbox[2]) - int(bbox[0]))/2)
            h = int((int(bbox[3]) - int(bbox[1]))/2)
            center_x = int(bbox[0]) + w
            center_y = int(bbox[1]) + h

            if self.target_id is None:
                if center_x > self.tracking_box[0] and center_x < self.tracking_box[2] and \
                    center_y > self.tracking_box[1] and center_y < self.tracking_box[3]:
                    self.target_id = track.track_id
                    print('Target ID: ', self.target_id)
                    break

            if len(tracker.tracks) == 1:
                self.target_id = track.track_id

            if track.track_id == self.target_id:
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
                cv2.putText(frame, str(track.track_id), 
                                (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.circle(frame, (int(bbox[0]) + w, int(bbox[1]) + h), 5, (0, 0, 0), 2)
                cv2.line(frame, (208, 208), (int(bbox[0]) + w, int(bbox[1]) + h), (0, 0, 0), 2)

                if center_x < self.tracking_box[0] or center_x > self.tracking_box[2]:
                    bais_x = center_x - 208
                    print('Bais X: ', bais_x)

                    if bais_x > 0:
                        print('Turn Right')
                    elif bais_x < 0:
                        print('Turn Left')
        return frame

    def _ultra_track_once(self, frame):
        """
        呼叫 Ultralytics 內建 BoT-SORT (ReID) 追蹤一幀。
        回傳介面與你原本一致：
        {'tracks': [{'track_id': int, 'tlwh': [x,y,w,h]}, ...],
         'preproc_frame': frame}
        """
        results = self.yolo.track(frame, **self.ultra_kwargs)[0]

        tracks = []
        if results.boxes is not None and len(results.boxes) > 0:
            ids  = results.boxes.id        # (N,) 追蹤 ID（可能為 None）
            clss = results.boxes.cls       # (N,) 類別 id
            xyxy = results.boxes.xyxy      # (N,4)
            # 轉成 numpy
            ids_np  = ids.cpu().numpy().astype(int)  if ids  is not None else None
            cl_np   = clss.cpu().numpy().astype(int) if clss is not None else None
            xyxy_np = xyxy.cpu().numpy()
            for i in range(xyxy_np.shape[0]):
                # 僅保留 person（等效你原本 person_only=True）
                if cl_np is not None and cl_np[i] != PERSON_CLASS_ID:
                    continue
                x1, y1, x2, y2 = xyxy_np[i]
                tlwh = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
                tid  = int(ids_np[i]) if ids_np is not None else -1
                tracks.append({"track_id": tid, "tlwh": tlwh})
        return {"tracks": tracks, "preproc_frame": frame}


    async def warmup_tracker(self):
        loop = asyncio.get_event_loop()
        print('Initialing Ultralytics YOLO + BoT-SORT (background warmup)...', flush=True)

        def _load_yolo():
            # 優先使用 TensorRT / ONNX，再退回 .pt
            model_name = MODEL_NAME_
            base_dir = os.path.join(os.path.dirname(__file__), "model")  # 修正資料夾名稱
            imgsz = TRT_IMGSZ_
            candidates = [
                f"{model_name}_{imgsz}.engine",
                f"{model_name}.engine",
                f"{model_name}_{imgsz}.onnx",
                f"{model_name}.onnx",
                f"{model_name}.pt",
            ]
            for fname in candidates:
                fpath = os.path.join(base_dir, fname)
                if os.path.exists(fpath):
                    print(f"[YOLO] loading model: {fpath}")
                    return YOLO(fpath)
            raise FileNotFoundError(
                f"No model file found. Tried: {', '.join(candidates)} under {base_dir}"
            )

        self.yolo = await loop.run_in_executor(None, _load_yolo)

        # 先做一次 dummy 推論把引擎/權重與追蹤器建起來
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)

        _ = self.yolo.track(dummy, **self.ultra_kwargs)
        self.tracker_ready.set()
        print('YOLO tracker ready.', flush=True)     


    async def run_camera(self):
        loop = asyncio.get_event_loop()

        if not hasattr(self, "camera") or self.camera is None:
            self.init_camera()

        interval = 1.0 / max(1, TARGET_FPS)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]

        def pack_msg(msg_type: int, payload: bytes) -> bytes:
            # 1 byte type + 4 bytes big-endian length
            return struct.pack(">BI", msg_type, len(payload)) + payload

        while True:
            sock = None
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                sock.setblocking(False)

                print(f"[TCP] Connecting to {VIDEO_IP_}:{VIDEO_PORT_} ...")
                await loop.sock_connect(sock, (VIDEO_IP_, VIDEO_PORT_))
                print("[TCP] Connected")

                last_send = time.time()

                while True:
                    # === 每幀更新 FPS（1秒窗） ===
                    self._fps_counter += 1
                    _now_perf = time.perf_counter()
                    elap = _now_perf - self._fps_last_t
                    if elap >= 1.0:
                        self._fps_value = self._fps_counter / elap
                        self._fps_counter = 0
                        self._fps_last_t = _now_perf

                    # --- 抓影像 ---
                    ret, frame = self.camera.read()
                    if not ret or frame is None:
                        print('No Frame, reinit camera...')
                        try: self.camera.release()
                        except: pass
                        await asyncio.sleep(1)
                        self.init_camera()
                        continue
                    frame = cv2.flip(frame, -1)  # 180度翻轉

                    # --- 追蹤計時 ---
                    trk_ms = 0.0
                    if self.yolo is not None and self.tracker_ready.is_set():
                        _t0 = time.perf_counter()
                        out = self._ultra_track_once(frame)
                        trk_ms = (time.perf_counter() - _t0) * 1000.0
                        raw_tracks = out['tracks']
                        send_frame = out.get('preproc_frame', frame)
                    else:
                        raw_tracks = []
                        send_frame = frame

                    await asyncio.sleep(0)

                    # === 依鎖定目標決定送哪些框 ===
                    now_ts = time.monotonic()
                    if self.target_id is not None:
                        found = next((t for t in raw_tracks if int(t['track_id']) == int(self.target_id)), None)
                        if found is not None:
                            self._last_seen_ts = now_ts
                            use_tracks = [found]
                        else:
                            if self._last_seen_ts is not None and (now_ts - self._last_seen_ts) > self._lost_timeout:
                                # 通知 PC：目標丟失
                                evt = json.dumps({"event":"target_lost", "id": int(self.target_id)}).encode('utf-8')
                                await loop.sock_sendall(sock, struct.pack(">BI", MSG_EVENT, len(evt)) + evt)

                                print('[Track] Target lost -> stop AutoTracking & clear lock')
                                self.target_id = None
                                self.tracking_flag = False
                                self._last_seen_ts = None

                                self.ctrl.reset()

                                # 關閉 AutoTracking 並 flush CSV
                                self.AutoTracking_flag = False
                                try: self._csv_f.flush()
                                except: pass

                                use_tracks = raw_tracks   # 追丟當幀起恢復全部框
                            else:
                                use_tracks = []
                    else:
                        use_tracks = raw_tracks

                    # === 疊字顯示 FPS / 延遲（在 encode 前畫上） ===
                    if SHOW_FPS_OVERLAY:
                        # txt = f"FPS:{self._fps_value:.1f}  TRT:{trk_ms:.1f}ms  ENC:{self._last_enc_ms:.1f}ms"
                        txt = f"FPS:{self._fps_value:.1f}"
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        scale = 0.7
                        thick = 2
                        color = (0, 255, 0)
                        margin = 10

                        (tw, th), baseline = cv2.getTextSize(txt, font, scale, thick)
                        H, W = send_frame.shape[:2]

                        # 右上
                        x = W - tw - margin
                        y = margin + th  # 由上方算 baseline


                        cv2.putText(send_frame, txt, (x, y), font, scale, color, thick, cv2.LINE_AA)

                    # === 鎖定模式中心紅色十字（瞄準心；包含 bias_y 偏移） ===
                    # if self.tracking_flag and (self.target_id is not None):
                    if self.target_id is not None:
                        H, W = send_frame.shape[:2]
                        fx, fy = W * 0.5, H * 0.5

                        cx = int(fx)
                        # 將十字中心往下偏移 bias_y（例：0.12 代表比畫面中線再往下 12% 畫面高度）
                        cy_expect = int(round(fy * (1.0 + float(self.ctrl.bias_y))))
                        # 防越界
                        cy_expect = max(0, min(H - 1, cy_expect))

                        size = max(8, min(W, H) // 40)  # 十字半長度
                        thickness = 2
                        color = (0, 0, 255)             # BGR: 紅色

                        # 橫線
                        cv2.line(send_frame, (cx - size, cy_expect), (cx + size, cy_expect), color, thickness, cv2.LINE_AA)
                        # 直線
                        cv2.line(send_frame, (cx, cy_expect - size), (cx, cy_expect + size), color, thickness, cv2.LINE_AA)
                        # 中心小點（可選）
                        # cv2.circle(send_frame, (cx, cy_expect), 2, color, -1, cv2.LINE_AA)


                    # === 自動追蹤控制 ===
                    if self.AutoTracking_flag and self.tracking_flag and (self.target_id is not None):
                        H, W = send_frame.shape[:2]

                        # --- draw lock box (100x100) centered at (fx, fy*(1+bias_y)) ---
                        fx, fy = W * 0.5, H * 0.5
                        cy_expect = fy * (1.0 + float(self.ctrl.bias_y))
                        box_size = int(round(getattr(self.ctrl, "lock_box", 100.0)))  # 預設畫 100x100
                        half = box_size // 2

                        x1 = max(0, int(fx) - half)
                        y1 = max(0, int(cy_expect) - half)
                        x2 = min(W - 1, int(fx) + half)
                        y2 = min(H - 1, int(cy_expect) + half)
                        # cv2.rectangle(send_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 綠框
                        draw_dashed_rect(send_frame, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2, dash=10, gap=6)  # 紅色虛線

                        # 只針對當前鎖定 target_id 控制
                        tgt = next((t for t in raw_tracks if int(t["track_id"]) == int(self.target_id)), None)

                        # dt 計算
                        now_ct = time.monotonic()
                        dt = None if (self._last_control_ts is None) else (now_ct - self._last_control_ts)
                        self._last_control_ts = now_ct

                        if tgt is not None:
                            tlwh = tuple(map(float, tgt["tlwh"]))
                            vx, vy, vz, yaw_rate = self.ctrl.compute(tlwh, (W, H), dt=dt)
                            vy = 0.0  # 車子式：不做側移

                            # 單一 MAVLink 命令：vx, vy, vz, yaw_rate
                            try:
                                self.drone_command.set_velocity_yawrate(vx, vy, vz, yaw_rate)
                            except Exception as e:
                                print("[AutoTrack] set_velocity_yawrate error:", e)

                            # 寫 CSV
                            self._csv_w.writerow([
                                f"{now_ct:.6f}", int(self.target_id),
                                f"{vx:.6f}", f"{vy:.6f}", f"{vz:.6f}", f"{yaw_rate:.6f}"
                            ])
                            self._csv_f.flush()
                        else:
                            # 這幀看不到目標：重置控制器（避免積分）
                            self.ctrl.reset()

                    
                    # 1) JPEG 影像（計算編碼時間）
                    _e0 = time.perf_counter()
                    ok, jpg = cv2.imencode('.jpg', send_frame, encode_param)
                    enc_ms = (time.perf_counter() - _e0) * 1000.0
                    if not ok:
                        await asyncio.sleep(0); continue
                    jpg_bytes = jpg.tobytes()
                    await loop.sock_sendall(sock, struct.pack(">BI", MSG_JPEG, len(jpg_bytes)) + jpg_bytes)

                    # 記錄本幀的編碼時間（下幀疊字會顯示最新值）
                    self._last_trk_ms = trk_ms
                    self._last_enc_ms = enc_ms

                    # 2) META（加上 fps 與延遲資訊）
                    meta = {
                        "w": int(send_frame.shape[1]),
                        "h": int(send_frame.shape[0]),
                        "tracks": [
                            {"id": int(t["track_id"]),
                             "tlwh": [int(t["tlwh"][0]), int(t["tlwh"][1]),
                                      int(t["tlwh"][2]), int(t["tlwh"][3])]}
                            for t in use_tracks
                        ],
                    }
                    meta_bytes = json.dumps(meta, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
                    await loop.sock_sendall(sock, struct.pack(">BI", MSG_META, len(meta_bytes)) + meta_bytes)

                    # 控制節奏
                    sleep_left = interval - (time.time() - last_send)
                    if sleep_left > 0:
                        await asyncio.sleep(min(sleep_left, 0.050))
                    # 無論如何都讓出一次，確保公平排程
                    await asyncio.sleep(0)
                    last_send = time.time()
            except asyncio.CancelledError:
                break
            except ConnectionRefusedError:
                await asyncio.sleep(5)
            except (OSError, ConnectionResetError):
                await asyncio.sleep(5)
            finally:
                if sock:
                    try: sock.close()
                    except: pass



    async def send_data(self):
        loop = asyncio.get_event_loop()

        print('Waiting for info socket...')
        msg, info_client_address = await loop.run_in_executor(None, self.info_server.recvfrom, self.buffer_size)
        print('Info Server Received:', msg.decode(), 'from', info_client_address)

        print("Start sending data...")
        while True:
            drone_msg = await loop.run_in_executor(None, self.drone_master.recv_msg)
            if drone_msg is not None:
                if drone_msg.get_type() == 'ATTITUDE':
                    data = {'get_type': drone_msg.get_type(),'roll': drone_msg.roll, 'pitch': drone_msg.pitch, 'yaw': drone_msg.yaw,
                            'rollspeed': drone_msg.rollspeed, 'pitchspeed': drone_msg.pitchspeed, 'yawspeed': drone_msg.yawspeed}
                elif drone_msg.get_type() == 'GLOBAL_POSITION_INT':
                    data = {'get_type': drone_msg.get_type(), 'lat': drone_msg.lat, 'lon': drone_msg.lon, 'alt': drone_msg.alt}
                elif drone_msg.get_type() == 'GPS_RAW_INT':
                    data = {'get_type': drone_msg.get_type(), 'eph': drone_msg.eph, 'satellites_visible': drone_msg.satellites_visible}
                elif drone_msg.get_type() == 'HEARTBEAT' and drone_msg.type == 2:
                    data = {'get_type': drone_msg.get_type(), 'custom_mode': drone_msg.custom_mode, 'base_mode': drone_msg.base_mode}
                else:
                    continue

                payload = json.dumps(data, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
                await loop.run_in_executor(None, self.info_server.sendto, payload, info_client_address)
                # await asyncio.sleep(0.005)

    async def receive_command(self):
        loop = asyncio.get_event_loop()

        print('Waiting for keyboard socket...')
        msg, keyboard_address = await loop.run_in_executor(None, self.keyboard_server.recvfrom, self.buffer_size)
        print('Keyboard Server Received:', msg.decode(), 'from', keyboard_address)

        print("Start receiving command...")
        while True:
            data = await loop.run_in_executor(None, self.keyboard_server.recv, self.buffer_size)
            data = data.decode()
            print('Keyboard Server Received:', data)
            if data == 'close':
                # 關閉 AutoTracking + flush CSV
                self.AutoTracking_flag = False
                try: self._csv_f.flush()
                except: pass
                self.camera_task.cancel()
                self.send_task.cancel()
                break

            if data == 'Forward':
                self.drone_command.set_position_velocity(0.5, 0, 0)
            elif data == 'Backward':
                self.drone_command.set_position_velocity(-0.5, 0, 0)
            elif data == 'Left':
                self.drone_command.set_position_velocity(0, -0.5, 0)
            elif data == 'Right':
                self.drone_command.set_position_velocity(0, 0.5, 0)
            elif data == 'Raise':
                self.drone_command.set_position_velocity(0, 0, -0.2)
            elif data == 'Drop':
                self.drone_command.set_position_velocity(0, 0, 0.2)
            elif data == 'Y_Right':
                self.drone_command.turning_yaw(10, 1)
            elif data == 'Y_Left':
                self.drone_command.turning_yaw(10, -1)
            elif data == 'Takeoff':
                self.drone_command.takeoff(3)
            elif data == 'Land':
                self.drone_command.land()
                # 視為取消 → 關 AutoTracking + flush CSV
                self.AutoTracking_flag = False
                try: self._csv_f.flush()
                except: pass
            elif data == 'Arm':
                self.drone_command.select_mode(4)
                self.drone_command.arm_disarm(1)
            elif data == 'Disarm':
                self.drone_command.arm_disarm(0)
                # 視為取消 → 關 AutoTracking + flush CSV
                self.AutoTracking_flag = False
                try: self._csv_f.flush()
                except: pass
            elif data == 'Start Tracking':
                # 僅開旗標；實際是否執行要有 tracking_flag=True + target_id
                self.AutoTracking_flag = True
            elif data == 'Stop Tracking':
                # 取消 → 關 AutoTracking + flush CSV
                self.AutoTracking_flag = False
                try: self._csv_f.flush()
                except: pass
            elif data.startswith('SelectID:'):
                try:
                    self.target_id = int(data.split(':', 1)[1])
                    self.tracking_flag = True
                    self._last_seen_ts = time.monotonic()
                    print('[Control] Lock target_id =', self.target_id)
                except Exception:
                    pass
            elif data == 'Reset Selection':
                # 視為取消 → 關 AutoTracking + flush CSV
                self.tracking_flag = False
                self.target_id = None
                self._last_seen_ts = None
                self.AutoTracking_flag = False
                try: self._csv_f.flush()
                except: pass
                print('[Control] Reset selection: unlock target/stop tracking lock')

            await asyncio.sleep(0.01)
        print('Finish receiving command.')
    
    async def run(self):
        # Run video sending concurrently
        warmup_task = asyncio.create_task(self.warmup_tracker())
        self.camera_task = asyncio.create_task(self.run_camera())
        self.receive_task = asyncio.create_task(self.receive_command())
        self.send_task = asyncio.create_task(self.send_data())
        await asyncio.gather(self.camera_task, self.receive_task, self.send_task, warmup_task)
        print("Finish sending video.")

if __name__ == '__main__':
    udp_hub = MavUdpHub(
        serial_dev=config.serial_dev,
        baud=config.baud,
        peers=config.peers,
    )
    udp_hub.start()
    print("[MAV-UDP-HUB] Started")

    main_server = MainServer()
    try:
        asyncio.run(main_server.run())
    except KeyboardInterrupt:
        main_server.camera.release()
        print("KeyboardInterrupt by user.")
    except asyncio.exceptions.CancelledError:
        main_server.camera.release()
        print("Finish server socket.")
    finally:
        try: main_server._csv_f.flush(); main_server._csv_f.close()
        except: pass
