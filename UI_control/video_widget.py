'''
增加BBox選框功能
'''
# video_widget_v3.py  (Py38-friendly)
from PyQt6 import QtWidgets, QtCore, QtGui
import cv2, os, socket, struct, json, numpy as np, time
from typing import Optional, Tuple, Dict, List

# 協定常數（同條 TCP 多路）
MSG_JPEG = 1  # 影像
MSG_META = 2  # 追蹤框 JSON
MSG_EVENT = 3 # 事件（ex: target_lost）

# --- 可點擊的顯示面板 ---
class _OverlayLabel(QtWidgets.QLabel):
    clicked = QtCore.pyqtSignal(int, int)  # x, y in widget coords
    def mousePressEvent(self, e: QtGui.QMouseEvent):
        if hasattr(e, "position"):
            pos = e.position()
            x, y = int(pos.x()), int(pos.y())
        else:
            x, y = e.x(), e.y()
        self.clicked.emit(x, y)
        super().mousePressEvent(e)

# === TCP 接收執行緒（PC 當 server，等 Jetson 連進來） ===
class TcpReceiverThread(QtCore.QThread):
    frame_trigger  = QtCore.pyqtSignal(object)  # emit BGR ndarray
    tracks_trigger = QtCore.pyqtSignal(dict)    # {'w':int,'h':int,'tracks':[{'id':..,'tlwh':[x,y,w,h]},...]}
    event_trigger  = QtCore.pyqtSignal(dict)    # ex: {'event':'target_lost','id':17}

    def __init__(self, bind_ip: str, port: int, parent=None):
        super().__init__(parent)
        self.bind_ip = bind_ip
        self.port = port
        self._running = True
        self._server: Optional[socket.socket] = None
        self._conn: Optional[socket.socket] = None

    
    def _recv_exact(self, n: int) -> Optional[bytes]:
        buf = b""
        while len(buf) < n and self._running:
            try:
                chunk = self._conn.recv(n - len(buf))  # type: ignore
                if not chunk:            # 對端關閉
                    return None
                buf += chunk
            except socket.timeout:
                # 不要當錯誤處理；繼續等資料（可小睡避免忙等）
                QtCore.QThread.msleep(10)
                continue
            except OSError:
                return None
        return buf

    def _read_one_message(self) -> Optional[Tuple[int, bytes]]:
        """
        回傳 (msg_type, payload)
        - 新協定：1 byte type + 4 bytes len
        - 舊協定：4 bytes len（只有 JPEG），自動視為 (MSG_JPEG, payload)
        """
        b0 = self._recv_exact(1)
        if not b0:
            return None
        t0 = b0[0]

        if t0 in (MSG_JPEG, MSG_META, MSG_EVENT):
            blen = self._recv_exact(4)
            if not blen:
                return None
            (length,) = struct.unpack(">I", blen)
            if length <= 0 or length > 50_000_000:
                return None
            payload = self._recv_exact(length)
            if payload is None:
                return None
            return (t0, payload)

        # 舊協定 fallback：b0 是長度第 1 byte，再讀 3 byte 補齊
        b123 = self._recv_exact(3)
        if not b123:
            return None
        (length,) = struct.unpack(">I", b0 + b123)
        if length <= 0 or length > 50_000_000:
            return None
        payload = self._recv_exact(length)
        if payload is None:
            return None
        return (MSG_JPEG, payload)

    def run(self):
        try:
            self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # self._conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
            self._server.bind((self.bind_ip, self.port))
            self._server.listen(1)
            print(f"\n[TCP] Listening on {self.bind_ip}:{self.port} ...")
            self._server.settimeout(1.0)

            while self._running:
                try:
                    self._conn, addr = self._server.accept()
                except socket.timeout:
                    continue
                print("[TCP] Client connected:", addr)
                self._conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self._conn.settimeout(2.0)

                try:
                    while self._running:
                        msg = self._read_one_message()
                        if msg is None:
                            break
                        mtype, payload = msg

                        if mtype == MSG_JPEG:
                            frame = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
                            if frame is not None:
                                self.frame_trigger.emit(frame)

                        elif mtype == MSG_META:
                            try:
                                meta = json.loads(payload.decode('utf-8', 'ignore'))
                                if isinstance(meta, dict) and 'tracks' in meta:
                                    self.tracks_trigger.emit(meta)
                            except Exception:
                                pass

                        elif mtype == MSG_EVENT:
                            try:
                                evt = json.loads(payload.decode('utf-8','ignore'))
                                if isinstance(evt, dict):
                                    self.event_trigger.emit(evt)
                            except Exception:
                                pass

                except Exception as e:
                    print("[TCP] Error:", e)
                finally:
                    if self._conn:
                        try: self._conn.close()
                        except: pass
                        self._conn = None
                    print("[TCP] Client disconnected")
        finally:
            if self._server:
                try: self._server.close()
                except: pass
                self._server = None
            print("[TCP] Server stopped")

    def stop(self):
        self._running = False
        try:
            if self._conn:
                self._conn.shutdown(socket.SHUT_RDWR)
        except:
            pass

# === UI + 覆繪 + 點選 ===
class VideoWidget(QtWidgets.QWidget):
    target_selected = QtCore.pyqtSignal(int)
    target_lost     = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        self.pwd = os.path.dirname(os.path.abspath(__file__))

        self.video_frame = _OverlayLabel('Video Widget')
        self.video_frame.setFixedSize(640, 480)
        self.video_frame.clicked.connect(self._on_clicked)

        layout = QtWidgets.QGridLayout()
        layout.addWidget(self.video_frame, 0, 0, 1, 1)
        layout.setContentsMargins(0,0,0,0)
        self.setLayout(layout)

        # 狀態
        self._last_bgr: Optional[np.ndarray] = None
        self._img_wh = (640, 480)
        self._tracks: List[Dict[str, object]] = []
        self._select_mode = False
        self._locked_id: Optional[int] = None

        # 追丟橫幅顯示截止時間（monotonic 秒）
        self._lost_banner_deadline: Optional[float] = None

        self._rx_thread: Optional[TcpReceiverThread] = None

        self.no_video()
        print('(Widget) Video Widget Initialized')

    def init(self, host_ip, port, _ignored):
        if self._rx_thread and self._rx_thread.isRunning():
            self._rx_thread.stop()
            self._rx_thread.wait(500)

        bind_ip = "0.0.0.0"
        self._rx_thread = TcpReceiverThread(bind_ip=bind_ip, port=port)
        self._rx_thread.frame_trigger.connect(self.update_frame)
        self._rx_thread.tracks_trigger.connect(self.update_tracks)
        self._rx_thread.event_trigger.connect(self._on_event)
        self._rx_thread.start()
        print(f'(Widget)(Video) TCP server started on {bind_ip}:{port}')

    def pause(self):
        if self._rx_thread and self._rx_thread.isRunning():
            print('(Socket)(Video) Video Socket Paused')
            self._rx_thread.stop()
            self._rx_thread.wait(500)

    def close(self):
        print('(Widget) Video Widget Closed')
        self.no_video()
        if self._rx_thread and self._rx_thread.isRunning():
            self._rx_thread.stop()
            self._rx_thread.wait(500)

    # 外部 API：開 / 關 選取模式
    def enable_select_mode(self, enable=True):
        self._select_mode = enable
        self._render()

    # 外部 API：重置（退出選取 + 清掉當前鎖定）
    def reset_selection(self):
        self._locked_id = None
        self._select_mode = False
        self._render()

    # slots
    @QtCore.pyqtSlot(object)
    def update_frame(self, frame_bgr):
        if frame_bgr is None:
            return
        self._last_bgr = frame_bgr
        self._img_wh = (frame_bgr.shape[1], frame_bgr.shape[0])
        self._render()

    @QtCore.pyqtSlot(dict)
    def update_tracks(self, meta: Dict):
        try:
            w = int(meta.get('w', self._img_wh[0])); h = int(meta.get('h', self._img_wh[1]))
            self._img_wh = (w, h)
            tlist: List[Dict[str, object]] = []
            for t in meta.get('tracks', []):
                tid = int(t.get('id'))
                x,y,ww,hh = [int(v) for v in t.get('tlwh', [0,0,0,0])]
                tlist.append({'id': tid, 'tlwh': [x,y,ww,hh]})
            self._tracks = tlist

            if self._locked_id is not None:
                self._tracks = [t for t in self._tracks if int(t['id']) == int(self._locked_id)]
        except Exception:
            self._tracks = []
        self._render()  # ← 只呼叫一次

    @QtCore.pyqtSlot(dict)
    def _on_event(self, evt: dict):
        if evt.get("event") == "target_lost":
            print("[Video] Target lost event from server")
            self._locked_id = None
            self._lost_banner_deadline = time.monotonic() + 2.5  # 顯示 2.5 秒
            self.enable_select_mode(True)  # 回到可選取
            self.target_lost.emit()        # 通知上層（控制面板改 UI）
            self._render()

    def _on_clicked(self, x: int, y: int):
        if not self._select_mode or self._last_bgr is None:
            return
        W, H = self.video_frame.width(), self.video_frame.height()  # 640x480
        iw, ih = self._img_wh
        px = int(x * iw / max(1, W))
        py = int(y * ih / max(1, H))
        for t in self._tracks:
            tx, ty, tw, th = t['tlwh']  # type: ignore
            if tx <= px <= tx+tw and ty <= py <= py+th:  # type: ignore
                self._select_mode = False
                self._locked_id = int(t['id'])
                self.target_selected.emit(self._locked_id)
                break
        self._render()

    def _render(self):
        if self._last_bgr is None:
            self.no_video()
            return

        canvas = cv2.resize(self._last_bgr, (640, 480))
        iw, ih = self._img_wh
        sx = 640 / max(1, iw)
        sy = 480 / max(1, ih)

        for t in self._tracks:
            x,y,w,h = t['tlwh']  # type: ignore
            rx1 = int(x * sx); ry1 = int(y * sy)
            rx2 = int((x+w) * sx); ry2 = int((y+h) * sy)
            cv2.rectangle(canvas, (rx1, ry1), (rx2, ry2), (0,255,0), 2)
            cv2.putText(canvas, f"ID:{t['id']}", (rx1, max(12, ry1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # 選取提示
        if self._select_mode:
            cv2.putText(canvas, "Click to select target",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        # 追丟橫幅（2.5s）
        if self._lost_banner_deadline is not None:
            if time.monotonic() < self._lost_banner_deadline:
                cv2.rectangle(canvas, (0,0), (640,40), (0,0,0), -1)
                cv2.putText(canvas, "target lost, click to select another",
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            else:
                self._lost_banner_deadline = None

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, ch*w, QtGui.QImage.Format.Format_RGB888)
        self.video_frame.setPixmap(QtGui.QPixmap.fromImage(qimg))

    def no_video(self):
        path = os.path.join(self.pwd, 'no_videos_available.jpg')
        if os.path.exists(path):
            image = cv2.imread(path)
        else:
            image = np.full((480, 640, 3), 40, np.uint8)
            cv2.putText(image, "No Video", (220, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2, cv2.LINE_AA)
        self.update_frame(image)


