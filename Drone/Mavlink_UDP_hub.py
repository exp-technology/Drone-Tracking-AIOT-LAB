# --- MAVLink UDP Hub (drop-in) ---
import threading, socket, select, serial, time

class MavUdpHub:
    """
    Bridge raw MAVLink bytes between serial (/dev/fcu) and multiple UDP peers.
    Each peer uses one UDP socket bound to a *local* port; we send to the peer's
    listening port, and receive replies on the same bound port.
    """
    def __init__(self,
                 serial_dev="/dev/fcu",
                 baud=115200,
                 peers=None,         # list of dicts: {"name": "mavros", "dst": ("127.0.0.1", 14550), "bind": ("127.0.0.1", 14650)}
                 ser_read_chunk=512):
        assert peers, "peers required"
        self.ser = serial.Serial(serial_dev, baudrate=baud, timeout=0)
        self.ser_read_chunk = ser_read_chunk
        self.stop = threading.Event()

        self.socks = []
        for p in peers:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(p["bind"])              # 固定本機來源埠，讓對端能回傳
            s.setblocking(False)
            self.socks.append({"name": p["name"], "sock": s, "dst": p["dst"]})

        self.t1 = threading.Thread(target=self._pump_serial_to_udp, daemon=True)
        self.t2 = threading.Thread(target=self._pump_udp_to_serial, daemon=True)

    def start(self):
        print("[UDP-HUB] start; serial={}, peers={}".format(self.ser.port, [(p["name"], p["dst"], p["sock"].getsockname()) for p in self.socks]))
        self.t1.start(); self.t2.start()

    def close(self):
        self.stop.set()
        try: self.t1.join(timeout=1)
        except: pass
        try: self.t2.join(timeout=1)
        except: pass
        for p in self.socks:
            try: p["sock"].close()
            except: pass
        try: self.ser.close()
        except: pass
        print("[UDP-HUB] closed")

    # 串口 → UDP 廣播
    def _pump_serial_to_udp(self):
        while not self.stop.is_set():
            try:
                n = self.ser.in_waiting
                data = self.ser.read(self.ser_read_chunk if n == 0 else min(n, 4096))
                if data:
                    for p in self.socks:
                        try:
                            p["sock"].sendto(data, p["dst"])
                        except OSError:
                            pass
                else:
                    time.sleep(0.001)
            except Exception as e:
                # 串口短暫中斷時稍等再試
                time.sleep(0.05)

    # UDP → 串口（雙向回寫）
    def _pump_udp_to_serial(self):
        socks = [p["sock"] for p in self.socks]
        while not self.stop.is_set():
            try:
                r, _, _ = select.select(socks, [], [], 0.05)
                for s in r:
                    try:
                        data, addr = s.recvfrom(4096)
                        if data:
                            self.ser.write(data)
                    except BlockingIOError:
                        pass
                    except OSError:
                        pass
            except Exception:
                time.sleep(0.01)
