from PyQt6 import QtWidgets, QtCore
from Widge_Component import Label_Component

import socket
import asyncio
import json
import math


class MessageWidget(QtWidgets.QWidget):
    flight_mode_def = {0: 'Stabilize', 2: 'AltHold', 4: 'Guided', 5: 'Loiter', 9: 'Land'}
    recv_command = {0: 'HEARTBEAT', 24: 'GPS_RAW_INT', 27: 'RAW_IMU', 30: 'ATTITUDE', 33: 'GLOBAL_POSITION_INT'}
    # 74: 'VFR_HUD', 76: 'COMMAND_ACK'}
    time_step = 100000

    read_task = None

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Drone Information')

        # Label's Variable
        self.flight_mode = 0
        self.GPS_HDOP = 0
        self.satellites_visible = 0
        self.attitude = [0, 0, 0, 0, 0, 0]
        self.GPS_int = [0, 0, 0]
        data = None

        # Information Layout
        info_widget = QtWidgets.QWidget()
        info_widget.setStyleSheet('background-color:rgb(176, 185, 182); border-radius:10px;')
        info_layout = QtWidgets.QGridLayout(info_widget)
        self.label_1 = Label_Component('Flight Mode:', 18)  # Flight Mode, GPS HDOP, Satellites Visible
        self.label_2 = Label_Component('Yaw(deg)', 18)  # Speed, Yaw(deg), Pitch(deg), Roll(deg)
        self.label_3 = Label_Component('GPS Altitude:', 18)  # Altitude, Latitude, Longitude, Relative Altitude
        self.label_4 = Label_Component('Pitch Speed(m/s):', 18)  # Pitch Speed, Roll Speed, Yaw Speed

        info_layout.addWidget(self.label_1, 0, 0, 1, 1)
        info_layout.addWidget(self.label_2, 0, 1, 1, 1)
        # info_layout.addWidget(self.label_3, 0, 2, 1, 1)
        info_layout.addWidget(self.label_4, 0, 2, 1, 1)

        # All Layout
        all_layout = QtWidgets.QGridLayout()
        all_layout.addWidget(info_widget, 0, 0, 1, 1)

        self.setLayout(all_layout)
        self.label_init()

        # Message Socket
        self.message_socket = MessageSocket()
        self.message_task = None
        self.message_socket.update_label.connect(self.update_label)

        print('(Widget)(Message) Message Widget Initialized')
    
    def init(self, host_ip, port, buffer_size):
        self.host_ip = host_ip
        self.port = port
        self.buffer_size = buffer_size
        self.message_socket.init(host_ip, port, buffer_size)
        self.message_socket.connect()
        self.message_socket.connect_event.clear()
        
        if self.message_task is None or self.message_task.done() or self.message_task.cancelled():
            # print('(Widget)(Video) Video Task Created')
            self.message_task = asyncio.create_task(self.message_socket.receiving_message())
    
    def pause(self):
        print('(Widget)(Message) Message Socket Paused')
        self.message_socket.message_client.sendto(b'Pause System', (self.host_ip, self.port))
        self.message_socket.connect_event.set()
    
    def close(self):
        print('(Widget)(Message) Message Widget Closed')
        self.label_init()
        print("(Widget)(Message) Closing message socket...")
        self.message_socket.message_client.sendto(b'Stop System', (self.host_ip, self.port))
        if self.message_task:
            self.message_task.cancel()
        self.message_socket.close()
    
    def label_init(self):
        label_1_text = (f'Flight Mode: {0}\n'
                        f'GPS HDOP: {0}\n'
                        f'Satellites Visible: {0}')
        self.label_1.setText(label_1_text)
        label_2_text = (f'Roll(deg): {0}\n'
                        f'Pitch(deg): {0}\n'
                        f'Yaw(deg): {0}')
        self.label_2.setText(label_2_text)
        label_4_text = (f'Altitude: {0}\n'
                        f'Latitude: {0}\n'
                        f'Longitude: {0}')
        self.label_4.setText(label_4_text)
    
    @QtCore.pyqtSlot(dict)
    def update_label(self, label_format):
        label_1_text = (f"Flight Mode: {label_format['flight_mode']}\n"
                        f"GPS HDOP: {label_format['gps_hdop']}\n"
                        f"Satellites Visible: {label_format['satellites_visible']}")
        self.label_1.setText(label_1_text)
        label_2_text = (f"Roll(deg): {label_format['roll']}\n"
                        f"Pitch(deg): {label_format['pitch']}\n"
                        f"Yaw(deg): {label_format['yaw']}")
        self.label_2.setText(label_2_text)
        label_4_text = (f"Altitude: {label_format['altitude']}\n"
                        f"Latitude: {label_format['latitude']}\n"
                        f"Longitude: {label_format['longitude']}")
        self.label_4.setText(label_4_text)

class MessageSocket(QtCore.QObject):
    update_label = QtCore.pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self.connect_event = asyncio.Event()
    
    def init(self, host_ip, port, buffer_size):
        self.host_ip = host_ip
        self.port = port
        self.buffer_size = buffer_size * 1024
        self.message_client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.message_client.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, self.buffer_size)
        # self.message_client.settimeout(10.0)
        print(f'(Socket)(Message) Message Socket Initialized: {self.host_ip}:{self.port}')
    
    def connect(self):
        message = b'Start System'
        print('(Socket)(Message) Sending message:', message)
        self.message_client.sendto(message, (self.host_ip, self.port))
    
    def close(self):
        print('(Socket)(Message) Message Socket Closed')
        self.message_client.close()
    
    async def receiving_message(self):
        print('(Socket)(Message)(async) Start receiving message...')
        loop = asyncio.get_event_loop()
        label_format = {'flight_mode': None, 'gps_hdop': None, 'satellites_visible': None,
                        'roll': None, 'pitch': None, 'yaw': None,
                        'rollspeed': None, 'pitchspeed': None, 'yawspeed': None,
                        'altitude': None, 'latitude': None, 'longitude': None}
        while not self.connect_event.is_set():
            try:
                msg, _ = await loop.run_in_executor(None, self.message_client.recvfrom, self.buffer_size)
                data = json.loads(msg)
                if data['get_type'] == 'HEARTBEAT':
                    label_format['flight_mode'] = self.get_flight_mode(data)
                elif data['get_type'] == 'GPS_RAW_INT':
                    label_format['gps_hdop'], label_format['satellites_visible'] = self.get_gps_raw_int(data)
                elif data['get_type'] == 'ATTITUDE':
                    label_format['roll'], label_format['pitch'], label_format['yaw'] = self.get_attitude(data)
                elif data['get_type'] == 'GLOBAL_POSITION_INT':
                    label_format['altitude'], label_format['latitude'], label_format['longitude'] = self.get_gps_int(data)
                
                # Update labels
                self.update_label.emit(label_format)               

            except socket.timeout:
                print("(Socket)(Message)(async) Socket timeout")
                self.connect_event.set()
                break
            except Exception as e:
                print(f"(Socket)(Message)(async) Error: {e}")
                self.connect_event.set()
                break
        print('(Socket)(Message)(async) Stop receiving message...')
    
    def get_flight_mode(self, msg):
        return msg['custom_mode']
    
    def get_gps_raw_int(self, msg):
        # gps_hdop = msg['eph']
        gps_hdop = msg.get('eph', 0)
        # satellites_visible = msg['s_v']
        satellites_visible = msg.get('satellites_visible', msg.get('s_v', 0))
        return gps_hdop, satellites_visible
    
    def get_attitude(self, msg):
        roll = math.degrees(msg.get('roll', 0.0))
        pitch = math.degrees(msg.get('pitch', 0.0))
        yaw = math.degrees(msg.get('yaw', 0.0))
        # 讓 yaw 落在 0~360（可選）
        yaw = (yaw + 360.0) % 360.0
        # 取兩位小數（你想顯示 200.xx）
        return [round(roll, 2), round(pitch, 2), round(yaw, 2)]
    
    def get_gps_int(self, msg):
        alt_m = msg.get('alt', 0) / 1000.0       # mm -> m
        lat = msg.get('lat', 0) / 1e7            # 1e7 deg -> deg
        lon = msg.get('lon', 0) / 1e7
        return [round(alt_m, 2), round(lat, 7), round(lon, 7)]
