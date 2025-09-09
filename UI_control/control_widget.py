from PyQt6 import QtWidgets, QtCore
from Widge_Component import Label_Component, Button_Component

import asyncio
import socket


class ControlWidget(QtWidgets.QWidget):
    keyboard_control_task = None
    tracking_flag = False
    request_select_mode = QtCore.pyqtSignal()
    request_reset_selection = QtCore.pyqtSignal()
    def __init__(self):
        super().__init__()
        self.setObjectName('Keyboard Widget')

        # Control Layout
        Direction_title_label = Label_Component('Pitch / Roll Control Plant', 24)
        Direction_title_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)
        self.forward_button = Button_Component('Forward\n(W)', 16, (100, 70))
        self.backward_button = Button_Component('Backward\n(S)', 16, (100, 70))
        self.right_button = Button_Component('Right\n(D)', 16, (100, 70))
        self.left_button = Button_Component('Left\n(A)', 16, (100, 70))

        # Direction Layout
        self.direction_widget = QtWidgets.QWidget()
        self.direction_widget.setStyleSheet('background-color:rgb(119, 182, 225); border-radius:10px;')
        direction_layout = QtWidgets.QGridLayout(self.direction_widget)

        direction_layout.addWidget(Direction_title_label, 0, 0, 1, 3)
        direction_layout.addWidget(self.forward_button, 1, 1, 1, 1)
        direction_layout.addWidget(self.backward_button, 3, 1, 1, 1)
        direction_layout.addWidget(self.right_button, 2, 2, 1, 1)
        direction_layout.addWidget(self.left_button, 2, 0, 1, 1)

        direction_layout.setRowStretch(0, 1)
        direction_layout.setRowStretch(1, 1)
        direction_layout.setRowStretch(2, 1)
        direction_layout.setRowStretch(3, 1)

        direction_layout.setColumnStretch(0, 1)
        direction_layout.setColumnStretch(1, 1)
        direction_layout.setColumnStretch(2, 1)

        # Alt Layout
        self.altitude_widget = QtWidgets.QWidget()
        self.altitude_widget.setStyleSheet('background-color:rgb(156, 194, 220); border-radius:10px;')
        altitude_layout = QtWidgets.QGridLayout(self.altitude_widget)

        alt_title_label = Label_Component('Alt / Yaw Control Plant', 24)
        alt_title_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)
        self.raise_button = Button_Component('Raise\n(I)', 16, (100, 70))
        self.down_button = Button_Component('Down\n(K)', 16, (100, 70))
        self.yaw_right_button = Button_Component('Right\n(L)', 16, (100, 70))
        self.yaw_left_button = Button_Component('Left\n(J)', 16, (100, 70))

        altitude_layout.addWidget(alt_title_label, 0, 0, 1, 3)
        altitude_layout.addWidget(self.raise_button, 1, 1, 1, 1)
        altitude_layout.addWidget(self.down_button, 3, 1, 1, 1)
        altitude_layout.addWidget(self.yaw_right_button, 2, 2, 1, 1)
        altitude_layout.addWidget(self.yaw_left_button, 2, 0, 1, 1)

        altitude_layout.setRowStretch(0, 4)
        altitude_layout.setRowStretch(1, 1)
        altitude_layout.setRowStretch(2, 1)
        altitude_layout.setRowStretch(3, 1)

        altitude_layout.setColumnStretch(0, 1)
        altitude_layout.setColumnStretch(1, 1)
        altitude_layout.setColumnStretch(2, 1)

        # Arm / Takeoff / Land Layout
        func_widget = QtWidgets.QWidget()
        func_widget.setStyleSheet('background-color:rgb(76, 207, 174); border-radius:10px')
        func_layout = QtWidgets.QVBoxLayout(func_widget)

        self.select_button = Button_Component('Select ID', 16, (110, 40))
        self.tracking_button = Button_Component('Tracking', 16, (110, 40))
        self.arm_disarm_button = Button_Component('Arm', 16, (110, 40))
        self.takeoff_button = Button_Component('TakeOFF', 16, (110, 40))
        self.land_button = Button_Component('Land', 16, (110, 40))

        func_layout.addWidget(self.select_button, 1)
        func_layout.addWidget(self.tracking_button, 1)
        func_layout.addWidget(self.arm_disarm_button, 1)
        func_layout.addWidget(self.takeoff_button, 1)
        func_layout.addWidget(self.land_button, 1)

        # Combine all widget
        all_layout = QtWidgets.QGridLayout()
        all_layout.addWidget(self.altitude_widget, 1, 0, 1, 1)
        all_layout.addWidget(self.direction_widget, 1, 1, 1, 1)
        all_layout.addWidget(func_widget, 1, 2, 1, 1)

        self.setLayout(all_layout)

        # Set Socket
        self.control_socket = ControlSocket()
    
    def init(self, host_ip, port, buffer_size):
        self.host_ip = host_ip
        self.port = port
        self.buffer_size = buffer_size
        self.control_socket.init(host_ip, port, buffer_size)
        self.control_socket.connect_event.clear()

        # Connect button signals
        self.forward_button.clicked.connect(self.forward)
        self.backward_button.clicked.connect(self.backward)
        self.right_button.clicked.connect(self.right)
        self.left_button.clicked.connect(self.left)
        self.raise_button.clicked.connect(self.raise_altitude)
        self.down_button.clicked.connect(self.lower_altitude)
        self.yaw_right_button.clicked.connect(self.yaw_right)
        self.yaw_left_button.clicked.connect(self.yaw_left)
        self.arm_disarm_button.clicked.connect(self.arm_disarm)
        self.takeoff_button.clicked.connect(self.takeoff)
        self.land_button.clicked.connect(self.land)

        self.tracking_button.clicked.connect(self.tracking)

        self._select_active = False  # 本地是否處在選取/已選狀態
        self.select_button.setText('Select ID')
        self.select_button.clicked.connect(self._on_select_clicked)

        for button in [
            self.forward_button,
            self.backward_button,
            self.right_button,
            self.left_button,
            self.raise_button,
            self.down_button,
            self.yaw_right_button,
            self.yaw_left_button,
            self.tracking_button,
            self.arm_disarm_button,
            self.takeoff_button,
            self.land_button
        ]:
            button.setDisabled(False)
    
    def pause(self):
        print('(Widget)(Control) Control Socket Paused')
        for button in [
            self.forward_button,
            self.backward_button,
            self.right_button,
            self.left_button,
            self.raise_button,
            self.down_button,
            self.yaw_right_button,
            self.yaw_left_button,
            self.tracking_button,
            self.arm_disarm_button,
            self.takeoff_button,
            self.land_button
        ]:
            button.setDisabled(True)
    
    def tracking(self):
        if self.tracking_flag:
            self.tracking_button.setText('Tracking')
            self.control_socket.send_control_command('Stop Tracking')
            self.tracking_flag = False
        else:
            self.tracking_button.setText('Stop Tracking')
            self.control_socket.send_control_command('Start Tracking')
            self.tracking_flag = True
    
    def takeoff(self):
        self.control_socket.send_control_command('Takeoff')
    
    def land(self):
        self.control_socket.send_control_command('Land')
    
    def arm_disarm(self):
        self.control_socket.send_control_command(self.arm_disarm_button.text())
        if self.arm_disarm_button.text() == 'Arm':
            self.arm_disarm_button.setText('Disarm')
        else:
            self.arm_disarm_button.setText('Arm')
    
    def forward(self):
        self.control_socket.send_control_command('Forward')
        
    def backward(self):
        self.control_socket.send_control_command('Backward')

    def right(self):
        self.control_socket.send_control_command('Right')

    def left(self):
        self.control_socket.send_control_command('Left')

    def raise_altitude(self):
        self.control_socket.send_control_command('Raise')

    def lower_altitude(self):
        self.control_socket.send_control_command('Drop')

    def yaw_right(self):
        self.control_socket.send_control_command('Y_Right')

    def yaw_left(self):
        self.control_socket.send_control_command('Y_Left')
    
    def close(self):
        print('(Widget)(Control) Control Widget Closed')
        self.control_socket.close()

    def _on_select_clicked(self):
        if not self._select_active:
            # 進入選取模式
            self._select_active = True
            self.select_button.setText('Reset')
            self.request_select_mode.emit()
        else:
            # Reset：退出選取 + 清除鎖定 + 告知伺服端停止 Tracking
            self._select_active = False
            self.select_button.setText('Select ID')
            try:
                self.control_socket.send_control_command('Stop Tracking')
            except Exception as e:
                print("(Control) Stop Tracking send error:", e)
            self.control_socket.send_control_command('Reset Selection')
            self.request_reset_selection.emit()

    @QtCore.pyqtSlot(int)
    def on_target_selected(self, tid: int):
        # VideoWidget 點到框後會呼叫到這裡
        self.control_socket.send_control_command(f"SelectID:{tid}")
        # UI 顯示目前已經有鎖定，如果你有追蹤開關，也可以自動打開
        self._select_active = True
        self.select_button.setText('Reset')
        print(f"(Control) Selected target id = {tid}")

    @QtCore.pyqtSlot()
    def on_target_lost(self):
        # 收到 PC 視訊端的 target_lost（MainWindows 會把 signal 轉進來）
        self._select_active = False
        self.select_button.setText('Select ID')
        # 如果你有 Tracking 按鈕，這裡同步把狀態收回
        self.tracking_flag = False
        try:
            self.tracking_button.setText('Tracking')
        except Exception:
            pass

class ControlSocket(QtCore.QObject):
    def __init__(self):
        super().__init__()
        self.host_ip = None
        self.port = None
        self.buffer_size = None
        self.control_client = None
        self.connect_event = asyncio.Event()

    def init(self, host_ip, port, buffer_size):
        self.host_ip = host_ip
        self.port = port
        self.buffer_size = buffer_size * 1024
        self.control_client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.control_client.setblocking(False)
        self.connect_event.clear()
        print('(Socket)(Control) Control Socket Initialized')

        message = b'Start Control Socket'
        self.control_client.sendto(message, (self.host_ip, self.port))
        print('(Socket)(Control) Sending message:', message)
    
    def close(self):
        print('(Socket)(Control) Control Socket Closed')
        self.control_client.sendto(b'Stop Control Socket', (self.host_ip, self.port))
        self.control_client.close()
    
    def send_control_command(self, command):
        try:
            self.control_client.sendto(command.encode(), (self.host_ip, self.port))
            print('(Socket)(Control) Sending command:', command)
        except Exception as e:
            print('(Socket)(Control) Error sending command:', e)
    
    async def receiving_message(self):
        loop = asyncio.get_event_loop()
        while not self.connect_event.is_set():
            msg, _ = await loop.run_in_executor(None, self.control_client.recvfrom, self.buffer_size)
            print("(Socket)(Control)(async) Received message:", msg)
