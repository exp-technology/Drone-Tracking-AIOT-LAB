from PyQt6 import QtWidgets, QtCore, QtGui
from Widge_Component import Label_Component, Button_Component
import sys
from qasync import QEventLoop
import asyncio

# Import stream modules
from video_widget import VideoWidget

# Import Message modules
from message_widget import MessageWidget

# Import Control modules
from control_widget import ControlWidget

VIDEO_PORT = 5000
MESSAGE_PORT = 5001
CONTROL_PORT = 5002

class MainWindows(QtWidgets.QWidget):
    drone_master = None

    def __init__(self):
        super().__init__()
        self.connect_flag = False
        self.setWindowTitle('Manual Drone Control System')
        self.setGeometry(0, 0, 1500, 500)

        self.title_label = Label_Component('Drone Control System')
        self.title_label.setFixedSize(350, 70)  # Set the size of the area
        self.title_label.setStyleSheet("QLabel {"
                                       "    background-color: rgb(238, 96, 73); padding: 10px; border: 2px solid #ddd;"
                                       "    border-radius: 10px; font-size: 26px; font-family: Arial; font-weight:bold;"
                                       "    box-shadow: 5px 5px 15px rgba(0, 0, 0, 0.3);"
                                       "}")

        # IP_Widget
        self.ip_widget = IPWidget()
        # Stream Widget
        self.stream_widget = VideoWidget()
        # Message Widget
        self.message_widget = MessageWidget()
        # Control Widget
        self.control_widget = ControlWidget()

        # Set Layout
        layout = QtWidgets.QGridLayout()
        layout.addWidget(self.stream_widget, 0, 0, 3, 1)
        layout.addWidget(self.title_label, 0, 1, 1, 2)
        layout.addWidget(self.ip_widget, 0, 3, 1, 2)
        layout.addWidget(self.message_widget, 1, 1, 1, 4)
        layout.addWidget(self.control_widget, 2, 1, 1, 4)

        self.setLayout(layout)

        # Connect signals
        self.ip_widget.trigger.connect(self.connect_server)
        
        # 進入選取
        self.control_widget.request_select_mode.connect(
            lambda: self.stream_widget.enable_select_mode(True)
        )
        
        # Reset：退出選取 + 清掉當前鎖定（VideoWidget 會立即清畫面上的鎖定框）
        self.control_widget.request_reset_selection.connect(
            self.stream_widget.reset_selection
        )
        
        # Video 端選到框 → 交給 ControlWidget 送指令
        self.stream_widget.target_selected.connect(
            self.control_widget.on_target_selected
        )
        
        # 追丟：讓 ControlWidget 自行把 UI（Select/Tracking）收回
        self.stream_widget.target_lost.connect(
            self.control_widget.on_target_lost
        )
            
    def connect_server(self):
        ip = self.ip_widget.ip_LineEdit.text()
        if not self.connect_flag:
            self.message_widget.init(ip, MESSAGE_PORT, 64)
            self.stream_widget.init(ip, VIDEO_PORT,1024)
            self.control_widget.init(ip, CONTROL_PORT, 64)
            self.ip_widget.connect_button.setText('Disconnect')
            self.ip_widget.ip_LineEdit.setEnabled(False)
            self.connect_flag = True
        else:
            self.message_widget.pause()
            self.stream_widget.pause()
            self.control_widget.pause()
            self.ip_widget.connect_button.setText('Connect')
            self.ip_widget.ip_LineEdit.setEnabled(True)
            self.connect_flag = False
            print('Disconnected from server')
    
    def closeEvent(self, event):
        if self.connect_flag:
            self.message_widget.pause()
            self.stream_widget.pause()
            QtCore.QTimer.singleShot(1000, self.close)
        self.stream_widget.close()
        self.message_widget.close()
        self.control_widget.close()

    def on_target_selected(self, tid: int):
        print("[UI] Selected target id =", tid)
        server_ip = self.ip_widget.ip_LineEdit.text()  # 飛機端 IP
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.sendto(f"SelectID:{tid}".encode('utf-8'), (server_ip, CONTROL_PORT))
        finally:
            s.close()


class IPWidget(QtWidgets.QWidget):
    CONNECT_FLAG = False
    trigger = QtCore.pyqtSignal()
    
    def __init__(self):
        super().__init__()
        # IP_Widget
        ip_widget = QtWidgets.QWidget()
        ip_widget.setFixedHeight(70)
        ip_widget.setStyleSheet("background-color: rgb(240, 179, 85); border-radius: 10px;")
        
        comport_label = Label_Component("IP Address:", 22)
        self.connect_button = Button_Component('Connect', 16, (140, 40))
        self.connect_button.clicked.connect(self.connect_server)

        # Set baudrate
        self.ip_LineEdit = QtWidgets.QLineEdit(ip_widget)
        regular_expression = QtCore.QRegularExpression('^\d{6}$')
        validator = QtGui.QRegularExpressionValidator(regular_expression, self)
        self.ip_LineEdit.setValidator(validator)
        self.ip_LineEdit.setStyleSheet("QLineEdit{"
                                             "   border: 2px solid black; border-radius: 3px; background: white;"
                                             "   font-size: 16px; font-weight: bold;}")
        self.ip_LineEdit.setText('192.168.50.111')

        serial_layout = QtWidgets.QHBoxLayout(ip_widget)
        serial_layout.addWidget(comport_label, 1)
        serial_layout.addWidget(self.ip_LineEdit, 1)
        serial_layout.addWidget(self.connect_button, 1)
        
        # Set Layout
        layout = QtWidgets.QGridLayout()
        layout.addWidget(ip_widget, 0, 0, 1, 1)
        self.setLayout(layout)
    
    def connect_server(self):
        self.trigger.emit()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)

    main_windows = MainWindows()
    main_windows.show()

    with loop:
        try:
            print("All asyncio tasks started")
            loop.run_forever()
            print("All asyncio tasks finished")
        except asyncio.CancelledError:
            print("All asyncio tasks cancelled")

