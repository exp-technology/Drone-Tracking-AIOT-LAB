# 本機 IP
HOST_IP = "192.168.50.111"  

# PC IP
PC_IP = "192.168.50.110" 

# Model
TRT_IMGSZ = 512
MODEL = "v8n_mix005_v2_1000"

# FCU
serial_dev="/dev/fcu"
baud=115200
peers=[
        # 給 MAVROS：我們從本機 14650 送到 MAVROS 監聽的 14550，MAVROS 會回到 14650
        {"name": "mavros", "dst": ("127.0.0.1", 14550), "bind": ("127.0.0.1", 14650)},
        # 給 你的控制端：我們從本機 14651 送到控制端監聽的 14551，控制端回到 14651
        {"name": "app",    "dst": ("127.0.0.1", 14551), "bind": ("127.0.0.1", 14651)},
    ]