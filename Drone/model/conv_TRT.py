from ultralytics import YOLO
m = YOLO("v8n_mix005_v4.pt")
m.export(format="engine", half=True, device=0, imgsz=512, dynamic=False, workspace=2)
# 會在當前目錄產生 yolov8n.engine（帶 metadata)
