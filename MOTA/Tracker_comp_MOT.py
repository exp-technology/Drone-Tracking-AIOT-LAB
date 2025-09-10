'''
新增MOT格式
'''


import cv2
import pandas as pd
from datetime import datetime
import json
import os

import argparse

# 新增 argparse 支援 dataset_name 設定
parser = argparse.ArgumentParser(description="Compare multiple trackers and output annotated video with metrics.")
parser.add_argument("--dataset", type=str, default="250506_0036_001", help="Dataset name (e.g., A2A_002 or multi_drone_241016_001 or 250506_0036_001)")
parser.add_argument("--test_name", type=str, default="v8n_mix005_v4", help="Test name, e.g., AGDS_5000")
args = parser.parse_args()

# 指定資料集
dataset_name = args.dataset
test_name = args.test_name



# dataset_name = 'A2A_002'
# dataset_name = 'multi_drone_241016_001'
# video_path = f'Roboflow/output/{dataset_name}_output_video.mp4'
# gt_path = f'./motmetrics/MOTA/{dataset_name}/{dataset_name}_gt.txt'

video_path = f'videos/{dataset_name}_output_video.mp4'
gt_path = f'ground_truth/{dataset_name}_gt.txt'

tracker_names = [
                'sort', 
                'bytetrack', 
                'botsort', 
                'deepsort'
                ]


input_path = f"track_demo_results/{test_name}/{dataset_name}"
tracker_colors = {
    'bytetrack': (0, 255, 0),
    'deepsort': (0, 0, 255),
    'sort': (0, 255, 255),
    'botsort': (255, 0, 255),
    'ocsort': (0, 128, 255)
}

overlay_image_path = 'image/tracker.jpg'
scale = 1
current_time = datetime.now().strftime('%Y%m%d_%H%M')
output_name = f'tracking_comparison_{current_time}'
output_path = f'./output/{output_name}'
output_video_path = f'{output_path}/{output_name}.mp4'
output_txt_path = f'{output_path}/{output_name}.txt'

os.makedirs(output_path, exist_ok=True)

# === 載入 Ground Truth 與 Tracker ===
columns = ['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'class', 'visibility']
gt = pd.read_csv(gt_path, header=None, names=columns)
tracker_dfs = {}
for name in tracker_names:
    tracker_dfs[name] = pd.read_csv(f'{input_path}/{name}_pred.txt', header=None, names=columns)

# === 函數：IoU + P/R ===
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)


json_results = {key: [] for key in tracker_names}
json_fps_dict = {key: 0 for key in tracker_names}

# === 初始化累加器 ===
total_iou = {name: 0 for name in tracker_names}
valid_frames = {name: 0 for name in tracker_names}
precision_dict = {name: 0 for name in tracker_names}
recall_dict = {name: 0 for name in tracker_names}
f1_dict = {name: 0 for name in tracker_names}

# === 初始化影片輸出 ===
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / scale)
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / scale)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, 30, (frame_width, frame_height))

frame_id = 1
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (frame_width, frame_height))

    # 畫 GT
    gt_frame = gt[gt['frame'] == frame_id]
    # for _, row in gt_frame.iterrows():
    #     x, y, w, h = map(int, [row.x, row.y, row.w, row.h])
    #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #     cv2.putText(frame, f"GT-{int(row.id)}", (x, y - 5),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 畫預測框 + IoU 計算
    for name in tracker_names:
        pred_frame = tracker_dfs[name][tracker_dfs[name]['frame'] == frame_id]
        matched = 0
        for _, row in pred_frame.iterrows():
            x, y, w, h = map(int, [row.x, row.y, row.w, row.h])
            color = tracker_colors[name]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{name[:2].upper()}-{int(row.id)}", (x, y + h + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # 與 GT 計算 IoU
            for _, gt_row in gt_frame.iterrows():
                iou = compute_iou((x, y, w, h), (gt_row.x, gt_row.y, gt_row.w, gt_row.h))
                if iou > 0.5:
                    matched += 1
                    total_iou[name] += iou
                    break
        if len(gt_frame) > 0:
            precision = matched / len(pred_frame) if len(pred_frame) > 0 else 0
            recall = matched / len(gt_frame)
            f1 = 2 * precision * recall / (precision + recall + 1e-6)

            precision_dict[name] += precision
            recall_dict[name] += recall
            f1_dict[name] += f1
            valid_frames[name] += 1

    # 疊加 overlay 圖示
    # 插入圖片在畫面中間下方
    overlay_image = cv2.imread(overlay_image_path, cv2.IMREAD_UNCHANGED)

    # 確保 overlay_image 存在
    if overlay_image is not None:
        # 調整 overlay_image 大小與當前影像大小自適應
        overlay_height, overlay_width = overlay_image.shape[:2]
        scale_factor = frame_width / overlay_width
        new_overlay_width = int(overlay_width * scale_factor)
        new_overlay_height = int(overlay_height * scale_factor)
        resized_overlay_image = cv2.resize(overlay_image, (new_overlay_width, new_overlay_height))

    # 確保 overlay_image 包含 alpha 通道
    if overlay_image.shape[2] == 3:
        # 添加 alpha 通道
        overlay_image = cv2.cvtColor(overlay_image, cv2.COLOR_BGR2BGRA)

    resized_overlay_image = cv2.resize(overlay_image, (new_overlay_width, new_overlay_height))

    # 計算插入位置
    x_offset = (frame_width - new_overlay_width) // 2
    y_offset = frame_height - new_overlay_height

    # 插入 overlay_image 到 frame
    for c in range(0, 3):
        frame[y_offset:y_offset+new_overlay_height, x_offset:x_offset+new_overlay_width, c] = \
            resized_overlay_image[:, :, c] * (resized_overlay_image[:, :, 3] / 255.0) + \
            frame[y_offset:y_offset+new_overlay_height, x_offset:x_offset+new_overlay_width, c] * (1.0 - resized_overlay_image[:, :, 3] / 255.0)

    out.write(frame)
    cv2.imshow("Tracking Comparison", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cv2.waitKey(1)
    frame_id += 1

cap.release()
out.release()
cv2.destroyAllWindows()


# 讀取每個追蹤器的結果和 FPS 從 JSON 檔案
for tracker_name in tracker_names:
    results_path = f'{input_path}/tracker_results_{tracker_name}.json'
    # results_path = f'./tracker_results_{tracker_name}.json'
    with open(results_path, 'r') as f:
        tracker_results = json.load(f)
        json_results[tracker_name] = tracker_results['results']
        json_fps_dict[tracker_name] = tracker_results['fps_dict']

        with open(output_txt_path, 'a', encoding='utf-8') as fps_file:
            fps_file.write(f"{tracker_name} 平均 FPS: {json_fps_dict[tracker_name]:.2f}\n")
        print(f"{tracker_name} 平均 FPS: {json_fps_dict[tracker_name]:.2f}")



# === 輸出數據 ===
with open(output_txt_path, 'a') as f:
    f.write("\n")
    for name in tracker_names:
        if valid_frames[name] > 0:
            avg_iou = total_iou[name] / valid_frames[name]
            avg_precision = precision_dict[name] / valid_frames[name]
            avg_recall = recall_dict[name] / valid_frames[name]
            avg_f1 = f1_dict[name] / valid_frames[name]
            f.write(f"{name} IoU: {avg_iou:.2f}\n")
            f.write(f"{name} Precision: {avg_precision:.2f}\n")
            f.write(f"{name} Recall: {avg_recall:.2f}\n")
            f.write(f"{name} F1 Score: {avg_f1:.2f}\n\n")
            print(f"{name} → IoU: {avg_iou:.2f}, Precision: {avg_precision:.2f}, Recall: {avg_recall:.2f}, F1: {avg_f1:.2f}")
