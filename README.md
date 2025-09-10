# Drone-Tracking-AIOT-LAB

A research-oriented platform for real‑time human target detection, multi-object identity tracking, and autonomous MAV (Micro Aerial Vehicle) control using Ultralytics YOLO + BoT‑SORT (ReID) and MAVLink. This repository accompanies (or is derived from) the associated paper (see Citation section). The code integrates onboard perception, low-latency streaming, target selection, PID-based auto pursuit, and mixed manual/autonomous control.

> NOTE: Replace paper-specific metadata (title, authors, DOI, results metrics) in the placeholders below once finalized.

---
## 1. Abstract (Placeholder)
Provide a concise abstract here summarizing: problem motivation, algorithmic approach (YOLO + BoT‑SORT + ReID + PID servo loop), system integration (onboard -> ground), and key performance highlights (e.g., FPS, latency, success rate, tracking stability). Example format:

```
This work presents an embedded real-time human tracking and control framework for small aerial robots. Using YOLO + BoT-SORT with ReID embedding fusion and a geometry+PID controller, the system streams visual feedback while providing safe autonomous velocity/yaw regulation relative to a selected target. Experiments demonstrate XX FPS at 512× resolution on <hardware>, mean ID switch rate of X.X, and stable pursuit within ±Δ positional error at 5–10 m range.
```

---
## 2. Key Contributions
- Unified pipeline: detection, multi-ID association (BoT-SORT + ReID), target locking, auto pursuit.
- Robust target retention: selective track forwarding + event-based lost-target recovery.
- Cross-layer communication design: single multiplexed TCP stream (VIDEO + META + EVENT) + dual UDP channels (telemetry / control) + MAVLink hub bridging serial & multiple peers.
- Modular AutoTrackController with tunable PID domains (vx, vz, yaw_rate) + lock box & dead-zones for stability.
- Asynchronous, non-blocking architecture (asyncio + threaded receiver) minimizing control loop jitter.
- Reproducible control logging (`autotrack_cmds.csv`) for post-flight analysis.

---
## 3. System Overview
```
 Onboard (Drone SBC)                          Ground Station (PC GUI)
 ───────────────────────────────             ─────────────────────────────
  Camera --> YOLO + BoT-SORT (ReID) --+      TCP Server (VideoWidget)
                                      |----> META (tracks JSON)
  PID AutoTrack Controller <----------+<---- Target Selection (GUI)
  MAVLink Command Output (velocity/yaw)      ControlWidget (manual + auto)
  Serial FCU <-> MAVLink UDP Hub            Telemetry Display (MessageWidget)
  CSV Logger (vx,vy,vz,yaw_rate)            User Selection / Start / Stop
```

---
## 4. Demo Video (Media Showcase)
Add links / thumbnails for demonstration clips here.

Example (replace with real URLs):

[![System Demo (YouTube)](https://img.youtube.com/vi/XXXXXXXXXXX/hqdefault.jpg)](https://www.youtube.com/watch?v=XXXXXXXXXXX "Drone Tracking Demo")

Alternative Markdown snippet for a locally hosted GIF (place file under `docs/media/`):
```
![Auto Tracking Demo](docs/media/autotrack_demo.gif)
```
Recommended clips:
- Real-time tracking & ID lock (single target)
- Multi-person scene with target switching
- Lost-target event and recovery
- PID stabilization (side-by-side: raw vs. control outputs)

Provide FPS / latency overlay in at least one clip for quantitative illustration.

---
## 5. Datasets (Drone)

We constructed a dataset comprising 2,850 synthetic samples and 5,481 real samples sourced from publicly available datasets. For retraining the object detection model, the training set included 5,904 samples (3,912 real and 1,992 synthetic), while the validation set contained 2,427 samples (1,569 real and 858 synthetic). The complete dataset is publicly accessible:

[![Drone Dataset (Roboflow)]](https://universe.roboflow.com/aiotlab-lnkrh/mix_v005-z2ksj/dataset/2)

---
## 6. Methodology
### 6.1 Detection & Tracking
- Detector: Ultralytics YOLO (model selectable, TensorRT engine preferred when available).
- Tracker: BoT‑SORT with optional ReID; creates stable IDs via motion + appearance cues.
- Track filtering: Only person class retained (`classes=[0]`).

### 6.2 Target Locking Strategy
- GUI sends `SelectID:<id>` over control UDP once a user clicks a bounding box.
- Server restricts outbound META to the locked ID (bandwidth optimization) unless lost.
- If ID disappears longer than `_lost_timeout`, server emits EVENT `{"event":"target_lost"}` and resets lock state.

### 6.3 Auto Tracking Control Loop
Given normalized errors:
- Horizontal: dx = (cx - fx)/fx → yaw_rate PID
- Vertical bias: dy_err = (cy - fy)/fy − bias_y → vertical velocity (vz) PID
- Range proxy: mean box side vs (min_box, max_box) → forward velocity (vx) PID with hysteresis region (dead band) for stability.

### 6.4 Controller Safeguards
- Integral windup limiting per axis.
- Dead-zone thresholds (min_vx, min_vz, min_yaw) to suppress micro jitter.
- Automatic reset when frame lacks target.

### 6.5 Communication Architecture
| Channel | Transport | Direction | Purpose |
|---------|-----------|-----------|---------|
| Video Stream (JPEG) | TCP | Drone → PC | Compressed frames |
| META (tracks JSON) | TCP (mux) | Drone → PC | Selected or full track set |
| EVENT | TCP (mux) | Drone → PC | Target loss notifications |
| Telemetry | UDP | Drone → PC | MAVLink-derived attitude / GPS / mode JSON |
| Control / Selection | UDP | PC → Drone | Manual motion, arming, selection, tracking state |
| MAVLink Hub | Serial↔UDP | Local | Bridges FCU to multiple UDP peers |

---
## 7. Repository Structure
```
Drone-Tracking-AIOT-LAB/
  Drone/
    Main_Drone.py                 # Core server runtime
    config.py                     # IP, model, hub settings
    mav_function.py               # MAVLink command helpers
    Mavlink_UDP_hub.py            # Serial <-> multi-UDP relay
    drone_autotrack_controller.py # PID + auto pursuit logic
    model/                        # .engine / .pt / yaml artifacts
  UI_control/
    Main_UI.py                       # Ground GUI entry
    video_widget.py               # TCP receiver + overlay + selection
    message_widget.py             # Telemetry display (UDP)
    control_widget.py             # Manual + auto tracking controls
    Widge_Component.py            # Shared UI elements
    no_videos_available.jpg       # Placeholder frame
```

---
## 8. Dependencies
| Category | Packages |
|----------|----------|
| Core Vision | ultralytics, opencv-python, numpy |
| Control / MAV | pymavlink, pyserial |
| GUI | PyQt6, qasync |
| Async / Stdlib | asyncio, json, csv, struct, time |

Install (example):
```
pip install --upgrade pip
pip install ultralytics opencv-python numpy pymavlink pyserial PyQt6 qasync
```
(Optional) Add TensorRT runtime if deploying `.engine` models.

---
## 9. Configuration (`Drone/config.py`)
```
HOST_IP = "<DRONE_SBC_IP>"    # Bound + source for UDP services
PC_IP   = "<GROUND_PC_IP>"     # Destination for TCP video
MODEL   = "Drone" # Base name (tries <name>_<imgsz>.engine etc.)
TRT_IMGSZ = 512
```

---
## 10. Running the System
### 10.1 Onboard Side
```
cd Drone
python Main_Drone.py
```
Wait for: `Initial Server OK`, `YOLO tracker ready.`

### 10.2 Ground GUI
```
cd UI_control
python Main_UI.py
```
Enter `HOST_IP` → Connect.

---
## 11. Target Selection & Auto Tracking Workflow
1. Press Select (enter selection mode — overlay prompts).
2. Click a bounding box → GUI emits `SelectID:<id>`.
3. Server locks ID (`tracking_flag=True`).
4. Start Tracking → enables PID auto pursuit (`AutoTracking_flag=True`).
5. Loss detection → EVENT issued; GUI re-enters selection mode.
6. Reset Selection → clears ID & stops auto mode.

---
## 16. Safety Notice
Autonomous velocity control can induce unsafe states:
- Always maintain manual RC override.
- Test PIDs at low altitude first.
- Avoid operation near people, reflective glass, or GPS-denied cluttered interiors.
- Log every flight session for traceability.
