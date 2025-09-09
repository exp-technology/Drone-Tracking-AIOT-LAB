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
## 4. Methodology
### 4.1 Detection & Tracking
- Detector: Ultralytics YOLO (model selectable, TensorRT engine preferred when available).
- Tracker: BoT‑SORT with optional ReID; creates stable IDs via motion + appearance cues.
- Track filtering: Only person class retained (`classes=[0]`).

### 4.2 Target Locking Strategy
- GUI sends `SelectID:<id>` over control UDP once a user clicks a bounding box.
- Server restricts outbound META to the locked ID (bandwidth optimization) unless lost.
- If ID disappears longer than `_lost_timeout`, server emits EVENT `{"event":"target_lost"}` and resets lock state.

### 4.3 Auto Tracking Control Loop
Given normalized errors:
- Horizontal: dx = (cx - fx)/fx → yaw_rate PID
- Vertical bias: dy_err = (cy - fy)/fy − bias_y → vertical velocity (vz) PID
- Range proxy: mean box side vs (min_box, max_box) → forward velocity (vx) PID with hysteresis region (dead band) for stability.

### 4.4 Controller Safeguards
- Integral windup limiting per axis.
- Dead-zone thresholds (min_vx, min_vz, min_yaw) to suppress micro jitter.
- Automatic reset when frame lacks target.

### 4.5 Communication Architecture
| Channel | Transport | Direction | Purpose |
|---------|-----------|-----------|---------|
| Video Stream (JPEG) | TCP | Drone → PC | Compressed frames |
| META (tracks JSON) | TCP (mux) | Drone → PC | Selected or full track set |
| EVENT | TCP (mux) | Drone → PC | Target loss notifications |
| Telemetry | UDP | Drone → PC | MAVLink-derived attitude / GPS / mode JSON |
| Control / Selection | UDP | PC → Drone | Manual motion, arming, selection, tracking state |
| MAVLink Hub | Serial↔UDP | Local | Bridges FCU to multiple UDP peers |

---
## 5. Repository Structure
```
Drone-Tracking-AIOT-LAB/
  Drone/
    Main_Drone.py                       # Core server runtime
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
  paper/
    otcn_aiot_lab2025.pdf         # Reference paper (provide citation)
```

---
## 6. Dependencies
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
## 7. Configuration (`Drone/config.py`)
```
HOST_IP = "<DRONE_SBC_IP>"    # Bound + source for UDP services
PC_IP   = "<GROUND_PC_IP>"     # Destination for TCP video
MODEL   = "v8n_mix005_v2_1000" # Base name (tries <name>_<imgsz>.engine etc.)
TRT_IMGSZ = 512
peers = [
  {"name":"mavros","dst":("127.0.0.1",14550),"bind":("127.0.0.1",14650)},
  {"name":"app",   "dst":("127.0.0.1",14551),"bind":("127.0.0.1",14651)},
]
```

---
## 8. Running the System
### 8.1 Onboard Side
```
cd Drone
python Main_Drone.py
```
Wait for: `Initial Server OK`, `YOLO tracker ready.`

### 8.2 Ground GUI
```
cd UI_control
python Main_UI.py
```
Enter `HOST_IP` → Connect.

---
## 9. Target Selection & Auto Tracking Workflow
1. Press Select (enter selection mode — overlay prompts).
2. Click a bounding box → GUI emits `SelectID:<id>`.
3. Server locks ID (`tracking_flag=True`).
4. Start Tracking → enables PID auto pursuit (`AutoTracking_flag=True`).
5. Loss detection → EVENT issued; GUI re-enters selection mode.
6. Reset Selection → clears ID & stops auto mode.

---
## 10. Logging & Reproducibility
- Control log: `autotrack_cmds.csv` with (time, target_id, vx, vy, vz, yaw_rate).
- Recommended to also capture raw telemetry (extend `send_data`).
- For experiments include: hardware specs, model variant, image size, average FPS, ID switches, loss recovery latency.

---
## 11. Performance Metrics (Insert From Paper)
Add a table like:
| Metric | Value | Notes |
|--------|-------|-------|
| Mean FPS (512 img) | XX.X | YOLO + tracking + encode |
| End-to-end latency | XXX ms | Camera → display |
| ID Switch rate | X.XX / min | BoT-SORT config ... |
| Target loss recovery | X.XX s | `_lost_timeout=...` |
| Control steady-state error | ±X px / ±Y deg | After lock |

---
## 12. Tuning Guide
| Parameter | File | Effect |
|-----------|------|--------|
| `bias_y` | `drone_autotrack_controller.py` (constructor) | Vertical framing offset |
| `min_box/max_box/lock_box` | same | Distance proxy scaling |
| `pid_*` tuples | same | Responsiveness vs. stability |
| `_lost_timeout` | `Main_Drone.py` | Robustness to momentary occlusion |
| `conf`, `track_buffer`, `max_age` | `ultra_kwargs` in `Main_Drone.py` | Detection recall vs. stability |
| JPEG quality | `JPEG_QUALITY` | Bandwidth vs. artifact level |

---
## 13. Extending the Framework
- Multi-class tracking: relax `classes=[0]`.
- Add depth / range fusion (e.g., stereo or LiDAR) for better distance control.
- Replace velocity PIDs with MPC or RL policy (interface in `set_velocity_yawrate`).
- Introduce safety geofencing & emergency failsafe (hover/land on anomalies).
- Add dataset recording: save synchronized frames + META for offline training.

---
## 14. Limitations
- Assumes stable illumination; heavy motion blur degrades ReID.
- Single locked target; no dynamic priority switching policy implemented.
- No explicit obstacle avoidance integrated in control loop.
- Loss recovery purely timeout-based (no semantic re-acquire logic yet).

---
## 15. Safety Notice
Autonomous velocity control can induce unsafe states:
- Always maintain manual RC override.
- Test PIDs at low altitude first.
- Avoid operation near people, reflective glass, or GPS-denied cluttered interiors.
- Log every flight session for traceability.

---
## 16. Citation (Fill In)
If you use this repository, please cite:
```
@inproceedings{<placeholder_key>,
  title     = {<Paper Title>},
  author    = {<Authors>},
  booktitle = {<Conference / Journal>},
  year      = {2025},
  pages     = {--},
}
```

---
## 17. License
(Choose one) MIT / Apache-2.0 / BSD-3-Clause. Add a LICENSE file before public release.

---
## 18. Quick Checklist
- [ ] IP & model configured (`config.py`)
- [ ] Model artifact (.engine or .pt) present
- [ ] Drone main runs & prints YOLO warmup complete
- [ ] GUI displays video + telemetry
- [ ] Target selection & EVENT loss works
- [ ] Auto tracking stable (no oscillatory yaw)
- [ ] CSV log populated and flushed

---
## 19. Acknowledgments
(Insert acknowledgments: funding agencies, lab groups, open-source projects.)

---
## 20. Demo Video (Media Showcase)
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
## 21. Datasets (Drone / Human Tracking)
Document the datasets used for training / evaluation. If proprietary or internally collected, indicate collection protocol and compliance.

Template:
| Dataset | Source / Link | Usage | License | Notes |
|---------|---------------|-------|---------|-------|
| <Name A> | <URL / Internal> | Detector pre-training | <License> | Resolution range, class filtering |
| <Name B> | <URL / Internal> | ReID embedding tuning | <License> | Anchor / positive strategy |
| Internal Flight Set | In-house capture | Controller tuning / latency tests | Restricted | Contains onboard camera motion & varied lighting |

Suggested public datasets (replace / prune as applicable):
- VisDrone (object detection / tracking over aerial perspective)
- MOT17 / MOT20 (for multi-object tracking baseline, ReID embeddings)
- CrowdHuman (dense human detection pre-training)
- COCO (general person class robustness)

If you created a custom drone flight dataset:
- Acquisition hardware (camera sensor, resolution, lens FOV)
- Annotation format (YOLO txt / COCO JSON / custom)
- Frame count & split (train/val/test)
- Environmental diversity (indoor/outdoor, illumination, altitude bands)
- Ethical & privacy considerations (face blurring / consent)

Data Preparation Example:
```
# Convert internal annotations to YOLO format
python tools/convert_internal_to_yolo.py \
    --src data/internal_json/ \
    --dst data/yolo_labels/ \
    --img-root data/images/
```
Include augmentation strategies (scale jitter, motion blur synth, brightness/contrast, random occlusion) if they influenced performance.

Compliance & Safety:
- Ensure datasets do not include sensitive personal identifiers.
- Follow local UAV regulations during data capture.
- Retain raw logs (flight telemetry) for reproducibility and audit.

---
Need the README compressed (short form) or a Chinese version again? Request further modifications.
