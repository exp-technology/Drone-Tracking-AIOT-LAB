# Multi-Object Tracking Evaluation System

This system allows you to run and evaluate multiple tracking algorithms including SORT, ByteTrack, DeepSORT and BotSORT.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### 1. Run Tracking

Use track_demo.py to run tracking algorithms:

```bash
python tracker/track_demo.py 
```

Parameters:
- `--dataset`: Dataset name (e.g. "250506_0036_001", "multi_drone_241016_001") 
- `--save_videos`: Save tracking result videos

This will generate tracking results in `track_demo_results/[test_name]/[dataset_name]`

### 2. Evaluate Tracking Performance

Use py_motmetrics.py to calculate MOTA and other metrics:

```bash
python py_motmetrics.py 
```

Parameters:
- `--dataset`: Dataset name
- `--test_name`: Test name (e.g. "v8n_mix005_v4")

### 3. Visualize Results

Use Tracker_comp_MOT.py to visually compare different trackers:

```bash
python Tracker_comp_MOT.py 
```

Output:
- Visualization video (`output/tracking_comparison_[timestamp].mp4`)
- Metrics report (`output/tracking_comparison_[timestamp].txt`)

## Directory Structure

```
├── tracker/              # Tracking implementations
│   ├── track_demo.py    # Main tracking script  
│   └── trackers/        # Tracker implementations
├── ground_truth/        # Ground truth data
├── py_motmetrics.py     # MOTA evaluation script
├── Tracker_comp_MOT.py  # Visualization script
└── videos/              # Test videos
```

The workflow is:
1. Run tracking algorithms
2. Calculate tracking metrics  
3. Visualize and compare