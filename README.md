# RiskSight

**A lightweight classical computer-vision pipeline that highlights visually risky regions in FPV drone video.** RiskSight combines apparent motion, object boundaries, and dominant lines into an interpretable hazard-awareness overlay—without a trained model or depth sensor.

## Demo

### Vehicle Interior

<table>
  <tr>
    <th>Original</th>
    <th>RiskSight</th>
  </tr>
  <tr>
    <td width="50%">
      <img src="assets/vehicle_original.gif" alt="Original vehicle interior footage" width="100%">
    </td>
    <td width="50%">
      <img src="assets/vehicle_risksight.gif" alt="RiskSight vehicle interior overlay" width="100%">
    </td>
  </tr>
</table>

### FPV Drone

<table>
  <tr>
    <th>Original</th>
    <th>RiskSight</th>
  </tr>
  <tr>
    <td width="50%">
      <img src="assets/drone_original.gif" alt="Original FPV drone footage" width="100%">
    </td>
    <td width="50%">
      <img src="assets/drone_risksight.gif" alt="RiskSight FPV drone overlay" width="100%">
    </td>
  </tr>
</table>

FPV pilots navigate cluttered environments from a single forward-facing camera, where nearby structures can be difficult to judge at speed. RiskSight explores whether lightweight, interpretable visual cues can highlight obstacle-dense regions without stereo depth, LiDAR, training data, or a learned model. Its output is a **heuristic visual risk score**, not a calibrated probability of danger or a replacement for flight-safety systems.

## How RiskSight Works

```text
video → resize/grayscale/blur → Canny edges ─┐
                              Hough lines ───┼→ weighted fusion → threshold → heatmap overlay
                  dense Farneback flow ─────┘
```

### 1. Structural preprocessing

<p align="center">
  <img src="assets/pipeline_edges.png" alt="Original drone frame, blurred grayscale frame, and Canny edge map" width="100%">
</p>

Each RGB frame is resized, converted to grayscale, and blurred with a Gaussian kernel to reduce image noise. Canny edge detection then extracts object contours and structural boundaries; dilation and smoothing expand those thin responses into an edge-influence map for fusion.

### 2. Dominant geometry

<p align="center">
  <img src="assets/pipeline_hough.png" alt="Canny edges and detected Hough lines in an FPV drone frame" width="100%">
</p>

The Probabilistic Hough Transform detects dominant linear structures in the edge map. These lines can correspond to walls, poles, pipes, beams, rooflines, and other elongated geometry that may matter during navigation.

### 3. Apparent motion

<p align="center">
  <img src="assets/pipeline_flow.png" alt="Consecutive FPV drone frames and dense optical-flow magnitude" width="100%">
</p>

Dense Farneback optical flow estimates pixel motion between consecutive grayscale frames. The normalized magnitude map emphasizes strong apparent motion, but it also includes motion caused by the drone's own camera—an important limitation rather than a direct measurement of depth.

### 4. Risk fusion

<p align="center">
  <img src="assets/pipeline_fusion.png" alt="Original FPV frame, motion score, fused risk map, and final RiskSight overlay" width="100%">
</p>

The normalized motion, edge, and line maps are combined using the implementation's fixed heuristic:

```text
risk = 0.70 × motion + 0.20 × edges + 0.10 × lines
```

Responses below the 45th percentile are suppressed, then the map is smoothed, converted to a heatmap, and blended with the original frame. The FPV sequence demonstrates the primary navigation use case; the vehicle-interior demo shows the same pipeline operating in a second, structurally different environment. Evaluation is qualitative—no calibrated safety or depth benchmark is claimed.

## Setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

For development and tests, use `python -m pip install -e '.[dev]'`.

## Usage

Process one or more videos:

```bash
risksight path/to/video.mp4 --output-dir outputs
```

With no video arguments, the command looks for the original project samples at `data/car.MP4` and `data/Drone.mp4`. For a quick smoke test:

```bash
risksight data/Drone.mp4 --max-frames 50
```

The legacy entry point remains available after installation:

```bash
python main.py
```

Inputs must be videos OpenCV can decode. Frames are resized to 640 pixels wide while retaining aspect ratio. Outputs are written under `outputs/`; source videos and generated videos are intentionally ignored because they are large.

## Repository structure

```text
src/risksight/
├── cli.py              # command-line orchestration
├── config.py           # preserved algorithm parameters
├── preprocessing.py    # grayscale, blur, normalization
├── structure.py        # Canny and Hough features
├── motion.py           # dense Farneback optical flow
├── pipeline.py         # weighted risk fusion
├── video.py            # video decoding and resize
└── visualization.py    # figures and overlay videos
tests/                  # stable behavioral checks
main.py                 # backward-compatible launcher
```

## Limitations

- Optical flow measures **apparent image motion**, including camera ego-motion. It does not independently distinguish an approaching object from motion caused by the drone itself.
- Highly textured terrain, vegetation, and roads can create strong flow or edge responses even when immediate collision risk is low.
- The method has no explicit depth or semantic understanding and may miss low-texture hazards.
- Fusion weights are fixed heuristics, and results have been evaluated qualitatively rather than against a labeled safety benchmark.
- The current implementation loads a video into memory before producing all demo artifacts.

Potential next steps include estimating global camera motion and using residual flow, evaluating on a larger labeled video set, testing adaptive fusion, and comparing the classical baseline with monocular-depth methods such as MiDaS or Depth Anything V2.

## Project context

RiskSight was developed as a Stanford CS131 computer-vision project focused on interpretable, training-free obstacle-awareness cues for fast FPV navigation.
