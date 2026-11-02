# Vision-Based Human Robot Interaction

Gesture and motion control of a simulated robot using a webcam. CS 485 Physical AI.

## Setup

MediaPipe needs a supported Python version. This was tested with Python 3.10 on macOS.

```bash
python3.10 -m venv .venv
.venv/bin/pip install -r requirements.txt
mkdir -p models && curl -o models/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

## Gesture module

```bash
.venv/bin/python gesture_detection.py
```

Press `q` to quit.

| Gesture | Command |
|---|---|
| Open hand | `stop` |
| Fist | `close_gripper` |
| Point left | `move_left` |
| Point right | `move_right` |

Each frame produces a `command`, `tracking_status` and `timestamp`. The command is `None` when no gesture is active, and `stop` when no hand is visible.

```python
with GestureDetector(mirrored=True) as detector:
    output = detector.process(frame, timestamp=capture_time)
```

Set `mirrored=True` if the frame is flipped into selfie view, otherwise `mirrored=False`.

## Tests

```bash
.venv/bin/python -m pytest
```
