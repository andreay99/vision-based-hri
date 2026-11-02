"""Hand gesture module (Person 1).

Turns webcam frames into robot commands using MediaPipe hand tracking.

    open hand       -> "stop"
    fist            -> "close_gripper"
    point left      -> "move_left"
    point right     -> "move_right"

In the integrated app, Person 4 owns the webcam and passes each frame in:

    with GestureDetector() as detector:
        output = detector.process(frame, timestamp=capture_time)
        output.command, output.tracking_status, output.timestamp

Run this file directly to test with your own webcam:

    python gesture_detection.py
    python gesture_detection.py --camera 1    # use a different camera
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import time

import cv2
import mediapipe as mp
import numpy as np

DEFAULT_MODEL_PATH = Path(__file__).parent / "models" / "hand_landmarker.task"

# Gestures
OPEN_HAND = "open_hand"
FIST = "fist"
POINT_LEFT = "point_left"
POINT_RIGHT = "point_right"
UNKNOWN = "unknown"

# Commands
STOP = "stop"
CLOSE_GRIPPER = "close_gripper"
MOVE_LEFT = "move_left"
MOVE_RIGHT = "move_right"

GESTURE_TO_COMMAND = {
    OPEN_HAND: STOP,
    FIST: CLOSE_GRIPPER,
    POINT_LEFT: MOVE_LEFT,
    POINT_RIGHT: MOVE_RIGHT,
    UNKNOWN: None,
}

# Tracking status
TRACKING = "tracking"
LOST = "lost"

# MediaPipe hand landmark indices
WRIST = 0
INDEX_MCP = 5
# (PIP joint, fingertip) for the index, middle, ring and pinky fingers.
# The thumb is ignored: it is unreliable and no gesture depends on it.
INDEX_FINGER = (6, 8)
OTHER_FINGERS = ((10, 12), (14, 16), (18, 20))

# Tuning values. These are starting points to adjust using Week 4 accuracy data.
# A finger is judged by (wrist-to-tip distance) / (wrist-to-PIP distance).
# Ratios between the two thresholds count as neither, which gives UNKNOWN.
EXTENDED_RATIO = 1.2
CURLED_RATIO = 1.0
# A point must be this many times more horizontal than vertical (about 34 degrees).
HORIZONTAL_DOMINANCE = 1.5
# Consecutive frames required before a new command takes effect.
CONFIRM_FRAMES = 5
# Stopping is the safety command, so it takes effect faster than the others.
STOP_CONFIRM_FRAMES = 2


@dataclass(frozen=True)
class GestureOutput:
    """Result for one frame.

    command: filtered command to act on, or None for no command.
    tracking_status: TRACKING or LOST.
    timestamp: capture time of the frame, passed through for latency logging.
    gesture: unfiltered gesture for this frame, used to measure accuracy.
    """

    command: Optional[str]
    tracking_status: str
    timestamp: float
    gesture: Optional[str] = None


def _finger_ratio(world: np.ndarray, finger: tuple) -> float:
    pip, tip = finger
    wrist = world[WRIST]
    pip_distance = np.linalg.norm(world[pip] - wrist)
    if pip_distance == 0:
        return 0.0
    return float(np.linalg.norm(world[tip] - wrist) / pip_distance)


def classify_gesture(world: np.ndarray, image_xy: np.ndarray, mirrored: bool = True) -> str:
    """Classify one hand.

    world: 21x3 landmarks in metres, used for finger bend because they do not
        depend on how far the hand is from the camera.
    image_xy: 21x2 landmarks in pixels, used for pointing direction.
    mirrored: True if the frame was flipped into a selfie view. Left and right
        are always from the user's point of view.
    """
    index = _finger_ratio(world, INDEX_FINGER)
    others = [_finger_ratio(world, finger) for finger in OTHER_FINGERS]

    if index > EXTENDED_RATIO and all(r > EXTENDED_RATIO for r in others):
        return OPEN_HAND
    if index < CURLED_RATIO and all(r < CURLED_RATIO for r in others):
        return FIST
    if index > EXTENDED_RATIO and all(r < CURLED_RATIO for r in others):
        dx, dy = image_xy[INDEX_FINGER[1]] - image_xy[INDEX_MCP]
        if abs(dx) < HORIZONTAL_DOMINANCE * abs(dy) or dx == 0:
            return UNKNOWN
        points_image_left = dx < 0
        return POINT_LEFT if points_image_left == mirrored else POINT_RIGHT
    return UNKNOWN


class CommandFilter:
    """Debounces gestures so a brief misdetection cannot change the command.

    A gesture must be seen for several frames in a row before its command takes
    effect. The active command then repeats on every frame until replaced.
    """

    def __init__(self, confirm_frames: int = CONFIRM_FRAMES, stop_confirm_frames: int = STOP_CONFIRM_FRAMES):
        self.confirm_frames = confirm_frames
        self.stop_confirm_frames = stop_confirm_frames
        self.reset()

    def reset(self) -> None:
        self.active = None
        self._candidate = None
        self._count = 0

    def update(self, gesture: str) -> Optional[str]:
        if gesture == self._candidate:
            self._count += 1
        else:
            self._candidate = gesture
            self._count = 1

        command = GESTURE_TO_COMMAND[gesture]
        needed = self.stop_confirm_frames if command == STOP else self.confirm_frames
        if self._count >= needed:
            self.active = command
        return self.active


class GestureDetector:
    """Runs hand tracking and gesture filtering on a stream of video frames."""

    def __init__(self, model_path=DEFAULT_MODEL_PATH, mirrored: bool = True):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Hand model not found at {model_path}. Download it from "
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                "hand_landmarker/float16/1/hand_landmarker.task"
            )
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=1,
        )
        self._landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
        self._filter = CommandFilter()
        self._mirrored = mirrored
        self._last_ms = -1
        self.last_landmarks = None

    def process(self, frame: np.ndarray, timestamp: Optional[float] = None) -> GestureOutput:
        """Process one BGR frame. timestamp is the capture time in seconds.

        If no hand is visible the command is STOP, so the robot stops even if a
        caller only reads the command and ignores tracking_status.
        """
        if timestamp is None:
            timestamp = time.monotonic()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        # MediaPipe requires strictly increasing millisecond timestamps.
        self._last_ms = max(self._last_ms + 1, int(timestamp * 1000))
        result = self._landmarker.detect_for_video(image, self._last_ms)

        if not result.hand_landmarks:
            self.last_landmarks = None
            self._filter.reset()
            return GestureOutput(STOP, LOST, timestamp)

        height, width = frame.shape[:2]
        self.last_landmarks = np.array(
            [(p.x * width, p.y * height) for p in result.hand_landmarks[0]]
        )
        world = np.array([(p.x, p.y, p.z) for p in result.hand_world_landmarks[0]])

        gesture = classify_gesture(world, self.last_landmarks, self._mirrored)
        return GestureOutput(self._filter.update(gesture), TRACKING, timestamp, gesture)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def draw_output(frame: np.ndarray, output: GestureOutput, landmarks: Optional[np.ndarray]) -> None:
    """Draw the hand skeleton and current output onto the frame."""
    if landmarks is not None:
        points = landmarks.astype(int)
        for connection in mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS:
            cv2.line(frame, tuple(points[connection.start]), tuple(points[connection.end]), (0, 200, 0), 2)
        for point in points:
            cv2.circle(frame, tuple(point), 4, (0, 0, 255), -1)

    lines = (
        f"tracking: {output.tracking_status}",
        f"gesture: {output.gesture}",
        f"command: {output.command}",
    )
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (10, 30 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


def main() -> None:
    """Standalone development view. Press q to quit."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--camera", type=int, default=0, help="camera number to use (default: 0)")
    args = parser.parse_args()

    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera}. Check the number and camera permissions.")

    try:
        with GestureDetector(mirrored=True) as detector:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                timestamp = time.monotonic()
                frame = cv2.flip(frame, 1)
                output = detector.process(frame, timestamp)
                draw_output(frame, output, detector.last_landmarks)
                cv2.imshow("Gesture detection (q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
