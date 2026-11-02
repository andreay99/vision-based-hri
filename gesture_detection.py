"""Hand gesture module (Person 1).

Tracks a hand in webcam frames using MediaPipe.

In the integrated app, Person 4 owns the webcam and passes each frame in:

    with GestureDetector() as detector:
        output = detector.process(frame, timestamp=capture_time)
        output.command, output.tracking_status, output.timestamp

Run this file directly to test with your own webcam:

    python gesture_detection.py
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import time

import cv2
import mediapipe as mp
import numpy as np

DEFAULT_MODEL_PATH = Path(__file__).parent / "models" / "hand_landmarker.task"

# Command sent when no hand is visible
STOP = "stop"

# Tracking status
TRACKING = "tracking"
LOST = "lost"

@dataclass(frozen=True)
class GestureOutput:
    """Result for one frame.

    command: STOP when no hand is visible, otherwise None for now.
    tracking_status: TRACKING or LOST.
    timestamp: capture time of the frame, passed through for latency logging.
    gesture: gesture detected in this frame.
    """

    command: Optional[str]
    tracking_status: str
    timestamp: float
    gesture: Optional[str] = None


class GestureDetector:
    """Runs hand tracking on a stream of video frames."""

    def __init__(self, model_path=DEFAULT_MODEL_PATH):
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
            return GestureOutput(STOP, LOST, timestamp)

        height, width = frame.shape[:2]
        self.last_landmarks = np.array(
            [(p.x * width, p.y * height) for p in result.hand_landmarks[0]]
        )
        return GestureOutput(None, TRACKING, timestamp)

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
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        raise RuntimeError("Could not open the webcam. Check camera permissions.")

    try:
        with GestureDetector() as detector:
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
