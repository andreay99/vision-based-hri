"""Gesture accuracy test (Person 1, Week 4).

Prompts for each gesture in random order, records what the module detected, and
reports accuracy per gesture and which gestures get confused. An attempt counts
as correct if the filtered command at the end of the attempt matches the prompt,
since that is what the robot would act on.

Run one session per participant, using an ID rather than a name:

    python gesture_accuracy.py --participant p1
    python gesture_accuracy.py --participant p1 --attempts 10 --camera 1

Combine the results from every session:

    python gesture_accuracy.py --summarize results/*.csv
"""

import argparse
from collections import Counter
import csv
from datetime import datetime
from pathlib import Path
import random
import time

import cv2

from gesture_detection import (
    FIST,
    GESTURE_TO_COMMAND,
    LOST,
    OPEN_HAND,
    POINT_LEFT,
    POINT_RIGHT,
    GestureDetector,
    draw_output,
)

GESTURES = (OPEN_HAND, FIST, POINT_LEFT, POINT_RIGHT)
COMMAND_TO_GESTURE = {command: gesture for gesture, command in GESTURE_TO_COMMAND.items() if command}
NO_COMMAND = "no_command"

GET_READY_SECONDS = 3.0
CAPTURE_SECONDS = 1.5

RESULTS_DIR = Path(__file__).parent / "results"
FIELDS = ("participant", "attempt", "expected", "predicted", "frames", "matching_frames", "lost_frames")


def predicted_label(output) -> str:
    """Name what the robot would act on for this output."""
    if output.tracking_status == LOST:
        return LOST
    return COMMAND_TO_GESTURE.get(output.command, NO_COMMAND)


def make_attempts(attempts_per_gesture: int) -> list:
    """Return every gesture the given number of times, in random order."""
    attempts = list(GESTURES) * attempts_per_gesture
    random.shuffle(attempts)
    return attempts


def run_attempt(capture, detector, expected, label, show, clock=time.monotonic):
    """Run one attempt. Returns its measurements, or None if the session ended."""
    start = clock()
    output = None
    frames = matching_frames = lost_frames = 0

    while True:
        elapsed = clock() - start
        if elapsed >= GET_READY_SECONDS + CAPTURE_SECONDS:
            break

        ok, frame = capture.read()
        if not ok:
            return None
        timestamp = clock()
        frame = cv2.flip(frame, 1)
        output = detector.process(frame, timestamp)

        capturing = elapsed >= GET_READY_SECONDS
        if capturing:
            frames += 1
            matching_frames += output.gesture == expected
            lost_frames += output.tracking_status == LOST

        total = GET_READY_SECONDS + CAPTURE_SECONDS
        status = f"HOLD IT {total - elapsed:.1f}s" if capturing else f"get ready {GET_READY_SECONDS - elapsed:.1f}s"
        if not show(frame, output, detector, [label, f"{expected}: {status}"]):
            return None

    return {
        "expected": expected,
        "predicted": predicted_label(output),
        "frames": frames,
        "matching_frames": matching_frames,
        "lost_frames": lost_frames,
    }


def run_session(capture, detector, participant, attempts, show, clock=time.monotonic) -> list:
    """Run the attempts in order. Keeps finished attempts if the session ends early."""
    rows = []
    for number, expected in enumerate(attempts, start=1):
        label = f"Attempt {number}/{len(attempts)}"
        result = run_attempt(capture, detector, expected, label, show, clock)
        if result is None:
            break
        rows.append({"participant": participant, "attempt": number, **result})
    return rows


def show_on_screen(frame, output, detector, lines) -> bool:
    """Draw the current state. Returns False if the user pressed q."""
    draw_output(frame, output, detector.last_landmarks)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (10, 150 + 45 * i), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    cv2.imshow("Gesture accuracy test (q to quit)", frame)
    return cv2.waitKey(1) & 0xFF != ord("q")


def write_results(rows, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_results(paths) -> list:
    rows = []
    for path in paths:
        with open(path, newline="") as file:
            rows.extend(csv.DictReader(file))
    return rows


def format_summary(rows) -> str:
    """Build the report: accuracy per gesture, confusions and tracking failures."""
    if not rows:
        return "No attempts recorded."

    lines = [f"Participants: {len({row['participant'] for row in rows})}", ""]
    for gesture in GESTURES:
        predictions = Counter(row["predicted"] for row in rows if row["expected"] == gesture)
        total = sum(predictions.values())
        if not total:
            lines.append(f"{gesture:12} no attempts")
            continue
        correct = predictions[gesture]
        line = f"{gesture:12} {correct}/{total} ({correct / total:.0%})"
        confused = [f"{name} x{count}" for name, count in predictions.most_common() if name != gesture]
        if confused:
            line += "  confused with: " + ", ".join(confused)
        lines.append(line)

    correct = sum(row["expected"] == row["predicted"] for row in rows)
    lost_attempts = sum(row["predicted"] == LOST for row in rows)
    lost_frames = sum(int(row["lost_frames"]) for row in rows)
    frames = sum(int(row["frames"]) for row in rows)
    lines += [
        "",
        f"Overall: {correct}/{len(rows)} ({correct / len(rows):.0%})",
        f"Tracking failures: {lost_attempts} attempts ended with tracking lost, "
        f"{lost_frames}/{frames} captured frames lost",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure gesture accuracy.")
    parser.add_argument("--participant", help="participant ID, for example p1")
    parser.add_argument("--attempts", type=int, default=10, help="attempts per gesture (default: 10)")
    parser.add_argument("--camera", type=int, default=0, help="camera number to use (default: 0)")
    parser.add_argument("--summarize", nargs="+", metavar="CSV", help="report on saved results instead of testing")
    args = parser.parse_args()

    if args.summarize:
        print(format_summary(read_results(args.summarize)))
        return
    if not args.participant:
        parser.error("--participant is required when running a test")

    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera}. Check the number and camera permissions.")

    try:
        with GestureDetector(mirrored=True) as detector:
            rows = run_session(capture, detector, args.participant, make_attempts(args.attempts), show_on_screen)
    finally:
        capture.release()
        cv2.destroyAllWindows()

    if rows:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = RESULTS_DIR / f"gesture_accuracy_{args.participant}_{stamp}.csv"
        write_results(rows, path)
        print(f"Saved {len(rows)} attempts to {path}\n")
    print(format_summary(rows))


if __name__ == "__main__":
    main()
