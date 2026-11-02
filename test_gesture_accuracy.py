"""Tests for the accuracy test logic. These use fake cameras and detectors, so
they need no webcam or model file.
"""

import numpy as np

from gesture_accuracy import (
    GESTURES,
    NO_COMMAND,
    format_summary,
    make_attempts,
    predicted_label,
    read_results,
    run_session,
    write_results,
)
from gesture_detection import (
    CLOSE_GRIPPER,
    FIST,
    LOST,
    MOVE_LEFT,
    OPEN_HAND,
    POINT_LEFT,
    STOP,
    TRACKING,
    GestureOutput,
)


class FakeCamera:
    def read(self):
        return True, np.zeros((4, 4, 3), dtype=np.uint8)


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        self.now += 0.1
        return self.now


class FakeDetector:
    last_landmarks = None

    def __init__(self, output):
        self.output = output

    def process(self, frame, timestamp):
        return self.output


def always_continue(frame, output, detector, lines):
    return True


def make_row(expected, predicted, participant="p1", lost_frames=0):
    return {
        "participant": participant,
        "attempt": 1,
        "expected": expected,
        "predicted": predicted,
        "frames": 10,
        "matching_frames": 10,
        "lost_frames": lost_frames,
    }


def test_predicted_label():
    assert predicted_label(GestureOutput(CLOSE_GRIPPER, TRACKING, 0.0)) == FIST
    assert predicted_label(GestureOutput(None, TRACKING, 0.0)) == NO_COMMAND
    assert predicted_label(GestureOutput(STOP, TRACKING, 0.0)) == OPEN_HAND


def test_lost_tracking_is_not_counted_as_open_hand():
    assert predicted_label(GestureOutput(STOP, LOST, 0.0)) == LOST


def test_make_attempts_covers_each_gesture_equally():
    attempts = make_attempts(3)
    assert len(attempts) == 3 * len(GESTURES)
    assert all(attempts.count(gesture) == 3 for gesture in GESTURES)


def test_session_records_correct_attempt():
    detector = FakeDetector(GestureOutput(CLOSE_GRIPPER, TRACKING, 0.0, FIST))
    rows = run_session(FakeCamera(), detector, "p1", [FIST], always_continue, FakeClock())
    assert len(rows) == 1
    assert rows[0]["predicted"] == FIST
    assert rows[0]["frames"] > 0
    assert rows[0]["matching_frames"] == rows[0]["frames"]
    assert rows[0]["lost_frames"] == 0


def test_session_records_wrong_and_lost_attempts():
    detector = FakeDetector(GestureOutput(MOVE_LEFT, TRACKING, 0.0, POINT_LEFT))
    assert run_session(FakeCamera(), detector, "p1", [FIST], always_continue, FakeClock())[0]["predicted"] == POINT_LEFT

    detector = FakeDetector(GestureOutput(STOP, LOST, 0.0))
    row = run_session(FakeCamera(), detector, "p1", [OPEN_HAND], always_continue, FakeClock())[0]
    assert row["predicted"] == LOST
    assert row["lost_frames"] == row["frames"]


def test_quitting_keeps_finished_attempts():
    calls = {"count": 0}

    def quit_during_second_attempt(frame, output, detector, lines):
        calls["count"] += 1
        return "Attempt 2" not in lines[0]

    detector = FakeDetector(GestureOutput(CLOSE_GRIPPER, TRACKING, 0.0, FIST))
    rows = run_session(FakeCamera(), detector, "p1", [FIST, FIST, FIST], quit_during_second_attempt, FakeClock())
    assert [row["attempt"] for row in rows] == [1]


def test_camera_failure_ends_session():
    class BrokenCamera:
        def read(self):
            return False, None

    detector = FakeDetector(GestureOutput(CLOSE_GRIPPER, TRACKING, 0.0, FIST))
    assert run_session(BrokenCamera(), detector, "p1", [FIST], always_continue, FakeClock()) == []


def test_summary_reports_accuracy_and_confusions():
    rows = [
        make_row(FIST, FIST),
        make_row(FIST, FIST),
        make_row(FIST, FIST),
        make_row(FIST, OPEN_HAND),
        make_row(OPEN_HAND, LOST, participant="p2", lost_frames=10),
    ]
    summary = format_summary(rows)
    assert "Participants: 2" in summary
    assert "3/4 (75%)  confused with: open_hand x1" in summary
    assert "0/1 (0%)  confused with: lost x1" in summary
    assert "point_left   no attempts" in summary
    assert "Overall: 3/5 (60%)" in summary
    assert "1 attempts ended with tracking lost, 10/50 captured frames lost" in summary


def test_summary_with_no_attempts():
    assert format_summary([]) == "No attempts recorded."


def test_saved_results_summarize_the_same(tmp_path):
    rows = [make_row(FIST, FIST), make_row(POINT_LEFT, NO_COMMAND)]
    path = tmp_path / "nested" / "results.csv"
    write_results(rows, path)
    assert format_summary(read_results([path])) == format_summary(rows)
