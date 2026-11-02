"""Tests for the gesture logic. These use made-up hand poses, so they need no
webcam or model file. Accuracy on real hands is measured separately in Week 4.
"""

import numpy as np

from gesture_detection import (
    CLOSE_GRIPPER,
    FIST,
    MOVE_LEFT,
    MOVE_RIGHT,
    OPEN_HAND,
    POINT_LEFT,
    POINT_RIGHT,
    STOP,
    UNKNOWN,
    CommandFilter,
    classify_gesture,
)

FINGERS = {"index": (6, 8), "middle": (10, 12), "ring": (14, 16), "pinky": (18, 20)}


def make_world(extended=(), half_bent=()):
    """Build a hand at the origin. Listed fingers are straight or half bent; the rest are curled."""
    world = np.zeros((21, 3))
    for name, (pip, tip) in FINGERS.items():
        world[pip] = (0, 0.13, 0)
        if name in extended:
            ratio = 1.35
        elif name in half_bent:
            ratio = 1.1
        else:
            ratio = 0.7
        world[tip] = (0, 0.13 * ratio, 0)
    return world


def make_image(dx, dy):
    """Build image landmarks where the index finger points along (dx, dy)."""
    image = np.zeros((21, 2))
    image[5] = (300, 300)
    image[8] = (300 + dx, 300 + dy)
    return image


POINTING = make_world(extended=("index",))
ANY_IMAGE = make_image(0, 0)


def test_open_hand():
    assert classify_gesture(make_world(extended=FINGERS), ANY_IMAGE) == OPEN_HAND


def test_fist():
    assert classify_gesture(make_world(), ANY_IMAGE) == FIST


def test_point_left_and_right_in_mirrored_frame():
    assert classify_gesture(POINTING, make_image(-100, 0), mirrored=True) == POINT_LEFT
    assert classify_gesture(POINTING, make_image(100, 0), mirrored=True) == POINT_RIGHT


def test_point_direction_swaps_in_unmirrored_frame():
    assert classify_gesture(POINTING, make_image(-100, 0), mirrored=False) == POINT_RIGHT
    assert classify_gesture(POINTING, make_image(100, 0), mirrored=False) == POINT_LEFT


def test_pointing_up_or_at_camera_is_unknown():
    assert classify_gesture(POINTING, make_image(0, -100)) == UNKNOWN
    assert classify_gesture(POINTING, make_image(0, 0)) == UNKNOWN
    assert classify_gesture(POINTING, make_image(100, -100)) == UNKNOWN


def test_ambiguous_fingers_are_unknown():
    assert classify_gesture(make_world(extended=("index", "middle")), make_image(100, 0)) == UNKNOWN
    assert classify_gesture(make_world(half_bent=FINGERS), ANY_IMAGE) == UNKNOWN


def feed(command_filter, gesture, frames):
    return [command_filter.update(gesture) for _ in range(frames)]


def test_command_needs_consecutive_frames():
    assert feed(CommandFilter(), FIST, 5) == [None, None, None, None, CLOSE_GRIPPER]


def test_stop_confirms_faster_than_other_commands():
    assert feed(CommandFilter(), OPEN_HAND, 2) == [None, STOP]


def test_single_misdetection_does_not_change_command():
    command_filter = CommandFilter()
    feed(command_filter, POINT_LEFT, 5)
    assert command_filter.update(FIST) == MOVE_LEFT
    assert command_filter.update(POINT_LEFT) == MOVE_LEFT


def test_switching_commands():
    command_filter = CommandFilter()
    feed(command_filter, POINT_LEFT, 5)
    assert feed(command_filter, POINT_RIGHT, 5)[-1] == MOVE_RIGHT


def test_unknown_clears_command_after_confirmation():
    command_filter = CommandFilter()
    feed(command_filter, POINT_LEFT, 5)
    assert feed(command_filter, UNKNOWN, 5) == [MOVE_LEFT, MOVE_LEFT, MOVE_LEFT, MOVE_LEFT, None]


def test_reset_clears_active_command():
    command_filter = CommandFilter()
    feed(command_filter, FIST, 5)
    command_filter.reset()
    assert command_filter.active is None
    assert command_filter.update(FIST) is None
