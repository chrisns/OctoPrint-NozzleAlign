# coding=utf-8
import numpy as np
import pytest

from nozzlealign_pkg import vision

from fakes import FakeBridge, FakeCamera, rotation_matrix


def make_frame(cx, cy, radius=40.0, size=(800, 1280), background=210.0, ink=25.0):
    height, width = size
    image = np.full((height, width), background, dtype=np.float32)
    ys, xs = np.mgrid[0:height, 0:width]
    image[(xs - cx) ** 2 + (ys - cy) ** 2 <= radius ** 2] = ink
    return image


def test_frame_health_spots_a_dead_stream():
    dead = np.full((800, 1280), 128.0, dtype=np.float32)
    dead += np.random.default_rng(1).normal(0.0, 2.0, dead.shape)
    mean, deviation = vision.frame_health(dead)
    assert mean == pytest.approx(128.0, abs=1.0)
    assert deviation < 5.0
    assert vision.is_blank(dead)


def test_frame_health_accepts_a_real_picture():
    frame = make_frame(640, 400)
    assert not vision.is_blank(frame)


def test_contour_finds_the_tip():
    frame = make_frame(700.0, 350.0, radius=50.0)
    x, y, confidence, details = vision.detect_contour(frame, invert=True)
    assert x == pytest.approx(700.0, abs=1.5)
    assert y == pytest.approx(350.0, abs=1.5)
    assert confidence > 0.7
    assert details["radius"] == pytest.approx(50.0, abs=2.0)


def test_contour_respects_the_region_of_interest():
    frame = make_frame(640.0, 400.0, radius=30.0)
    # a decoy blob outside the region of interest
    frame[50:110, 50:110] = 25.0
    x, y, _, _ = vision.detect_contour(frame, roi=(0.5, 0.5, 0.4), invert=True)
    assert x == pytest.approx(640.0, abs=2.0)
    assert y == pytest.approx(400.0, abs=2.0)


def test_contour_rejects_a_speck():
    frame = np.full((800, 1280), 210.0, dtype=np.float32)
    frame[400:403, 640:643] = 25.0
    with pytest.raises(vision.DetectionError):
        vision.detect_contour(frame, invert=True, min_area=200)


def test_hough_finds_a_circle():
    frame = make_frame(500.0, 300.0, radius=60.0)
    x, y, _, _ = vision.detect_hough(frame, min_radius=40, max_radius=90, param2=25)
    assert x == pytest.approx(500.0, abs=4.0)
    assert y == pytest.approx(300.0, abs=4.0)


def test_template_matches_to_sub_pixel():
    reference = make_frame(640.0, 400.0, radius=35.0)
    patch = reference[360:440, 600:680].copy()
    moved = make_frame(660.0, 415.0, radius=35.0)
    x, y, score, _ = vision.detect_template(moved, patch)
    assert x == pytest.approx(660.0, abs=1.0)
    assert y == pytest.approx(415.0, abs=1.0)
    assert score > 0.9


def test_detect_tip_dispatches_to_the_named_strategy():
    frame = make_frame(640.0, 400.0)
    x, _, _, _ = vision.detect_tip(frame, strategy="contour", invert=True)
    assert x == pytest.approx(640.0, abs=2.0)
    with pytest.raises(vision.DetectionError):
        vision.detect_tip(frame, strategy="template", template=None)


def test_camera_model_agrees_with_the_detector():
    """The fake camera and the real detector must see the same tip."""
    machine = FakeBridge()
    machine.position_xyz = [150.0, 175.0, 5.0]
    matrix = rotation_matrix(60.0, 8.0)
    camera = FakeCamera(machine, matrix, (150.0, 175.0), (640.0, 400.0))
    x, y, _, _ = vision.detect_contour(camera.frame(), invert=True)
    expected = camera.tip_pixel()
    assert x == pytest.approx(expected[0], abs=2.0)
    assert y == pytest.approx(expected[1], abs=2.0)
