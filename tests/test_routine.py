# coding=utf-8
"""End to end test of the closed loop against a simulated printer and camera."""

import numpy as np
import pytest

from nozzlealign_pkg import routine, vision

from fakes import FakeBridge, FakeCamera, rotation_matrix

CAMERA_XY = (150.0, 175.0)
CENTRE_PX = (640.0, 400.0)


def make_config(**overrides):
    config = dict(
        snapshot_url="http://camera/frame.jpeg",
        http_timeout=5.0,
        frame_average=1,
        camera_x=CAMERA_XY[0],
        camera_y=CAMERA_XY[1],
        camera_z=3.0,
        safe_z=50.0,
        home_first=True,
        feedrate=3000,
        fine_feedrate=600,
        z_feedrate=600,
        move_timeout=30.0,
        home_timeout=60.0,
        tool_settle_s=0.0,
        probe_distance=1.0,
        tolerance_mm=0.005,
        max_passes=8,
        max_correction_mm=5.0,
        target_x_px=CENTRE_PX[0],
        target_y_px=CENTRE_PX[1],
        strategy="contour",
        motion_threshold=4.0,
        motion_min_area=60,
        motion_min_circularity=0.25,
        motion_max_extent=0.4,
        motion_probe_mm=0.6,
        contour_invert=True,
        blur=5,
        min_area=200,
        min_radius=10,
        max_radius=200,
        hough_param2=30,
        min_confidence=0.3,
        roi=None,
        template=None,
        nominal_offset_x=26.0,
        nominal_offset_y=0.0,
        offset_limit_mm=3.0,
        save_to_eeprom=True,
    )
    config.update(overrides)
    return config


class Recorder(object):
    def __init__(self):
        self.messages = []

    def __call__(self, payload):
        self.messages.append(payload)


class NullLogger(object):
    def info(self, *args, **kwargs):
        pass

    def exception(self, *args, **kwargs):
        pass


def build(monkeypatch, tool_error, scale=60.0, rotation=7.0, **config_overrides):
    bridge = FakeBridge(tool_error=tool_error)
    matrix = rotation_matrix(scale, rotation)
    camera = FakeCamera(bridge, matrix, CAMERA_XY, CENTRE_PX)
    monkeypatch.setattr(
        vision, "average_frames", lambda url, count=1, timeout=1.0, settle=0: camera.frame()
    )
    job = routine.CalibrationRoutine(bridge, make_config(**config_overrides), Recorder(), NullLogger())
    return bridge, camera, job


def test_measures_a_known_tool_error(monkeypatch):
    error = (0.42, -0.31)
    bridge, _, job = build(monkeypatch, error)
    result = job._execute()
    # nozzle 0 has no error, so it centres at the camera position
    assert result["position_0"] == pytest.approx(list(CAMERA_XY), abs=0.01)
    # nozzle 1 has to be commanded short by its own error
    assert result["correction"] == pytest.approx([-error[0], -error[1]], abs=0.02)
    assert result["residual_0_mm"] <= 0.005
    assert result["residual_1_mm"] <= 0.005


def test_reports_the_measured_scale_and_rotation(monkeypatch):
    _, _, job = build(monkeypatch, (0.1, 0.1), scale=72.0, rotation=15.0)
    result = job._execute()
    assert result["px_per_mm"] == pytest.approx(72.0, rel=0.05)
    # the fake camera mirrors Y, so the reported angle is the mirrored one
    assert abs(result["rotation_deg"]) == pytest.approx(15.0, abs=1.5)


def test_offers_both_sign_conventions(monkeypatch):
    _, _, job = build(monkeypatch, (0.25, 0.0))
    result = job._execute()
    stored = result["stored_offset"]
    correction = result["correction"]
    assert result["candidates"]["plus"] == pytest.approx(
        [stored[0] + correction[0], stored[1] + correction[1]], abs=1e-6
    )
    assert result["candidates"]["minus"] == pytest.approx(
        [stored[0] - correction[0], stored[1] - correction[1]], abs=1e-6
    )


def test_zero_error_measures_zero(monkeypatch):
    _, _, job = build(monkeypatch, (0.0, 0.0))
    result = job._execute()
    assert result["correction"] == pytest.approx([0.0, 0.0], abs=0.02)


def test_refuses_a_correction_beyond_the_limit(monkeypatch):
    _, _, job = build(monkeypatch, (0.2, 0.0), max_correction_mm=0.05)
    with pytest.raises(routine.CalibrationError) as caught:
        job._execute()
    assert "beyond" in str(caught.value)


def test_travels_through_the_safe_z(monkeypatch):
    bridge, _, job = build(monkeypatch, (0.1, 0.1))
    job._execute()
    lowered = [i for i, c in enumerate(bridge.sent) if "Z3.000" in c]
    raised = [i for i, c in enumerate(bridge.sent) if "Z50.000" in c]
    assert lowered and raised
    # every descent onto the camera is preceded by a climb to the safe height
    for index in lowered:
        assert any(r < index for r in raised)


def test_stops_when_aborted(monkeypatch):
    _, _, job = build(monkeypatch, (0.1, 0.1))
    job.abort()
    with pytest.raises(routine.Aborted):
        job._execute()


def test_fails_clearly_on_a_blank_camera(monkeypatch):
    bridge = FakeBridge()

    def blank(url, count=1, timeout=1.0, settle=0):
        raise vision.CaptureError("camera frame is blank")

    monkeypatch.setattr(vision, "average_frames", blank)
    job = routine.CalibrationRoutine(bridge, make_config(), Recorder(), NullLogger())
    with pytest.raises(vision.CaptureError):
        job._execute()


def test_fails_when_the_offset_cannot_be_read(monkeypatch):
    bridge, _, job = build(monkeypatch, (0.1, 0.1))
    bridge.read_hotend_offset = lambda tool=1, timeout=30.0: None
    with pytest.raises(routine.CalibrationError) as caught:
        job._execute()
    assert "M218" in str(caught.value)
