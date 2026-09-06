# coding=utf-8
"""End to end tests of the closed loop against a simulated printer and camera."""

import numpy as np
import pytest

from nozzlealign_pkg import routine, vision
from nozzlealign_pkg.settings import DEFAULTS

from fakes import BoreCamera, FakeBridge

cv2 = pytest.importorskip("cv2")

# Where the active T0 nozzle sits over the lens in the fake machine frame.
CAMERA_XY = (175.3, 284.9)
FRAME = (400, 640)


def make_config(**overrides):
    config = dict(DEFAULTS)
    config.update(
        snapshot_url="http://camera/frame.jpeg",
        frame_average=1,
        camera_x=CAMERA_XY[0],
        camera_y=CAMERA_XY[1],
        camera_z=30.0,
        home_first=True,
        tool_settle_s=0.0,
        settle_s=0.0,
        target_x_px=FRAME[1] / 2.0,
        target_y_px=FRAME[0] / 2.0,
        search_span_mm=6.0,
        search_step_mm=4.0,
        search_radius_px=300,
        bore_search_radius_px=220,
        track_search_px=120,
        focus_template_px=60,
        max_passes=10,
    )
    config.update(overrides)
    return config


class Recorder(object):
    def __init__(self):
        self.messages = []

    def __call__(self, payload):
        self.messages.append(payload)

    def text(self):
        return "\n".join(m.get("message", "") for m in self.messages)

    def last(self):
        return self.messages[-1]


class NullLogger(object):
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def exception(self, *args, **kwargs):
        pass


def build(monkeypatch, bridge=None, camera_xy=CAMERA_XY, focus_z=32.0, **config_overrides):
    bridge = bridge or FakeBridge()
    camera = BoreCamera(bridge, camera_xy, focus_z=focus_z, size=FRAME)
    monkeypatch.setattr(
        vision, "average_frames",
        lambda url, count=1, timeout=1.0, settle=0, attempts=4: camera.frame())
    recorder = Recorder()
    cameras = []
    job = routine.CalibrationRoutine(
        bridge, make_config(**config_overrides), recorder, NullLogger(),
        on_camera=cameras.append)
    job.cameras = cameras
    return bridge, camera, job, recorder


def run(job):
    job.run()
    return job.result


# -- the answer ---------------------------------------------------------------

def test_recovers_the_true_offset_despite_backlash(monkeypatch):
    bridge, _, job, recorder = build(monkeypatch)
    result = run(job)
    assert result is not None, recorder.text()
    assert result["new_offset"][0] == pytest.approx(bridge.true_offset[0], abs=0.01)
    assert result["new_offset"][1] == pytest.approx(bridge.true_offset[1], abs=0.01)
    assert result["correction"][0] == pytest.approx(25.20 - 25.56, abs=0.01)


def test_a_correct_offset_measures_a_zero_correction(monkeypatch):
    bridge = FakeBridge(stored_offset=(25.56, 0.64, -0.891), true_offset=(25.56, 0.64))
    _, _, job, recorder = build(monkeypatch, bridge=bridge)
    result = run(job)
    assert result is not None, recorder.text()
    assert abs(result["correction"][0]) < 0.01
    assert abs(result["correction"][1]) < 0.01


def test_the_corrected_offset_matches_the_machine():
    # Measured 2026-09-06: T0 on target at X175.2744 Y284.9063, T1 on the
    # same pixel at X174.7822 Y284.6174, with X25.07 Y0.35 stored.
    new = routine.corrected_offset(
        (25.07, 0.35, -0.891), (175.2744, 284.9063), (174.7822, 284.6174))
    assert new[0] == pytest.approx(25.562, abs=0.001)
    assert new[1] == pytest.approx(0.639, abs=0.001)


# -- the raised nozzle ---------------------------------------------------------

def test_a_raised_t1_nozzle_over_the_lens_is_noticed_and_skipped(monkeypatch):
    """The camera is where T1's raised nozzle sits under T0, as it was on the machine."""
    bridge = FakeBridge()
    raised = np.array(CAMERA_XY) - bridge.true_offset - bridge.lift_shift
    _, _, job, recorder = build(monkeypatch, bridge=bridge, camera_xy=CAMERA_XY,
                                camera_x=raised[0], camera_y=raised[1])
    result = run(job)
    assert result is not None, recorder.text()
    assert "raised T1 nozzle" in recorder.text()
    assert result["new_offset"][0] == pytest.approx(bridge.true_offset[0], abs=0.01)
    assert result["new_offset"][1] == pytest.approx(bridge.true_offset[1], abs=0.01)
    assert result["camera"]["camera_x"] == pytest.approx(CAMERA_XY[0], abs=0.05)


# -- focus -----------------------------------------------------------------------

def test_the_focus_height_is_measured_not_assumed(monkeypatch):
    _, _, job, recorder = build(monkeypatch, focus_z=31.4, camera_z=29.0)
    result = run(job)
    assert result is not None, recorder.text()
    assert result["focus_z"] == pytest.approx(31.4, abs=0.35)
    assert job.cameras[-1]["camera_z"] == pytest.approx(31.4, abs=0.35)


def test_no_z_ever_goes_below_the_floor(monkeypatch):
    bridge, _, job, recorder = build(monkeypatch, focus_z=22.0, camera_z=24.0, min_z=23.0)
    run(job)
    assert min(bridge.z_commands()) >= 23.0
    assert "peak" in recorder.text() or job.result is not None


def test_a_camera_height_below_the_floor_is_refused_before_any_move(monkeypatch):
    bridge, _, job, recorder = build(monkeypatch, camera_z=15.0, min_z=20.0)
    run(job)
    assert job.result is None
    assert recorder.last()["type"] == "failed"
    assert "floor" in recorder.last()["message"]
    # only the park move to the safe height, no travel and no descent
    assert all(z >= DEFAULTS["safe_z"] for z in bridge.z_commands())
    assert not any(c.startswith("G1 X") for c in bridge.sent)


# -- failure and safety --------------------------------------------------------

def test_a_lost_connection_fails_cleanly(monkeypatch):
    bridge = FakeBridge(fail_after=40)
    _, _, job, recorder = build(monkeypatch, bridge=bridge)
    run(job)
    assert job.result is None
    assert recorder.last()["type"] == "failed"


def test_an_abort_parks_high_on_t0(monkeypatch):
    bridge, _, job, recorder = build(monkeypatch)
    original = job._build_pixel_map

    def abort_then_map():
        job.abort()
        return original()

    job._build_pixel_map = abort_then_map
    run(job)
    assert recorder.last()["type"] == "aborted"
    assert bridge.sent[-1] == "T0"
    assert bridge.logical[2] == pytest.approx(DEFAULTS["safe_z"])


def test_failure_to_converge_is_an_error_not_an_answer(monkeypatch):
    _, _, job, recorder = build(monkeypatch, max_passes=1, tolerance_mm=0.0001)
    run(job)
    assert job.result is None
    assert "did not settle" in recorder.last()["message"]


def test_travel_goes_through_the_safe_height(monkeypatch):
    bridge, _, job, _ = build(monkeypatch)
    run(job)
    heights = bridge.z_commands()
    assert heights[0] == pytest.approx(DEFAULTS["safe_z"])
    assert heights[-1] == pytest.approx(DEFAULTS["safe_z"])
    assert bridge.sent[-1] == "T0"


def test_the_camera_is_found_when_it_has_moved_a_little(monkeypatch):
    bridge, _, job, recorder = build(monkeypatch, camera_x=CAMERA_XY[0] + 4.5,
                                     camera_y=CAMERA_XY[1] - 3.0)
    result = run(job)
    assert result is not None, recorder.text()
    assert "found the bore" in recorder.text()
    assert result["new_offset"][0] == pytest.approx(bridge.true_offset[0], abs=0.01)


def test_the_camera_is_found_anywhere_on_the_bed(monkeypatch):
    """The camera sits 100 mm from where the settings say."""
    bridge, _, job, recorder = build(
        monkeypatch, camera_xy=(80.0, 150.0),
        bed_x_min=40.0, bed_x_max=205.0, bed_y_min=60.0, bed_y_max=270.0,
        bed_step_x=55.0, bed_step_y=35.0)
    result = run(job)
    assert result is not None, recorder.text()
    assert "searching the whole bed" in recorder.text()
    assert "over the camera near" in recorder.text()
    assert result["new_offset"][0] == pytest.approx(bridge.true_offset[0], abs=0.01)
    assert result["new_offset"][1] == pytest.approx(bridge.true_offset[1], abs=0.01)
    assert result["camera"]["camera_x"] == pytest.approx(80.0, abs=0.05)


def test_no_toolhead_anywhere_is_a_clear_error(monkeypatch):
    bridge, camera, job, recorder = build(
        monkeypatch, camera_xy=(600.0, 600.0),
        bed_x_min=40.0, bed_x_max=150.0, bed_y_min=60.0, bed_y_max=130.0)
    run(job)
    assert job.result is None
    assert "never came into view" in recorder.last()["message"]
    assert min(bridge.z_commands()) >= DEFAULTS["min_z"]


def test_every_setting_the_routine_reads_is_declared():
    import re
    import pathlib

    source = (pathlib.Path(__file__).resolve().parents[1]
              / "octoprint_nozzlealign" / "routine.py").read_text()
    keys = set(re.findall(r'cfg\[\s*"([a-z_]+)"\s*\]', source))
    keys |= set(re.findall(r'cfg\.get\(\s*"([a-z_]+)"', source))
    keys |= set(re.findall(r'self\._cfg\[\s*"([a-z_]+)"\s*\]', source))
    missing = keys - set(DEFAULTS)
    assert not missing, "routine reads settings with no default: %s" % sorted(missing)
