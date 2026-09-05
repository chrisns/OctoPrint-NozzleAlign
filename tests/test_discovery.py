# coding=utf-8
"""The camera finds itself. Nobody types a coordinate."""

import numpy as np
import pytest

from nozzlealign_pkg import discovery, vision

from fakes import FakeBridge, PerspectiveCamera
from test_routine import NullLogger, Recorder, make_config

CAMERA_XY = (162.4, 171.8)
LENS_Z = 18.0
FOCUS_Z = 34.0


def discovery_config(**overrides):
    config = make_config()
    config.update(
        strategy="motion",
        motion_threshold=4.0,
        motion_min_area=60,
        motion_min_circularity=0.25,
        motion_max_extent=0.4,
        motion_probe_mm=0.6,
        search_z=80.0,
        search_centre_x=160.0,
        search_centre_y=175.0,
        search_span_mm=120.0,
        search_points=3,
        search_probe_mm=3.0,
        coarse_step=10.0,
        fine_step=2.0,
        min_z=12.0,
        lens_clearance_mm=6.0,
        max_blob_fraction=0.15,
        focus_window_px=240,
        focus_drop_ratio=0.6,
        template_size_px=96,
        discovery_tolerance_mm=0.2,
        max_passes=8,
    )
    config.update(overrides)
    return config


def build(monkeypatch, camera_xy=CAMERA_XY, bright=False, **overrides):
    bridge = FakeBridge()
    camera = PerspectiveCamera(
        bridge, camera_xy, lens_z=LENS_Z, focus_z=FOCUS_Z, bright=bright
    )
    monkeypatch.setattr(
        vision, "average_frames",
        lambda url, count=1, timeout=1.0, settle=0: camera.frame(),
    )
    stored = {}

    def keep(result, template):
        stored["result"] = result
        stored["template"] = template

    job = discovery.DiscoveryRoutine(
        bridge, discovery_config(**overrides), Recorder(), NullLogger(), on_result=keep
    )
    return bridge, camera, job, stored


# -- the motion detector ----------------------------------------------------


def test_motion_finds_a_dark_nozzle_in_both_frames(monkeypatch):
    bridge = FakeBridge()
    camera = PerspectiveCamera(bridge, CAMERA_XY, LENS_Z, FOCUS_Z)
    bridge.position_xyz = [CAMERA_XY[0], CAMERA_XY[1], FOCUS_Z]
    before_px = camera.tip_pixel()
    before = camera.frame()
    bridge.position_xyz[0] += 2.0
    after_px = camera.tip_pixel()
    after = camera.frame()
    found_a, found_b, _ = vision.locate_by_motion(before, after)
    assert found_a == pytest.approx(tuple(before_px), abs=3.0)
    assert found_b == pytest.approx(tuple(after_px), abs=3.0)


def test_motion_works_for_a_lit_nozzle_too(monkeypatch):
    """The nozzle may be bright against a dark enclosure, or the reverse."""
    bridge = FakeBridge()
    camera = PerspectiveCamera(bridge, CAMERA_XY, LENS_Z, FOCUS_Z, bright=True)
    bridge.position_xyz = [CAMERA_XY[0], CAMERA_XY[1], FOCUS_Z]
    before_px = camera.tip_pixel()
    before = camera.frame()
    bridge.position_xyz[1] += 2.0
    after_px = camera.tip_pixel()
    after = camera.frame()
    found_a, found_b, _ = vision.locate_by_motion(before, after)
    assert found_a == pytest.approx(tuple(before_px), abs=3.0)
    assert found_b == pytest.approx(tuple(after_px), abs=3.0)


def test_motion_complains_when_nothing_moves():
    still = np.full((400, 400), 200.0, dtype=np.float32)
    still += np.random.default_rng(2).normal(0.0, 1.5, still.shape)
    with pytest.raises(vision.DetectionError):
        vision.locate_by_motion(still, still.copy())


# -- the lens estimate ------------------------------------------------------


def test_lens_estimate_recovers_the_lens_plane():
    lens_z, constant = 18.0, 900.0
    samples = [(z, constant / (z - lens_z)) for z in (80.0, 60.0, 45.0, 35.0)]
    assert discovery.estimate_lens_z(samples) == pytest.approx(lens_z, abs=0.01)


def test_lens_estimate_needs_two_heights():
    assert discovery.estimate_lens_z([(80.0, 12.0)]) is None
    assert discovery.estimate_lens_z([(80.0, 12.0), (80.0, 12.0)]) is None


# -- focus ------------------------------------------------------------------


def test_sharpness_peaks_at_the_focal_plane():
    bridge = FakeBridge()
    camera = PerspectiveCamera(bridge, CAMERA_XY, LENS_Z, FOCUS_Z)
    scores = {}
    for z in (60.0, 50.0, FOCUS_Z, 26.0):
        bridge.position_xyz = [CAMERA_XY[0], CAMERA_XY[1], z]
        frame = camera.frame()
        scores[z] = vision.sharpness(frame, camera.tip_pixel(), 240)
    assert scores[FOCUS_Z] == max(scores.values())


# -- the whole routine ------------------------------------------------------


def test_discovery_finds_the_camera(monkeypatch):
    bridge, camera, job, stored = build(monkeypatch)
    result = job._execute()
    assert result["camera_x"] == pytest.approx(CAMERA_XY[0], abs=0.3)
    assert result["camera_y"] == pytest.approx(CAMERA_XY[1], abs=0.3)
    assert result["camera_z"] == pytest.approx(FOCUS_Z, abs=3.0)
    assert result["lens_z"] == pytest.approx(LENS_Z, abs=3.0)
    assert stored["result"]["camera_x"] == result["camera_x"]
    assert stored["template"] is not None


def test_discovery_searches_a_grid_when_the_camera_is_off_centre(monkeypatch):
    off_centre = (215.0, 130.0)
    bridge, camera, job, stored = build(monkeypatch, camera_xy=off_centre)
    result = job._execute()
    assert result["camera_x"] == pytest.approx(off_centre[0], abs=0.3)
    assert result["camera_y"] == pytest.approx(off_centre[1], abs=0.3)


def test_discovery_never_goes_below_the_lens(monkeypatch):
    bridge, camera, job, stored = build(monkeypatch)
    job._execute()
    heights = [
        float(c.split("Z")[1].split(" ")[0])
        for c in bridge.sent
        if c.startswith("G1 Z") or (c.startswith("G1") and " Z" in c)
    ]
    assert heights, "the routine never moved in Z"
    assert min(heights) > LENS_Z


def test_discovery_respects_the_hard_floor(monkeypatch):
    bridge, camera, job, stored = build(monkeypatch, min_z=55.0)
    result = job._execute()
    assert result["camera_z"] >= 55.0


def test_discovery_gives_up_clearly_when_the_nozzle_never_appears(monkeypatch):
    bridge = FakeBridge()

    def empty(url, count=1, timeout=1.0, settle=0):
        frame = np.full((800, 1280), 200.0, dtype=np.float32)
        return frame + np.random.default_rng(3).normal(0.0, 1.5, frame.shape)

    monkeypatch.setattr(vision, "average_frames", empty)
    job = discovery.DiscoveryRoutine(
        bridge, discovery_config(), Recorder(), NullLogger()
    )
    with pytest.raises(discovery.CalibrationError) as caught:
        job._execute()
    assert "could not see the nozzle" in str(caught.value)


def test_motion_ignores_the_gantry_beam():
    """Moving the toolhead also moves the gantry, which is not the nozzle."""
    height, width = 800, 1280
    def render(nozzle_x, beam_y):
        image = np.full((height, width), 200.0, dtype=np.float32)
        # the gantry beam: a long thin bar across the whole frame
        image[int(beam_y):int(beam_y) + 26, :] = 40.0
        # the nozzle: a small compact disc
        ys, xs = np.mgrid[0:height, 0:width]
        image[(xs - nozzle_x) ** 2 + (ys - 620) ** 2 <= 22 ** 2] = 30.0
        return image + np.random.default_rng(4).normal(0.0, 1.5, image.shape)

    before = render(600.0, 200.0)
    after = render(660.0, 232.0)
    found_a, found_b, details = vision.locate_by_motion(before, after)
    assert found_a == pytest.approx((600.0, 620.0), abs=6.0)
    assert found_b == pytest.approx((660.0, 620.0), abs=6.0)
    assert details["circularity"] > 0.5


@pytest.mark.parametrize("rotation", [0.0, 37.0, 91.5, -128.0, 179.0])
def test_discovery_copes_with_any_camera_rotation(monkeypatch, rotation):
    """The camera is never square to the machine axes, and need not be.

    The pixel map is a full 2x2 matrix measured from two known moves, so it
    absorbs rotation, scale and mirroring together. No setting describes the
    camera orientation, and none has to.
    """
    bridge = FakeBridge()
    camera = PerspectiveCamera(
        bridge, CAMERA_XY, lens_z=LENS_Z, focus_z=FOCUS_Z, rotation=rotation
    )
    monkeypatch.setattr(
        vision, "average_frames",
        lambda url, count=1, timeout=1.0, settle=0: camera.frame(),
    )
    job = discovery.DiscoveryRoutine(
        bridge, discovery_config(), Recorder(), NullLogger()
    )
    result = job._execute()
    assert result["camera_x"] == pytest.approx(CAMERA_XY[0], abs=0.3)
    assert result["camera_y"] == pytest.approx(CAMERA_XY[1], abs=0.3)


@pytest.mark.parametrize("rotation", [0.0, 37.0, -128.0])
def test_calibration_copes_with_any_camera_rotation(monkeypatch, rotation):
    """The offset measurement is just as indifferent to camera orientation."""
    from nozzlealign_pkg import routine as routine_module
    from test_routine import build as build_calibration

    error = (0.31, -0.24)
    _, _, job = build_calibration(monkeypatch, error, rotation=rotation)
    result = job._execute()
    assert result["correction"] == pytest.approx([-error[0], -error[1]], abs=0.02)
