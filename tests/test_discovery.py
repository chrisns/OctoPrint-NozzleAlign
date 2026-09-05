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
        motion_min_coverage=0.5,
        motion_probe_mm=0.6,
        search_z=80.0,
        search_x_min=25.0,
        search_x_max=295.0,
        search_y_min=25.0,
        search_y_max=325.0,
        coarse_max_correction_mm=250.0,
        raster_spacing_x=90.0,
        raster_spacing_y=60.0,
        search_probe_mm=3.0,
        map_check_scale=0.7,
        map_check_tolerance=0.25,
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


def test_motion_will_not_call_a_frame_wide_change_the_nozzle():
    """The gantry beam is in view too, so nothing here can be trusted as a tip.

    A round difference blob is not enough. When the two blobs account for only
    a small part of everything that moved, some of it is structure other than
    the nozzle, so the measurement stays coarse and the caller is told not to
    treat the position as a nozzle.

    Only the decision is asserted here. Shift accuracy for a rigidly moving
    structure is covered by the two tests above.
    """
    height, width = 800, 1280
    # a real beam carries bolts and edges, so its sideways movement is visible
    beam = np.random.default_rng(9).normal(40.0, 25.0, (26, width))

    def render(dx, dy):
        image = np.full((height, width), 200.0, dtype=np.float32)
        top = int(200 + dy)
        image[top:top + 26, :] = np.roll(beam, int(dx), axis=1)
        # the nozzle: a small compact disc, moving with it
        ys, xs = np.mgrid[0:height, 0:width]
        image[(xs - (600 + dx)) ** 2 + (ys - (620 + dy)) ** 2 <= 22 ** 2] = 30.0
        return image + np.random.default_rng(4).normal(0.0, 1.5, image.shape)

    measured = vision.measure_motion(render(0, 0), render(24, 9))
    assert measured["compact"] is False
    with pytest.raises(vision.DetectionError):
        vision.locate_by_motion(render(0, 0), render(24, 9))


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


# -- the two regimes --------------------------------------------------------


def test_measure_motion_reports_a_compact_nozzle():
    height, width = 600, 900
    def render(cx):
        image = np.full((height, width), 200.0, dtype=np.float32)
        ys, xs = np.mgrid[0:height, 0:width]
        image[(xs - cx) ** 2 + (ys - 300) ** 2 <= 25 ** 2] = 30.0
        return image + np.random.default_rng(5).normal(0.0, 1.5, image.shape)

    measured = vision.measure_motion(render(400.0), render(452.0))
    assert measured["compact"] is True
    assert measured["position"] == pytest.approx((400.0, 300.0), abs=3.0)
    assert measured["shift"] == pytest.approx((52.0, 0.0), abs=3.0)


def test_measure_motion_falls_back_to_phase_correlation_at_long_range():
    """Far from the camera the whole toolhead moves and no blob is compact.

    Blob centroids would be badly wrong there, because a small move of a large
    object only changes a thin crescent at each edge. Phase correlation over the
    region that changed measures the real displacement instead.
    """
    height, width = 600, 900
    rng = np.random.default_rng(6)
    texture = rng.normal(0.0, 40.0, (120, 160)).astype(np.float32)
    import cv2
    body = cv2.resize(texture, (520, 420), interpolation=cv2.INTER_CUBIC) + 120.0

    def render(x0, y0):
        image = np.full((height, width), 200.0, dtype=np.float32)
        image[y0:y0 + body.shape[0], x0:x0 + body.shape[1]] = body
        return image + rng.normal(0.0, 1.0, image.shape)

    before = render(150, 90)
    after = render(150 + 18, 90 + 7)
    measured = vision.measure_motion(before, after)
    assert measured["compact"] is False
    assert measured["shift"] == pytest.approx((18.0, 7.0), abs=2.0)


def test_discovery_starts_coarse_and_finishes_on_the_nozzle(monkeypatch):
    """The search steers on the whole toolhead, then refines on the nozzle."""
    bridge = FakeBridge()
    camera = PerspectiveCamera(
        bridge, CAMERA_XY, lens_z=LENS_Z, focus_z=FOCUS_Z,
        nozzle_visible_below=45.0,
    )
    monkeypatch.setattr(
        vision, "average_frames",
        lambda url, count=1, timeout=1.0, settle=0: camera.frame(),
    )
    job = discovery.DiscoveryRoutine(
        bridge, discovery_config(), Recorder(), NullLogger()
    )
    result = job._execute()
    assert result["camera_x"] == pytest.approx(CAMERA_XY[0], abs=0.6)
    assert result["camera_y"] == pytest.approx(CAMERA_XY[1], abs=0.6)
    # the answer must come from a height where the nozzle really did separate
    assert result["camera_z"] <= 45.0


def test_discovery_says_so_when_the_nozzle_never_separates(monkeypatch):
    bridge = FakeBridge()
    camera = PerspectiveCamera(
        bridge, CAMERA_XY, lens_z=LENS_Z, focus_z=FOCUS_Z,
        nozzle_visible_below=0.0,
    )
    monkeypatch.setattr(
        vision, "average_frames",
        lambda url, count=1, timeout=1.0, settle=0: camera.frame(),
    )
    job = discovery.DiscoveryRoutine(
        bridge, discovery_config(), Recorder(), NullLogger()
    )
    with pytest.raises(discovery.CalibrationError) as caught:
        job._execute()
    assert "never separated" in str(caught.value)


# -- the whole job, camera position included --------------------------------


@pytest.mark.parametrize("camera_xy", [(162.4, 171.8), (58.0, 305.0), (271.0, 44.0)])
def test_full_calibration_from_scratch_wherever_the_camera_sits(monkeypatch, camera_xy):
    """Put the camera anywhere on the bed and press one button.

    Nothing about the camera is remembered between runs, so this test hands the
    routine no coordinates at all and moves the camera between cases.
    """
    error = (0.37, -0.29)
    bridge = FakeBridge(tool_error=error)
    camera = PerspectiveCamera(
        bridge, camera_xy, lens_z=LENS_Z, focus_z=FOCUS_Z, rotation=23.0
    )
    monkeypatch.setattr(
        vision, "average_frames",
        lambda url, count=1, timeout=1.0, settle=0: camera.frame(),
    )
    config = discovery_config(camera_x=None, camera_y=None, camera_z=None)
    job = discovery.FullCalibration(bridge, config, Recorder(), NullLogger())
    result = job._execute()

    assert result["camera"]["camera_x"] == pytest.approx(camera_xy[0], abs=0.5)
    assert result["camera"]["camera_y"] == pytest.approx(camera_xy[1], abs=0.5)
    assert result["correction"] == pytest.approx([-error[0], -error[1]], abs=0.03)


def test_full_calibration_ignores_a_stale_stored_position(monkeypatch):
    """A camera that has been moved must not be looked for where it used to be."""
    error = (0.2, 0.15)
    actual = (240.0, 90.0)
    bridge = FakeBridge(tool_error=error)
    camera = PerspectiveCamera(bridge, actual, lens_z=LENS_Z, focus_z=FOCUS_Z)
    monkeypatch.setattr(
        vision, "average_frames",
        lambda url, count=1, timeout=1.0, settle=0: camera.frame(),
    )
    # the settings still hold where the camera used to be, and a wrong height
    stale = discovery_config(camera_x=60.0, camera_y=300.0, camera_z=FOCUS_Z - 12.0)
    job = discovery.FullCalibration(bridge, stale, Recorder(), NullLogger())
    result = job._execute()
    assert result["camera"]["camera_x"] == pytest.approx(actual[0], abs=0.5)
    assert result["camera"]["camera_y"] == pytest.approx(actual[1], abs=0.5)
    assert result["correction"] == pytest.approx([-error[0], -error[1]], abs=0.03)


# -- rejecting things that move but are not the toolhead --------------------


class SwingingTube(PerspectiveCamera):
    """A camera that can only see the filament tube, not the toolhead.

    This is what the real rig did on 2026-09-05. Something moved whenever the
    machine moved, and two probe moves happily produced a pixel map from it, but
    it was a bowden tube swinging overhead. A tube does not travel in proportion
    to the commanded move, so a third move in a new direction exposes it.
    """

    def tip_pixel(self):
        physical = self.machine.physical_xy()
        offset = physical - self.camera_xy
        # a hanging tube swings: it lags, saturates, and barely follows Y
        swing = np.array([
            18.0 * np.tanh(offset[0] / 25.0),
            4.0 * np.tanh(offset[1] / 40.0),
        ])
        return self.centre_px + self.matrix().dot(swing)


def test_a_swinging_tube_is_not_mistaken_for_the_toolhead(monkeypatch):
    bridge = FakeBridge()
    camera = SwingingTube(bridge, CAMERA_XY, lens_z=LENS_Z, focus_z=FOCUS_Z)
    monkeypatch.setattr(
        vision, "average_frames",
        lambda url, count=1, timeout=1.0, settle=0: camera.frame(),
    )
    job = discovery.DiscoveryRoutine(
        bridge, discovery_config(), Recorder(), NullLogger()
    )
    with pytest.raises(discovery.CalibrationError) as caught:
        job._execute()
    # however it gives up, it must point at the camera rather than report a
    # confident answer derived from the tube
    assert "camera" in str(caught.value)


def test_the_map_check_rejects_a_tube_directly(monkeypatch):
    """The third move is what exposes it, so test that step on its own."""
    bridge = FakeBridge()
    camera = SwingingTube(bridge, CAMERA_XY, lens_z=LENS_Z, focus_z=FOCUS_Z)
    monkeypatch.setattr(
        vision, "average_frames",
        lambda url, count=1, timeout=1.0, settle=0: camera.frame(),
    )
    job = discovery.DiscoveryRoutine(
        bridge, discovery_config(), Recorder(), NullLogger()
    )
    # stand well off to the side, where a swinging tube stops tracking the move
    bridge.position_xyz = [CAMERA_XY[0] + 30.0, CAMERA_XY[1] + 20.0, 60.0]
    with pytest.raises(vision.DetectionError) as caught:
        job._pixel_map_here(3.0)
    assert "does not track the toolhead" in str(caught.value)


def test_the_map_check_passes_for_a_real_toolhead(monkeypatch):
    """The same check must not reject a camera that is aimed properly."""
    bridge = FakeBridge()
    camera = PerspectiveCamera(bridge, CAMERA_XY, lens_z=LENS_Z, focus_z=FOCUS_Z)
    monkeypatch.setattr(
        vision, "average_frames",
        lambda url, count=1, timeout=1.0, settle=0: camera.frame(),
    )
    job = discovery.DiscoveryRoutine(
        bridge, discovery_config(), Recorder(), NullLogger()
    )
    bridge.position_xyz = [CAMERA_XY[0], CAMERA_XY[1], 60.0]
    matrix, _, _, _ = job._pixel_map_here(3.0)
    assert geometry_scale(matrix) > 0


def geometry_scale(matrix):
    from nozzlealign_pkg import geometry
    return geometry.pixels_per_mm(matrix)
