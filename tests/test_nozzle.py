# coding=utf-8
"""Finding nozzles once the camera is aimed at them and focused."""

import numpy as np
import pytest

from nozzlealign_pkg import nozzle

SCALE = 20.7          # px/mm at Z60 on this machine
SEPARATION = 25.20    # mm, the offset the firmware applies


def scene(centres=((420.0, 300.0), (941.4, 300.0)), radius=110.0, size=(800, 1280)):
    """Two bright annular nozzles on a dark textured toolhead."""
    import cv2
    height, width = size
    rng = np.random.default_rng(5)
    image = np.full((height, width), 60.0, dtype=np.float32)
    image += rng.normal(0.0, 12.0, image.shape)
    ys, xs = np.mgrid[0:height, 0:width]
    for cx, cy in centres:
        r = np.hypot(xs - cx, ys - cy)
        image[r <= radius] = 210.0                 # the bright nozzle body
        image[r <= radius * 0.22] = 40.0           # the dark orifice
    return cv2.GaussianBlur(image, (5, 5), 1)


def test_the_two_nozzles_are_the_strongest_circles():
    found = nozzle.find_circles(scene(), min_radius_px=60, max_radius_px=160)
    assert len(found) >= 2
    centres = sorted((c["x"], c["y"]) for c in found[:2])
    assert centres[0][0] == pytest.approx(420.0, abs=6)
    assert centres[1][0] == pytest.approx(941.4, abs=6)


def test_refinement_beats_the_hough_grid():
    truth = (420.35, 300.65)
    image = scene(centres=(truth, (941.4, 300.0)))
    rough = nozzle.find_circles(image, min_radius_px=60, max_radius_px=160)
    near = min(rough, key=lambda c: abs(c["x"] - truth[0]) + abs(c["y"] - truth[1]))
    fine = nozzle.refine_centre(image, near)
    rough_error = np.hypot(near["x"] - truth[0], near["y"] - truth[1])
    fine_error = np.hypot(fine["x"] - truth[0], fine["y"] - truth[1])
    assert fine_error < 1.0
    assert fine_error <= rough_error


def test_a_clean_rim_fits_tighter_than_a_dirty_one():
    """The fit residual is a quality score, which the left nozzle needs."""
    import cv2
    clean = scene(centres=((420.0, 300.0),))
    dirty = clean.copy()
    rng = np.random.default_rng(11)
    ys, xs = np.mgrid[0:800, 0:1280]
    blob = np.hypot(xs - 400, ys - 285) < 70          # burnt filament, off centre
    dirty[blob] = 150.0 + rng.normal(0.0, 40.0, int(blob.sum()))
    dirty = cv2.GaussianBlur(dirty, (5, 5), 1)
    rough = dict(x=420.0, y=300.0, r=110.0)
    assert (nozzle.refine_centre(clean, rough)["residual"]
            < nozzle.refine_centre(dirty, rough)["residual"])


def test_the_known_spacing_confirms_the_right_pair():
    found = nozzle.find_circles(scene(), min_radius_px=60, max_radius_px=160)
    pair = nozzle.pick_pair(found, SEPARATION, SCALE)
    assert pair["separation_mm"] == pytest.approx(SEPARATION, abs=0.6)
    assert abs(pair["dy_mm"]) < 0.6


def test_a_wrong_scale_is_caught_rather_than_believed():
    """A separation that does not match cannot be the two nozzles."""
    found = nozzle.find_circles(scene(), min_radius_px=60, max_radius_px=160)
    with pytest.raises(nozzle.NozzleError):
        nozzle.pick_pair(found, SEPARATION, scale_px_mm=12.0)


def test_fit_circle_is_exact_on_exact_points():
    angles = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    xs = 512.25 + 97.5 * np.cos(angles)
    ys = 333.75 + 97.5 * np.sin(angles)
    fit = nozzle.fit_circle(xs, ys)
    assert fit["x"] == pytest.approx(512.25, abs=1e-6)
    assert fit["y"] == pytest.approx(333.75, abs=1e-6)
    assert fit["r"] == pytest.approx(97.5, abs=1e-6)
    assert fit["residual"] < 1e-6


def test_fit_circle_needs_three_points():
    with pytest.raises(nozzle.NozzleError):
        nozzle.fit_circle([1.0, 2.0], [1.0, 2.0])


def test_an_empty_frame_finds_nothing_rather_than_something():
    flat = np.full((800, 1280), 120.0, dtype=np.float32)
    assert nozzle.find_circles(flat, min_radius_px=60, max_radius_px=160) == []
