# coding=utf-8
"""Finding the nozzle tips as the nearest points on the toolhead."""

import numpy as np
import pytest

from nozzlealign_pkg import tips

FOCAL = 1339.0
MOVE = 3.0
LENS_Z = -4.8


def layered_pair(background_mm=600.0, body_mm=180.0, tip_mm=150.0,
                 tip_centres=((420, 300), (640, 300)), size=(600, 900)):
    """Two frames of a body with two small tips on it, after a bed move.

    Each layer slides by its own amount, because a bed move translates the
    camera and near things slide further. The tips are small on purpose: that is
    the whole difficulty, and it is what defeats a patch based method.
    """
    import cv2
    height, width = size
    rng = np.random.default_rng(3)

    def texture(seed, scale):
        source = rng.normal(120.0, 45.0, (height // scale, width // scale))
        return cv2.resize(source.astype(np.float32), (width, height),
                          interpolation=cv2.INTER_CUBIC)

    background = texture(1, 3)
    body = texture(2, 2) + 20.0
    tip = texture(3, 2) - 30.0

    body_mask = np.zeros((height, width), np.float32)
    body_mask[120:480, 220:840] = 1.0
    tip_mask = np.zeros((height, width), np.float32)
    ys, xs = np.mgrid[0:height, 0:width]
    for cx, cy in tip_centres:
        tip_mask[(xs - cx) ** 2 + (ys - cy) ** 2 <= 26 ** 2] = 1.0

    def slide(image, mm):
        shift = FOCAL * MOVE / mm
        matrix = np.float32([[1, 0, shift], [0, 1, 0]])
        return cv2.warpAffine(image, matrix, (width, height),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    def compose(moved):
        far = slide(background, background_mm) if moved else background
        mid = slide(body, body_mm) if moved else body
        mid_mask = slide(body_mask, body_mm) if moved else body_mask
        near = slide(tip, tip_mm) if moved else tip
        near_mask = slide(tip_mask, tip_mm) if moved else tip_mask
        out = far * (1 - mid_mask) + mid * mid_mask
        return out * (1 - near_mask) + near * near_mask

    noise = rng.normal(0.0, 0.8, (height, width)).astype(np.float32)
    return compose(False) + noise, compose(True) + noise


def test_dense_depth_separates_tip_from_body_from_background():
    a, b = layered_pair()
    distance, magnitude = tips.dense_depth(a, b, MOVE, FOCAL)
    # sample each layer well inside itself
    tip_here = float(np.median(distance[290:310, 410:430]))
    body_here = float(np.median(distance[400:440, 300:360]))
    background_here = float(np.median(distance[520:560, 60:140]))
    assert tip_here == pytest.approx(150.0, rel=0.15)
    assert body_here == pytest.approx(180.0, rel=0.15)
    assert background_here > 300.0


def test_find_tips_picks_the_two_small_near_features():
    a, b = layered_pair()
    distance, _ = tips.dense_depth(a, b, MOVE, FOCAL)
    found = tips.find_tips(distance, commanded_z=150.0 + LENS_Z, lens_z=LENS_Z,
                           tolerance_mm=6.0, window=41)
    assert found, "no tips found at the tip plane"
    centres = [(tip["x"], tip["y"]) for tip in found[:4]]
    for want in ((420, 300), (640, 300)):
        assert any(abs(cx - want[0]) < 40 and abs(cy - want[1]) < 40
                   for cx, cy in centres), "missed the tip at %s" % (want,)


def test_the_body_is_not_mistaken_for_a_tip():
    a, b = layered_pair()
    distance, _ = tips.dense_depth(a, b, MOVE, FOCAL)
    found = tips.find_tips(distance, commanded_z=150.0 + LENS_Z, lens_z=LENS_Z,
                           tolerance_mm=6.0, window=41)
    # nothing accepted may be sitting at the body's depth
    assert all(abs(tip["depth"] - 180.0) > 8.0 for tip in found)


# -- persistence ------------------------------------------------------------


def test_noise_that_moves_between_repeats_is_dropped():
    steady = dict(x=500.0, y=300.0, depth=150.0)
    first = [steady, dict(x=200.0, y=100.0, depth=149.0)]
    second = [dict(x=503.0, y=302.0, depth=150.4), dict(x=800.0, y=600.0, depth=149.5)]
    third = [dict(x=498.0, y=299.0, depth=150.2), dict(x=100.0, y=700.0, depth=149.8)]
    survivors = tips.persistent_tips([first, second, third])
    assert len(survivors) == 1
    assert survivors[0]["x"] == pytest.approx(500.3, abs=3.0)
    assert survivors[0]["seen"] == 3


def test_the_live_nozzle_is_the_lowest_survivor():
    survivors = [dict(x=1.0, y=1.0, depth=155.0, seen=3, spread=1.0),
                 dict(x=2.0, y=2.0, depth=150.0, seen=3, spread=1.0)]
    assert tips.active_tip(sorted(survivors, key=lambda t: t["depth"]))["depth"] == 150.0


def test_nothing_surviving_is_an_error_not_a_guess():
    with pytest.raises(tips.TipError):
        tips.active_tip([])


def test_one_measurement_cannot_judge_persistence():
    with pytest.raises(tips.TipError):
        tips.persistent_tips([[dict(x=1.0, y=1.0, depth=150.0)]])
