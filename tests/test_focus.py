# coding=utf-8
"""The focus sweep: heights, sharpness, tracking and the peak."""

import numpy as np
import pytest

from nozzlealign_pkg import focus


def blurred_bore(sigma, centre=(300.0, 200.0), size=(400, 600)):
    """A dark ring with a bright collar, blurred by ``sigma`` pixels."""
    import cv2

    height, width = size
    ys, xs = np.mgrid[0:height, 0:width]
    radius = np.hypot(xs - centre[0], ys - centre[1])
    image = np.full((height, width), 90.0, dtype=np.float32)
    image[radius <= 34] = 230.0
    image[(radius >= 6) & (radius <= 14)] = 20.0
    if sigma > 0.3:
        k = int(sigma * 6) | 1
        image = cv2.GaussianBlur(image, (k, k), sigma)
    return image


def test_heights_run_from_high_to_low_inside_the_limits():
    heights = focus.plan_heights(start=30.0, span=4.0, step=1.0, floor=28.0, ceiling=200.0)
    assert heights == [34.0, 33.0, 32.0, 31.0, 30.0, 29.0, 28.0]
    assert heights == sorted(heights, reverse=True)


def test_heights_never_go_below_the_floor():
    heights = focus.plan_heights(start=22.0, span=6.0, step=2.0, floor=20.0, ceiling=200.0)
    assert min(heights) == 20.0
    assert all(h >= 20.0 for h in heights)


def test_no_room_to_focus_is_an_error():
    with pytest.raises(focus.FocusError):
        focus.plan_heights(start=10.0, span=2.0, step=1.0, floor=20.0, ceiling=200.0)


def test_sharpness_falls_with_blur():
    sharp = focus.sharpness(blurred_bore(0.0), (300, 200))
    soft = focus.sharpness(blurred_bore(3.0), (300, 200))
    softer = focus.sharpness(blurred_bore(6.0), (300, 200))
    assert sharp > soft > softer


def test_tracking_follows_a_moved_bore():
    first = blurred_bore(1.0, centre=(300.0, 200.0))
    template = focus.cut(first, (300, 200), 45)
    later = blurred_bore(1.5, centre=(330.0, 215.0))
    found, confidence = focus.track(later, template, near=(300, 200))
    assert abs(found[0] - 330.0) < 2.0
    assert abs(found[1] - 215.0) < 2.0
    assert confidence > 0.7


def test_the_peak_is_refined_between_samples():
    samples = [(z, 1000.0 - 40.0 * (z - 30.4) ** 2) for z in (28.0, 29.0, 30.0, 31.0, 32.0)]
    z, peak = focus.best_height(samples)
    assert abs(z - 30.4) < 0.05
    assert peak == max(s[1] for s in samples)


def test_a_peak_at_the_end_of_the_sweep_is_not_trusted():
    rising = [(28.0, 100.0), (29.0, 200.0), (30.0, 400.0)]
    assert not focus.is_peaked(rising)
    peaked = [(28.0, 100.0), (29.0, 400.0), (30.0, 120.0)]
    assert focus.is_peaked(peaked)
