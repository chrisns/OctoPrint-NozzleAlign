# coding=utf-8
"""Depth from a bed move, checked against a simulated camera and real numbers."""

import numpy as np
import pytest

from nozzlealign_pkg import depth


def scene_pair(near_shift, far_shift, size=(600, 900)):
    """Two frames of a near object on a far background, after a bed move.

    A bed move translates the camera, so everything slides sideways and near
    things slide further. The near object slides as a whole, mask included,
    which is what makes it a different depth rather than just different texture.
    """
    import cv2
    height, width = size
    rng = np.random.default_rng(7)
    # fine texture matters: phase correlation needs high frequencies, and a
    # heavily upsampled random field has none
    far = cv2.resize(rng.normal(120.0, 45.0, (300, 450)).astype(np.float32),
                     (width, height), interpolation=cv2.INTER_CUBIC)
    near = cv2.resize(rng.normal(90.0, 55.0, (200, 300)).astype(np.float32),
                      (width, height), interpolation=cv2.INTER_CUBIC)
    mask = np.zeros((height, width), dtype=np.float32)
    mask[60:560, 220:760] = 1.0

    def shift(image, dx):
        matrix = np.float32([[1, 0, dx], [0, 1, 0]])
        return cv2.warpAffine(image, matrix, (width, height),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    def compose(dn, df):
        m = shift(mask, dn)
        return shift(near, dn) * m + shift(far, df) * (1.0 - m)

    noise = rng.normal(0.0, 1.0, (height, width)).astype(np.float32)
    return compose(0.0, 0.0) + noise, compose(near_shift, far_shift) + noise


def test_near_things_shift_more_and_read_as_nearer():
    focal, move = 1339.0, 2.0
    near_d, far_d = 150.0, 600.0
    a, b = scene_pair(focal * move / near_d, focal * move / far_d)
    rows = depth.patch_shifts(a, b, patch=192, step=64)
    assert len(rows) > 20
    distance, speed = depth.depths(rows, move, focal)
    centre, nearest, confident = depth.nearest_region(rows, distance)
    assert nearest == pytest.approx(near_d, rel=0.15)
    # the nearest patches must sit on the near object, not on the background
    assert 220 < centre[0] < 760
    assert 60 < centre[1] < 560


def test_a_flat_picture_is_refused_rather_than_guessed():
    flat = np.full((400, 400), 128.0, dtype=np.float32)
    rows = depth.patch_shifts(flat, flat.copy(), patch=192, step=64)
    assert len(rows) == 0


def test_low_confidence_patches_do_not_get_to_be_nearest():
    rows = np.array([
        [100.0, 100.0, 40.0, 0.0, 0.20],   # huge shift, no confidence
        [200.0, 200.0, 17.3, 0.0, 0.95],
        [260.0, 200.0, 17.2, 0.0, 0.92],
        [320.0, 200.0, 17.4, 0.0, 0.90],
    ])
    # the noisy patch reads as by far the nearest, which is exactly the trap
    distance, _ = depth.depths(rows, 2.0, 1339.0)
    centre, nearest, confident = depth.nearest_region(rows, distance)
    assert confident == 3
    assert nearest == pytest.approx(1339.0 * 2.0 / 17.4, rel=0.05)
    assert centre[0] > 150          # not the noisy patch at x=100


def test_nearest_region_complains_when_there_is_nothing_to_work_with():
    rows = np.array([[10.0, 10.0, 5.0, 0.0, 0.1], [20.0, 20.0, 5.0, 0.0, 0.1]])
    distance, _ = depth.depths(rows, 2.0, 1339.0)
    with pytest.raises(depth.DepthError):
        depth.nearest_region(rows, distance)


def test_the_reliable_move_shrinks_as_the_picture_magnifies():
    # 192 px patch, 8.65 px/mm as measured at Z150 on the real machine
    assert depth.max_reliable_move_mm(192, 8.65) == pytest.approx(5.55, rel=0.01)
    # closer in, the same patch can only measure a smaller move
    assert depth.max_reliable_move_mm(192, 27.3) == pytest.approx(1.76, rel=0.01)


def test_lens_plane_from_real_measurements():
    """The heights and scales measured on the machine on 2026-09-05."""
    samples = [(150.0, 8.027), (130.0, 9.271), (110.0, 10.962),
               (90.0, 12.921), (75.0, 15.552)]
    lens_z, focal = depth.fit_lens_plane(samples)
    assert lens_z == pytest.approx(-6.1, abs=3.0)
    assert focal == pytest.approx(1258.0, rel=0.1)
    # and the fit must reproduce what was measured
    for z, measured in samples:
        assert depth.scale_at(focal, z - lens_z) == pytest.approx(measured, rel=0.05)


def test_lens_plane_needs_more_than_one_height():
    with pytest.raises(depth.DepthError):
        depth.fit_lens_plane([(150.0, 8.0)])
    with pytest.raises(depth.DepthError):
        depth.fit_lens_plane([(150.0, 8.0), (150.0, 8.1)])


def test_focal_round_trips():
    focal = depth.focal_from_scale(8.65, 154.8)
    assert depth.scale_at(focal, 154.8) == pytest.approx(8.65)


def test_two_separate_low_regions_do_not_average_into_the_gap():
    """A toolhead can show two low regions, and the answer must pick one.

    Averaging across both lands the answer between them, on neither, and which
    group happens to be nearest flips as the toolhead moves. A loop steering on
    that average oscillates instead of converging, which is exactly what the
    machine did before this was fixed.
    """
    rows = np.array([
        # a tight group of four on the left, all confident
        [200.0, 400.0, 17.3, 0.0, 0.95],
        [240.0, 400.0, 17.3, 0.0, 0.94],
        [200.0, 440.0, 17.2, 0.0, 0.93],
        [240.0, 440.0, 17.2, 0.0, 0.92],
        # a looser pair 700 px away at almost the same depth
        [940.0, 300.0, 17.25, 0.0, 0.90],
        [980.0, 300.0, 17.15, 0.0, 0.89],
    ])
    distance, _ = depth.depths(rows, 2.0, 1339.0)
    centre, nearest, _ = depth.nearest_region(rows, distance)
    # the answer must sit on the heavier left group, not midway between the two
    assert 180 < centre[0] < 260
    assert 380 < centre[1] < 460


def test_a_single_group_is_unaffected():
    rows = np.array([
        [600.0, 500.0, 17.3, 0.0, 0.95],
        [640.0, 500.0, 17.2, 0.0, 0.94],
        [600.0, 540.0, 17.25, 0.0, 0.93],
    ])
    distance, _ = depth.depths(rows, 2.0, 1339.0)
    centre, _, _ = depth.nearest_region(rows, distance)
    assert centre[0] == pytest.approx(613.0, abs=15)
    assert centre[1] == pytest.approx(513.0, abs=15)
