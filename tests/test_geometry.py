# coding=utf-8
import math
import numpy as np
import pytest

from nozzlealign_pkg.geometry import (
    GeometryError,
    build_pixel_map,
    mm_per_pixel,
    pixel_error_to_mm,
    pixels_per_mm,
    rotation_degrees,
)


def synthetic(scale, rotation_deg, flip_y=True):
    """Pixels per mm matrix for a camera at a known scale and rotation."""
    angle = math.radians(rotation_deg)
    matrix = scale * np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )
    if flip_y:
        matrix = np.array([[1.0, 0.0], [0.0, -1.0]]).dot(matrix)
    return matrix


def test_build_map_recovers_scale_and_rotation():
    truth = synthetic(scale=80.0, rotation_deg=0.0, flip_y=False)
    origin = np.array([640.0, 400.0])
    distance = 1.0
    matrix = build_pixel_map(
        origin,
        origin + truth.dot([distance, 0.0]),
        origin + truth.dot([0.0, distance]),
        distance,
    )
    assert np.allclose(matrix, truth)
    assert pixels_per_mm(matrix) == pytest.approx(80.0)
    assert mm_per_pixel(matrix) == pytest.approx(1.0 / 80.0)
    assert rotation_degrees(matrix) == pytest.approx(0.0)


def test_build_map_handles_rotation_and_mirroring():
    truth = synthetic(scale=52.5, rotation_deg=30.0, flip_y=True)
    origin = np.array([100.0, 200.0])
    matrix = build_pixel_map(
        origin, origin + truth.dot([2.0, 0.0]), origin + truth.dot([0.0, 2.0]), 2.0
    )
    assert np.allclose(matrix, truth)
    assert pixels_per_mm(matrix) == pytest.approx(52.5)


def test_pixel_error_round_trips():
    truth = synthetic(scale=64.0, rotation_deg=12.0, flip_y=True)
    origin = np.array([640.0, 400.0])
    matrix = build_pixel_map(
        origin, origin + truth.dot([1.0, 0.0]), origin + truth.dot([0.0, 1.0]), 1.0
    )
    target = np.array([640.0, 400.0])
    current = target + truth.dot([0.37, -0.22])
    dx, dy = pixel_error_to_mm(matrix, current, target)
    assert dx == pytest.approx(-0.37, abs=1e-9)
    assert dy == pytest.approx(0.22, abs=1e-9)


def test_zero_distance_is_rejected():
    with pytest.raises(GeometryError):
        build_pixel_map([0, 0], [1, 1], [2, 2], 0)


def test_parallel_moves_are_rejected():
    with pytest.raises(GeometryError):
        build_pixel_map([0.0, 0.0], [10.0, 0.0], [20.0, 0.0], 1.0)
