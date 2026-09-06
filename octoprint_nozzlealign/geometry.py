# coding=utf-8
"""Mapping between camera pixels and machine millimetres.

The map is measured, not configured.  The routine commands two known moves and
watches how far the tip travels in the image.  That single step captures the
scale, the camera rotation and any mirroring at once, so nobody has to measure
a reference object with a ruler or tell the plugin which way up the camera is.
"""

from __future__ import absolute_import

import numpy as np


class GeometryError(Exception):
    """The measured moves do not describe a usable map."""


def build_pixel_map(origin_px, x_move_px, y_move_px, distance_mm):
    """Return the 2x2 matrix of pixels per millimetre.

    ``origin_px`` is the tip before the probe moves.  ``x_move_px`` is the tip
    after moving ``distance_mm`` along machine X, and ``y_move_px`` the same for
    machine Y.  Column 0 is the pixel displacement caused by one millimetre of
    X, column 1 the same for Y.
    """
    if distance_mm == 0:
        raise GeometryError("probe distance must not be zero")
    origin = np.asarray(origin_px, dtype=float)
    column_x = (np.asarray(x_move_px, dtype=float) - origin) / float(distance_mm)
    column_y = (np.asarray(y_move_px, dtype=float) - origin) / float(distance_mm)
    matrix = np.column_stack([column_x, column_y])
    determinant = float(np.linalg.det(matrix))
    if abs(determinant) < 1e-6:
        raise GeometryError(
            "the two probe moves are not independent (determinant %.3g); "
            "check that the tip is visible and that detection is stable"
            % determinant
        )
    return matrix


def build_pixel_map_from_shifts(shift_x, shift_y, distance_mm):
    """Return the 2x2 matrix of pixels per millimetre from two measured shifts.

    ``shift_x`` is the image displacement caused by moving ``distance_mm`` along
    machine X, and ``shift_y`` the same for machine Y.
    """
    if distance_mm == 0:
        raise GeometryError("probe distance must not be zero")
    column_x = np.asarray(shift_x, dtype=float) / float(distance_mm)
    column_y = np.asarray(shift_y, dtype=float) / float(distance_mm)
    matrix = np.column_stack([column_x, column_y])
    determinant = float(np.linalg.det(matrix))
    if abs(determinant) < 1e-6:
        raise GeometryError(
            "the two probe moves are not independent (determinant %.3g); "
            "check that the nozzle is visible and that detection is stable"
            % determinant
        )
    return matrix


def pixels_per_mm(matrix):
    """Average pixel scale, useful for reporting and for sanity limits."""
    matrix = np.asarray(matrix, dtype=float)
    return float(np.sqrt(abs(np.linalg.det(matrix))))


def mm_per_pixel(matrix):
    """Inverse of :func:`pixels_per_mm`."""
    scale = pixels_per_mm(matrix)
    if scale == 0:
        raise GeometryError("pixel scale is zero")
    return 1.0 / scale


def rotation_degrees(matrix):
    """Angle between machine X and the image X axis, for reporting only."""
    matrix = np.asarray(matrix, dtype=float)
    return float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0])))


def pixel_error_to_mm(matrix, current_px, target_px):
    """Machine move in millimetres that puts ``current_px`` onto ``target_px``."""
    matrix = np.asarray(matrix, dtype=float)
    error = np.asarray(target_px, dtype=float) - np.asarray(current_px, dtype=float)
    return tuple(np.linalg.solve(matrix, error))


def validate_map(matrix, min_scale=30.0, max_scale=130.0, max_cos=0.35):
    """Whether a measured pixel map can be trusted to steer with.

    Two failures both produced a matrix that looked like a matrix and was not
    one. A probe move partly eaten by backlash gave 12.6 px/mm with the two
    columns 0.98 aligned, which is nearly singular, and solving with it asked
    for a move of 118 px in the wrong direction. A probe measured while the
    nozzle drifted out of frame gave 213 px/mm.

    So a map is checked before it is used: the scale must be physically
    plausible, and the two columns must be close to perpendicular, because a
    camera looking at a moving stage cannot make them anything else.
    """
    matrix = np.asarray(matrix, dtype=float)
    scale = float(abs(np.linalg.det(matrix)) ** 0.5)
    first, second = matrix[:, 0], matrix[:, 1]
    norms = np.linalg.norm(first) * np.linalg.norm(second)
    if norms <= 0:
        return False, scale, 1.0
    cosine = abs(float(first @ second) / norms)
    ok = bool(min_scale < scale < max_scale and cosine < max_cos)
    return ok, scale, cosine


def backlash_free(dx, dy, backoff=1.0):
    """The pair of moves that arrives at (dx, dy) from a fixed direction.

    An axis gives back less than it was asked for when it reverses, so a probe
    measured after a reversal is short and the map comes out wrong. Approaching
    every measurement from the same side takes the slack up first. On this
    machine that turned a 12.6 px/mm map with parallel columns into a 73 px/mm
    map with square ones, from the same 1.5 mm probe.
    """
    return (dx - backoff, dy - backoff), (backoff, backoff)
