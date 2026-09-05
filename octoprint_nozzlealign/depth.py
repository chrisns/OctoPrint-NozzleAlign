# coding=utf-8
"""Depth from a bed move, which is how the toolhead is found on this machine.

The three axes of a Snapmaker A350 do different things to this picture, and the
difference is what makes the toolhead findable at all.

* X moves the toolhead. The bed, and a camera sitting on it, stay still, so the
  background is static and the toolhead isolates by subtraction.
* Y moves the bed, and the camera with it, while the toolhead and the room stay
  put. That is a pure camera translation, so a point at distance ``D`` from the
  lens shifts by ``focal * dY / D`` pixels. Near things shift more.
* Z moves the toolhead towards the camera. The picture expands about the point
  straight above the lens.

Depth comes from the Y move. It needs no appearance model, no training and no
focus of expansion to fit, and it survives the blur and the glare that a lens
set for long range produces at short range. The Z move looks equivalent but is
not: every depth then hangs on locating the point above the lens, and patches
near that point have almost no radius to divide by, so their ratios are noise.
"""

from __future__ import absolute_import

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - opencv is a runtime dependency
    cv2 = None


class DepthError(Exception):
    """The bed move did not produce a usable depth measurement."""


# A tapered phase correlation window stops being reliable much beyond a quarter
# of its width, and it then reports a shift that quietly saturates rather than
# failing. A saturated shift looks like a plausible depth and is not one.
RELIABLE_FRACTION = 0.25


def max_reliable_move_mm(patch_px, px_per_mm):
    """Largest bed move a patch of this size can measure at this scale."""
    if px_per_mm <= 0:
        raise DepthError("pixel scale must be positive")
    return RELIABLE_FRACTION * patch_px / float(px_per_mm)


def patch_shifts(frame_a, frame_b, patch=192, step=64,
                 min_response=0.10, min_texture=4.0):
    """Local displacement for a grid of patches, with a confidence for each.

    Returns an array of rows ``(x, y, dx, dy, response)``. Featureless patches
    and low confidence patches are dropped rather than guessed at.
    """
    if cv2 is None:
        raise DepthError("opencv is not installed")
    a = np.asarray(frame_a, dtype=np.float32)
    b = np.asarray(frame_b, dtype=np.float32)
    if a.shape != b.shape:
        raise DepthError("the two frames are different sizes")
    height, width = a.shape
    if height < patch or width < patch:
        raise DepthError("the frame is smaller than one patch")
    window = cv2.createHanningWindow((patch, patch), cv2.CV_32F)
    rows = []
    for y in range(0, height - patch + 1, step):
        for x in range(0, width - patch + 1, step):
            pa = np.ascontiguousarray(a[y:y + patch, x:x + patch], dtype=np.float32)
            if pa.std() < min_texture:
                continue
            pb = np.ascontiguousarray(b[y:y + patch, x:x + patch], dtype=np.float32)
            (dx, dy), response = cv2.phaseCorrelate(pa, pb, window)
            if response < min_response:
                continue
            rows.append((x + patch / 2.0, y + patch / 2.0, dx, dy, response))
    return np.array(rows) if rows else np.zeros((0, 5))


def depths(rows, move_mm, focal_px, min_shift=0.3):
    """Distance from the lens for each patch, in millimetres."""
    if focal_px <= 0:
        raise DepthError("focal length must be positive")
    speed = np.hypot(rows[:, 2], rows[:, 3])
    with np.errstate(divide="ignore", invalid="ignore"):
        distance = np.where(speed > min_shift,
                            focal_px * abs(float(move_mm)) / speed, np.inf)
    return distance, speed


def nearest_region(rows, distance, response_floor=0.35, take=8, radius=140.0,
                   band=0.08):
    """Where the nearest confident patches sit, and how far away they are.

    Low confidence patches throw wild distances, so they are excluded before
    anything is called "nearest". Without that the answer is whichever patch
    happened to correlate on noise, and that patch is always the "nearest" one
    because noise reads as a large shift.

    The cut is the higher of a fixed floor and half the best confidence in this
    picture. A fixed floor alone does not travel: a sharp, well lit scene
    correlates at 0.9 while a blurred one tops out near 0.5, and what matters
    either way is being in the same class as the best patch rather than clearing
    an absolute bar.
    """
    if len(rows) == 0:
        raise DepthError("no patches to work with")
    floor = max(float(response_floor), 0.5 * float(rows[:, 4].max()))
    keep = np.isfinite(distance) & (rows[:, 4] >= floor)
    if keep.sum() < 3:
        raise DepthError(
            "only %d confident patches; the bed move may be too small or the "
            "picture too flat" % int(keep.sum()))
    kept_rows, kept_distance = rows[keep], distance[keep]

    # Only patches at genuinely the nearest depth are candidates. Clustering
    # first and depth second would let a large far group outvote the near one
    # simply by having more members.
    closest = float(kept_distance.min())
    at_front = kept_distance <= closest * (1.0 + band)
    front_rows = kept_rows[at_front]
    front_distance = kept_distance[at_front]
    order = np.argsort(front_distance)[:take]
    points = front_rows[order][:, :2]
    weights = front_rows[order][:, 4]

    # Even at one depth the nearest patches are not always one place. A toolhead
    # can present two low regions, and averaging across both lands the answer in
    # the gap between them, which is on neither. Worse, which group wins flips
    # as the toolhead moves, so a loop steering on that average oscillates
    # instead of converging. Take the heaviest group and ignore the rest.
    group = _tightest_group(points, weights, radius)
    centre = np.average(points[group], axis=0, weights=weights[group])
    return ((float(centre[0]), float(centre[1])),
            float(front_distance[order][group].min()), int(keep.sum()))


def _tightest_group(points, weights, radius):
    """Indices of the heaviest cluster of points within ``radius`` of each other."""
    best, best_weight = None, -1.0
    for index in range(len(points)):
        near = np.hypot(points[:, 0] - points[index, 0],
                        points[:, 1] - points[index, 1]) <= radius
        total = float(weights[near].sum())
        if total > best_weight:
            best, best_weight = near, total
    return best


def focal_from_scale(px_per_mm, distance_mm):
    """Focal length in pixels, from a known scale at a known distance."""
    return float(px_per_mm) * float(distance_mm)


def scale_at(focal_px, distance_mm):
    """Pixels per millimetre for something this far from the lens."""
    if distance_mm <= 0:
        raise DepthError("distance must be positive")
    return float(focal_px) / float(distance_mm)


def fit_lens_plane(samples):
    """Height of the lens above the bed, from scale measured at several heights.

    The scale of anything at the toolhead is ``focal / (Z - lens)``, so ``1 /
    scale`` is a straight line in Z that crosses zero at the lens. Fitting that
    line gives the lens height from measurements taken nowhere near the camera,
    which is the only safe way to learn it.
    """
    usable = [(z, s) for z, s in samples if s > 0]
    if len(usable) < 2:
        raise DepthError("need at least two heights")
    heights = np.array([z for z, _ in usable], dtype=float)
    inverse = np.array([1.0 / s for _, s in usable], dtype=float)
    if np.ptp(heights) < 1e-6:
        raise DepthError("all the samples are at the same height")
    slope, intercept = np.polyfit(heights, inverse, 1)
    if abs(slope) < 1e-12:
        raise DepthError("the scale does not change with height")
    lens_z = float(-intercept / slope)
    focal = float(1.0 / slope)
    return lens_z, focal
