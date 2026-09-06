# coding=utf-8
"""Locate nozzles in a focused, upward looking view.

Once the camera is aimed at the nozzles and focused on them, a nozzle is the
most circular thing in the picture, and the strongest circles in the frame are
the two hotends. That is far easier than anything the earlier work needed,
because the earlier work was trying to find a nozzle in a picture that did not
really contain one.

Two other approaches were tried on the same frames and found nothing. TAXY's
YOLOv8 nozzle model wants a much closer view, even on upscaled crops. The TAMV
``SimpleBlobDetector`` recipe wants the nozzle to be a dark blob on a light
field, which is not what this lens and lighting produce.

A Hough circle centre is only good to a pixel or two, so the centre is refined
afterwards by fitting a circle to the edge points around it. At about 20 px/mm
one pixel is 0.05 mm, and the refinement is worth roughly a factor of five.
"""

from __future__ import absolute_import

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - opencv is a runtime dependency
    cv2 = None


class NozzleError(Exception):
    """No usable nozzle was found."""


def find_circles(image, min_radius_px, max_radius_px, param2=60, blur=9,
                 min_separation_px=120, take=6):
    """The strongest circles in the frame, largest first."""
    if cv2 is None:
        raise NozzleError("opencv is not installed")
    gray = image
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    gray = cv2.GaussianBlur(gray, (blur | 1, blur | 1), 2)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, 1.2, min_separation_px,
        param1=120, param2=param2,
        minRadius=int(min_radius_px), maxRadius=int(max_radius_px))
    if circles is None:
        return []
    ordered = sorted(circles[0], key=lambda c: -c[2])[:take]
    return [dict(x=float(c[0]), y=float(c[1]), r=float(c[2])) for c in ordered]


def refine_centre(image, circle, band=0.30, canny_low=60, canny_high=160):
    """Fit a circle to the edge points around a rough one, for sub-pixel centre.

    A Hough centre lands on the accumulator grid, so it is good to a pixel or
    two. Fitting the actual edge does much better, and the fit residual doubles
    as a quality score: a clean nozzle rim fits tightly, a nozzle caked in burnt
    filament does not.
    """
    if cv2 is None:
        raise NozzleError("opencv is not installed")
    gray = image
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 1), canny_low, canny_high)
    ys, xs = np.nonzero(edges)
    if len(xs) < 20:
        raise NozzleError("no edges to fit")
    radius = np.hypot(xs - circle["x"], ys - circle["y"])
    keep = (radius > circle["r"] * (1 - band)) & (radius < circle["r"] * (1 + band))
    if keep.sum() < 20:
        raise NozzleError("too few edge points around the circle")
    return fit_circle(xs[keep].astype(float), ys[keep].astype(float))


def fit_circle(xs, ys):
    """Least squares circle through a set of points.

    Solves the linear form of ``(x - a)^2 + (y - b)^2 = r^2``, which is exact
    rather than iterative and needs no starting guess.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if len(xs) < 3:
        raise NozzleError("need at least three points to fit a circle")
    design = np.column_stack([xs, ys, np.ones(len(xs))])
    rhs = xs ** 2 + ys ** 2
    solution, *_ = np.linalg.lstsq(design, rhs, rcond=None)
    cx = solution[0] / 2.0
    cy = solution[1] / 2.0
    radius = float(np.sqrt(max(solution[2] + cx * cx + cy * cy, 0.0)))
    residual = float(np.std(np.hypot(xs - cx, ys - cy) - radius))
    return dict(x=float(cx), y=float(cy), r=radius, residual=residual,
                points=int(len(xs)))


def pick_pair(nozzles, expected_mm, scale_px_mm, tolerance_mm=3.0):
    """The two circles whose spacing matches the offset the firmware applies.

    Two nozzles are a known distance apart, so that distance is a free check on
    whether the right two circles were found. Anything else is a coincidence.
    """
    best = None
    for i in range(len(nozzles)):
        for j in range(i + 1, len(nozzles)):
            dx = nozzles[j]["x"] - nozzles[i]["x"]
            dy = nozzles[j]["y"] - nozzles[i]["y"]
            separation = float(np.hypot(dx, dy)) / float(scale_px_mm)
            error = abs(separation - float(expected_mm))
            if error <= tolerance_mm and (best is None or error < best["error"]):
                best = dict(error=error, first=nozzles[i], second=nozzles[j],
                            separation_mm=separation,
                            dx_mm=dx / float(scale_px_mm),
                            dy_mm=dy / float(scale_px_mm))
    if best is None:
        raise NozzleError(
            "no two circles are %.2f mm apart; either the scale is wrong or "
            "these are not the two nozzles" % expected_mm)
    return best


# --------------------------------------------------------------------------
# the detector that actually worked on the machine
# --------------------------------------------------------------------------


def _ring_kernels(inner_r, ring_lo, ring_hi):
    size = int(ring_hi) * 2 + 1
    centre = size // 2
    ys, xs = np.mgrid[0:size, 0:size]
    radius = np.hypot(xs - centre, ys - centre)
    disc = (radius <= inner_r).astype(np.float32)
    ring = ((radius >= ring_lo) & (radius <= ring_hi)).astype(np.float32)
    return disc / max(disc.sum(), 1.0), ring / max(ring.sum(), 1.0)


def find_bore(image, centre=None, search_radius=430, inner_r=12, ring_lo=18,
              ring_hi=34, refine=9):
    """Locate the bore by its shape: a dark hole inside a bright collar.

    Neither half identifies it alone. A burnt cone has darker patches than the
    bore, and a glint off the heater block is brighter than the collar. Both
    were tried on real frames and both failed, picking a reflection 260 px away
    from the nozzle.

    The pair is distinctive. Every pixel is scored by how much brighter its
    surrounding annulus is than the disc at its centre, and the peak wins. A
    lone glint has no dark core; a dark smear has no bright collar.

    On the machine this repeated to 0.01 px over five frames, which at 73 px/mm
    is about a fifth of a micron.
    """
    if cv2 is None:
        raise NozzleError("opencv is not installed")
    gray = image
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(np.clip(gray, 0, 255).astype(np.float32), (5, 5), 1)
    disc, ring = _ring_kernels(inner_r, ring_lo, ring_hi)
    score = (cv2.filter2D(gray, cv2.CV_32F, ring)
             - cv2.filter2D(gray, cv2.CV_32F, disc))

    height, width = gray.shape
    cx, cy = centre or (width / 2.0, height / 2.0)
    ys, xs = np.mgrid[0:height, 0:width]
    score[np.hypot(xs - cx, ys - cy) > search_radius] = -1e9

    peak = np.unravel_index(int(np.argmax(score)), score.shape)
    y, x = int(peak[0]), int(peak[1])
    half = refine // 2
    y0, y1 = max(0, y - half), min(height, y + half + 1)
    x0, x1 = max(0, x - half), min(width, x + half + 1)
    window = np.maximum(score[y0:y1, x0:x1] - score[y0:y1, x0:x1].min(), 0.0)
    if window.sum() <= 0:
        return dict(x=float(x), y=float(y), score=float(score[y, x]))
    gy, gx = np.mgrid[y0:y1, x0:x1]
    return dict(x=float((gx * window).sum() / window.sum()),
                y=float((gy * window).sum() / window.sum()),
                score=float(score[y, x]))
