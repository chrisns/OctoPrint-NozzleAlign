# coding=utf-8
"""Locate the nozzle bore in a focused, upward looking view.

Looking straight up a nozzle, the bore is a small hole in the flat tip, and
the flat tip around it catches the light as a bright collar. The pair is what
the detector scores. Hough circles, blob detectors, TAXY's YOLO model and the
TAMV recipe were all tried on frames from this machine and picked something
else: a glint, a burnt patch, or nothing.
"""

from __future__ import absolute_import

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - opencv is a runtime dependency
    cv2 = None


class NozzleError(Exception):
    """No usable nozzle was found."""


def _ring_kernels(inner_r, ring_lo, ring_hi, core_r=5):
    size = int(ring_hi) * 2 + 1
    centre = size // 2
    ys, xs = np.mgrid[0:size, 0:size]
    radius = np.hypot(xs - centre, ys - centre)
    disc = ((radius >= core_r) & (radius <= inner_r)).astype(np.float32)
    ring = ((radius >= ring_lo) & (radius <= ring_hi)).astype(np.float32)
    return disc / max(disc.sum(), 1.0), ring / max(ring.sum(), 1.0)


def find_bore(image, centre=None, search_radius=430, inner_r=12, ring_lo=18,
              ring_hi=34, refine=9, core_r=5):
    """Locate the bore by its shape: a dark hole inside a bright collar.

    The very middle of the hole is left out of the score. Once the lens was
    focused properly the bore showed a bright spot at its centre, light coming
    back off the inside of the nozzle, with the dark bore wall as a ring around
    it. Scoring the wall and the collar, and ignoring the middle, finds both
    the dark hole and the lit one.

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
    disc, ring = _ring_kernels(inner_r, ring_lo, ring_hi, core_r)
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
