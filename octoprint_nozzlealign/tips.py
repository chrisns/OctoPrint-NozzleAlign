# coding=utf-8
"""Find the nozzle tips as the nearest points on the toolhead.

Patch correlation cannot do this, and it took a night to see why. A 192 pixel
patch spans 22 mm at this camera's scale, and a nozzle is a few millimetres
across, so any patch holding the tip is dominated by the body around it and
reports the body's depth. The tip never wins a vote that large. That is why the
patch method kept answering "the nearest thing is 20 mm above the tip": it was
measuring the body every time.

Dense optical flow has no window. A bed move is a pure camera translation, so
every pixel's flow magnitude gives its own depth, ``D = focal * move / |flow|``.
A tip is then a local minimum of that depth field whose height comes back as the
commanded Z, because Z is defined as the height of the tip above the bed.

Everything here uses numpy and opencv only. ``cv2.erode`` is a minimum filter on
a greyscale image, which is all the local minimum search needs, so the print PC
gains no new dependency.
"""

from __future__ import absolute_import

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - opencv is a runtime dependency
    cv2 = None


class TipError(Exception):
    """No usable tip was found."""


def dense_depth(frame_a, frame_b, move_mm, focal_px, min_flow_px=0.4):
    """Distance from the lens for every pixel, from one bed move."""
    if cv2 is None:
        raise TipError("opencv is not installed")
    a = np.clip(frame_a, 0, 255).astype(np.uint8)
    b = np.clip(frame_b, 0, 255).astype(np.uint8)
    if a.shape != b.shape:
        raise TipError("the two frames are different sizes")
    flow = _flow(a, b)
    magnitude = np.hypot(flow[..., 0], flow[..., 1])
    with np.errstate(divide="ignore", invalid="ignore"):
        distance = np.where(magnitude > min_flow_px,
                            float(focal_px) * abs(float(move_mm)) / magnitude,
                            np.inf)
    return distance, magnitude


def _flow(a, b):
    if hasattr(cv2, "DISOpticalFlow_create"):
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        dis.setUseSpatialPropagation(True)
        return dis.calc(a, b, None)
    return cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 4, 31, 5, 7, 1.5, 0)


def find_tips(distance, commanded_z, lens_z, tolerance_mm=5.0, window=61,
              min_area_px=40, max_area_px=4000, blur=7):
    """Local depth minima that sit at the height of the nozzle tip."""
    if cv2 is None:
        raise TipError("opencv is not installed")
    far = float(np.nanmax(distance[np.isfinite(distance)])) if np.isfinite(distance).any() else 1e6
    filled = np.where(np.isfinite(distance), distance, far * 10.0).astype(np.float32)
    smooth = cv2.GaussianBlur(filled, (blur | 1, blur | 1), 0)
    # eroding a greyscale image is a minimum filter, so a pixel that survives is
    # the smallest value in its neighbourhood
    lowest = cv2.erode(smooth, np.ones((window | 1, window | 1), np.uint8))
    height = smooth + float(lens_z)
    seeds = ((smooth <= lowest + 1e-6)
             & (np.abs(height - float(commanded_z)) <= float(tolerance_mm)))
    seeds = cv2.dilate(seeds.astype(np.uint8), np.ones((9, 9), np.uint8))
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(seeds, 8)
    found = []
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < min_area_px or area > max_area_px:
            continue
        mask = labels == index
        depth_here = float(np.median(smooth[mask]))
        found.append(dict(x=float(centroids[index][0]), y=float(centroids[index][1]),
                          area=area, depth=depth_here,
                          height=depth_here + float(lens_z)))
    found.sort(key=lambda tip: tip["depth"])
    return found


def persistent_tips(repeats, radius_px=18.0, min_seen=None):
    """Keep only the tips that turn up in most of several repeated measurements.

    A single measurement cannot tell a tip from a lucky patch of flow noise.
    Noise moves between repeats; a tip does not. This is the same lesson the
    depth work taught, and it costs only a few more frames.
    """
    if len(repeats) < 2:
        raise TipError("need at least two measurements to judge persistence")
    if min_seen is None:
        min_seen = (len(repeats) + 1) // 2 + 1
        min_seen = min(min_seen, len(repeats))
    survivors = []
    for candidate in repeats[0]:
        seen, xs, ys, depths = 1, [candidate["x"]], [candidate["y"]], [candidate["depth"]]
        for later in repeats[1:]:
            match = _closest(later, candidate, radius_px)
            if match is not None:
                seen += 1
                xs.append(match["x"]); ys.append(match["y"])
                depths.append(match["depth"])
        if seen >= min_seen:
            survivors.append(dict(x=float(np.mean(xs)), y=float(np.mean(ys)),
                                  depth=float(np.mean(depths)), seen=seen,
                                  spread=float(np.hypot(np.std(xs), np.std(ys)))))
    survivors.sort(key=lambda tip: tip["depth"])
    return survivors


def _closest(tips, target, radius):
    best, best_distance = None, radius
    for tip in tips:
        gap = float(np.hypot(tip["x"] - target["x"], tip["y"] - target["y"]))
        if gap <= best_distance:
            best, best_distance = tip, gap
    return best


def active_tip(survivors):
    """The tip of the live nozzle, which is the lowest thing on the toolhead."""
    if not survivors:
        raise TipError(
            "no tip survived the repeats; the nozzle may be out of view, or out "
            "of focus enough that the flow field is noise")
    return survivors[0]
