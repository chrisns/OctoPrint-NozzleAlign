# coding=utf-8
"""Find the height at which the nozzle bore is sharpest.

The lens is fixed, so the toolhead is the only thing that can move to the focal
plane. The camera mount, the bed height and which nozzle is lowered all shift
that plane by a millimetre or two, so the height is measured on every run
rather than trusted from a setting.

The bore is tracked from one height to the next with a template, because the
picture grows as the nozzle comes closer and a fixed pixel would drift off it.
Sharpness is the variance of the Laplacian over a small window on the bore,
so a bright background cannot fool it.
"""

from __future__ import absolute_import

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - opencv is a runtime dependency
    cv2 = None


class FocusError(Exception):
    """No usable focus was found."""


def plan_heights(start, span, step, floor, ceiling):
    """Heights to visit, from high to low, never outside the floor and ceiling.

    Descending order matters. Every height is then reached from above, so a
    missed step can only leave the nozzle higher than asked, never lower.
    """
    if step <= 0:
        raise FocusError("the focus step must be positive")
    top = min(float(start) + float(span), float(ceiling))
    bottom = max(float(start) - float(span), float(floor))
    if bottom > top:
        raise FocusError(
            "no room to focus between Z%.2f and Z%.2f" % (bottom, top))
    heights = []
    z = top
    while z >= bottom - 1e-9:
        heights.append(round(z, 3))
        z -= float(step)
    if heights[-1] > bottom + 1e-9:
        heights.append(round(bottom, 3))
    return heights


def sharpness(frame, centre, size=80):
    """Variance of the Laplacian in a window on the bore."""
    if cv2 is None:
        raise FocusError("opencv is not installed")
    image = np.clip(frame, 0, 255).astype(np.uint8)
    half = int(size) // 2
    x, y = int(round(centre[0])), int(round(centre[1]))
    y0, y1 = max(0, y - half), min(image.shape[0], y + half)
    x0, x1 = max(0, x - half), min(image.shape[1], x + half)
    if y1 - y0 < 16 or x1 - x0 < 16:
        raise FocusError("the bore is too close to the edge of the frame")
    return float(cv2.Laplacian(image[y0:y1, x0:x1], cv2.CV_64F).var())


def cut(frame, centre, half):
    x, y = int(round(centre[0])), int(round(centre[1]))
    y0, y1 = max(0, y - half), min(frame.shape[0], y + half)
    x0, x1 = max(0, x - half), min(frame.shape[1], x + half)
    return np.clip(frame[y0:y1, x0:x1], 0, 255).astype(np.uint8)


def track(frame, template, near, search=140):
    """Where the template landed in this frame, and how well it matched."""
    if cv2 is None:
        raise FocusError("opencv is not installed")
    image = np.clip(frame, 0, 255).astype(np.uint8)
    x, y = int(round(near[0])), int(round(near[1]))
    y0, y1 = max(0, y - search), min(image.shape[0], y + search)
    x0, x1 = max(0, x - search), min(image.shape[1], x + search)
    window = image[y0:y1, x0:x1]
    th, tw = template.shape[:2]
    if window.shape[0] <= th or window.shape[1] <= tw:
        raise FocusError("lost the bore near the edge of the frame")
    result = cv2.matchTemplate(window, template, cv2.TM_CCOEFF_NORMED)
    _, confidence, _, location = cv2.minMaxLoc(result)
    found = (x0 + location[0] + tw / 2.0, y0 + location[1] + th / 2.0)
    return found, float(confidence)


def best_height(samples):
    """The height with the sharpest picture, refined between the samples.

    A parabola through the peak and its two neighbours puts the answer
    between steps. The refinement is only trusted when it lands inside that
    span, otherwise the sampled peak stands.
    """
    if not samples:
        raise FocusError("no focus samples")
    ordered = sorted(samples, key=lambda s: s[0])
    index = max(range(len(ordered)), key=lambda i: ordered[i][1])
    z, peak = ordered[index]
    if 0 < index < len(ordered) - 1:
        (z0, s0), (z1, s1), (z2, s2) = ordered[index - 1:index + 2]
        denominator = (s0 - 2 * s1 + s2)
        if denominator < 0 and abs(z1 - z0 - (z2 - z1)) < 1e-6:
            step = z1 - z0
            shift = 0.5 * step * (s0 - s2) / denominator
            if abs(shift) <= step:
                return float(z1 + shift), float(peak)
    return float(z), float(peak)


def is_peaked(samples, ratio=0.6):
    """Whether the sharpest sample stands clearly above the ends of the sweep.

    If the sharpness only rises towards one end, the focal plane is outside
    the range that was swept and the answer would be the range limit, not the
    focus. That is reported as a failure rather than believed.
    """
    if len(samples) < 3:
        return False
    ordered = sorted(samples, key=lambda s: s[0])
    peak = max(s[1] for s in ordered)
    if peak <= 0:
        return False
    low, high = ordered[0][1], ordered[-1][1]
    return bool(max(low, high) < peak * ratio)
