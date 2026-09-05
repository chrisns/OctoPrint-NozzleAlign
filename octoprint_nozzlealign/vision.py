# coding=utf-8
"""Frame capture and nozzle tip detection.

Frames come from go2rtc over HTTP.  The plugin never opens the camera device
itself: go2rtc already holds it, and a UVC device only allows one consumer.
"""

from __future__ import absolute_import

import io
import math
import time

import numpy as np
import requests
from PIL import Image

try:
    import cv2
except ImportError:  # pragma: no cover - opencv is a runtime dependency
    cv2 = None


class CaptureError(Exception):
    """The camera did not give us a usable frame."""


class DetectionError(Exception):
    """No nozzle tip was found in the frame."""


# --------------------------------------------------------------------------
# capture
# --------------------------------------------------------------------------


def fetch_frame(url, timeout=10.0, attempts=4, retry_delay=1.0):
    """Fetch one JPEG frame and return it as a float32 greyscale array.

    go2rtc stops the camera process while nothing is watching, and the first
    request after that answers 200 with an empty body while ffmpeg starts.  The
    retry covers that cold start.
    """
    last = 0
    for attempt in range(attempts):
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        last = len(response.content)
        if last >= 1000:
            image = Image.open(io.BytesIO(response.content)).convert("L")
            return np.asarray(image, dtype=np.float32)
        if attempt < attempts - 1:
            time.sleep(retry_delay)
    raise CaptureError(
        "the camera returned %d bytes after %d attempts; check the go2rtc "
        "stream at %s" % (last, attempts, url)
    )


def frame_health(frame):
    """Return (mean, standard deviation) so a dead stream can be spotted.

    A broken MJPEG path decodes to flat mid grey: mean near 128 and a standard
    deviation below about 5.  A real picture is far noisier than that.
    """
    return float(frame.mean()), float(frame.std())


def is_blank(frame, min_std=8.0):
    """True when the frame carries no picture."""
    return frame_health(frame)[1] < min_std


def average_frames(url, count=8, timeout=10.0, settle=0):
    """Average several frames to cut sensor noise.

    ``settle`` frames are fetched and thrown away first, which lets the camera
    finish reacting to a move before the measurement starts.
    """
    for _ in range(settle):
        fetch_frame(url, timeout)
    total = None
    for _ in range(count):
        frame = fetch_frame(url, timeout)
        total = frame if total is None else total + frame
    stacked = total / float(count)
    if is_blank(stacked):
        raise CaptureError(
            "camera frame is blank (mean %.1f, sd %.1f); check the go2rtc stream"
            % frame_health(stacked)
        )
    return stacked


# --------------------------------------------------------------------------
# detection
# --------------------------------------------------------------------------


def _require_cv2():
    if cv2 is None:
        raise DetectionError(
            "opencv is not installed; run pip install opencv-python-headless"
        )


def _roi_slice(shape, roi):
    """Turn a fractional roi (cx, cy, size) into array slices and an origin."""
    height, width = shape
    cx, cy, size = roi
    half = size / 2.0
    x0 = max(0, int((cx - half) * width))
    x1 = min(width, int((cx + half) * width))
    y0 = max(0, int((cy - half) * height))
    y1 = min(height, int((cy + half) * height))
    if x1 - x0 < 16 or y1 - y0 < 16:
        raise DetectionError("region of interest is too small")
    return (slice(y0, y1), slice(x0, x1)), (x0, y0)


def detect_contour(frame, roi=None, invert=True, blur=5, min_area=200):
    """Find the tip as the centre of the largest blob after an Otsu threshold.

    ``invert`` selects a dark nozzle against a bright background, which is what
    an upward camera with a lit ring sees.
    """
    _require_cv2()
    work = frame
    origin = (0, 0)
    if roi:
        slices, origin = _roi_slice(frame.shape, roi)
        work = frame[slices]
    image = np.clip(work, 0, 255).astype(np.uint8)
    if blur > 1:
        image = cv2.GaussianBlur(image, (blur | 1, blur | 1), 0)
    flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, mask = cv2.threshold(image, 0, 255, flag + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise DetectionError("no contour found")
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < min_area:
        raise DetectionError("largest contour is only %.0f px" % area)
    (x, y), radius = cv2.minEnclosingCircle(largest)
    # circularity doubles as the confidence score
    perimeter = cv2.arcLength(largest, True)
    confidence = 0.0
    if perimeter > 0:
        confidence = min(1.0, 4.0 * math.pi * area / (perimeter * perimeter))
    return (x + origin[0], y + origin[1], confidence, {"radius": radius, "area": area})


def detect_hough(frame, roi=None, min_radius=10, max_radius=200, blur=5, param2=30):
    """Find the tip as the strongest circle, which suits a lit brass orifice."""
    _require_cv2()
    work = frame
    origin = (0, 0)
    if roi:
        slices, origin = _roi_slice(frame.shape, roi)
        work = frame[slices]
    image = np.clip(work, 0, 255).astype(np.uint8)
    if blur > 1:
        image = cv2.GaussianBlur(image, (blur | 1, blur | 1), 0)
    circles = cv2.HoughCircles(
        image,
        cv2.HOUGH_GRADIENT,
        dp=1.0,
        minDist=max(10, min_radius * 2),
        param1=100,
        param2=param2,
        minRadius=int(min_radius),
        maxRadius=int(max_radius),
    )
    if circles is None:
        raise DetectionError("no circle found")
    circles = np.squeeze(circles, axis=0)
    x, y, radius = circles[0]
    return (
        float(x) + origin[0],
        float(y) + origin[1],
        1.0,
        {"radius": float(radius), "candidates": int(len(circles))},
    )


def detect_template(frame, template, roi=None):
    """Find the tip by matching a patch the user picked once, to sub-pixel.

    This is the fallback that works whatever the optics look like, because the
    user supplies the appearance rather than the code guessing it.
    """
    _require_cv2()
    work = frame
    origin = (0, 0)
    if roi:
        slices, origin = _roi_slice(frame.shape, roi)
        work = frame[slices]
    image = np.clip(work, 0, 255).astype(np.float32)
    patch = np.asarray(template, dtype=np.float32)
    if patch.shape[0] >= image.shape[0] or patch.shape[1] >= image.shape[1]:
        raise DetectionError("template is larger than the search area")
    result = cv2.matchTemplate(image, patch, cv2.TM_CCOEFF_NORMED)
    _, score, _, location = cv2.minMaxLoc(result)
    x, y = location
    dx, dy = _subpixel_peak(result, x, y)
    centre_x = x + dx + patch.shape[1] / 2.0 + origin[0]
    centre_y = y + dy + patch.shape[0] / 2.0 + origin[1]
    return (centre_x, centre_y, float(score), {"score": float(score)})


def _subpixel_peak(surface, x, y):
    """Refine a correlation peak with a parabola through its neighbours."""
    height, width = surface.shape
    if x <= 0 or y <= 0 or x >= width - 1 or y >= height - 1:
        return 0.0, 0.0
    def refine(a, b, c):
        denominator = a - 2.0 * b + c
        if abs(denominator) < 1e-9:
            return 0.0
        return float(np.clip(0.5 * (a - c) / denominator, -1.0, 1.0))
    dx = refine(surface[y, x - 1], surface[y, x], surface[y, x + 1])
    dy = refine(surface[y - 1, x], surface[y, x], surface[y + 1, x])
    return dx, dy


# --------------------------------------------------------------------------
# motion based location
# --------------------------------------------------------------------------


def locate_by_motion(frame_a, frame_b, threshold=4.0, min_area=60,
                     min_circularity=0.25, max_extent=0.4):
    """Find the nozzle in two frames taken either side of a known move.

    The nozzle is the only thing in the picture that moves, so subtracting the
    two frames isolates it without any model of what a nozzle looks like.  This
    works whether the nozzle is darker or brighter than its background, so it
    needs no tuning before the camera has ever seen a nozzle.

    Returns ``(position_in_a, position_in_b, details)``.
    """
    _require_cv2()
    a = np.asarray(frame_a, dtype=np.float32)
    b = np.asarray(frame_b, dtype=np.float32)
    if a.shape != b.shape:
        raise DetectionError("the two frames are different sizes")
    difference = cv2.GaussianBlur(a - b, (9, 9), 0)
    noise = float(np.median(np.abs(difference - np.median(difference)))) * 1.4826
    level = max(threshold, threshold * noise)

    positive = _best_blob(difference > level, min_area, min_circularity, max_extent)
    negative = _best_blob(difference < -level, min_area, min_circularity, max_extent)
    if positive is None or negative is None:
        raise DetectionError(
            "no moving object found; the nozzle may be outside the field of view"
        )

    # Decide which blob holds the nozzle in frame a.  At the nozzle's position
    # in a, frame a shows the nozzle; at its position in b, frame a shows the
    # background.  So the blob whose pixels differ most from the background in
    # frame a is the one that holds the nozzle in a.
    background = float(np.median(a))
    lift_positive = abs(_masked_mean(a, positive["mask"]) - background)
    lift_negative = abs(_masked_mean(a, negative["mask"]) - background)
    if lift_positive >= lift_negative:
        in_a, in_b = positive, negative
    else:
        in_a, in_b = negative, positive

    details = dict(
        area_a=in_a["area"],
        area_b=in_b["area"],
        radius=in_a["radius"],
        circularity=in_a["circularity"],
        contrast=float(max(lift_positive, lift_negative)),
        noise=noise,
    )
    return in_a["centre"], in_b["centre"], details


def _best_blob(mask, min_area, min_circularity=0.25, max_extent=0.4):
    """The most nozzle shaped moving region in a mask.

    Size alone is the wrong test.  Moving the toolhead also moves the gantry
    beam, which paints a long thin sliver of change with a large area.  A nozzle
    paints a compact blob, so the score rewards area and roundness together and
    rejects anything that spans a large part of the frame.
    """
    height, width = mask.shape
    binary = mask.astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_score = 0.0
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w > width * max_extent or h > height * max_extent:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        circularity = min(1.0, 4.0 * math.pi * area / (perimeter * perimeter))
        if circularity < min_circularity:
            continue
        score = area * circularity
        if score > best_score:
            best_score = score
            best = (contour, area, circularity)
    if best is None:
        return None
    contour, area, circularity = best
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return None
    centre = (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])
    _, radius = cv2.minEnclosingCircle(contour)
    filled = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(filled, [contour], -1, 1, thickness=-1)
    return dict(
        centre=centre, area=area, radius=float(radius),
        circularity=circularity, mask=filled.astype(bool),
    )


def _masked_mean(frame, mask):
    values = frame[mask]
    if values.size == 0:
        return 0.0
    return float(values.mean())


def sharpness(frame, centre=None, size=200):
    """Variance of the Laplacian, which peaks when the image is in focus."""
    _require_cv2()
    image = np.clip(frame, 0, 255).astype(np.uint8)
    if centre is not None:
        half = size // 2
        x, y = int(centre[0]), int(centre[1])
        y0 = max(0, y - half)
        y1 = min(image.shape[0], y + half)
        x0 = max(0, x - half)
        x1 = min(image.shape[1], x + half)
        if y1 - y0 >= 16 and x1 - x0 >= 16:
            image = image[y0:y1, x0:x1]
    return float(cv2.Laplacian(image, cv2.CV_64F).var())


def cut_template(frame, centre, size=96):
    """Take a square patch around a point, for later template matching."""
    half = int(size) // 2
    x, y = int(centre[0]), int(centre[1])
    y0 = max(0, y - half)
    y1 = min(frame.shape[0], y + half)
    x0 = max(0, x - half)
    x1 = min(frame.shape[1], x + half)
    patch = frame[y0:y1, x0:x1]
    if patch.shape[0] < 16 or patch.shape[1] < 16:
        raise DetectionError("the nozzle is too close to the edge of the frame")
    return np.array(patch, dtype=np.float32)


def detect_tip(frame, strategy="contour", template=None, **options):
    """Locate the nozzle tip and return (x, y, confidence, details)."""
    if strategy == "template":
        if template is None:
            raise DetectionError("template strategy needs a stored template")
        return detect_template(frame, template, roi=options.get("roi"))
    if strategy == "hough":
        return detect_hough(frame, **options)
    return detect_contour(frame, **options)
