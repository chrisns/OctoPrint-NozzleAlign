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


def measure_motion(frame_a, frame_b, threshold=4.0, min_area=60,
                   min_circularity=0.25, max_extent=0.4, min_coverage=0.5):
    """Find what moved between two frames taken either side of a known move.

    Returns a dict with

    ``position``  where the moving thing sits in ``frame_a``
    ``shift``     how far it moved, to sub-pixel accuracy
    ``compact``   True when a single nozzle sized blob was isolated

    Two regimes matter, and the caller needs to know which one it got.

    Close to the camera the nozzle is the only thing in the frame, so it
    isolates cleanly as a compact blob and ``position`` really is the nozzle.

    Far from the camera the whole toolhead is in view and moves as one piece.
    No compact blob exists, and a small move only changes a thin crescent at
    each edge, so blob centroids would badly overstate the displacement.  In
    that regime the shift comes from phase correlation over the region that
    changed, which measures the movement of the structure itself, and
    ``position`` is the centre of that region rather than the nozzle.
    """
    _require_cv2()
    a = np.asarray(frame_a, dtype=np.float32)
    b = np.asarray(frame_b, dtype=np.float32)
    if a.shape != b.shape:
        raise DetectionError("the two frames are different sizes")
    difference = cv2.GaussianBlur(a - b, (9, 9), 0)
    noise = float(np.median(np.abs(difference - np.median(difference)))) * 1.4826
    level = max(threshold, threshold * noise)

    changed = np.abs(difference) > level
    if changed.sum() < min_area:
        raise DetectionError(
            "no moving object found; the nozzle may be outside the field of view"
        )
    box = _bounding_box(changed, pad=0, shape=a.shape)
    total_changed = float(changed.sum())

    # Coverage decides the regime, not blob shape and not how far apart the two
    # positions are.  A textured toolhead throws off plenty of small round
    # difference blobs, and any one of them would pass a roundness test while
    # being nothing to do with the nozzle.  When the nozzle is the only thing in
    # frame, its two positions account for nearly everything that changed.  When
    # the toolhead or the gantry is in frame as well, they do not.
    positive = _best_blob(difference > level, min_area, min_circularity, max_extent)
    negative = _best_blob(difference < -level, min_area, min_circularity, max_extent)
    coverage = 0.0
    if positive is not None and negative is not None:
        coverage = (positive["area"] + negative["area"]) / max(total_changed, 1.0)
        if coverage < min_coverage:
            positive = negative = None
    if positive is not None and negative is not None:
        in_a, in_b = _assign_by_contrast(a, positive, negative)
        shift = (
            in_b["centre"][0] - in_a["centre"][0],
            in_b["centre"][1] - in_a["centre"][1],
        )
        return dict(
            position=in_a["centre"],
            shift=shift,
            compact=True,
            details=dict(
                area=in_a["area"],
                radius=in_a["radius"],
                circularity=in_a["circularity"],
                coverage=coverage,
                noise=noise,
            ),
        )

    padded = _pad_box(box, pad=40, shape=a.shape)
    shift = _rigid_shift(a, difference, padded)
    ys, xs = np.nonzero(changed)
    return dict(
        position=(float(xs.mean()), float(ys.mean())),
        shift=shift,
        compact=False,
        details=dict(
            area=total_changed, coverage=coverage, noise=noise, box=list(padded)
        ),
    )


def locate_by_motion(frame_a, frame_b, threshold=4.0, min_area=60,
                     min_circularity=0.25, max_extent=0.4, min_coverage=0.5):
    """Where a compact moving object sits in each of two frames."""
    measured = measure_motion(
        frame_a, frame_b, threshold, min_area, min_circularity,
        max_extent, min_coverage,
    )
    if not measured["compact"]:
        raise DetectionError(
            "the moving region is not a single compact object; the camera is "
            "probably too far away to isolate the nozzle"
        )
    x, y = measured["position"]
    dx, dy = measured["shift"]
    return (x, y), (x + dx, y + dy), measured["details"]


def _assign_by_contrast(frame, positive, negative):
    """Decide which difference blob holds the object in the first frame.

    At the object's position in ``a``, frame ``a`` shows the object; at its
    position in ``b``, frame ``a`` shows the background.  So the blob whose
    pixels differ most from the background in ``a`` is the one in ``a``.  This
    works whether the object is darker or brighter than its surroundings.
    """
    background = float(np.median(frame))
    lift_positive = abs(_masked_mean(frame, positive["mask"]) - background)
    lift_negative = abs(_masked_mean(frame, negative["mask"]) - background)
    if lift_positive >= lift_negative:
        return positive, negative
    return negative, positive


def _bounding_box(mask, pad, shape):
    ys, xs = np.nonzero(mask)
    return _pad_box(
        (int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())), pad, shape
    )


def _pad_box(box, pad, shape):
    y0, y1, x0, x1 = box
    return (
        max(0, y0 - pad),
        min(shape[0], y1 + pad),
        max(0, x0 - pad),
        min(shape[1], x1 + pad),
    )


def _rigid_shift(frame_a, difference, box):
    """Displacement of a rigidly moving structure, from the difference alone.

    Correlating the two frames does not work here.  Most of the picture is a
    static background that dominates the correlation and pins the answer at
    zero, however far the toolhead actually moved.

    The difference image does not have that problem.  For a rigid shift ``s``
    it is ``D(x) = A(x) - A(x - s)``, whose autocorrelation carries a strong
    negative dip at ``+s`` and at ``-s``, and nothing from the static
    background at all.  The dip gives the distance; the two difference lobes
    give the direction.
    """
    y0, y1, x0, x1 = box
    patch = np.ascontiguousarray(difference[y0:y1, x0:x1], dtype=np.float32)
    if patch.shape[0] < 16 or patch.shape[1] < 16:
        raise DetectionError("the region that changed is too small to measure")
    patch = patch - patch.mean()
    window = np.outer(np.hanning(patch.shape[0]), np.hanning(patch.shape[1]))
    spectrum = np.fft.rfft2(patch * window)
    correlation = np.fft.fftshift(np.fft.irfft2(np.abs(spectrum) ** 2, patch.shape))

    centre_y, centre_x = patch.shape[0] // 2, patch.shape[1] // 2
    ys, xs = np.mgrid[0:patch.shape[0], 0:patch.shape[1]]
    radius = np.hypot(xs - centre_x, ys - centre_y)
    search = correlation.copy()
    search[radius < 3] = 0.0
    peak = np.unravel_index(np.argmin(search), search.shape)
    magnitude = (float(peak[1] - centre_x), float(peak[0] - centre_y))

    # The dip appears at both +s and -s, so the direction has to come from the
    # two lobes of the difference.  Which lobe is the starting position depends
    # on whether the object is darker or brighter than its background, so the
    # first frame decides it rather than an assumption about polarity.
    level = float(np.abs(patch).max()) * 0.25
    positive = patch > level
    negative = patch < -level
    if positive.sum() < 4 or negative.sum() < 4:
        return magnitude
    reference = np.ascontiguousarray(frame_a[y0:y1, x0:x1], dtype=np.float32)
    background = float(np.median(reference))
    lift_positive = abs(float(reference[positive].mean()) - background)
    lift_negative = abs(float(reference[negative].mean()) - background)
    start, end = (positive, negative) if lift_positive >= lift_negative else (negative, positive)
    sy, sx = np.nonzero(start)
    ey, ex = np.nonzero(end)
    direction = (float(ex.mean() - sx.mean()), float(ey.mean() - sy.mean()))
    if direction[0] * magnitude[0] + direction[1] * magnitude[1] < 0:
        return (-magnitude[0], -magnitude[1])
    return magnitude


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
