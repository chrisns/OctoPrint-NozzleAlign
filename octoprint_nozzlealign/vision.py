# coding=utf-8
"""Frame capture.

Frames come from go2rtc over HTTP.  The plugin never opens the camera device
itself: go2rtc already holds it, and a UVC device only allows one consumer.
Finding the nozzle in a frame lives in :mod:`nozzle`; focus in :mod:`focus`.
"""

from __future__ import absolute_import

import io
import threading
import time

import numpy as np
import requests
from PIL import Image


class CaptureError(Exception):
    """The camera did not give us a usable frame."""


class DetectionError(Exception):
    """No nozzle was found in the frame."""


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


class FrameStream(object):
    """The camera's MJPEG stream, read in the background, latest frame kept.

    Fetching ``frame.jpeg`` costs about a second per frame on this rig, and a
    run takes hundreds of frames. The stream delivers them as the camera
    makes them, so a frame is ready the moment it is asked for, and a frame
    taken after a move is guaranteed to be one the camera made after the
    move, because every frame carries the time it arrived.
    """

    def __init__(self, url, timeout=10.0):
        self.url = url
        self.timeout = float(timeout)
        self._latest = None          # (arrived, bytes)
        self._lock = threading.Lock()
        self._changed = threading.Condition(self._lock)
        self._stop = threading.Event()
        self._thread = None
        self.error = None

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._read, name="nozzlealign-stream")
        self._thread.daemon = True
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def _read(self):
        while not self._stop.is_set():
            try:
                response = requests.get(self.url, stream=True, timeout=self.timeout)
                response.raise_for_status()
                buffer = b""
                for chunk in response.iter_content(chunk_size=16384):
                    if self._stop.is_set():
                        break
                    buffer += chunk
                    while True:
                        start = buffer.find(b"\xff\xd8")
                        end = buffer.find(b"\xff\xd9", start + 2) if start >= 0 else -1
                        if start < 0 or end < 0:
                            if len(buffer) > 4 * 1024 * 1024:
                                buffer = b""
                            break
                        jpeg = buffer[start:end + 2]
                        buffer = buffer[end + 2:]
                        with self._changed:
                            self._latest = (time.time(), jpeg)
                            self._changed.notify_all()
            except Exception as exception:  # pragma: no cover - network
                self.error = str(exception)
                if self._stop.wait(1.0):
                    break

    def frame_after(self, moment, timeout=20.0):
        """A frame the camera made after ``moment``, decoded to greyscale float32."""
        deadline = time.time() + timeout
        with self._changed:
            while self._latest is None or self._latest[0] <= moment:
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise CaptureError("no frame from the stream within %.0fs%s" % (
                        timeout, "; " + self.error if self.error else ""))
                self._changed.wait(remaining)
            arrived, jpeg = self._latest
        image = Image.open(io.BytesIO(jpeg)).convert("L")
        frame = np.asarray(image, dtype=np.float32)
        if is_blank(frame):
            raise CaptureError("the stream frame is blank; check the go2rtc stream")
        return frame, arrived

    def average(self, count=1):
        """The mean of ``count`` frames all made after now."""
        moment = time.time()
        total = None
        for _ in range(int(count)):
            frame, moment = self.frame_after(moment)
            total = frame if total is None else total + frame
        return total / float(count)


def frame_health(frame):
    """Return (mean, standard deviation) so a dead stream can be spotted.

    A broken MJPEG path decodes to flat mid grey: mean near 128 and a standard
    deviation below about 5.  A real picture is far noisier than that.
    """
    return float(frame.mean()), float(frame.std())


def is_blank(frame, min_std=8.0):
    """True when the frame carries no picture."""
    return frame_health(frame)[1] < min_std


def average_frames(url, count=8, timeout=10.0, settle=0, attempts=4):
    """Average several frames to cut sensor noise.

    ``settle`` frames are fetched and thrown away first, which lets the camera
    finish reacting to a move before the measurement starts.
    """
    for _ in range(settle):
        fetch_frame(url, timeout, attempts)
    total = None
    for _ in range(count):
        frame = fetch_frame(url, timeout, attempts)
        total = frame if total is None else total + frame
    stacked = total / float(count)
    if is_blank(stacked):
        raise CaptureError(
            "camera frame is blank (mean %.1f, sd %.1f); check the go2rtc stream"
            % frame_health(stacked)
        )
    return stacked


def _texture(frame, sigma=25.0):
    """The frame with its lighting taken out: what is left is texture."""
    import cv2

    image = np.asarray(frame, dtype=np.float32)
    size = int(sigma * 4) | 1
    return image - cv2.GaussianBlur(image, (size, size), sigma)


def motion_blob(before, after, threshold=12.0):
    """The fraction of the frame covered by the largest patch of moved texture.

    The toolhead over the camera is one big connected patch. Its cable chain
    in the distance, or a flicker of its light, is thin or scattered, and
    scores little here even when many pixels changed.
    """
    import cv2

    difference = np.abs(_texture(after) - _texture(before))
    mask = (difference > float(threshold)).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return 0.0
    largest = stats[1:, cv2.CC_STAT_AREA].max()
    return float(largest) / float(mask.size)


def moving_region(before, after, threshold=12.0):
    """The largest patch of moved texture: its centroid, its box and its fraction.

    Returns ``(centroid, (x0, y0, x1, y1), fraction)`` or ``(None, None, 0)``.
    """
    import cv2

    difference = np.abs(_texture(after) - _texture(before))
    mask = (difference > float(threshold)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return None, None, 0.0
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x0, y0, w, h, area = stats[largest]
    centroid = (float(centroids[largest][0]), float(centroids[largest][1]))
    return centroid, (int(x0), int(y0), int(x0 + w), int(y0 + h)), float(area) / float(mask.size)


def shift_in_box(before, after, box):
    """How far the picture inside ``box`` moved between two frames, in pixels."""
    import cv2

    x0, y0, x1, y1 = box
    a = _texture(before)[y0:y1, x0:x1]
    b = _texture(after)[y0:y1, x0:x1]
    if a.shape[0] < 32 or a.shape[1] < 32:
        return (0.0, 0.0), 0.0
    window = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(a, b, window)
    return (float(dx), float(dy)), float(response)


