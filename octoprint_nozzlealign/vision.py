# coding=utf-8
"""Frame capture.

Frames come from go2rtc over HTTP.  The plugin never opens the camera device
itself: go2rtc already holds it, and a UVC device only allows one consumer.
Finding the nozzle in a frame lives in :mod:`nozzle`; focus in :mod:`focus`.
"""

from __future__ import absolute_import

import io
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


def motion_fraction(before, after, threshold=12.0):
    """The fraction of the frame that changed between two frames.

    X moves the toolhead and nothing else the camera can see, so a nudge in
    X changes the picture only where the toolhead is. This is the coarse
    test for whether the toolhead is over the camera at all.
    """
    difference = np.abs(np.asarray(after, dtype=np.float32) - np.asarray(before, dtype=np.float32))
    return float((difference > float(threshold)).mean())
