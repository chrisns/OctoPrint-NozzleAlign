# coding=utf-8
"""The closed loop XY nozzle alignment routine.

The routine drives the toolhead over a camera that sits on the bed, so every
motion goes through a safe Z first and every Z command passes one floor check.

What one run does, in order:

1. Home and select T0.
2. Find the camera. Nothing is remembered about where it is: the head sweeps
   the bed at the search height, watching for the toolhead to come into the
   picture, closes in on it, descends and finds the bore in a ring.
3. Sweep Z and settle at the height where the bore is sharpest. The camera
   mount, the bed and the lift mechanism all move the focal plane, so it is
   measured rather than trusted.
4. Build the pixel map from two probe moves, approached from one direction so
   backlash cannot shorten them, and refuse the map if it is not plausible.
5. Steer the bore onto the target pixel and read the machine position.
6. Select T1 and command the same position. The firmware applies the stored
   offset, so the T1 bore must now appear near the same pixel. If nothing is
   there, the nozzle just centred was the raised T1 nozzle, not the active T0
   one, and the run moves over by the nominal offset and starts again.
7. Steer T1 onto the same pixel and read the position. Both nozzles on one
   pixel means the scale and distortion cancel; only the machine positions
   matter.

The result is what T1 needed beyond T0, and the offset that would make that
zero.
"""

from __future__ import absolute_import

import threading
import time

import numpy as np

from . import focus, geometry, nozzle, vision
from .gcode import Timeout, format_offset_command


class Aborted(Exception):
    """The operator stopped the routine."""


class CalibrationError(Exception):
    """The routine could not finish."""


class NotConverged(CalibrationError):
    """A real nozzle was centred but never settled within the tolerance."""


class CalibrationRoutine(threading.Thread):
    """Measures the XY offset between nozzle 0 and nozzle 1."""

    def __init__(self, bridge, settings, notify, logger, on_camera=None):
        super(CalibrationRoutine, self).__init__()
        self.daemon = True
        self.name = "nozzlealign-routine"
        self._bridge = bridge
        self._cfg = settings
        self._notify = notify
        self._logger = logger
        self._on_camera = on_camera
        self._abort = threading.Event()
        self.result = None
        self.error = None
        self._target = None
        self._template = None
        self._last_bore = None
        self._stream = None

    # -- control ----------------------------------------------------------

    def abort(self):
        self._abort.set()

    def _check_abort(self):
        if self._abort.is_set():
            raise Aborted("stopped by the operator")

    def _progress(self, step, message, **extra):
        payload = dict(type="progress", step=step, message=message)
        payload.update(extra)
        self._logger.info("[%s] %s", step, message)
        self._notify(payload)

    # -- capture ----------------------------------------------------------

    def _frame(self, settle=1):
        """An averaged frame made after now."""
        self._check_abort()
        if self._stream is not None:
            return self._stream.average(int(self._cfg["frame_average"]))
        return vision.average_frames(
            self._cfg["snapshot_url"],
            count=int(self._cfg["frame_average"]),
            timeout=float(self._cfg["http_timeout"]),
            settle=settle,
        )

    def _frame_now(self):
        """One frame made after now, as fast as the camera gives it."""
        self._check_abort()
        if self._stream is not None:
            return self._stream.frame_after(time.time())[0]
        return vision.fetch_frame(self._cfg["snapshot_url"], float(self._cfg["http_timeout"]))

    def _open_stream(self):
        url = self._cfg.get("stream_url")
        if url:
            self._stream = vision.FrameStream(url, float(self._cfg["http_timeout"])).start()

    def _close_stream(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream = None

    def _find_bore(self, frame, near=None, radius=None):
        cfg = self._cfg
        return nozzle.find_bore(
            frame,
            centre=tuple(near) if near is not None else tuple(self._target),
            search_radius=float(radius if radius is not None else cfg["bore_search_radius_px"]),
            inner_r=int(cfg["bore_inner_r"]),
            ring_lo=int(cfg["bore_ring_lo"]),
            ring_hi=int(cfg["bore_ring_hi"]),
            core_r=int(cfg["bore_core_r"]),
        )

    def _measure_bore(self, near=None, radius=None, settle=1):
        """The bore's pixel position, or an error if nothing bore shaped is there.

        The first find is a wide search scored strictly. After that the bore
        is followed with a template cut from the last frame, and the detector
        only refines the centre close to where the template landed. The bore's
        contrast changes across the frame as the lighting does, and on T1 it
        fell below the strict bar one probe move away while a burnt patch on
        the cone scored higher; the template is what keeps the lock.
        """
        cfg = self._cfg
        frame = self._frame(settle=settle)
        if self._template is not None and near is None:
            spot, confidence = focus.track(
                frame, self._template, self._last_bore, int(cfg["track_search_px"]))
            if confidence >= float(cfg["track_min_match"]):
                found = self._find_bore(frame, spot, int(cfg["track_lock_px"]))
                if found["score"] >= float(cfg["bore_track_score"]):
                    self._remember(frame, found)
                    return np.array([found["x"], found["y"]], dtype=float), found["score"]
        if near is None and self._last_bore is not None:
            near = self._last_bore
        found = self._find_bore(frame, near, radius)
        if found["score"] < float(cfg["bore_min_score"]):
            raise CalibrationError(
                "no nozzle bore in view (score %.0f, need %.0f)"
                % (found["score"], float(cfg["bore_min_score"])))
        self._remember(frame, found)
        return np.array([found["x"], found["y"]], dtype=float), found["score"]

    def _remember(self, frame, found):
        self._last_bore = (found["x"], found["y"])
        self._template = focus.cut(frame, self._last_bore, int(self._cfg["focus_template_px"]) // 2)

    def _forget(self):
        """Drop the template, for when a different nozzle comes into view."""
        self._template = None
        self._last_bore = None

    def _bore_in_view(self, near=None, radius=None):
        try:
            return self._measure_bore(near, radius)
        except CalibrationError:
            return None, 0.0

    # -- motion -----------------------------------------------------------

    def _move_absolute(self, x=None, y=None, feedrate=None):
        self._check_abort()
        parts = ["G1"]
        if x is not None:
            parts.append("X%.4f" % x)
        if y is not None:
            parts.append("Y%.4f" % y)
        parts.append("F%d" % int(feedrate or self._cfg["feedrate"]))
        return self._bridge.run(["G90", " ".join(parts)], timeout=float(self._cfg["move_timeout"]))

    def _move_relative(self, dx=0.0, dy=0.0, feedrate=None):
        self._check_abort()
        command = "G1 X%.4f Y%.4f F%d" % (dx, dy, int(feedrate or self._cfg["fine_feedrate"]))
        return self._bridge.run(
            ["G91", command, "G90"], timeout=float(self._cfg["move_timeout"])
        )

    def _move_z(self, z, feedrate=None):
        """The one place a Z is commanded, so the floor is checked once."""
        self._check_abort()
        floor = float(self._cfg["min_z"])
        ceiling = max(float(self._cfg["safe_z"]), float(self._cfg["search_z"]))
        if z < floor - 1e-9:
            raise CalibrationError(
                "refusing Z%.3f: below the %.1f mm floor" % (z, floor))
        if z > ceiling + 1e-9:
            raise CalibrationError(
                "refusing Z%.3f: above the safe height %.1f" % (z, ceiling))
        return self._bridge.run(
            ["G90", "G1 Z%.4f F%d" % (z, int(feedrate or self._cfg["z_feedrate"]))],
            timeout=float(self._cfg["move_timeout"]),
        )

    def _settle_move(self, dx=0.0, dy=0.0):
        """Arrive at a point from one fixed direction, so backlash is taken up."""
        first, second = geometry.backlash_free(dx, dy, float(self._cfg["backlash_backoff_mm"]))
        self._move_relative(*first)
        self._move_relative(*second)
        time.sleep(float(self._cfg["settle_s"]))

    def _arrive(self, x, y):
        """Absolute version of :meth:`_settle_move`."""
        backoff = float(self._cfg["backlash_backoff_mm"])
        self._move_absolute(x=x - backoff, y=y - backoff)
        self._move_absolute(x=x, y=y, feedrate=self._cfg["fine_feedrate"])
        time.sleep(float(self._cfg["settle_s"]))

    def _retract_to_safe_z(self):
        self._move_z(float(self._cfg["safe_z"]))

    def _go_to_camera(self, x, y, z):
        """Reach the camera point through the safe Z, never across the bed low."""
        self._retract_to_safe_z()
        self._arrive(x, y)
        self._move_z(z)

    def _select_tool(self, tool):
        self._progress("tool", "selecting T%d" % tool)
        self._retract_to_safe_z()
        self._forget()
        self._bridge.run(["T%d" % tool], timeout=float(self._cfg["move_timeout"]))
        time.sleep(float(self._cfg["tool_settle_s"]))

    def _position(self):
        position = self._bridge.position(timeout=float(self._cfg["move_timeout"]))
        if position is None:
            raise CalibrationError("the printer did not report its position")
        return np.array(position[:2], dtype=float)

    # -- finding the bore -------------------------------------------------

    def _frame_quick(self):
        return self._frame_now()

    def _moved_fraction_after_nudge(self):
        """Nudge X and report how much of the picture one moving thing covers."""
        cfg = self._cfg
        nudge = float(cfg["motion_nudge_mm"])
        before = self._frame_quick()
        self._move_relative(dx=nudge)
        time.sleep(float(cfg["settle_s"]))
        after = self._frame_quick()
        self._move_relative(dx=-nudge)
        return vision.motion_blob(before, after, float(cfg["motion_threshold"]))

    def _search_bed(self):
        """Find the camera anywhere on the bed, then descend onto it.

        Nothing is assumed about where the camera is. At Z90, where the
        camera sees about 75 by 47 mm, the head sweeps X back and forth in
        steps, one row of Y at a time, and each frame is compared with the
        one before. The bed and the camera do not move with X, so only the
        toolhead can change the texture in the picture. A sighting is
        confirmed with a nudge, which must move one large connected patch of
        texture: the toolhead's cable chain and the flicker of its light
        change scattered pixels, not a patch. Then the head walks on along X
        until the toolhead leaves the picture, and up and down Y likewise;
        the middle of each stretch is the middle of the toolhead to within a
        step. The toolhead is wider than the field at this height, so a
        climb to where the most of it moves finds its middle. Both
        nozzles are within half an offset of that. Returns the ring span to
        search for the bore at the working height.
        """
        cfg = self._cfg
        x_min, x_max = float(cfg["bed_x_min"]), float(cfg["bed_x_max"])
        y_min, y_max = float(cfg["bed_y_min"]), float(cfg["bed_y_max"])
        feed = int(cfg["sweep_feedrate"])
        threshold = float(cfg["motion_threshold"])
        minimum = float(cfg["motion_min_blob"])
        step = float(cfg["sweep_step_mm"])

        def clamp(x, y):
            return min(max(x, x_min), x_max), min(max(y, y_min), y_max)

        def nudge_at(x, y):
            x, y = clamp(x, y)
            self._check_abort()
            self._move_absolute(x=x, y=y, feedrate=feed)
            time.sleep(float(cfg["settle_s"]))
            fraction = self._moved_fraction_after_nudge()
            self._progress("bed", "X%.0f Y%.0f: %.0f%% of the texture moved" % (x, y, fraction * 100),
                           x=x, y=y, moved=fraction)
            return fraction

        # -- sighting: a rectangle growing out from the middle of the bed --------
        # The camera is usually put down near the middle, so the search starts
        # there and grows outward. Frames can only be compared along X within
        # one row, because a Y move carries the camera with the bed and changes
        # everything, so each ring scans its two new rows along X and extends
        # every older row by one column each side, comparing the new frame
        # with the one kept at that row's edge.
        height = float(cfg["search_z"])
        self._move_z(height)
        row_gap = float(cfg["bed_row_mm"])
        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0
        self._progress("bed", "searching for the camera from the middle of the bed outwards at Z%.0f" % height)
        edges = {}       # (x, y) -> frame at a scanned edge of a row

        def sweep_row(y, x_from, x_to):
            """Move along the row in one go, watching for the toolhead.

            Frames are taken while the head moves, and each is compared with
            the one before. The first pair that differs by one big patch of
            texture puts the toolhead in view; where the head was then is
            read from the clock and the feedrate, good to a step or so, and
            the nudge that follows confirms it.
            """
            self._check_abort()
            self._move_absolute(x=x_from, y=y, feedrate=feed)
            time.sleep(float(cfg["settle_s"]))
            self._frame_quick()
            previous = self._frame_now()
            speed = float(cfg["sweep_feedrate"]) / 60.0
            distance = abs(x_to - x_from)
            direction = 1.0 if x_to > x_from else -1.0
            self._bridge.send(["G90", "G1 X%.4f F%d" % (x_to, int(cfg["sweep_feedrate"]))])
            started = time.time()
            hit = None
            while time.time() - started < distance / speed + float(cfg["sweep_overrun_s"]):
                self._check_abort()
                frame = self._frame_now()
                now = time.time() - started
                if hit is None and vision.motion_blob(previous, frame, threshold) >= minimum:
                    hit = x_from + direction * min(distance, speed * max(0.0, now - float(cfg["sweep_lag_s"])))
                previous = frame
            self._bridge.run([], timeout=float(cfg["move_timeout"]))
            return hit

        def look(x, y, compare):
            """Move, take a frame, and say whether the texture changed since ``compare``."""
            self._check_abort()
            self._move_absolute(x=x, y=y, feedrate=feed)
            frame = self._frame_quick()
            # The toolhead 20 mm further on is one big patch of moved texture.
            # The cable chain and the light are scattered pixels and are not.
            changed = compare is not None and \
                vision.motion_blob(compare, frame, threshold) >= minimum
            return frame, changed

        def confirmed(x, y, fraction_hint):
            self._progress("bed", "something changed at X%.0f Y%.0f; checking" % (x, y), x=x, y=y)
            if nudge_at(x, y) >= minimum:
                return True
            return False

        sighting = None
        ring = 0
        while sighting is None:
            half_x, half_y = ring * step, ring * row_gap
            xs = [x for x in (cx + i * step for i in range(-ring, ring + 1)) if x_min - 1e-6 <= x <= x_max + 1e-6]
            ys = [y for y in (cy + j * row_gap for j in range(-ring, ring + 1)) if y_min - 1e-6 <= y <= y_max + 1e-6]
            if ring > 0 and cx - half_x < x_min - 1e-6 and cx + half_x > x_max + 1e-6 \
                    and cy - half_y < y_min - 1e-6 and cy + half_y > y_max + 1e-6:
                break
            new_rows = [y for y in (cy - half_y, cy + half_y) if y in ys] if ring > 0 else [cy]
            old_rows = [y for y in ys if y not in new_rows]
            new_columns = [x for x in (cx - half_x, cx + half_x) if x in xs] if ring > 0 else []
            # the new rows, scanned along X in one move each
            for index, y in enumerate(dict.fromkeys(new_rows)):
                order = xs if index % 2 == 0 else xs[::-1]
                if len(order) > 1:
                    hit = sweep_row(y, order[0], order[-1])
                    if hit is not None:
                        near_x = min(xs, key=lambda x: abs(x - hit))
                        for x in (near_x, near_x - step, near_x + step):
                            if x_min - 1e-6 <= x <= x_max + 1e-6 and confirmed(x, y, None):
                                sighting = (x, y, 1 if order is xs else -1)
                                break
                    if sighting is not None:
                        break
                    for x in (xs[0], xs[-1]):
                        self._move_absolute(x=x, y=y, feedrate=feed)
                        edges[(x, y)] = self._frame_quick()
                else:
                    self._move_absolute(x=order[0], y=y, feedrate=feed)
                    edges[(order[0], y)] = self._frame_quick()
                if sighting is not None:
                    break
            if sighting is not None or ring == 0:
                if sighting is None and ring == 0:
                    ring += 1
                    continue
                break
            # the older rows, extended by the new columns
            for x in dict.fromkeys(new_columns):
                inner = x + step if x < cx else x - step
                direction = -1 if x < cx else 1
                for y in old_rows:
                    compare = edges.get((inner, y))
                    frame, changed = look(x, y, compare)
                    edges[(x, y)] = frame
                    if changed and confirmed(x, y, None):
                        sighting = (x, y, direction)
                        break
                if sighting is not None:
                    break
            for key in list(edges):
                if key[0] not in (xs[0], xs[-1]):
                    del edges[key]
            ring += 1
        if sighting is None:
            raise CalibrationError(
                "the toolhead never came into view over the bed; is the camera "
                "plugged in and pointing up?")

        # -- the middle of the toolhead, from where it leaves the picture -------
        x, y, direction = sighting
        last_x = x
        probe = x + direction * step
        while x_min <= probe <= x_max and nudge_at(probe, y) >= minimum:
            last_x = probe
            probe += direction * step
        first_x = x
        probe = x - direction * step
        while x_min <= probe <= x_max and nudge_at(probe, y) >= minimum:
            first_x = probe
            probe -= direction * step
        x_mid = (first_x + last_x) / 2.0
        last_y = first_y = y
        probe = y + step
        while probe <= y_max and nudge_at(x_mid, probe) >= minimum:
            last_y = probe
            probe += step
        probe = y - step
        while probe >= y_min and nudge_at(x_mid, probe) >= minimum:
            first_y = probe
            probe -= step
        y_mid = (first_y + last_y) / 2.0
        self._progress("bed", "the toolhead is over the camera near X%.0f Y%.0f" % (x_mid, y_mid))

        # -- close in: put the moving patch in the middle of the picture -------
        # The fraction that moves is flat and noisy across X, because the
        # toolhead is nearly as wide as the field, so it cannot place the
        # toolhead in X. The patch's position in the picture can. A nudge
        # gives the picture's direction and scale for machine X, and the
        # offset of the patch from the picture's centre along that direction
        # is how far to move. Two or three passes settle it.
        self._move_absolute(x=x_mid, y=y_mid, feedrate=feed)
        if abs(float(cfg["closein_z"]) - height) > 1e-6:
            self._move_z(float(cfg["closein_z"]))
        nudge = float(cfg["motion_nudge_mm"])
        x_here = x_mid
        for attempt in range(int(cfg["closein_max_steps"])):
            self._check_abort()
            time.sleep(float(cfg["settle_s"]))
            before = self._frame_quick()
            self._move_relative(dx=nudge)
            time.sleep(float(cfg["settle_s"]))
            after = self._frame_quick()
            self._move_relative(dx=-nudge)
            centroid, box, fraction = vision.moving_region(before, after, threshold)
            if centroid is None or fraction < minimum:
                raise CalibrationError(
                    "the toolhead was seen at X%.0f Y%.0f and then lost" % (x_here, y_mid))
            shift, response = vision.shift_in_box(before, after, box)
            scale = float(np.hypot(*shift)) / nudge
            if scale < float(cfg["closein_min_scale"]) or response < float(cfg["closein_min_response"]):
                self._progress("bed", "cannot read the picture's X direction (%.1f px/mm, response %.2f); "
                               "keeping X%.0f" % (scale, response, x_here))
                break
            unit = np.array(shift) / float(np.hypot(*shift))
            centre_px = np.array([before.shape[1] / 2.0, before.shape[0] / 2.0])
            offset_mm = float((np.array(centroid) - centre_px) @ unit) / scale
            self._progress("bed", "the toolhead is %.1f mm along X from the lens (%.1f px/mm, %.0f%% in view)"
                           % (offset_mm, scale, fraction * 100), offset=offset_mm, px_per_mm=scale)
            if abs(offset_mm) < float(cfg["closein_done_mm"]):
                break
            step_mm = max(-float(cfg["closein_max_move_mm"]), min(float(cfg["closein_max_move_mm"]), -offset_mm))
            x_here = min(max(x_here + step_mm, x_min), x_max)
            self._move_absolute(x=x_here, feedrate=feed)
        centre = (x_here, y_mid)
        self._progress("bed", "the middle of the toolhead is at X%.0f Y%.0f" % centre)
        # T0's active nozzle sits half an offset towards minus X from the
        # middle, and T1's raised one the other way. Starting the ring at
        # T0's side puts the active nozzle first in line.
        self._arrive(centre[0] - float(cfg["nominal_offset_x"]) / 2.0, centre[1])
        self._move_z(float(cfg["camera_z"]))
        time.sleep(float(cfg["settle_s"]))
        self._forget()
        return float(cfg["bed_search_span_mm"])

    def _bore_candidates(self, span):
        """Bores found around the current point at the working height.

        The camera is put down by hand, so it is rarely exactly where it was
        last time. A ring of points spaced by most of the field of view covers
        a few centimetres in a few moves. Every point is reached from the same
        direction, so the map built afterwards is not spoilt by backlash. The
        first candidate is at the starting point itself.
        """
        cfg = self._cfg
        step = float(cfg["search_step_mm"])
        origin = self._position()
        offsets = [(0.0, 0.0)]
        radius = step
        while radius <= float(span) + 1e-9:
            count = max(6, int(round(2 * np.pi * radius / step)))
            for index in range(count):
                angle = 2 * np.pi * index / count
                offsets.append((radius * np.cos(angle), radius * np.sin(angle)))
            radius += step
        for dx, dy in offsets:
            self._check_abort()
            if (dx, dy) != (0.0, 0.0):
                self._arrive(origin[0] + dx, origin[1] + dy)
            self._forget()
            where, score = self._bore_in_view(radius=cfg["search_radius_px"])
            if where is not None:
                self._progress(
                    "search",
                    "a bore %.1f mm from the starting point (score %.0f)"
                    % (float(np.hypot(dx, dy)), score))
                yield where

    def _focus_here(self, bore_px):
        """Sweep Z and stop at the sharpest height for this nozzle.

        A coarse sweep first, so a candidate that is not a nozzle is rejected
        in a few frames, then a fine sweep around the coarse peak.
        """
        cfg = self._cfg
        start = float(cfg["camera_z"])
        floor, ceiling = float(cfg["min_z"]), float(cfg["safe_z"])
        template = focus.cut(self._frame(), bore_px, int(cfg["focus_template_px"]) // 2)
        where = tuple(bore_px)
        samples = []
        seen = {}

        def sweep(heights):
            nonlocal template, where
            for z in heights:
                self._move_z(z)
                time.sleep(float(cfg["settle_s"]))
                frame = self._frame()
                where, confidence = focus.track(frame, template, where)
                value = focus.sharpness(frame, where, int(cfg["focus_window_px"]))
                samples.append((z, value))
                self._progress("focus", "Z%.2f: sharpness %.0f" % (z, value), z=z, sharpness=value)
                template = focus.cut(frame, where, int(cfg["focus_template_px"]) // 2)
                seen[z] = (template, where)

        coarse = focus.plan_heights(start, float(cfg["focus_span_mm"]), float(cfg["focus_step_mm"]),
                                    floor, ceiling)
        sweep(coarse)
        if not focus.is_peaked(samples, float(cfg["focus_peak_ratio"])):
            raise CalibrationError(
                "the focus did not peak inside Z%.1f to Z%.1f; the camera height "
                "or the lens has changed more than the sweep covers"
                % (min(coarse), max(coarse)))
        # Restart the tracking from the sharpest coarse frame: a template cut
        # from the blurred end of the sweep does not match a sharp picture.
        peak_z = max(samples, key=lambda sample: sample[1])[0]
        template, where = seen[peak_z]
        fine = [round(z, 2) for z in focus.plan_heights(
            peak_z, float(cfg["focus_step_mm"]) / 2.0, float(cfg["focus_fine_step_mm"]), floor, ceiling)]
        fine = [z for z in fine if all(abs(z - done) > 1e-3 for done, _ in samples)]
        sweep(sorted(fine, key=lambda z: abs(z - peak_z)))
        best, peak = focus.best_height(samples)
        best = min(max(best, min(coarse)), max(coarse))
        self._move_z(best)
        time.sleep(float(cfg["settle_s"]))
        self._progress("focus", "sharpest at Z%.2f" % best, z=best, sharpness=peak)
        # The bore has moved in the picture as the scale changed with Z;
        # the next measurement must look for it where it is now.
        frame = self._frame()
        where, _ = focus.track(frame, template, where)
        self._remember(frame, dict(x=where[0], y=where[1]))
        return best

    # -- calibration steps ------------------------------------------------

    def _build_pixel_map(self):
        """Measure how many pixels the bore moves per millimetre of machine travel."""
        distance = float(self._cfg["probe_distance"])
        attempts = int(self._cfg["map_attempts"])
        self._progress("map", "measuring the pixel to millimetre map")
        for attempt in range(attempts):
            origin, _ = self._measure_bore()
            self._settle_move(dx=distance)
            try:
                after_x, _ = self._measure_bore()
            finally:
                self._settle_move(dx=-distance)
            self._settle_move(dy=distance)
            try:
                after_y, _ = self._measure_bore()
            finally:
                self._settle_move(dy=-distance)
            matrix = geometry.build_pixel_map(origin, after_x, after_y, distance)
            ok, scale, cosine = geometry.validate_map(
                matrix, float(self._cfg["map_min_scale"]),
                float(self._cfg["map_max_scale"]), float(self._cfg["map_max_cos"]))
            if ok:
                self._progress(
                    "map",
                    "map ready: %.1f px/mm, camera rotated %.1f degrees"
                    % (scale, geometry.rotation_degrees(matrix)),
                    px_per_mm=scale, rotation=geometry.rotation_degrees(matrix))
                return matrix
            self._progress(
                "map", "map rejected (%.1f px/mm, columns %.2f aligned); trying again"
                % (scale, cosine))
        raise CalibrationError(
            "could not build a trustworthy pixel map in %d attempts" % attempts)

    def _centre_bore(self, matrix, label):
        """Move the active nozzle until its bore sits on the target pixel."""
        tolerance = float(self._cfg["tolerance_mm"])
        passes = int(self._cfg["max_passes"])
        limit = float(self._cfg["max_correction_mm"])
        residual = None
        for index in range(passes):
            current, score = self._measure_bore()
            dx, dy = geometry.pixel_error_to_mm(matrix, current, self._target)
            residual = float(np.hypot(dx, dy))
            self._progress(
                "centre",
                "%s pass %d: off by %.4f mm (bore score %.0f)" % (label, index + 1, residual, score),
                tool=label, passes=index + 1, residual_mm=residual)
            if residual <= tolerance:
                break
            if residual > limit:
                raise CalibrationError(
                    "%s wants a %.2f mm correction, which is beyond the %.2f mm limit"
                    % (label, residual, limit))
            self._settle_move(dx=dx, dy=dy)
        else:
            raise NotConverged(
                "%s did not settle within %.4f mm in %d passes (last residual %.4f mm)"
                % (label, tolerance, passes, residual))
        return self._position(), residual

    def _settle_on_active_nozzle(self):
        """Find the camera, put the nozzle in view on the target, and report.

        Returns the machine position, the residual, the focus height and the
        pixel map. A candidate that will not focus, map or centre was not a
        nozzle, and the search moves on to the next.
        """
        self._retract_to_safe_z()
        return self._settle_near(self._search_bed())

    def _settle_near(self, span):
        """Put the nozzle within ``span`` of the current point on the target."""
        cfg = self._cfg
        for bore in self._bore_candidates(span):
            try:
                z = self._focus_here(bore)
                matrix = self._build_pixel_map()
                position, residual = self._centre_bore(matrix, "T0")
                return position, residual, z, matrix
            except NotConverged:
                raise
            except (CalibrationError, focus.FocusError) as exception:
                self._progress("search", "that was not a nozzle: %s" % exception)
                self._move_z(float(cfg["camera_z"]))
        raise CalibrationError("the toolhead was over the camera but no nozzle bore was found")

    # -- main -------------------------------------------------------------

    def run(self):
        try:
            try:
                self._open_stream()
                self.result = self._execute()
            finally:
                self._park()
                self._close_stream()
            self._notify(dict(type="done", result=self.result))
        except Aborted as exception:
            self.error = str(exception)
            self._notify(dict(type="aborted", message=str(exception)))
        except (CalibrationError, focus.FocusError, nozzle.NozzleError,
                vision.CaptureError, vision.DetectionError,
                geometry.GeometryError, Timeout, RuntimeError) as exception:
            self.error = str(exception)
            self._logger.exception("calibration failed")
            self._notify(dict(type="failed", message=str(exception)))
        except Exception as exception:  # pragma: no cover - unexpected
            self.error = str(exception)
            self._logger.exception("calibration crashed")
            self._notify(dict(type="failed", message="unexpected error: %s" % exception))

    def _park(self):
        """Leave the head high and T0 selected, whatever happened.

        A failed run must not leave the nozzle sitting on the camera, and the
        next run must not start with a tool change at the working height.
        """
        try:
            if self._bridge.position(timeout=float(self._cfg["move_timeout"])) is None:
                return
            self._abort.clear()
            self._retract_to_safe_z()
            self._bridge.run(["T0"], timeout=float(self._cfg["move_timeout"]))
        except Exception as exception:  # pragma: no cover - best effort
            self._logger.warning("could not park after the run: %s", exception)

    def _execute(self):
        cfg = self._cfg
        self._check_abort()
        self._progress("preflight", "checking the camera and the printer")
        if not (float(cfg["min_z"]) < float(cfg["camera_z"]) < float(cfg["safe_z"])):
            raise CalibrationError(
                "the heights must satisfy floor %.1f < camera %.1f < safe %.1f"
                % (float(cfg["min_z"]), float(cfg["camera_z"]), float(cfg["safe_z"])))

        frame = self._frame()
        mean, deviation = vision.frame_health(frame)
        self._progress(
            "preflight",
            "camera frame mean %.1f, standard deviation %.1f" % (mean, deviation))
        self._target = np.array([
            float(cfg["target_x_px"]) if cfg.get("target_x_px") is not None else frame.shape[1] / 2.0,
            float(cfg["target_y_px"]) if cfg.get("target_y_px") is not None else frame.shape[0] / 2.0,
        ], dtype=float)

        stored = self._bridge.read_hotend_offset(tool=1)
        if stored is None:
            raise CalibrationError(
                "could not read the stored hotend offset with M218 or M503")
        self._progress(
            "preflight",
            "stored T1 offset is X%.3f Y%.3f" % (stored[0], stored[1]),
            stored_offset=[stored[0], stored[1]])

        if cfg["home_first"]:
            self._progress("home", "homing all axes")
            self._bridge.run(["G28"], timeout=float(cfg["home_timeout"]))

        self._select_tool(0)
        position_0, residual_0, z, matrix = self._settle_on_active_nozzle()
        self._progress(
            "centre", "nozzle 0 centred at X%.3f Y%.3f" % (position_0[0], position_0[1]))

        # The firmware applies the stored offset on the tool change, so T1's
        # bore must now appear within the stored offset's error of the same
        # pixel. If it does not, the nozzle just centred was the raised T1
        # nozzle: T0's active nozzle is a nominal offset away, on the +X side.
        self._select_tool(1)
        self._arrive(position_0[0], position_0[1])
        self._move_z(z)
        time.sleep(float(cfg["settle_s"]))
        near_px = float(cfg["active_check_mm"]) * geometry.pixels_per_mm(matrix)
        bore, _ = self._bore_in_view(radius=near_px)
        if bore is None:
            self._progress(
                "check",
                "no bore near the target under T1: the nozzle centred under T0 was "
                "the raised T1 nozzle; moving over by the nominal offset")
            self._select_tool(0)
            self._go_to_camera(
                position_0[0] + float(cfg["nominal_offset_x"]),
                position_0[1] + float(cfg["nominal_offset_y"]), z)
            self._forget()
            position_0, residual_0, z, matrix = self._settle_near(float(cfg["bed_search_span_mm"]))
            self._progress(
                "centre", "nozzle 0 centred at X%.3f Y%.3f" % (position_0[0], position_0[1]))
            self._select_tool(1)
            self._arrive(position_0[0], position_0[1])
            self._move_z(z)
            time.sleep(float(cfg["settle_s"]))
            bore, _ = self._bore_in_view(radius=near_px)
            if bore is None:
                raise CalibrationError(
                    "T1's bore is not within %.1f mm of where T0's was; the stored "
                    "offset is further out than this routine will chase"
                    % float(cfg["active_check_mm"]))

        matrix_1 = self._build_pixel_map()
        position_1, residual_1 = self._centre_bore(matrix_1, "T1")
        self._progress(
            "centre", "nozzle 1 centred at X%.3f Y%.3f" % (position_1[0], position_1[1]))
        self._retract_to_safe_z()

        correction = position_1 - position_0
        new_offset = corrected_offset(stored, position_0, position_1)
        self._progress(
            "result",
            "nozzle 1 needed X%+.4f Y%+.4f beyond nozzle 0; the offset that fixes "
            "that is X%.3f Y%.3f" % (correction[0], correction[1], new_offset[0], new_offset[1]))

        camera = dict(camera_x=float(position_0[0]), camera_y=float(position_0[1]),
                      camera_z=float(z))
        if self._on_camera is not None:
            self._on_camera(camera)

        return dict(
            stored_offset=[float(stored[0]), float(stored[1]), float(stored[2])],
            position_0=[float(position_0[0]), float(position_0[1])],
            position_1=[float(position_1[0]), float(position_1[1])],
            correction=[float(correction[0]), float(correction[1])],
            new_offset=[float(new_offset[0]), float(new_offset[1])],
            residual_0_mm=residual_0,
            residual_1_mm=residual_1,
            focus_z=float(z),
            px_per_mm=geometry.pixels_per_mm(matrix_1),
            rotation_deg=geometry.rotation_degrees(matrix_1),
            camera=camera,
        )


def corrected_offset(stored, position_0, position_1):
    """The stored offset that would put both nozzles on the same spot.

    The firmware moves the head by minus the offset when T1 is selected, so a
    larger offset moves T1's nozzle towards minus X. If T1 had to be commanded
    further along than T0 to reach the same spot, the offset is too small by
    that amount. Confirmed on the machine on 2026-09-06: changing the stored
    X by -0.130 moved the commanded gap by -0.129.
    """
    return (
        float(stored[0]) + (float(position_0[0]) - float(position_1[0])),
        float(stored[1]) + (float(position_0[1]) - float(position_1[1])),
    )


def offset_command(tool, x, y):
    """Re-exported for the API layer."""
    return format_offset_command(tool, x, y)
