# coding=utf-8
"""The closed loop XY nozzle alignment routine.

The routine drives the toolhead over a camera that sits on the bed, so every
motion goes through a safe Z first, every Z command passes one floor check,
and the whole thing refuses to start until the camera position has been set.

What one run does, in order:

1. Home, select T0 and go to the camera through the safe Z.
2. Find the bore. If it is not where the settings say, search around there at
   the working height, because the camera is never put down in quite the
   same place twice.
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
        self._check_abort()
        return vision.average_frames(
            self._cfg["snapshot_url"],
            count=int(self._cfg["frame_average"]),
            timeout=float(self._cfg["http_timeout"]),
            settle=settle,
        )

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
        ceiling = float(self._cfg["safe_z"])
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

    def _search_for_bore(self):
        """Look around the stored camera point at the working height.

        The camera is put down by hand, so it is rarely exactly where it was
        last time. A ring of points spaced by most of the field of view covers
        a few centimetres in a few moves. Every point is reached from the same
        direction, so the map built afterwards is not spoilt by backlash.
        """
        cfg = self._cfg
        span = float(cfg["search_span_mm"])
        step = float(cfg["search_step_mm"])
        origin = self._position()
        offsets = [(0.0, 0.0)]
        radius = step
        while radius <= span + 1e-9:
            count = max(6, int(round(2 * np.pi * radius / step)))
            for index in range(count):
                angle = 2 * np.pi * index / count
                offsets.append((radius * np.cos(angle), radius * np.sin(angle)))
            radius += step
        for dx, dy in offsets:
            self._check_abort()
            self._arrive(origin[0] + dx, origin[1] + dy)
            where, score = self._bore_in_view(radius=cfg["search_radius_px"])
            if where is not None:
                self._progress(
                    "search",
                    "found the bore %.1f mm from the stored point (score %.0f)"
                    % (float(np.hypot(dx, dy)), score))
                return where
        raise CalibrationError(
            "no nozzle bore within %.0f mm of X%.1f Y%.1f at the working height; "
            "run 'Find the camera' or set the camera position"
            % (span, origin[0], origin[1]))

    def _focus_here(self, bore_px):
        """Sweep Z and stop at the sharpest height for this nozzle."""
        cfg = self._cfg
        start = float(cfg["camera_z"])
        heights = focus.plan_heights(
            start, float(cfg["focus_span_mm"]), float(cfg["focus_step_mm"]),
            float(cfg["min_z"]), float(cfg["safe_z"]))
        samples = []
        template = focus.cut(self._frame(), bore_px, int(cfg["focus_template_px"]) // 2)
        where = tuple(bore_px)
        for z in heights:
            self._move_z(z)
            time.sleep(float(cfg["settle_s"]))
            frame = self._frame()
            where, confidence = focus.track(frame, template, where)
            value = focus.sharpness(frame, where, int(cfg["focus_window_px"]))
            samples.append((z, value))
            self._progress("focus", "Z%.2f: sharpness %.0f" % (z, value),
                           z=z, sharpness=value)
            template = focus.cut(frame, where, int(cfg["focus_template_px"]) // 2)
        if not focus.is_peaked(samples, float(cfg["focus_peak_ratio"])):
            raise CalibrationError(
                "the focus did not peak inside Z%.1f to Z%.1f; the camera height "
                "or the lens has changed more than the sweep covers"
                % (min(heights), max(heights)))
        best, peak = focus.best_height(samples)
        best = min(max(best, min(heights)), max(heights))
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
            raise CalibrationError(
                "%s did not settle within %.4f mm in %d passes (last residual %.4f mm)"
                % (label, tolerance, passes, residual))
        return self._position(), residual

    def _settle_on_active_nozzle(self):
        """Put T0's active nozzle on the target and return its position and focus.

        Returns the machine position, the focus height and the pixel map.
        """
        cfg = self._cfg
        bore = self._bore_in_view(radius=cfg["search_radius_px"])[0]
        if bore is None:
            self._progress("search", "no bore at the stored camera point; searching")
            bore = self._search_for_bore()
        z = self._focus_here(bore)
        matrix = self._build_pixel_map()
        position, residual = self._centre_bore(matrix, "T0")
        return position, residual, z, matrix

    # -- main -------------------------------------------------------------

    def run(self):
        try:
            try:
                self.result = self._execute()
            finally:
                self._park()
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

        frame = vision.average_frames(
            cfg["snapshot_url"], count=2, timeout=float(cfg["http_timeout"]))
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
        self._progress("move", "moving nozzle 0 over the camera")
        self._go_to_camera(float(cfg["camera_x"]), float(cfg["camera_y"]), float(cfg["camera_z"]))

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
            position_0, residual_0, z, matrix = self._settle_on_active_nozzle()
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
