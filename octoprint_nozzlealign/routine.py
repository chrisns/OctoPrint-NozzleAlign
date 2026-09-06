# coding=utf-8
"""The closed loop XY nozzle alignment routine.

The routine drives the toolhead over a camera that sits on the bed, so every
motion goes through a safe Z first and the whole thing refuses to start until
the camera position has been set.
"""

from __future__ import absolute_import

import threading
import time

import numpy as np

from . import geometry, nozzle, vision
from .gcode import Timeout, format_offset_command


class Aborted(Exception):
    """The operator stopped the routine."""


class CalibrationError(Exception):
    """The routine could not finish."""


class CalibrationRoutine(threading.Thread):
    """Measures the XY offset between nozzle 0 and nozzle 1."""

    def __init__(self, bridge, settings, notify, logger):
        super(CalibrationRoutine, self).__init__()
        self.daemon = True
        self.name = "nozzlealign-routine"
        self._bridge = bridge
        self._cfg = settings
        self._notify = notify
        self._logger = logger
        self._abort = threading.Event()
        self.result = None
        self.error = None

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

    def _motion_probe(self, dx=0.0, dy=0.0):
        """Move by a known amount and report what moved in the picture.

        The toolhead is the only thing the camera can see that moves, so this
        finds it without any model of what a nozzle looks like.  The toolhead is
        left where it started.  See :func:`vision.measure_motion` for what the
        returned dict holds and why ``compact`` matters.
        """
        before = self._frame()
        self._move_relative(dx=dx, dy=dy)
        after = self._frame()
        self._move_relative(dx=-dx, dy=-dy)
        return vision.measure_motion(
            before,
            after,
            threshold=float(self._cfg["motion_threshold"]),
            min_area=int(self._cfg["motion_min_area"]),
            min_circularity=float(self._cfg["motion_min_circularity"]),
            max_extent=float(self._cfg["motion_max_extent"]),
            min_coverage=float(self._cfg["motion_min_coverage"]),
        )

    def _measure_tip(self, settle=2):
        """Average frames, detect the tip, and return its pixel position."""
        if self._cfg["strategy"] == "circle":
            return self._measure_circle(settle)
        if self._cfg["strategy"] == "motion":
            measured = self._motion_probe(dx=float(self._cfg["motion_probe_mm"]))
            if not measured["compact"]:
                raise CalibrationError(
                    "the nozzle did not separate from the rest of the toolhead; "
                    "the camera is too far away to measure an offset"
                )
            return (
                np.array(measured["position"], dtype=float),
                1.0,
                measured["details"],
            )
        frame = self._frame(settle=settle)
        options = dict(roi=self._cfg.get("roi"))
        strategy = self._cfg["strategy"]
        if strategy == "contour":
            options.update(
                invert=bool(self._cfg["contour_invert"]),
                blur=int(self._cfg["blur"]),
                min_area=int(self._cfg["min_area"]),
            )
        elif strategy == "hough":
            options.update(
                blur=int(self._cfg["blur"]),
                min_radius=int(self._cfg["min_radius"]),
                max_radius=int(self._cfg["max_radius"]),
                param2=int(self._cfg["hough_param2"]),
            )
        x, y, confidence, details = vision.detect_tip(
            frame,
            strategy=strategy,
            template=self._cfg.get("template"),
            **options
        )
        if confidence < float(self._cfg["min_confidence"]):
            raise CalibrationError(
                "tip detection confidence %.2f is below the %.2f limit"
                % (confidence, float(self._cfg["min_confidence"]))
            )
        return np.array([x, y], dtype=float), confidence, details

    # -- motion -----------------------------------------------------------

    def _measure_circle(self, settle=2):
        """Find the nozzle as a circle and refine its centre by fitting the rim.

        This is the strategy to use once the camera is aimed at the nozzles and
        focused on them. A nozzle is then the most circular thing in the picture,
        which is far easier than anything the motion strategies have to do, and
        the rim fit gives a sub-pixel centre plus a residual that says how
        trustworthy it is. A nozzle caked in burnt filament fits loosely and says
        so.
        """
        cfg = self._cfg
        frame = self._frame(settle=settle)
        circles = nozzle.find_circles(
            frame,
            min_radius_px=int(cfg["circle_min_radius_px"]),
            max_radius_px=int(cfg["circle_max_radius_px"]),
            param2=int(cfg["circle_param2"]),
        )
        margin = float(cfg["circle_edge_margin"])
        height, width = frame.shape
        central = [c for c in circles
                   if margin * width < c["x"] < (1 - margin) * width
                   and margin * height < c["y"] < (1 - margin) * height]
        if not central:
            raise CalibrationError(
                "no nozzle shaped circle in the middle of the frame; check the "
                "camera is aimed at the nozzle and in focus")
        rough = central[0]
        try:
            fine = nozzle.refine_centre(frame, rough)
            drift = ((fine["x"] - rough["x"]) ** 2 + (fine["y"] - rough["y"]) ** 2) ** 0.5
            if drift > rough["r"] * float(cfg["circle_max_drift"]):
                fine = dict(rough, residual=float("nan"))
        except nozzle.NozzleError:
            fine = dict(rough, residual=float("nan"))
        details = dict(radius=fine["r"], residual=fine.get("residual", float("nan")))
        return np.array([fine["x"], fine["y"]], dtype=float), 1.0, details

    def _move_absolute(self, x=None, y=None, z=None, feedrate=None):
        self._check_abort()
        parts = ["G1"]
        if x is not None:
            parts.append("X%.4f" % x)
        if y is not None:
            parts.append("Y%.4f" % y)
        if z is not None:
            parts.append("Z%.4f" % z)
        parts.append("F%d" % int(feedrate or self._cfg["feedrate"]))
        return self._bridge.run(["G90", " ".join(parts)], timeout=float(self._cfg["move_timeout"]))

    def _move_relative(self, dx=0.0, dy=0.0, feedrate=None):
        self._check_abort()
        command = "G1 X%.4f Y%.4f F%d" % (dx, dy, int(feedrate or self._cfg["fine_feedrate"]))
        return self._bridge.run(
            ["G91", command, "G90"], timeout=float(self._cfg["move_timeout"])
        )

    def _go_to_camera(self):
        """Reach the camera point through the safe Z, never across the bed low."""
        cfg = self._cfg
        self._check_abort()
        self._bridge.run(
            ["G90", "G1 Z%.3f F%d" % (float(cfg["safe_z"]), int(cfg["z_feedrate"]))],
            timeout=float(cfg["move_timeout"]),
        )
        self._move_absolute(x=float(cfg["camera_x"]), y=float(cfg["camera_y"]))
        self._bridge.run(
            ["G90", "G1 Z%.3f F%d" % (float(cfg["camera_z"]), int(cfg["z_feedrate"]))],
            timeout=float(cfg["move_timeout"]),
        )

    def _retract_to_safe_z(self):
        self._bridge.run(
            ["G90", "G1 Z%.3f F%d" % (float(self._cfg["safe_z"]), int(self._cfg["z_feedrate"]))],
            timeout=float(self._cfg["move_timeout"]),
        )

    # -- calibration steps ------------------------------------------------

    def _build_pixel_map(self):
        """Measure how many pixels the tip moves per millimetre of machine travel."""
        distance = float(self._cfg["probe_distance"])
        self._progress("map", "measuring the pixel to millimetre map")
        origin, _, _ = self._measure_tip()

        self._move_relative(dx=distance)
        after_x, _, _ = self._measure_tip()
        self._move_relative(dx=-distance)

        self._move_relative(dy=distance)
        after_y, _, _ = self._measure_tip()
        self._move_relative(dy=-distance)

        matrix = geometry.build_pixel_map(origin, after_x, after_y, distance)
        self._progress(
            "map",
            "map ready: %.1f px/mm, camera rotated %.1f degrees"
            % (geometry.pixels_per_mm(matrix), geometry.rotation_degrees(matrix)),
            px_per_mm=geometry.pixels_per_mm(matrix),
            rotation=geometry.rotation_degrees(matrix),
        )
        return matrix

    def _centre_tip(self, matrix, target_px, label):
        """Move the active nozzle until its tip sits on ``target_px``."""
        tolerance = float(self._cfg["tolerance_mm"])
        passes = int(self._cfg["max_passes"])
        position = None
        residual = None
        for index in range(passes):
            self._check_abort()
            current, confidence, _ = self._measure_tip()
            dx, dy = geometry.pixel_error_to_mm(matrix, current, target_px)
            residual = (dx * dx + dy * dy) ** 0.5
            self._progress(
                "centre",
                "%s pass %d: off by %.4f mm (confidence %.2f)"
                % (label, index + 1, residual, confidence),
                tool=label,
                passes=index + 1,
                residual_mm=residual,
            )
            if residual <= tolerance:
                break
            if residual > float(self._cfg["max_correction_mm"]):
                raise CalibrationError(
                    "%s wants a %.2f mm correction, which is beyond the %.2f mm limit"
                    % (label, residual, float(self._cfg["max_correction_mm"]))
                )
            self._move_relative(dx=dx, dy=dy)
            position = None
        position = self._bridge.position(timeout=float(self._cfg["move_timeout"]))
        if position is None:
            raise CalibrationError("the printer did not report its position")
        return np.array(position[:2], dtype=float), residual

    def _select_tool(self, tool):
        self._progress("tool", "selecting T%d" % tool)
        self._bridge.run(["T%d" % tool], timeout=float(self._cfg["move_timeout"]))
        time.sleep(float(self._cfg["tool_settle_s"]))

    # -- main -------------------------------------------------------------

    def run(self):
        try:
            self.result = self._execute()
            self._notify(dict(type="done", result=self.result))
        except Aborted as exception:
            self.error = str(exception)
            self._notify(dict(type="aborted", message=str(exception)))
        except (CalibrationError, vision.CaptureError, vision.DetectionError,
                geometry.GeometryError, Timeout, RuntimeError) as exception:
            self.error = str(exception)
            self._logger.exception("calibration failed")
            self._notify(dict(type="failed", message=str(exception)))
        except Exception as exception:  # pragma: no cover - unexpected
            self.error = str(exception)
            self._logger.exception("calibration crashed")
            self._notify(dict(type="failed", message="unexpected error: %s" % exception))

    def _execute(self):
        cfg = self._cfg
        self._check_abort()
        self._progress("preflight", "checking the camera and the printer")

        frame = vision.average_frames(
            cfg["snapshot_url"], count=2, timeout=float(cfg["http_timeout"])
        )
        mean, deviation = vision.frame_health(frame)
        self._progress(
            "preflight",
            "camera frame mean %.1f, standard deviation %.1f" % (mean, deviation),
        )

        stored = self._bridge.read_hotend_offset(tool=1)
        if stored is None:
            raise CalibrationError(
                "could not read the stored hotend offset with M218 or M503"
            )
        self._progress(
            "preflight",
            "stored T1 offset is X%.3f Y%.3f" % (stored[0], stored[1]),
            stored_offset=[stored[0], stored[1]],
        )

        if cfg["home_first"]:
            self._progress("home", "homing all axes")
            self._bridge.run(["G28"], timeout=float(cfg["home_timeout"]))

        self._select_tool(0)
        self._progress("move", "moving nozzle 0 over the camera")
        self._go_to_camera()

        matrix = self._build_pixel_map()

        target = np.array(
            [
                float(cfg["target_x_px"]) if cfg["target_x_px"] is not None else frame.shape[1] / 2.0,
                float(cfg["target_y_px"]) if cfg["target_y_px"] is not None else frame.shape[0] / 2.0,
            ],
            dtype=float,
        )

        position_0, residual_0 = self._centre_tip(matrix, target, "T0")
        self._progress(
            "centre",
            "nozzle 0 centred at X%.3f Y%.3f" % (position_0[0], position_0[1]),
        )

        self._retract_to_safe_z()
        self._select_tool(1)
        self._move_absolute(x=position_0[0], y=position_0[1])
        self._bridge.run(
            ["G90", "G1 Z%.3f F%d" % (float(cfg["camera_z"]), int(cfg["z_feedrate"]))],
            timeout=float(cfg["move_timeout"]),
        )

        position_1, residual_1 = self._centre_tip(matrix, target, "T1")
        self._progress(
            "centre",
            "nozzle 1 centred at X%.3f Y%.3f" % (position_1[0], position_1[1]),
        )

        self._retract_to_safe_z()

        correction = position_1 - position_0
        self._progress(
            "result",
            "nozzle 1 needed X%+.4f Y%+.4f more than nozzle 0"
            % (correction[0], correction[1]),
        )

        return dict(
            stored_offset=[stored[0], stored[1], stored[2]],
            position_0=[float(position_0[0]), float(position_0[1])],
            position_1=[float(position_1[0]), float(position_1[1])],
            correction=[float(correction[0]), float(correction[1])],
            residual_0_mm=residual_0,
            residual_1_mm=residual_1,
            px_per_mm=geometry.pixels_per_mm(matrix),
            rotation_deg=geometry.rotation_degrees(matrix),
            candidates=candidate_offsets(stored, correction),
        )


def candidate_offsets(stored, correction):
    """Both sign conventions for the new offset.

    Marlin's hotend offset sign is not documented for the Snapmaker toolhead, so
    the routine reports both and the apply step verifies which one is right by
    re-measuring.
    """
    return dict(
        plus=[float(stored[0] + correction[0]), float(stored[1] + correction[1])],
        minus=[float(stored[0] - correction[0]), float(stored[1] - correction[1])],
    )


def offset_command(tool, x, y):
    """Re-exported for the API layer."""
    return format_offset_command(tool, x, y)
