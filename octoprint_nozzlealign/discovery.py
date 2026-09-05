# coding=utf-8
"""Finds the camera on the bed without being told where it is.

Nothing here needs a coordinate from the operator.  The nozzle is the only
moving object the camera can see, so the routine wobbles the toolhead and
watches which pixels change.  That gives the nozzle position in the image and,
from a known move, the pixel to millimetre map.  Driving the nozzle to the
image centre then gives the camera position in machine coordinates.

The descent to the focal plane is guarded three ways: a measured estimate of
where the lens actually is, a hard floor from the settings, and a limit on how
much of the frame the nozzle may fill.
"""

from __future__ import absolute_import

import numpy as np

from . import geometry, vision
from .routine import CalibrationError, CalibrationRoutine


class DiscoveryRoutine(CalibrationRoutine):
    """Locates the camera and stores the result on the plugin."""

    def __init__(self, bridge, settings, notify, logger, on_result=None):
        super(DiscoveryRoutine, self).__init__(bridge, settings, notify, logger)
        self.name = "nozzlealign-discovery"
        self._on_result = on_result

    # -- measurement ------------------------------------------------------

    def _pixel_map_here(self, distance):
        """Measure the pixel map and where the toolhead is at this height.

        Far from the camera the whole toolhead fills the frame and no single
        nozzle blob exists.  The map is still correct there, because it comes
        from how far the picture shifted, so the search can steer on it.  Only
        once ``compact`` is true does the position refer to the nozzle itself.
        """
        probe_x = self._motion_probe(dx=distance)
        probe_y = self._motion_probe(dy=distance)
        matrix = geometry.build_pixel_map_from_shifts(
            probe_x["shift"], probe_y["shift"], distance
        )
        position = np.array(probe_x["position"], dtype=float)
        compact = bool(probe_x["compact"] and probe_y["compact"])
        return matrix, position, compact, probe_x["details"]

    # -- search -----------------------------------------------------------

    def _candidate_points(self):
        """Where to stand while looking for the camera.

        The middle of the bed comes first, because the camera's field of view is
        wide enough that the toolhead is usually already in it.  After that the
        bed is covered in a serpentine sweep, spaced closely enough that the
        camera cannot fall between two points.
        """
        cfg = self._cfg
        x0, x1 = float(cfg["search_x_min"]), float(cfg["search_x_max"])
        y0, y1 = float(cfg["search_y_min"]), float(cfg["search_y_max"])
        first = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
        columns = _span(x0, x1, max(5.0, float(cfg["raster_spacing_x"])))
        rows = _span(y0, y1, max(5.0, float(cfg["raster_spacing_y"])))
        points = [first]
        for index, y in enumerate(rows):
            ordered = columns if index % 2 == 0 else list(reversed(columns))
            points += [(x, y) for x in ordered]
        return points

    def _search_for_the_nozzle(self, distance):
        """Stand at each candidate point until the camera can see the toolhead.

        One probe move is enough to answer "can the camera see anything move
        here", and it is the only test that actually settles it.  Comparing
        still frames looked cheaper, but it does not work: this camera sees the
        toolhead from most of the bed, so there is no empty view to compare a
        frame against.
        """
        points = self._candidate_points()
        last_error = None
        for index, (x, y) in enumerate(points):
            self._check_abort()
            self._progress(
                "search",
                "looking from X%.0f Y%.0f (%d of %d)" % (x, y, index + 1, len(points)),
            )
            self._move_absolute(x=x, y=y)
            try:
                self._motion_probe(dx=distance)
            except vision.DetectionError as exception:
                last_error = str(exception)
                continue
            try:
                matrix, origin, compact, _ = self._pixel_map_here(distance)
            except (vision.DetectionError, geometry.GeometryError) as exception:
                last_error = str(exception)
                continue
            self._progress(
                "search",
                "saw the toolhead at %.0f, %.0f px, %.1f px/mm%s"
                % (
                    origin[0], origin[1], geometry.pixels_per_mm(matrix),
                    "" if compact else " (whole toolhead, not yet the nozzle)",
                ),
            )
            return matrix, origin
        raise CalibrationError(
            "could not see the nozzle: nothing moved in the picture from "
            "anywhere on the bed (%s). Check that the camera is plugged in, "
            "points up, and that its stream is live" % last_error
        )

    def _centre_here(self, matrix, target, distance, label):
        """Put the nozzle on the target pixel using motion based location."""
        tolerance = float(self._cfg["discovery_tolerance_mm"])
        for index in range(int(self._cfg["max_passes"])):
            self._check_abort()
            origin = self._motion_probe(dx=distance)["position"]
            dx, dy = geometry.pixel_error_to_mm(matrix, origin, target)
            error = (dx * dx + dy * dy) ** 0.5
            self._progress(
                "centre", "%s pass %d: off by %.3f mm" % (label, index + 1, error)
            )
            if error <= tolerance:
                return
            if error > float(self._cfg["coarse_max_correction_mm"]):
                raise CalibrationError(
                    "%s wants a %.1f mm correction, which looks wrong" % (label, error)
                )
            self._move_relative(dx=dx, dy=dy)

    # -- main -------------------------------------------------------------

    def _execute(self):
        cfg = self._cfg
        self._check_abort()
        self._progress("preflight", "checking the camera")
        frame = self._frame(settle=2)
        mean, deviation = vision.frame_health(frame)
        self._progress(
            "preflight", "camera mean %.1f, standard deviation %.1f" % (mean, deviation)
        )
        target = np.array([frame.shape[1] / 2.0, frame.shape[0] / 2.0])

        if cfg["home_first"]:
            self._progress("home", "homing all axes")
            self._bridge.run(["G28"], timeout=float(cfg["home_timeout"]))

        search_z = float(cfg["search_z"])
        distance = float(cfg["search_probe_mm"])
        self._progress("move", "rising to Z%.1f before travelling" % search_z)
        self._bridge.run(
            ["G90", "G1 Z%.3f F%d" % (search_z, int(cfg["z_feedrate"]))],
            timeout=float(cfg["move_timeout"]),
        )

        matrix, _ = self._search_for_the_nozzle(distance)
        self._centre_here(matrix, target, distance, "search height")

        samples = [(search_z, geometry.pixels_per_mm(matrix))]
        position = self._bridge.position(timeout=float(cfg["move_timeout"]))
        camera_xy = np.array(position[:2], dtype=float)
        self._progress(
            "camera",
            "camera is under X%.3f Y%.3f" % (camera_xy[0], camera_xy[1]),
        )

        best = self._descend(target, samples)

        self._progress(
            "done",
            "camera at X%.3f Y%.3f, focus at Z%.3f, turned %.1f degrees from the "
            "machine axes, lens plane near Z%.2f"
            % (best["x"], best["y"], best["z"], best["rotation"], best["lens_z"]),
        )
        result = dict(
            camera_x=float(best["x"]),
            camera_y=float(best["y"]),
            camera_z=float(best["z"]),
            lens_z=float(best["lens_z"]),
            px_per_mm=float(best["px_per_mm"]),
            rotation_deg=float(best["rotation"]),
            matrix=best["matrix"],
            sharpness=float(best["sharpness"]),
            samples=[[float(z), float(s)] for z, s in samples],
        )
        if self._on_result is not None:
            self._on_result(result, best.get("template"))
        return result

    def _descend(self, target, samples):
        """Step down towards the lens, re-centring and watching the focus."""
        cfg = self._cfg
        distance = float(cfg["search_probe_mm"])
        floor = float(cfg["min_z"])
        clearance = float(cfg["lens_clearance_mm"])
        area_limit = float(cfg["max_blob_fraction"])

        z = float(cfg["search_z"])
        best = None
        for step in (float(cfg["coarse_step"]), float(cfg["fine_step"])):
            while True:
                self._check_abort()
                lens_z = estimate_lens_z(samples)
                limit = floor
                if lens_z is not None:
                    limit = max(floor, lens_z + clearance)
                next_z = z - step
                if next_z < limit:
                    self._progress(
                        "descend",
                        "stopping at Z%.2f; the floor is Z%.2f" % (z, limit),
                    )
                    break

                self._bridge.run(
                    ["G90", "G1 Z%.3f F%d" % (next_z, int(cfg["z_feedrate"]))],
                    timeout=float(cfg["move_timeout"]),
                )
                z = next_z
                try:
                    matrix, origin, compact, details = self._pixel_map_here(distance)
                except (vision.DetectionError, geometry.GeometryError) as exception:
                    self._progress(
                        "descend", "lost the toolhead at Z%.2f: %s" % (z, exception)
                    )
                    break

                frame = self._frame()
                frame_area = float(frame.shape[0] * frame.shape[1])
                if compact and details["area"] / frame_area > area_limit:
                    self._progress(
                        "descend",
                        "the nozzle fills %.0f%% of the frame at Z%.2f; stopping"
                        % (100.0 * details["area"] / frame_area, z),
                    )
                    break

                try:
                    self._centre_here(matrix, target, distance, "Z%.1f" % z)
                except CalibrationError as exception:
                    # A nonsense correction here means the measurement is not
                    # tracking the nozzle. Stop descending rather than abort, so
                    # the caller still hears why the run produced nothing.
                    self._progress(
                        "descend", "stopping the descent at Z%.2f: %s" % (z, exception)
                    )
                    break
                frame = self._frame()
                probe = self._motion_probe(dx=distance)
                origin = probe["position"]
                compact = compact and probe["compact"]
                focus = vision.sharpness(frame, origin, int(cfg["focus_window_px"]))
                scale = geometry.pixels_per_mm(matrix)
                samples.append((z, scale))
                position = self._bridge.position(timeout=float(cfg["move_timeout"]))
                self._progress(
                    "descend",
                    "Z%.2f: %.1f px/mm, focus %.0f%s"
                    % (z, scale, focus, "" if compact else " (toolhead, not the nozzle)"),
                    z=z, px_per_mm=scale, sharpness=focus, compact=compact,
                )
                if not compact:
                    continue
                if best is None or focus > best["sharpness"]:
                    best = dict(
                        x=position[0], y=position[1], z=z,
                        sharpness=focus, px_per_mm=scale,
                        rotation=geometry.rotation_degrees(matrix),
                        matrix=[list(row) for row in matrix],
                        template=vision.cut_template(
                            frame, origin, int(cfg["template_size_px"])
                        ),
                    )
                elif focus < best["sharpness"] * float(cfg["focus_drop_ratio"]):
                    self._progress(
                        "descend", "focus is falling away; the best was Z%.2f" % best["z"]
                    )
                    break
            if best is not None:
                # start the fine pass just above the best height
                z = min(z + step, float(cfg["search_z"]))

        if best is None:
            raise CalibrationError(
                "the nozzle never separated from the rest of the toolhead before "
                "the descent had to stop; lower the minimum Z, reduce the "
                "clearance above the lens, or check that the camera is aimed at "
                "the nozzle and in focus"
            )
        lens_z = estimate_lens_z(samples)
        best["lens_z"] = lens_z if lens_z is not None else float("nan")
        return best


def _span(low, high, step):
    """Points from low to high inclusive, at most ``step`` apart."""
    if high <= low:
        return [low]
    count = int(np.ceil((high - low) / step))
    return [low + (high - low) * i / count for i in range(count + 1)]


def estimate_lens_z(samples):
    """Estimate the Z of the lens plane from how the image scale grows.

    The image scale of an object is inversely proportional to its distance from
    the lens, so ``1 / scale`` falls linearly as Z falls.  Where that line
    crosses zero is the lens.  Two heights are enough, and more improve it.
    """
    usable = [(z, s) for z, s in samples if s > 0]
    if len(usable) < 2:
        return None
    heights = np.array([z for z, _ in usable], dtype=float)
    inverse = np.array([1.0 / s for _, s in usable], dtype=float)
    if np.ptp(heights) < 1e-6:
        return None
    slope, intercept = np.polyfit(heights, inverse, 1)
    if abs(slope) < 1e-9:
        return None
    return float(-intercept / slope)


class FullCalibration(DiscoveryRoutine):
    """Find the camera, then measure the offset, in one go.

    Nothing about the camera is remembered between runs.  The position, the
    focus height, the pixel scale and the camera rotation are all measured
    afresh every time, so the camera can be put anywhere on the bed and moved
    whenever you like.  A stored position would be worse than useless here: the
    routine lowers the nozzle onto the camera, and a stale Z would drive it into
    a camera that is no longer there.
    """

    def __init__(self, bridge, settings, notify, logger, on_result=None):
        super(FullCalibration, self).__init__(
            bridge, settings, notify, logger, on_result=on_result
        )
        self.name = "nozzlealign-calibration"

    def _execute(self):
        found = DiscoveryRoutine._execute(self)
        self._cfg = dict(self._cfg)
        self._cfg.update(
            camera_x=found["camera_x"],
            camera_y=found["camera_y"],
            camera_z=found["camera_z"],
            home_first=False,
        )
        self._progress(
            "measure",
            "camera found; measuring the offset between the two nozzles",
        )
        result = CalibrationRoutine._execute(self)
        result["camera"] = found
        return result
