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
        """Measure the pixel map and the nozzle position at the current height."""
        origin_x, moved_x, details = self._motion_probe(dx=distance)
        origin_y, moved_y, _ = self._motion_probe(dy=distance)
        origin = np.array(
            [
                (origin_x[0] + origin_y[0]) / 2.0,
                (origin_x[1] + origin_y[1]) / 2.0,
            ]
        )
        matrix = geometry.build_pixel_map(origin, moved_x, moved_y, distance)
        return matrix, origin, details

    # -- search -----------------------------------------------------------

    def _search_for_the_nozzle(self, distance):
        """Try the bed centre, then a widening grid, until the nozzle shows up."""
        cfg = self._cfg
        span = float(cfg["search_span_mm"])
        points = int(cfg["search_points"])
        centre = (float(cfg["search_centre_x"]), float(cfg["search_centre_y"]))

        offsets = [(0.0, 0.0)]
        if points > 1:
            step = span / float(points - 1)
            grid = [-span / 2.0 + step * i for i in range(points)]
            offsets += [(x, y) for y in grid for x in grid if (x, y) != (0.0, 0.0)]

        last_error = None
        for index, (dx, dy) in enumerate(offsets):
            self._check_abort()
            x, y = centre[0] + dx, centre[1] + dy
            self._progress(
                "search",
                "looking for the nozzle at X%.1f Y%.1f (%d of %d)"
                % (x, y, index + 1, len(offsets)),
            )
            self._move_absolute(x=x, y=y)
            try:
                matrix, origin, details = self._pixel_map_here(distance)
            except (vision.DetectionError, geometry.GeometryError) as exception:
                last_error = str(exception)
                continue
            self._progress(
                "search",
                "found the nozzle at %.0f, %.0f px, %.1f px/mm"
                % (origin[0], origin[1], geometry.pixels_per_mm(matrix)),
            )
            return matrix, origin
        raise CalibrationError(
            "could not see the nozzle move anywhere in the search area (%s); "
            "check that the camera points up at the toolhead" % last_error
        )

    def _centre_here(self, matrix, target, distance, label):
        """Put the nozzle on the target pixel using motion based location."""
        tolerance = float(self._cfg["discovery_tolerance_mm"])
        for index in range(int(self._cfg["max_passes"])):
            self._check_abort()
            origin, _, _ = self._motion_probe(dx=distance)
            dx, dy = geometry.pixel_error_to_mm(matrix, origin, target)
            error = (dx * dx + dy * dy) ** 0.5
            self._progress(
                "centre", "%s pass %d: off by %.3f mm" % (label, index + 1, error)
            )
            if error <= tolerance:
                return
            if error > float(self._cfg["max_correction_mm"]) * 20:
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
                    matrix, origin, details = self._pixel_map_here(distance)
                except (vision.DetectionError, geometry.GeometryError) as exception:
                    self._progress(
                        "descend", "lost the nozzle at Z%.2f: %s" % (z, exception)
                    )
                    break

                frame_area = float(np.prod(self._frame().shape))
                if details["area_a"] / frame_area > area_limit:
                    self._progress(
                        "descend",
                        "the nozzle fills %.0f%% of the frame at Z%.2f; stopping"
                        % (100.0 * details["area_a"] / frame_area, z),
                    )
                    break

                self._centre_here(matrix, target, distance, "Z%.1f" % z)
                frame = self._frame()
                origin, _, _ = self._motion_probe(dx=distance)
                focus = vision.sharpness(frame, origin, int(cfg["focus_window_px"]))
                scale = geometry.pixels_per_mm(matrix)
                samples.append((z, scale))
                position = self._bridge.position(timeout=float(cfg["move_timeout"]))
                self._progress(
                    "descend",
                    "Z%.2f: %.1f px/mm, focus %.0f" % (z, scale, focus),
                    z=z, px_per_mm=scale, sharpness=focus,
                )
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
            raise CalibrationError("never found a focused view of the nozzle")
        lens_z = estimate_lens_z(samples)
        best["lens_z"] = lens_z if lens_z is not None else float("nan")
        return best


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
