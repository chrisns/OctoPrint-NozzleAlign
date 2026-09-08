# coding=utf-8
"""Finding the camera on the bed, in three steps.

Nothing is assumed about where the camera is. The job splits cleanly:

1. **Sight it.** Sweep a rectangle growing out from the middle of the bed until the
   toolhead appears in the picture.
2. **Bracket it.** Walk out in X and then Y until the toolhead leaves the picture
   again. The middle of each stretch is the middle of the toolhead.
3. **Close in.** Steer on the position of the moving patch in the frame, because the
   fraction that moves is flat across X once the toolhead fills the field.

Each step is one method here. They used to be one function with five nested
functions inside it, and nobody could read it.

The `machine` argument is the small set of moves and pictures the search needs. The
calibration routine supplies it, and the tests supply a fake, so every step can be
exercised without a printer.
"""
import time

import numpy as np

from . import vision


class SearchError(Exception):
    """The camera could not be found."""


class Sighting(object):
    """Where the toolhead was first seen, and which way the sweep was going."""

    def __init__(self, x, y, direction):
        self.x = float(x)
        self.y = float(y)
        self.direction = int(direction)


class BedSearch(object):
    """Finds the middle of the toolhead over the camera."""

    def __init__(self, machine, config, progress):
        self._machine = machine
        self._cfg = config
        self._progress = progress
        self.x_min = float(config["bed_x_min"])
        self.x_max = float(config["bed_x_max"])
        self.y_min = float(config["bed_y_min"])
        self.y_max = float(config["bed_y_max"])
        self.feed = int(config["sweep_feedrate"])
        self.threshold = float(config["motion_threshold"])
        self.minimum = float(config["motion_min_blob"])
        self.step = float(config["sweep_step_mm"])
        self.row_gap = float(config["bed_row_mm"])
        self.settle = float(config["settle_s"])

    # -- the whole job ----------------------------------------------------

    def run(self):
        """Find the camera and leave the toolhead over it. Returns the ring span."""
        self._machine.move_z(float(self._cfg["search_z"]))
        self._say("searching for the camera from the middle of the bed outwards at Z%.0f"
                  % float(self._cfg["search_z"]))
        sighting = self.sight()
        if sighting is None:
            raise SearchError(
                "the toolhead never came into view over the bed; is the camera "
                "plugged in and pointing up?")
        x_mid, y_mid = self.bracket(sighting)
        centre = self.close_in(x_mid, y_mid)
        self._say("the middle of the toolhead is at X%.0f Y%.0f" % centre)
        # T0's active nozzle sits half an offset towards minus X from the middle, and
        # T1's raised one the other way. Starting on T0's side puts the active nozzle
        # first in line.
        self._machine.arrive(centre[0] - float(self._cfg["nominal_offset_x"]) / 2.0,
                             centre[1])
        self._machine.move_z(float(self._cfg["camera_z"]))
        time.sleep(self.settle)
        self._machine.forget()
        return float(self._cfg["bed_search_span_mm"])

    # -- step 1: sight the toolhead ---------------------------------------

    def sight(self):
        """Grow a rectangle out from the middle of the bed. Returns a Sighting.

        Frames can only be compared along X within one row. A Y move carries the
        camera with the bed and changes the whole picture, so each ring scans its two
        new rows along X, then extends every older row by one column each side.
        """
        centre_x = (self.x_min + self.x_max) / 2.0
        centre_y = (self.y_min + self.y_max) / 2.0
        edges = {}
        ring = 0
        while True:
            columns = self._columns(centre_x, ring)
            rows = self._rows(centre_y, ring)
            if ring > 0 and self._covered(centre_x, centre_y, ring):
                return None

            new_rows, old_rows, new_columns = self._ring_parts(
                centre_x, centre_y, ring, columns, rows)

            found = self._scan_rows(new_rows, columns, edges)
            if found is not None:
                return found
            if ring > 0:
                found = self._scan_columns(new_columns, old_rows, centre_x, edges)
                if found is not None:
                    return found
                self._trim(edges, columns)
            ring += 1

    def _columns(self, centre_x, ring):
        return [x for x in (centre_x + i * self.step for i in range(-ring, ring + 1))
                if self.x_min - 1e-6 <= x <= self.x_max + 1e-6]

    def _rows(self, centre_y, ring):
        return [y for y in (centre_y + j * self.row_gap for j in range(-ring, ring + 1))
                if self.y_min - 1e-6 <= y <= self.y_max + 1e-6]

    def _covered(self, centre_x, centre_y, ring):
        """True once the rectangle has grown past every edge of the search area."""
        half_x, half_y = ring * self.step, ring * self.row_gap
        return (centre_x - half_x < self.x_min - 1e-6
                and centre_x + half_x > self.x_max + 1e-6
                and centre_y - half_y < self.y_min - 1e-6
                and centre_y + half_y > self.y_max + 1e-6)

    def _ring_parts(self, centre_x, centre_y, ring, columns, rows):
        """Which rows are new this ring, which are old, and which columns are new."""
        if ring == 0:
            return [centre_y], [], []
        half_x, half_y = ring * self.step, ring * self.row_gap
        new_rows = [y for y in (centre_y - half_y, centre_y + half_y) if y in rows]
        old_rows = [y for y in rows if y not in new_rows]
        new_columns = [x for x in (centre_x - half_x, centre_x + half_x) if x in columns]
        return list(dict.fromkeys(new_rows)), old_rows, list(dict.fromkeys(new_columns))

    def _scan_rows(self, new_rows, columns, edges):
        """Sweep each new row along X in one continuous move."""
        for index, y in enumerate(new_rows):
            order = columns if index % 2 == 0 else columns[::-1]
            if len(order) <= 1:
                self._machine.move_absolute(x=order[0], y=y, feedrate=self.feed)
                edges[(order[0], y)] = self._machine.frame_quick()
                continue
            hit = self.sweep_row(y, order[0], order[-1])
            if hit is not None:
                found = self._confirm_near(hit, y, columns,
                                           1 if order is columns else -1)
                if found is not None:
                    return found
            for x in (columns[0], columns[-1]):
                self._machine.move_absolute(x=x, y=y, feedrate=self.feed)
                edges[(x, y)] = self._machine.frame_quick()
        return None

    def _confirm_near(self, hit, y, columns, direction):
        """A sweep places the toolhead to about one step. Check the three nearest."""
        nearest = min(columns, key=lambda x: abs(x - hit))
        for x in (nearest, nearest - self.step, nearest + self.step):
            if not self.x_min - 1e-6 <= x <= self.x_max + 1e-6:
                continue
            if self.confirm(x, y):
                return Sighting(x, y, direction)
        return None

    def _scan_columns(self, new_columns, old_rows, centre_x, edges):
        """Extend the older rows by one column each side."""
        for x in new_columns:
            inner = x + self.step if x < centre_x else x - self.step
            direction = -1 if x < centre_x else 1
            for y in old_rows:
                self._machine.move_absolute(x=x, y=y, feedrate=self.feed)
                frame = self._machine.frame_quick()
                compare = edges.get((inner, y))
                edges[(x, y)] = frame
                changed = (compare is not None
                           and vision.motion_blob(compare, frame, self.threshold)
                           >= self.minimum)
                if changed and self.confirm(x, y):
                    return Sighting(x, y, direction)
        return None

    @staticmethod
    def _trim(edges, columns):
        """Keep only the outermost columns. The inner ones are never compared again."""
        for key in list(edges):
            if key[0] not in (columns[0], columns[-1]):
                del edges[key]

    def sweep_row(self, y, x_from, x_to):
        """Move along one row in a single go, watching for the toolhead.

        Frames are taken while the head moves and each is compared with the one
        before. The first pair that differs by one big patch of texture puts the
        toolhead in view. Where the head was then comes from the clock and the
        feedrate, which is good to about a step, and the nudge that follows confirms
        it.
        """
        self._machine.check_abort()
        self._machine.move_absolute(x=x_from, y=y, feedrate=self.feed)
        time.sleep(self.settle)
        self._machine.frame_quick()
        previous = self._machine.frame_now()

        speed = float(self._cfg["sweep_feedrate"]) / 60.0
        distance = abs(x_to - x_from)
        direction = 1.0 if x_to > x_from else -1.0
        lag = float(self._cfg["sweep_lag_s"])
        self._machine.send(["G90", "G1 X%.4f F%d" % (x_to, self.feed)])

        started = time.time()
        deadline = distance / speed + float(self._cfg["sweep_overrun_s"])
        hit = None
        while time.time() - started < deadline:
            self._machine.check_abort()
            frame = self._machine.frame_now()
            elapsed = time.time() - started
            if hit is None and vision.motion_blob(previous, frame,
                                                  self.threshold) >= self.minimum:
                travelled = min(distance, speed * max(0.0, elapsed - lag))
                hit = x_from + direction * travelled
            previous = frame
        self._machine.wait()
        return hit

    def confirm(self, x, y):
        """Nudge at a place and say whether one big patch of texture moved.

        The toolhead is one patch. The cable chain and the flicker of the light are
        scattered pixels, and are rejected.
        """
        self._say("something changed at X%.0f Y%.0f; checking" % (x, y), x=x, y=y)
        return self.nudge_at(x, y) >= self.minimum

    def nudge_at(self, x, y):
        """Move somewhere, nudge, and return the fraction of texture that moved."""
        x = min(max(x, self.x_min), self.x_max)
        y = min(max(y, self.y_min), self.y_max)
        self._machine.check_abort()
        self._machine.move_absolute(x=x, y=y, feedrate=self.feed)
        time.sleep(self.settle)
        fraction = self._machine.moved_fraction_after_nudge()
        self._say("X%.0f Y%.0f: %.0f%% of the texture moved" % (x, y, fraction * 100),
                  x=x, y=y, moved=fraction)
        return fraction

    # -- step 2: bracket the toolhead -------------------------------------

    def bracket(self, sighting):
        """Walk out until the toolhead leaves the picture. Returns its middle."""
        first_x, last_x = self._edges_along_x(sighting)
        x_mid = (first_x + last_x) / 2.0
        first_y, last_y = self._edges_along_y(x_mid, sighting.y)
        y_mid = (first_y + last_y) / 2.0
        self._say("the toolhead is over the camera near X%.0f Y%.0f" % (x_mid, y_mid))
        return x_mid, y_mid

    def _edges_along_x(self, sighting):
        forward = self._walk(sighting.x, sighting.y, sighting.direction * self.step,
                             axis="x")
        backward = self._walk(sighting.x, sighting.y, -sighting.direction * self.step,
                              axis="x")
        return min(forward, backward), max(forward, backward)

    def _edges_along_y(self, x_mid, y):
        forward = self._walk(x_mid, y, self.step, axis="y")
        backward = self._walk(x_mid, y, -self.step, axis="y")
        return min(forward, backward), max(forward, backward)

    def _walk(self, x, y, delta, axis):
        """Step until the toolhead stops moving the picture. Returns the last hit."""
        last = x if axis == "x" else y
        probe = last + delta
        while self._inside(probe, axis):
            here = (probe, y) if axis == "x" else (x, probe)
            if self.nudge_at(*here) < self.minimum:
                break
            last = probe
            probe += delta
        return last

    def _inside(self, value, axis):
        if axis == "x":
            return self.x_min <= value <= self.x_max
        return self.y_min <= value <= self.y_max

    # -- step 3: close in -------------------------------------------------

    def close_in(self, x_mid, y_mid):
        """Steer on where the moving patch sits in the frame.

        The fraction that moves is flat and noisy across X, because the toolhead is
        nearly as wide as the field, so it cannot place the toolhead. The patch's
        position in the picture can. A nudge gives the picture's direction and scale
        for machine X, and the patch's offset from the centre along that direction is
        how far to move.
        """
        self._machine.move_absolute(x=x_mid, y=y_mid, feedrate=self.feed)
        height = float(self._cfg["closein_z"])
        if abs(height - float(self._cfg["search_z"])) > 1e-6:
            self._machine.move_z(height)

        x_here = x_mid
        for _ in range(int(self._cfg["closein_max_steps"])):
            self._machine.check_abort()
            offset, scale, fraction = self._read_offset(x_here, y_mid)
            if offset is None:
                self._say("cannot read the picture's X direction (%.1f px/mm); "
                          "keeping X%.0f" % (scale, x_here))
                break
            self._say("the toolhead is %.1f mm along X from the lens "
                      "(%.1f px/mm, %.0f%% in view)" % (offset, scale, fraction * 100),
                      offset=offset, px_per_mm=scale)
            if abs(offset) < float(self._cfg["closein_done_mm"]):
                break
            cap = float(self._cfg["closein_max_move_mm"])
            x_here = min(max(x_here + max(-cap, min(cap, -offset)), self.x_min),
                         self.x_max)
            self._machine.move_absolute(x=x_here, feedrate=self.feed)
        return x_here, y_mid

    def _read_offset(self, x_here, y_mid):
        """One nudge. Returns (offset_mm, px_per_mm, fraction_in_view).

        The offset is None when the picture cannot be read well enough to steer by.
        """
        nudge = float(self._cfg["motion_nudge_mm"])
        time.sleep(self.settle)
        before = self._machine.frame_quick()
        self._machine.move_relative(dx=nudge)
        time.sleep(self.settle)
        after = self._machine.frame_quick()
        self._machine.move_relative(dx=-nudge)

        centroid, box, fraction = vision.moving_region(before, after, self.threshold)
        if centroid is None or fraction < self.minimum:
            raise SearchError("the toolhead was seen at X%.0f Y%.0f and then lost"
                              % (x_here, y_mid))

        shift, response = vision.shift_in_box(before, after, box)
        length = float(np.hypot(*shift))
        scale = length / nudge
        if (scale < float(self._cfg["closein_min_scale"])
                or response < float(self._cfg["closein_min_response"])):
            return None, scale, fraction

        unit = np.array(shift) / length
        centre_px = np.array([before.shape[1] / 2.0, before.shape[0] / 2.0])
        offset = float((np.array(centroid) - centre_px) @ unit) / scale
        return offset, scale, fraction

    def _say(self, message, **extra):
        self._progress("bed", message, **extra)
