# coding=utf-8
"""A simulated printer and camera, so the closed loop can be tested offline.

The printer follows the firmware's real tool change: selecting T1 shifts the
head by minus the stored offset and leaves the logical position alone, so
T1's nozzle lands where T0's was only if the stored offset equals the true
separation. Each axis has backlash. The camera renders both nozzles wherever
they are: the active one at the commanded height, the other one raised by the
lift mechanism and shifted a little sideways by it, both blurred by how far
they sit from the focal plane.
"""

import math
import re

import numpy as np

MOVE_RE = re.compile(r"G1(?:\s+X(-?\d+\.?\d*))?(?:\s+Y(-?\d+\.?\d*))?(?:\s+Z(-?\d+\.?\d*))?")


def rotation_matrix(scale, degrees, flip_y=True):
    angle = math.radians(degrees)
    matrix = scale * np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )
    if flip_y:
        matrix = np.array([[1.0, 0.0], [0.0, -1.0]]).dot(matrix)
    return matrix


class FakeBridge(object):
    """Enough of GcodeBridge to drive the routine, with real tool semantics."""

    def __init__(self, stored_offset=(25.20, 0.32, -0.891), true_offset=(25.56, 0.64),
                 backlash_mm=0.12, lift_shift=(0.13, -0.02), fail_after=None):
        self.logical = [0.0, 0.0, 100.0]
        self.physical = [0.0, 0.0, 100.0]      # the logical axis after backlash
        self.last_direction = [0, 0, 0]
        self.tool = 0
        self.relative = False
        self.stored_offset = list(stored_offset)
        self.true_offset = np.asarray(true_offset, dtype=float)
        self.backlash_mm = float(backlash_mm)
        self.lift_shift = np.asarray(lift_shift, dtype=float)
        self.fail_after = fail_after
        self.sent = []
        self.interrupted = False
        self.flight = None      # (started, duration, start_xy, end_xy) of a move in progress

    # -- simulation -------------------------------------------------------

    def head_xy(self):
        """Where T0's nozzle is, in the machine frame."""
        self._advance()
        head = np.array(self.physical[:2], dtype=float)
        if self.tool == 1:
            head = head - np.array(self.stored_offset[:2], dtype=float)
        return head

    def nozzle_xy(self, tool):
        """Where a nozzle is, whether or not it is the active one."""
        head = self.head_xy()
        if tool == 1:
            head = head + self.true_offset
        if tool != self.tool:
            head = head + self.lift_shift
        return head

    def physical_xy(self):
        return self.nozzle_xy(self.tool)

    def z(self):
        return self.physical[2]

    def _move_axis(self, index, target):
        delta = target - self.logical[index]
        if delta == 0:
            return
        direction = 1 if delta > 0 else -1
        lost = 0.0
        if index < 2 and self.last_direction[index] not in (0, direction):
            lost = min(abs(delta), self.backlash_mm)
        self.physical[index] += delta - direction * lost
        self.logical[index] = target
        self.last_direction[index] = direction

    def _apply(self, command):
        self.sent.append(command)
        if self.fail_after is not None and len(self.sent) > self.fail_after:
            raise RuntimeError("printer is not connected")
        if command.startswith("G90"):
            self.relative = False
            return
        if command.startswith("G91"):
            self.relative = True
            return
        if command.startswith("G28"):
            self.logical = [0.0, 0.0, 100.0]
            self.physical = [0.0, 0.0, 100.0]
            self.last_direction = [0, 0, 0]
            return
        if command.startswith("T"):
            self.tool = int(command[1:].split()[0])
            return
        match = MOVE_RE.match(command)
        if command.startswith("G1") and match:
            for index, group in enumerate(match.groups()):
                if group is None:
                    continue
                value = float(group)
                target = self.logical[index] + value if self.relative else value
                self._move_axis(index, target)

    def _advance(self):
        """Let a move sent with ``send`` progress with the clock."""
        import time

        if self.flight is None:
            return
        started, duration, start, end = self.flight
        done = min(1.0, (time.time() - started) / duration) if duration > 0 else 1.0
        self.physical[0] = start[0] + (end[0] - start[0]) * done
        self.physical[1] = start[1] + (end[1] - start[1]) * done
        if done >= 1.0:
            self.flight = None

    # -- GcodeBridge surface ---------------------------------------------

    def send(self, commands):
        """A move that takes real time, so a watcher can see it happen."""
        import time

        for command in commands:
            match = MOVE_RE.match(command)
            feed = re.search(r"F(\d+)", command)
            if command.startswith("G1") and match and feed and not self.relative:
                start = list(self.physical[:2])
                self._apply(command)
                end = list(self.logical[:2])
                distance = float(np.hypot(end[0] - start[0], end[1] - start[1]))
                duration = distance / (float(feed.group(1)) / 60.0)
                self.physical[0], self.physical[1] = start
                self.flight = (time.time(), duration, start, end)
            else:
                self._apply(command)

    def run(self, commands, timeout=60.0):
        if self.flight is not None:
            started, duration, start, end = self.flight
            self.physical[0], self.physical[1] = end
            self.flight = None
        for command in commands:
            self._apply(command)
        return tuple(self.logical)

    def position(self, timeout=30.0):
        return tuple(self.logical)

    def query(self, command, timeout=30.0):
        return []

    def read_hotend_offset(self, tool=1, timeout=30.0):
        return tuple(self.stored_offset)

    def interrupt(self):
        self.interrupted = True

    # -- for the tests ----------------------------------------------------

    def z_commands(self):
        values = []
        for command in self.sent:
            match = MOVE_RE.match(command)
            if command.startswith("G1") and match and match.group(3) is not None:
                values.append(float(match.group(3)))
        return values


class BoreCamera(object):
    """An upward camera that renders both nozzle bores.

    The scale of the picture is inversely proportional to the distance from
    the lens, the picture is sharpest for a nozzle at ``focus_z`` above the
    lens, and the bore is drawn as a bright collar with a dark ring and a lit
    centre, which is what the real one looks like once the lens is focused.
    """

    def __init__(self, machine, camera_xy, lens_z=12.0, focus_z=32.0, lift_mm=2.4,
                 rotation=6.0, scale_constant=1500.0, size=(400, 640), blur_per_mm=1.6,
                 seed=3):
        self.machine = machine
        self.camera_xy = np.asarray(camera_xy, dtype=float)
        self.lens_z = float(lens_z)
        self.focus_z = float(focus_z)
        self.lift_mm = float(lift_mm)
        self.rotation = rotation
        self.scale_constant = float(scale_constant)
        self.size = size
        self.blur_per_mm = blur_per_mm
        self.centre_px = np.array([size[1] / 2.0, size[0] / 2.0])
        self.rng = np.random.default_rng(seed)
        # the underside of the toolhead: a textured square above the nozzles
        self.body_mm = 60.0
        self.body_height = 6.0
        self.texture = np.random.default_rng(seed + 1).normal(0.0, 1.0, (256, 256)).astype(np.float32)

    def scale(self, height):
        return self.scale_constant / max(1.0, height - self.lens_z)

    def matrix(self, height):
        return rotation_matrix(self.scale(height), self.rotation)

    def pixel(self, xy, height):
        return self.centre_px + self.matrix(height).dot(np.asarray(xy, dtype=float) - self.camera_xy)

    def _draw_body(self, image, xs, ys):
        """The toolhead underside, which moves with X and stays put with Y."""
        import cv2

        body_z = self.machine.z() + self.body_height
        centre = self.machine.head_xy() + self.machine.true_offset / 2.0
        inverse = np.linalg.inv(self.matrix(body_z))
        px = np.stack([xs - self.centre_px[0], ys - self.centre_px[1]], axis=-1).astype(np.float32)
        machine = px.reshape(-1, 2).dot(inverse.T).reshape(px.shape) + self.camera_xy - centre
        inside = (np.abs(machine[..., 0]) <= self.body_mm / 2) & (np.abs(machine[..., 1]) <= self.body_mm / 2)
        if not inside.any():
            return image
        u = np.floor(machine[..., 0] * 1.5).astype(int) % 256
        v = np.floor(machine[..., 1] * 1.5).astype(int) % 256
        layer = 130.0 + 16.0 * self.texture[v, u]
        defocus = abs(body_z - self.focus_z) * self.blur_per_mm
        # the real body keeps visible structure even far from focus, because
        # its features are millimetres across; cap the blur accordingly
        if defocus > 0.4:
            size_px = int(min(defocus, 3) * 6) | 1
            layer = cv2.GaussianBlur(layer.astype(np.float32), (size_px, size_px), min(defocus, 3))
        return np.where(inside, layer, image)

    def frame(self):
        import cv2

        height, width = self.size
        image = np.full((height, width), 90.0, dtype=np.float32)
        ys, xs = np.mgrid[0:height, 0:width]
        image = self._draw_body(image, xs, ys)
        for tool in (0, 1):
            raised = 0.0 if tool == self.machine.tool else self.lift_mm
            nozzle_z = self.machine.z() + raised
            cx, cy = self.pixel(self.machine.nozzle_xy(tool), nozzle_z)
            if not (-60 < cx < width + 60 and -60 < cy < height + 60):
                continue
            radius = np.hypot(xs - cx, ys - cy)
            layer = np.full_like(image, np.nan)
            layer[radius <= 34] = 230.0
            layer[(radius >= 6) & (radius <= 14)] = 20.0
            layer[radius <= 3] = 200.0
            defocus = abs(nozzle_z - self.focus_z) * self.blur_per_mm
            if defocus > 0.4:
                filled = np.where(np.isnan(layer), 90.0, layer)
                size_px = int(defocus * 6) | 1
                blurred = cv2.GaussianBlur(filled, (size_px, size_px), defocus)
                mask = cv2.GaussianBlur((~np.isnan(layer)).astype(np.float32),
                                        (size_px, size_px), defocus)
                image = image * (1 - mask) + blurred * mask
            else:
                image = np.where(np.isnan(layer), image, layer)
        image += self.rng.normal(0.0, 1.5, image.shape)
        return image
