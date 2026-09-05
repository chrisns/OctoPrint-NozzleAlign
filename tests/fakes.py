# coding=utf-8
"""A simulated printer and camera, so the closed loop can be tested offline."""

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


class FakeCamera(object):
    """Renders the active nozzle as a dark disc on a bright field."""

    def __init__(self, machine, matrix, camera_xy, centre_px, size=(800, 1280)):
        self.machine = machine
        self.matrix = np.asarray(matrix, dtype=float)
        self.camera_xy = np.asarray(camera_xy, dtype=float)
        self.centre_px = np.asarray(centre_px, dtype=float)
        self.size = size
        self.radius = 40.0

    def tip_pixel(self):
        physical = self.machine.physical_xy()
        return self.centre_px + self.matrix.dot(physical - self.camera_xy)

    def frame(self):
        height, width = self.size
        image = np.full((height, width), 210.0, dtype=np.float32)
        cx, cy = self.tip_pixel()
        ys, xs = np.mgrid[0:height, 0:width]
        mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= self.radius ** 2
        image[mask] = 25.0
        # a little texture so the frame never looks blank
        image += np.random.default_rng(0).normal(0.0, 2.0, image.shape)
        return image


class PerspectiveCamera(object):
    """An upward camera whose scale and focus both depend on height.

    The image scale of the nozzle is inversely proportional to its distance
    from the lens, and the picture is sharpest at one height.  Both are what the
    discovery routine measures, so the fake has to model both.
    """

    def __init__(self, machine, camera_xy, lens_z, focus_z, rotation=9.0,
                 scale_constant=900.0, size=(800, 1280), nozzle_mm=0.9,
                 bright=False, blur_per_mm=0.22, nozzle_visible_below=None):
        self.machine = machine
        self.camera_xy = np.asarray(camera_xy, dtype=float)
        self.lens_z = float(lens_z)
        self.focus_z = float(focus_z)
        self.rotation = rotation
        self.scale_constant = float(scale_constant)
        self.size = size
        self.nozzle_mm = nozzle_mm
        self.bright = bright
        self.blur_per_mm = blur_per_mm
        # above this height the whole toolhead is in view and the nozzle does
        # not separate from it; below it, only the nozzle is left in frame
        self.nozzle_visible_below = nozzle_visible_below
        self.centre_px = np.array([size[1] / 2.0, size[0] / 2.0])

    def scale(self):
        distance = max(1.0, self.machine.position_xyz[2] - self.lens_z)
        return self.scale_constant / distance

    def matrix(self):
        return rotation_matrix(self.scale(), self.rotation)

    def tip_pixel(self):
        physical = self.machine.physical_xy()
        return self.centre_px + self.matrix().dot(physical - self.camera_xy)

    def in_view(self):
        height, width = self.size
        x, y = self.tip_pixel()
        margin = self.nozzle_mm * self.scale()
        return margin < x < width - margin and margin < y < height - margin

    def nozzle_separates(self):
        if self.nozzle_visible_below is None:
            return True
        return self.machine.position_xyz[2] <= self.nozzle_visible_below

    def frame(self):
        import cv2

        height, width = self.size
        background = 40.0 if self.bright else 210.0
        ink = 220.0 if self.bright else 25.0
        image = np.full((height, width), background, dtype=np.float32)
        if self.in_view():
            cx, cy = self.tip_pixel()
            ys, xs = np.mgrid[0:height, 0:width]
            if self.nozzle_separates():
                radius = max(3.0, self.nozzle_mm * self.scale())
                image[(xs - cx) ** 2 + (ys - cy) ** 2 <= radius ** 2] = ink
            else:
                # the whole toolhead: a broad textured structure that moves as
                # one piece, with no compact nozzle to pick out
                span = min(18.0 * self.scale(), 0.35 * min(width, height))
                rng = np.random.default_rng(11)
                block = rng.normal(0.0, 45.0, (24, 24)).astype(np.float32)
                tile = cv2.resize(
                    block, (int(2 * span) + 2, int(2 * span) + 2),
                    interpolation=cv2.INTER_CUBIC,
                )
                x0 = int(cx - span)
                y0 = int(cy - span)
                for iy in range(tile.shape[0]):
                    ty = y0 + iy
                    if 0 <= ty < height:
                        tx0 = max(0, x0)
                        tx1 = min(width, x0 + tile.shape[1])
                        if tx1 > tx0:
                            image[ty, tx0:tx1] = (
                                background * 0.55
                                + tile[iy, tx0 - x0:tx1 - x0]
                            )
        defocus = abs(self.machine.position_xyz[2] - self.focus_z) * self.blur_per_mm
        if defocus > 0.4:
            sigma = float(defocus)
            size_px = int(sigma * 6) | 1
            image = cv2.GaussianBlur(image, (size_px, size_px), sigma)
        image += np.random.default_rng(0).normal(0.0, 1.5, image.shape)
        return image


class FakeBridge(object):
    """Enough of GcodeBridge to drive the routine."""

    def __init__(self, tool_error=(0.0, 0.0), stored_offset=(26.0, 0.0, -1.5)):
        self.position_xyz = [0.0, 0.0, 100.0]
        self.tool = 0
        self.relative = False
        self.tool_error = np.asarray(tool_error, dtype=float)
        self.stored_offset = tuple(stored_offset)
        self.sent = []

    # -- simulation -------------------------------------------------------

    def physical_xy(self):
        """Where the active nozzle really is."""
        logical = np.array(self.position_xyz[:2], dtype=float)
        return logical + (self.tool_error if self.tool == 1 else np.zeros(2))

    def _apply(self, command):
        self.sent.append(command)
        if command.startswith("G90"):
            self.relative = False
            return
        if command.startswith("G91"):
            self.relative = True
            return
        if command.startswith("G28"):
            self.position_xyz = [0.0, 0.0, 100.0]
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
                if self.relative:
                    self.position_xyz[index] += value
                else:
                    self.position_xyz[index] = value

    # -- GcodeBridge surface ---------------------------------------------

    def run(self, commands, timeout=60.0):
        for command in commands:
            self._apply(command)
        return tuple(self.position_xyz)

    def position(self, timeout=30.0):
        return tuple(self.position_xyz)

    def query(self, command, timeout=30.0):
        return []

    def read_hotend_offset(self, tool=1, timeout=30.0):
        return self.stored_offset
