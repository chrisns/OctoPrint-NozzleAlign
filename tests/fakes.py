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
