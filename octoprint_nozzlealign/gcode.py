# coding=utf-8
"""Serial helpers for the nozzle alignment routine.

OctoPrint's ``PrinterInterface.commands`` is fire and forget, so every step that
needs to know the printer actually finished has to be paired with a reply we can
recognise.  ``M114`` is used as that marker throughout: it is cheap, Snapmaker
supports it, and its reply carries the position we want anyway.
"""

from __future__ import absolute_import

import re
import threading

# "X:100.00 Y:200.00 Z:10.00 E:0.00 Count X: 8000 Y:16000 Z:8000"
_POSITION_RE = re.compile(
    r"X:\s*(?P<x>-?\d+\.?\d*)\s+Y:\s*(?P<y>-?\d+\.?\d*)\s+Z:\s*(?P<z>-?\d+\.?\d*)"
)
# "M218 T1 X26.00 Y0.00 Z-1.52" as echoed by M218 and M503
_OFFSET_RE = re.compile(
    r"M218\s+T(?P<t>\d+)"
    r"(?:\s+X(?P<x>-?\d+\.?\d*))?"
    r"(?:\s+Y(?P<y>-?\d+\.?\d*))?"
    r"(?:\s+Z(?P<z>-?\d+\.?\d*))?"
)
# Marlin also reports the offsets as bare triples after an "offset" header.
# Snapmaker answers a bare M218 with every triple on the header line itself:
#   echo:Hotend offsets: 0.00,0.00,0.000 25.20,0.32,-0.891
_TRIPLE_RE = re.compile(
    r"^\s*(?P<x>-?\d+\.\d+)[,\s]+(?P<y>-?\d+\.\d+)[,\s]+(?P<z>-?\d+\.\d+)\s*$"
)
_INLINE_TRIPLE_RE = re.compile(
    r"(?<![\d.-])(-?\d+\.\d+),(-?\d+\.\d+),(-?\d+\.\d+)(?![\d.])"
)


class Timeout(Exception):
    """The printer did not answer in time."""


class GcodeBridge(object):
    """Sends G-code and waits for the reply that proves it ran."""

    def __init__(self, printer, logger):
        self._printer = printer
        self._logger = logger
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._collecting = False
        self._lines = []
        self._position = None

    # -- OctoPrint hook ---------------------------------------------------

    def on_gcode_received(self, comm_instance, line, *args, **kwargs):
        if self._collecting and line:
            self._lines.append(line)
            match = _POSITION_RE.search(line)
            if match:
                self._position = (
                    float(match.group("x")),
                    float(match.group("y")),
                    float(match.group("z")),
                )
                self._event.set()
        return line

    # -- primitives -------------------------------------------------------

    def _run(self, commands, timeout):
        """Send commands, then M400 and M114, and collect every reply line."""
        if not self._printer.is_operational():
            raise RuntimeError("printer is not connected")
        with self._lock:
            self._lines = []
            self._position = None
            self._event.clear()
            self._collecting = True
            try:
                payload = list(commands) + ["M400", "M114"]
                self._printer.commands(payload, tags={"plugin:nozzlealign"})
                if not self._event.wait(timeout):
                    raise Timeout("no M114 reply within %.0fs" % timeout)
                return list(self._lines), self._position
            finally:
                self._collecting = False

    def position(self, timeout=30.0):
        """Return the current (x, y, z) in machine units."""
        _, position = self._run([], timeout)
        return position

    def run(self, commands, timeout=60.0):
        """Run commands and return the position once they have finished."""
        _, position = self._run(commands, timeout)
        return position

    def query(self, command, timeout=30.0):
        """Run one command and return every reply line it produced."""
        lines, _ = self._run([command], timeout)
        return lines

    # -- higher level -----------------------------------------------------

    def read_hotend_offset(self, tool=1, timeout=30.0):
        """Read the stored hotend offset for a tool, or None if unreadable.

        Tries M218 first because Snapmaker persists it in the toolhead module,
        then falls back to the M503 dump.
        """
        for command in ("M218", "M503"):
            offset = parse_hotend_offset(self.query(command, timeout), tool)
            if offset is not None:
                return offset
        return None


def parse_position(lines):
    """Pull an (x, y, z) triple out of an M114 reply."""
    for line in lines:
        match = _POSITION_RE.search(line)
        if match:
            return (
                float(match.group("x")),
                float(match.group("y")),
                float(match.group("z")),
            )
    return None


def parse_hotend_offset(lines, tool=1):
    """Pull the (x, y, z) hotend offset for ``tool`` out of reply lines.

    Handles the ``M218 T1 X.. Y.. Z..`` echo that both M218 and M503 produce,
    and the bare triple that some Snapmaker firmware prints after a header
    containing the word "offset".
    """
    for line in lines:
        match = _OFFSET_RE.search(line)
        if match and int(match.group("t")) == tool:
            return (
                float(match.group("x") or 0.0),
                float(match.group("y") or 0.0),
                float(match.group("z") or 0.0),
            )

    # Fallback: an "offset" header carrying one triple per hotend, either on the
    # header line itself or on the lines after it.
    triples = []
    seen_header = False
    for line in lines:
        if "offset" in line.lower():
            seen_header = True
            inline = _INLINE_TRIPLE_RE.findall(line)
            if len(inline) > tool:
                x, y, z = inline[tool]
                return (float(x), float(y), float(z))
            triples = []
            continue
        if not seen_header:
            continue
        match = _TRIPLE_RE.match(line.strip())
        if match:
            triples.append(
                (
                    float(match.group("x")),
                    float(match.group("y")),
                    float(match.group("z")),
                )
            )
        elif triples:
            break
    if len(triples) > tool:
        return triples[tool]
    return None


def format_offset_command(tool, x, y):
    """Build the M218 command that stores an XY hotend offset."""
    return "M218 T%d X%.3f Y%.3f" % (tool, x, y)
