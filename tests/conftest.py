# coding=utf-8
"""Load the plugin's pure modules without pulling in OctoPrint.

``octoprint_nozzlealign/__init__.py`` imports flask and octoprint.plugin, which
only exist on the printer host.  The vision, geometry, gcode and routine modules
have no such dependency, so the tests mount them under a synthetic package name
and leave the real package __init__ alone.
"""

import pathlib
import sys
import types

ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / "octoprint_nozzlealign"

if "nozzlealign_pkg" not in sys.modules:
    package = types.ModuleType("nozzlealign_pkg")
    package.__path__ = [str(PACKAGE_DIR)]
    sys.modules["nozzlealign_pkg"] = package
