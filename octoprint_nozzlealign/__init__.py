# coding=utf-8
"""XY nozzle alignment for a dual extruder Snapmaker, driven by a bed camera."""

from __future__ import absolute_import

import io
import threading

import flask
import octoprint.plugin

from . import routine, vision
from .gcode import (GcodeBridge, format_offset_command, read_back_matches,
                    within_toolhead_band)
from .settings import DEFAULTS

__plugin_name__ = "XY Nozzle Alignment"
__plugin_pythoncompat__ = ">=3.7,<4"


class NozzleAlignPlugin(
    octoprint.plugin.SettingsPlugin,
    octoprint.plugin.TemplatePlugin,
    octoprint.plugin.AssetPlugin,
    octoprint.plugin.SimpleApiPlugin,
    octoprint.plugin.BlueprintPlugin,
    octoprint.plugin.StartupPlugin,
    octoprint.plugin.ShutdownPlugin,
    octoprint.plugin.EventHandlerPlugin,
):
    def __init__(self):
        self._bridge = None
        self._routine = None
        self._routine_lock = threading.Lock()
        self._last_result = None

    # -- SettingsPlugin ---------------------------------------------------

    def get_settings_defaults(self):
        return dict(DEFAULTS)

    def get_settings_version(self):
        return 2

    def on_settings_migrate(self, target, current):
        # Version 1 stored the raised T1 nozzle's position as the camera
        # point and a floor equal to the working height. Both are replaced
        # by the measured values, and the floor by the real hard floor.
        if current is None or current < 2:
            for key in ("camera_x", "camera_y", "camera_z", "min_z",
                        "offset_limit_mm"):
                self._settings.set([key], DEFAULTS[key])

    # -- TemplatePlugin ---------------------------------------------------

    def is_template_autoescaped(self):
        return True

    def get_template_configs(self):
        return [
            dict(type="tab", name="Nozzle Align", template="nozzlealign_tab.jinja2",
                 custom_bindings=True),
            dict(type="settings", name="Nozzle Align",
                 template="nozzlealign_settings.jinja2", custom_bindings=True),
        ]

    # -- AssetPlugin ------------------------------------------------------

    def get_assets(self):
        return dict(js=["js/nozzlealign.js"], css=["css/nozzlealign.css"])

    # -- StartupPlugin ----------------------------------------------------

    def on_after_startup(self):
        self._bridge = GcodeBridge(self._printer, self._logger)
        self._logger.info("XY nozzle alignment ready")

    def on_shutdown(self):
        self._stop_routine()

    # -- EventHandlerPlugin -----------------------------------------------

    def on_event(self, event, payload):
        if event in ("Disconnected", "Error", "PrintStarted"):
            self._stop_routine(stop_motion=event == "PrintStarted")

    # -- gcode hook -------------------------------------------------------

    def gcode_received(self, comm_instance, line, *args, **kwargs):
        if self._bridge is None:
            return line
        return self._bridge.on_gcode_received(comm_instance, line, *args, **kwargs)

    # -- helpers ----------------------------------------------------------

    def _config(self):
        config = {}
        for key in DEFAULTS:
            config[key] = self._settings.get([key])
        return config

    def _notify(self, payload):
        self._plugin_manager.send_plugin_message(self._identifier, payload)

    def _stop_routine(self, stop_motion=True):
        """Stop the routine, and stop the machine, as fast as possible.

        The abort flag alone is not enough: the routine may be waiting on a
        move, so the bridge is woken too, and M410 halts whatever the planner
        still holds. The routine parks the head high afterwards.
        """
        with self._routine_lock:
            if self._routine is not None and self._routine.is_alive():
                self._routine.abort()
                if self._bridge is not None:
                    self._bridge.interrupt()
                if stop_motion and self._printer.is_operational():
                    self._printer.commands(["M410"], tags={"plugin:nozzlealign"})

    def _routine_running(self):
        return self._routine is not None and self._routine.is_alive()

    def _store_camera_position(self, camera):
        """Keep where the active T0 nozzle sat over the lens, and how high.

        The position is for information only; every run sweeps the bed for
        the camera from scratch. The height is where the next focus sweep
        starts, and no height below the floor is ever commanded.
        """
        for key in ("camera_x", "camera_y", "camera_z"):
            self._settings.set([key], round(float(camera[key]), 3))
        self._settings.save()
        self._logger.info(
            "camera found at X%.3f Y%.3f, focus at Z%.3f",
            camera["camera_x"], camera["camera_y"], camera["camera_z"])

    # -- SimpleApiPlugin --------------------------------------------------

    def is_api_protected(self):
        return True

    def get_api_commands(self):
        return dict(run=[], abort=[], apply=["x", "y"], read_offset=[])

    def on_api_command(self, command, data):
        try:
            handler = getattr(self, "_api_" + command)
        except AttributeError:
            return flask.abort(400, "unknown command")
        try:
            result = handler(data)
        except ValueError as exception:
            return flask.make_response(flask.jsonify(error=str(exception)), 400)
        except Exception as exception:
            self._logger.exception("api command %s failed", command)
            return flask.make_response(flask.jsonify(error=str(exception)), 500)
        return flask.jsonify(result or dict(ok=True))

    def on_api_get(self, request):
        return flask.jsonify(running=self._routine_running(), result=self._last_result)

    def _api_run(self, data):
        with self._routine_lock:
            if self._routine_running():
                raise ValueError("a routine is already running")
            if not self._printer.is_operational():
                raise ValueError("the printer is not connected")
            if self._printer.is_printing():
                raise ValueError("the printer is busy")
            self._last_result = None
            self._routine = _ResultKeepingRoutine(
                self, self._bridge, self._config(), self._notify, self._logger)
            self._routine.start()
        return dict(started=True)

    def _api_abort(self, data):
        self._stop_routine()
        return dict(aborted=True)

    def _api_read_offset(self, data):
        if not self._printer.is_operational():
            raise ValueError("the printer is not connected")
        if self._routine_running():
            raise ValueError("a routine is running")
        offset = self._bridge.read_hotend_offset(tool=1)
        if offset is None:
            raise ValueError("could not read the offset with M218 or M503")
        return dict(offset=list(offset))

    def _api_apply(self, data):
        if not self._printer.is_operational():
            raise ValueError("the printer is not connected")
        if self._printer.is_printing():
            raise ValueError("the printer is busy")
        if self._routine_running():
            raise ValueError("a routine is running")
        config = self._config()
        x = round(float(data["x"]), 2)
        y = round(float(data["y"]), 2)
        nominal_x = float(config["nominal_offset_x"])
        nominal_y = float(config["nominal_offset_y"])
        limit = float(config["offset_limit_mm"])
        if not within_toolhead_band(x, y, nominal_x, nominal_y, limit):
            raise ValueError(
                "X%.2f Y%.2f is more than %.1f mm from X%.2f Y%.2f; the toolhead "
                "would throw it away and reset to the default, so it is not sent"
                % (x, y, limit, nominal_x, nominal_y))
        before = self._bridge.read_hotend_offset(tool=1)
        commands = [format_offset_command(1, x, y)]
        if config["save_to_eeprom"]:
            commands.append("M500")
        self._bridge.run(commands, timeout=float(config["move_timeout"]))
        written = self._bridge.read_hotend_offset(tool=1)
        verified = read_back_matches((x, y), written)
        if not verified and before is not None and written is not None:
            # Put back what was there rather than leave a value the toolhead
            # chose for itself.
            self._logger.warning(
                "the firmware kept X%.2f Y%.2f instead of X%.2f Y%.2f; restoring",
                written[0], written[1], x, y)
            self._bridge.run(
                [format_offset_command(1, before[0], before[1])] + commands[1:],
                timeout=float(config["move_timeout"]))
            written = self._bridge.read_hotend_offset(tool=1)
        return dict(
            requested=[x, y],
            written=list(written) if written is not None else None,
            verified=bool(verified),
        )

    # -- BlueprintPlugin --------------------------------------------------

    @octoprint.plugin.BlueprintPlugin.route("/overlay.jpg", methods=["GET"])
    def overlay(self):
        """The camera view with the target and the detected bore drawn on it.

        Uses the same detector as the run, so what the operator sees is what
        the routine will steer on.
        """
        from PIL import Image, ImageDraw

        from . import nozzle

        config = self._config()
        frame = vision.average_frames(
            config["snapshot_url"], count=2,
            timeout=min(float(config["http_timeout"]), 5.0), attempts=1)
        image = Image.fromarray(frame.clip(0, 255).astype("uint8")).convert("RGB")
        draw = ImageDraw.Draw(image)
        width, height = image.size
        target_x = config["target_x_px"]
        target_y = config["target_y_px"]
        target_x = width / 2.0 if target_x is None else float(target_x)
        target_y = height / 2.0 if target_y is None else float(target_y)
        draw.line([(target_x, 0), (target_x, height)], fill=(0, 160, 255), width=1)
        draw.line([(0, target_y), (width, target_y)], fill=(0, 160, 255), width=1)
        found = nozzle.find_bore(
            frame, centre=(target_x, target_y),
            search_radius=float(config["search_radius_px"]),
            inner_r=int(config["bore_inner_r"]), ring_lo=int(config["bore_ring_lo"]),
            ring_hi=int(config["bore_ring_hi"]), core_r=int(config["bore_core_r"]))
        colour = (255, 60, 60) if found["score"] >= float(config["bore_min_score"]) else (255, 200, 0)
        x, y, radius = found["x"], found["y"], float(config["bore_ring_hi"])
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], outline=colour, width=2)
        draw.line([(x - 15, y), (x + 15, y)], fill=colour, width=2)
        draw.line([(x, y - 15), (x, y + 15)], fill=colour, width=2)
        draw.text((10, 10), "bore score %.0f" % found["score"], fill=colour)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        response = flask.make_response(buffer.getvalue())
        response.headers["Content-Type"] = "image/jpeg"
        response.headers["Cache-Control"] = "no-store"
        return response

    def is_blueprint_csrf_protected(self):
        return True


class _ResultKeepingRoutine(routine.CalibrationRoutine):
    """Runs the measurement and keeps the result for the UI."""

    def __init__(self, plugin, bridge, config, notify, logger):
        super(_ResultKeepingRoutine, self).__init__(
            bridge, config, notify, logger, on_camera=plugin._store_camera_position)
        self._plugin = plugin

    def run(self):
        super(_ResultKeepingRoutine, self).run()
        self._plugin._last_result = self.result


def __plugin_load__():
    global __plugin_implementation__
    global __plugin_hooks__
    __plugin_implementation__ = NozzleAlignPlugin()
    __plugin_hooks__ = {
        "octoprint.comm.protocol.gcode.received":
            __plugin_implementation__.gcode_received,
    }
