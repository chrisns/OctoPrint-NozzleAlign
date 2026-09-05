# coding=utf-8
"""XY nozzle alignment for a dual extruder Snapmaker, driven by a bed camera."""

from __future__ import absolute_import

import io
import threading

import flask
import octoprint.plugin

from . import discovery, geometry, routine, vision
from .gcode import GcodeBridge, format_offset_command

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
        self._last_discovery = None
        self._template = None

    # -- SettingsPlugin ---------------------------------------------------

    def get_settings_defaults(self):
        return dict(
            # camera
            snapshot_url="http://127.0.0.1:1984/api/frame.jpeg?src=nozzle_cam",
            stream_url="http://127.0.0.1:1984/api/stream.mjpeg?src=nozzle_cam",
            http_timeout=20.0,
            frame_average=8,
            # where the camera sits on the bed; the routine refuses to run
            # until these are set, because it drives the nozzle down onto it
            camera_x=None,
            camera_y=None,
            camera_z=None,
            safe_z=50.0,
            home_first=True,
            # motion
            feedrate=3000,
            fine_feedrate=600,
            z_feedrate=600,
            move_timeout=90.0,
            home_timeout=240.0,
            tool_settle_s=1.5,
            # closed loop
            probe_distance=1.0,
            tolerance_mm=0.005,
            max_passes=6,
            max_correction_mm=5.0,
            target_x_px=None,
            target_y_px=None,
            # camera discovery; nothing here is a coordinate you have to supply
            search_z=80.0,
            search_centre_x=160.0,
            search_centre_y=175.0,
            search_span_mm=120.0,
            search_points=3,
            search_probe_mm=3.0,
            coarse_step=10.0,
            fine_step=2.0,
            min_z=12.0,
            lens_clearance_mm=6.0,
            max_blob_fraction=0.15,
            focus_window_px=240,
            focus_drop_ratio=0.6,
            template_size_px=96,
            discovery_tolerance_mm=0.2,
            # detection
            strategy="motion",
            motion_threshold=4.0,
            motion_min_area=60,
            # a nozzle paints a compact blob; the gantry beam paints a sliver
            motion_min_circularity=0.25,
            motion_max_extent=0.4,
            motion_min_coverage=0.5,
            motion_probe_mm=0.6,
            contour_invert=True,
            blur=5,
            min_area=200,
            min_radius=10,
            max_radius=200,
            hough_param2=30,
            min_confidence=0.3,
            roi=None,
            # guard rails on what may be written to the firmware
            nominal_offset_x=26.0,
            nominal_offset_y=0.0,
            offset_limit_mm=3.0,
            save_to_eeprom=True,
        )

    def get_settings_version(self):
        return 1

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
            self._stop_routine()

    # -- gcode hook -------------------------------------------------------

    def gcode_received(self, comm_instance, line, *args, **kwargs):
        if self._bridge is None:
            return line
        return self._bridge.on_gcode_received(comm_instance, line, *args, **kwargs)

    # -- helpers ----------------------------------------------------------

    def _config(self):
        defaults = self.get_settings_defaults()
        config = {}
        for key in defaults:
            config[key] = self._settings.get([key])
        config["template"] = self._template
        return config

    def _notify(self, payload):
        self._plugin_manager.send_plugin_message(self._identifier, payload)

    def _stop_routine(self):
        with self._routine_lock:
            if self._routine is not None and self._routine.is_alive():
                self._routine.abort()

    def _routine_running(self):
        return self._routine is not None and self._routine.is_alive()

    def _require_camera_position(self, config):
        missing = [
            key for key in ("camera_x", "camera_y", "camera_z")
            if config.get(key) is None
        ]
        if missing:
            raise ValueError(
                "set the camera position first (%s); the routine lowers the "
                "nozzle onto the camera and will not guess where it is"
                % ", ".join(missing)
            )

    # -- SimpleApiPlugin --------------------------------------------------

    def is_api_protected(self):
        return True

    def get_api_commands(self):
        return dict(
            run=[],
            discover=[],
            verify=[],
            abort=[],
            apply=["x", "y"],
            preview=[],
            set_template=["x", "y", "size"],
            clear_template=[],
            read_offset=[],
        )

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
        return flask.jsonify(
            running=self._routine_running(),
            result=self._last_result,
            discovery=self._last_discovery,
            has_template=self._template is not None,
        )

    def _start(self, factory):
        with self._routine_lock:
            if self._routine_running():
                raise ValueError("a routine is already running")
            config = self._config()
            self._require_camera_position(config)
            if not self._printer.is_operational():
                raise ValueError("the printer is not connected")
            if self._printer.is_printing():
                raise ValueError("the printer is busy")
            self._routine = factory(config)
            self._routine.start()
        return dict(started=True)

    def _api_run(self, data):
        def factory(config):
            return _ResultKeepingRoutine(
                self, self._bridge, config, self._notify, self._logger
            )
        return self._start(factory)

    def _api_discover(self, data):
        """Find the camera without being told where it is."""
        with self._routine_lock:
            if self._routine_running():
                raise ValueError("a routine is already running")
            if not self._printer.is_operational():
                raise ValueError("the printer is not connected")
            if self._printer.is_printing():
                raise ValueError("the printer is busy")
            self._routine = discovery.DiscoveryRoutine(
                self._bridge,
                self._config(),
                self._notify,
                self._logger,
                on_result=self._store_camera_position,
            )
            self._routine.start()
        return dict(started=True)

    def _store_camera_position(self, result, template):
        for key in ("camera_x", "camera_y", "camera_z"):
            self._settings.set([key], result[key])
        self._settings.save()
        self._template = template
        self._last_discovery = result
        self._logger.info(
            "camera found at X%.3f Y%.3f Z%.3f", result["camera_x"],
            result["camera_y"], result["camera_z"],
        )

    def _api_verify(self, data):
        if self._last_result is None:
            raise ValueError("run a measurement first")
        return self._api_run(data)

    def _api_abort(self, data):
        self._stop_routine()
        return dict(aborted=True)

    def _api_read_offset(self, data):
        if not self._printer.is_operational():
            raise ValueError("the printer is not connected")
        offset = self._bridge.read_hotend_offset(tool=1)
        if offset is None:
            raise ValueError("could not read the offset with M218 or M503")
        return dict(offset=list(offset))

    def _api_apply(self, data):
        if not self._printer.is_operational():
            raise ValueError("the printer is not connected")
        if self._routine_running():
            raise ValueError("a routine is running")
        config = self._config()
        x = float(data["x"])
        y = float(data["y"])
        limit = float(config["offset_limit_mm"])
        nominal_x = float(config["nominal_offset_x"])
        nominal_y = float(config["nominal_offset_y"])
        if abs(x - nominal_x) > limit or abs(y - nominal_y) > limit:
            raise ValueError(
                "X%.3f Y%.3f is more than %.1f mm from the expected X%.3f Y%.3f; "
                "refusing to write it" % (x, y, limit, nominal_x, nominal_y)
            )
        commands = [format_offset_command(1, x, y)]
        if config["save_to_eeprom"]:
            commands.append("M500")
        self._bridge.run(commands, timeout=float(config["move_timeout"]))
        written = self._bridge.read_hotend_offset(tool=1)
        if written is None:
            raise ValueError(
                "wrote the offset but could not read it back; M218 may not be "
                "supported by this firmware"
            )
        matches = abs(written[0] - x) < 0.01 and abs(written[1] - y) < 0.01
        return dict(written=list(written), verified=bool(matches), requested=[x, y])

    def _api_preview(self, data):
        config = self._config()
        frame = vision.average_frames(
            config["snapshot_url"],
            count=int(config["frame_average"]),
            timeout=float(config["http_timeout"]),
        )
        mean, deviation = vision.frame_health(frame)
        detection = None
        message = None
        try:
            x, y, confidence, details = _detect(frame, config)
            detection = dict(x=x, y=y, confidence=confidence, details=_jsonable(details))
        except (vision.DetectionError, vision.CaptureError) as exception:
            message = str(exception)
        return dict(
            width=int(frame.shape[1]),
            height=int(frame.shape[0]),
            mean=mean,
            std=deviation,
            detection=detection,
            message=message,
        )

    def _api_set_template(self, data):
        config = self._config()
        frame = vision.average_frames(
            config["snapshot_url"],
            count=int(config["frame_average"]),
            timeout=float(config["http_timeout"]),
        )
        centre_x = int(data["x"])
        centre_y = int(data["y"])
        size = max(16, int(data.get("size") or 96))
        half = size // 2
        y0 = max(0, centre_y - half)
        y1 = min(frame.shape[0], centre_y + half)
        x0 = max(0, centre_x - half)
        x1 = min(frame.shape[1], centre_x + half)
        patch = frame[y0:y1, x0:x1]
        if patch.shape[0] < 16 or patch.shape[1] < 16:
            raise ValueError("pick a point further from the edge of the image")
        self._template = patch
        return dict(size=[int(patch.shape[1]), int(patch.shape[0])])

    def _api_clear_template(self, data):
        self._template = None
        return dict(ok=True)

    # -- BlueprintPlugin --------------------------------------------------

    @octoprint.plugin.BlueprintPlugin.route("/overlay.jpg", methods=["GET"])
    def overlay(self):
        from PIL import Image, ImageDraw

        config = self._config()
        frame = vision.average_frames(
            config["snapshot_url"],
            count=int(config["frame_average"]),
            timeout=float(config["http_timeout"]),
        )
        image = Image.fromarray(frame.clip(0, 255).astype("uint8")).convert("RGB")
        draw = ImageDraw.Draw(image)
        width, height = image.size
        target_x = config["target_x_px"]
        target_y = config["target_y_px"]
        target_x = width / 2.0 if target_x is None else float(target_x)
        target_y = height / 2.0 if target_y is None else float(target_y)
        draw.line([(target_x, 0), (target_x, height)], fill=(0, 160, 255), width=1)
        draw.line([(0, target_y), (width, target_y)], fill=(0, 160, 255), width=1)
        try:
            x, y, _, details = _detect(frame, config)
            radius = float(details.get("radius") or 20)
            draw.ellipse(
                [x - radius, y - radius, x + radius, y + radius],
                outline=(255, 60, 60), width=2,
            )
            draw.line([(x - 15, y), (x + 15, y)], fill=(255, 60, 60), width=2)
            draw.line([(x, y - 15), (x, y + 15)], fill=(255, 60, 60), width=2)
        except (vision.DetectionError, vision.CaptureError):
            pass
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        response = flask.make_response(buffer.getvalue())
        response.headers["Content-Type"] = "image/jpeg"
        response.headers["Cache-Control"] = "no-store"
        return response

    def is_blueprint_csrf_protected(self):
        return True


class _ResultKeepingRoutine(routine.CalibrationRoutine):
    """Stores the result on the plugin so the UI can fetch it after a reload."""

    def __init__(self, plugin, bridge, config, notify, logger):
        super(_ResultKeepingRoutine, self).__init__(bridge, config, notify, logger)
        self._plugin = plugin

    def run(self):
        super(_ResultKeepingRoutine, self).run()
        if self.result is not None:
            self._plugin._last_result = self.result


def _detect(frame, config):
    options = dict(roi=config.get("roi"))
    strategy = config["strategy"]
    if strategy == "contour":
        options.update(
            invert=bool(config["contour_invert"]),
            blur=int(config["blur"]),
            min_area=int(config["min_area"]),
        )
    elif strategy == "hough":
        options.update(
            blur=int(config["blur"]),
            min_radius=int(config["min_radius"]),
            max_radius=int(config["max_radius"]),
            param2=int(config["hough_param2"]),
        )
    return vision.detect_tip(
        frame, strategy=strategy, template=config.get("template"), **options
    )


def _jsonable(details):
    return {key: float(value) for key, value in (details or {}).items()}


def __plugin_load__():
    global __plugin_implementation__
    global __plugin_hooks__
    __plugin_implementation__ = NozzleAlignPlugin()
    __plugin_hooks__ = {
        "octoprint.comm.protocol.gcode.received":
            __plugin_implementation__.gcode_received,
    }
