$(function () {
    function NozzleAlignViewModel(parameters) {
        var self = this;

        self.settingsViewModel = parameters[0];
        self.loginState = parameters[1];

        self.running = ko.observable(false);
        self.live = ko.observable(false);
        self.log = ko.observable("");
        self.result = ko.observable(null);
        self.cameraText = ko.observable("");
        self.overlayUrl = ko.observable(
            OctoPrint.getBlueprintUrl("nozzlealign") + "overlay.jpg?t=" + Date.now()
        );

        var liveTimer = null;

        self.refreshPreview = function () {
            self.overlayUrl(
                OctoPrint.getBlueprintUrl("nozzlealign") + "overlay.jpg?t=" + Date.now()
            );
        };

        self.toggleLive = function () {
            self.live(!self.live());
            if (self.live()) {
                liveTimer = setInterval(self.refreshPreview, 1500);
            } else if (liveTimer) {
                clearInterval(liveTimer);
                liveTimer = null;
            }
        };

        self.append = function (line) {
            self.log(self.log() + line + "\n");
        };

        function errorText(response) {
            if (response && response.responseJSON && response.responseJSON.error) {
                return response.responseJSON.error;
            }
            return response && response.responseText ? response.responseText : "failed";
        }

        self.run = function () {
            self.log("");
            self.result(null);
            OctoPrint.simpleApiCommand("nozzlealign", "run", {})
                .done(function () {
                    self.running(true);
                })
                .fail(function (response) {
                    self.append("error: " + errorText(response));
                });
        };

        self.abort = function () {
            OctoPrint.simpleApiCommand("nozzlealign", "abort", {});
        };

        self.applyNew = function () {
            var pair = self.result().new_offset;
            OctoPrint.simpleApiCommand("nozzlealign", "apply", {x: pair[0], y: pair[1]})
                .done(function (data) {
                    self.append(
                        "wrote X" + data.requested[0].toFixed(2) + " Y" + data.requested[1].toFixed(2) +
                        "; firmware reports " + (data.written
                            ? "X" + data.written[0].toFixed(2) + " Y" + data.written[1].toFixed(2)
                            : "nothing") +
                        (data.verified ? " (verified)" : " (READ BACK DOES NOT MATCH; the previous value was put back)")
                    );
                })
                .fail(function (response) {
                    self.append("apply failed: " + errorText(response));
                });
        };

        function pair(values, digits) {
            return "X" + values[0].toFixed(digits) + "  Y" + values[1].toFixed(digits);
        }

        self.storedOffsetText = ko.pureComputed(function () {
            var r = self.result();
            return r ? pair(r.stored_offset, 3) : "";
        });
        self.correctionText = ko.pureComputed(function () {
            var r = self.result();
            return r ? pair(r.correction, 4) + " mm from nozzle 0" : "";
        });
        self.newOffsetText = ko.pureComputed(function () {
            var r = self.result();
            return r ? pair(r.new_offset, 3) : "";
        });
        self.residualText = ko.pureComputed(function () {
            var r = self.result();
            if (!r) return "";
            return "T0 " + r.residual_0_mm.toFixed(4) +
                   " mm, T1 " + r.residual_1_mm.toFixed(4) + " mm";
        });
        self.scaleText = ko.pureComputed(function () {
            var r = self.result();
            if (!r) return "";
            return r.px_per_mm.toFixed(1) + " px/mm, rotated " +
                   r.rotation_deg.toFixed(1) + " degrees";
        });
        self.applyLabel = ko.pureComputed(function () {
            var r = self.result();
            return r ? "Write " + pair(r.new_offset, 2) + " to the firmware" : "";
        });

        function showCamera(camera) {
            self.cameraText(
                "T0 over the lens at X" + camera.camera_x.toFixed(2) +
                " Y" + camera.camera_y.toFixed(2) +
                ", sharpest at Z" + camera.camera_z.toFixed(2)
            );
        }

        self.onDataUpdaterPluginMessage = function (plugin, data) {
            if (plugin !== "nozzlealign") return;
            if (data.type === "progress") {
                self.append(data.message);
                self.refreshPreview();
            } else if (data.type === "done") {
                self.running(false);
                self.result(data.result);
                if (data.result && data.result.camera) {
                    showCamera(data.result.camera);
                }
                self.append("finished");
            } else if (data.type === "failed") {
                self.running(false);
                self.append("failed: " + data.message);
            } else if (data.type === "aborted") {
                self.running(false);
                self.append("stopped");
            }
        };

        self.onBeforeBinding = function () {
            OctoPrint.simpleApiGet("nozzlealign").done(function (data) {
                self.running(!!data.running);
                if (data.result) {
                    self.result(data.result);
                    if (data.result.camera) showCamera(data.result.camera);
                }
            });
        };

        self.onTabChange = function (current) {
            if (current !== "#tab_plugin_nozzlealign" && self.live()) {
                self.toggleLive();
            }
        };
    }

    OCTOPRINT_VIEWMODELS.push({
        construct: NozzleAlignViewModel,
        dependencies: ["settingsViewModel", "loginStateViewModel"],
        elements: ["#nozzlealign_tab"]
    });
});
