$(function () {
    function NozzleAlignViewModel(parameters) {
        var self = this;

        self.settingsViewModel = parameters[0];
        self.loginState = parameters[1];

        self.running = ko.observable(false);
        self.live = ko.observable(false);
        self.log = ko.observable("");
        self.result = ko.observable(null);
        self.cameraWarning = ko.observable("");
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

        self.cameraText = ko.observable("");

        self.discover = function () {
            self.log("");
            self.result(null);
            OctoPrint.simpleApiCommand("nozzlealign", "discover", {})
                .done(function () {
                    self.running(true);
                })
                .fail(function (response) {
                    self.append("error: " + errorText(response));
                });
        };

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

        function errorText(response) {
            if (response && response.responseJSON && response.responseJSON.error) {
                return response.responseJSON.error;
            }
            return response && response.responseText ? response.responseText : "failed";
        }

        function applyOffset(pair) {
            OctoPrint.simpleApiCommand("nozzlealign", "apply", {x: pair[0], y: pair[1]})
                .done(function (data) {
                    self.append(
                        "wrote X" + pair[0].toFixed(3) + " Y" + pair[1].toFixed(3) +
                        "; firmware reports X" + data.written[0].toFixed(3) +
                        " Y" + data.written[1].toFixed(3) +
                        (data.verified ? " (verified)" : " (READ BACK DOES NOT MATCH)")
                    );
                })
                .fail(function (response) {
                    self.append("apply failed: " + errorText(response));
                });
        }

        self.applyPlus = function () {
            applyOffset(self.result().candidates.plus);
        };
        self.applyMinus = function () {
            applyOffset(self.result().candidates.minus);
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
            return r ? pair(r.correction, 4) + " mm" : "";
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
        self.plusLabel = ko.pureComputed(function () {
            var r = self.result();
            return r ? "Write " + pair(r.candidates.plus, 3) : "";
        });
        self.minusLabel = ko.pureComputed(function () {
            var r = self.result();
            return r ? "Write " + pair(r.candidates.minus, 3) : "";
        });

        self.onDataUpdaterPluginMessage = function (plugin, data) {
            if (plugin !== "nozzlealign") return;
            if (data.type === "progress") {
                self.append(data.message);
                self.refreshPreview();
            } else if (data.type === "done") {
                self.running(false);
                var found = null;
                if (data.result && data.result.camera_z !== undefined) {
                    found = data.result;
                } else if (data.result && data.result.camera) {
                    found = data.result.camera;
                    self.result(data.result);
                }
                if (found) {
                    self.cameraWarning("");
                    self.cameraText(
                        "X" + found.camera_x.toFixed(2) +
                        " Y" + found.camera_y.toFixed(2) +
                        ", focus at Z" + found.camera_z.toFixed(2) +
                        " (" + found.px_per_mm.toFixed(1) + " px/mm, turned " +
                        found.rotation_deg.toFixed(1) + " degrees from the machine axes)"
                    );
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
            // no camera position is required up front; every run measures it
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
