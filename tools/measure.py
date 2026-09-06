"""Measure the tool offset from a workstation, with the plugin's own modules.

Same steps as the plugin routine, driven through OctoPrint's HTTP API instead
of from inside it, so a change to the detector or the map can be tried on the
machine before the plugin is redeployed.

    python tools/measure.py            # measure and report
    python tools/measure.py --repeat 3 # repeat and report the spread

Nothing here writes to the firmware.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import rig                                                  # noqa: E402
from octoprint_nozzlealign import focus, geometry, nozzle   # noqa: E402
from octoprint_nozzlealign.settings import DEFAULTS         # noqa: E402

CFG = dict(DEFAULTS)
TARGET = np.array([640.0, 400.0])


class Tracker(object):
    """Find the bore once, then follow it with a template."""

    def __init__(self):
        self.template = None
        self.last = None

    def find(self, frame, near=None, radius=None):
        if self.template is not None and near is None:
            spot, confidence = focus.track(frame, self.template, self.last, CFG["track_search_px"])
            if confidence >= CFG["track_min_match"]:
                found = self._bore(frame, spot, CFG["track_lock_px"])
                if found["score"] >= CFG["bore_track_score"]:
                    return self._keep(frame, found)
        found = self._bore(frame, near or self.last or tuple(TARGET),
                           radius or CFG["bore_search_radius_px"])
        if found["score"] < CFG["bore_min_score"]:
            raise SystemExit("no bore in view (score %.0f)" % found["score"])
        return self._keep(frame, found)

    def _bore(self, frame, near, radius):
        return nozzle.find_bore(frame, tuple(near), radius, CFG["bore_inner_r"],
                                CFG["bore_ring_lo"], CFG["bore_ring_hi"], core_r=CFG["bore_core_r"])

    def _keep(self, frame, found):
        self.last = (found["x"], found["y"])
        self.template = focus.cut(frame, self.last, CFG["focus_template_px"] // 2)
        return np.array(self.last), found["score"]


def z_to(value):
    if value < CFG["min_z"] or value > CFG["safe_z"]:
        raise SystemExit("refusing Z%.2f" % value)
    rig.send(["G90", "G1 Z%.3f F%d" % (value, CFG["z_feedrate"]), "M400"], settle=6)


def settle_move(dx=0.0, dy=0.0):
    first, second = geometry.backlash_free(dx, dy, CFG["backlash_backoff_mm"])
    rig.rel(*first, settle=4)
    rig.rel(*second, settle=5)


def arrive(x, y):
    back = CFG["backlash_backoff_mm"]
    rig.moveto(x=x - back, y=y - back, settle=5)
    rig.moveto(x=x, y=y, settle=6)


def focus_here(tracker):
    heights = focus.plan_heights(CFG["camera_z"], CFG["focus_span_mm"], CFG["focus_step_mm"],
                                 CFG["min_z"], CFG["safe_z"])
    samples = []
    frame = rig.frame(count=3)
    template = focus.cut(frame, tracker.last, CFG["focus_template_px"] // 2)
    where = tracker.last
    for z in heights:
        z_to(z)
        frame = rig.frame(count=3)
        where, _ = focus.track(frame, template, where)
        value = focus.sharpness(frame, where, CFG["focus_window_px"])
        samples.append((z, value))
        print("   Z%.2f sharpness %.0f" % (z, value), flush=True)
        template = focus.cut(frame, where, CFG["focus_template_px"] // 2)
    if not focus.is_peaked(samples, CFG["focus_peak_ratio"]):
        raise SystemExit("the focus did not peak inside the sweep")
    best, _ = focus.best_height(samples)
    best = min(max(best, min(heights)), max(heights))
    z_to(best)
    tracker.find(rig.frame(count=3))
    print("   sharpest at Z%.2f" % best, flush=True)
    return best


def pixel_map(tracker):
    distance = CFG["probe_distance"]
    for attempt in range(CFG["map_attempts"]):
        origin, _ = tracker.find(rig.frame(count=4))
        settle_move(dx=distance)
        after_x, _ = tracker.find(rig.frame(count=4))
        settle_move(dx=-distance)
        settle_move(dy=distance)
        after_y, _ = tracker.find(rig.frame(count=4))
        settle_move(dy=-distance)
        matrix = geometry.build_pixel_map(origin, after_x, after_y, distance)
        ok, scale, cosine = geometry.validate_map(matrix, CFG["map_min_scale"],
                                                  CFG["map_max_scale"], CFG["map_max_cos"])
        print("   map %.1f px/mm, columns %.2f aligned%s" % (scale, cosine, "" if ok else " (rejected)"),
              flush=True)
        if ok:
            return matrix, scale
    raise SystemExit("could not build a trustworthy map")


def centre(tracker, matrix, label):
    for index in range(CFG["max_passes"]):
        where, score = tracker.find(rig.frame(count=4))
        dx, dy = geometry.pixel_error_to_mm(matrix, where, TARGET)
        residual = float(np.hypot(dx, dy))
        print("   %s pass %d: off by %.4f mm (score %.0f)" % (label, index + 1, residual, score), flush=True)
        if residual <= CFG["tolerance_mm"]:
            break
        if residual > CFG["max_correction_mm"]:
            raise SystemExit("%s wants a %.2f mm correction" % (label, residual))
        settle_move(dx, dy)
    else:
        raise SystemExit("%s did not settle" % label)
    return np.array(rig.position()[:2])


def measure():
    rig.require_ready()
    z_to(CFG["safe_z"])
    rig.send(["T0"], settle=15)
    arrive(CFG["camera_x"], CFG["camera_y"])
    z_to(CFG["camera_z"])
    tracker = Tracker()
    tracker.find(rig.frame(count=4), radius=CFG["search_radius_px"])
    z = focus_here(tracker)
    matrix, scale = pixel_map(tracker)
    position_0 = centre(tracker, matrix, "T0")
    print("T0 on target at X%.4f Y%.4f" % tuple(position_0))

    z_to(CFG["safe_z"])
    rig.send(["T1"], settle=20)
    arrive(*position_0)
    z_to(z)
    tracker = Tracker()
    near_px = CFG["active_check_mm"] * scale
    try:
        tracker.find(rig.frame(count=4), near=tuple(TARGET), radius=near_px)
    except SystemExit:
        raise SystemExit("no bore near the target under T1: the nozzle under T0 was the "
                         "raised T1 nozzle; move camera_x by the nominal offset")
    matrix_1, _ = pixel_map(tracker)
    position_1 = centre(tracker, matrix_1, "T1")
    print("T1 on target at X%.4f Y%.4f" % tuple(position_1))
    z_to(CFG["safe_z"])
    rig.send(["T0"], settle=15)
    return position_0, position_1, z


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--stored", type=float, nargs=2, default=None,
                        help="the stored X Y offset, if you want the corrected value printed")
    args = parser.parse_args()
    gaps = []
    for index in range(args.repeat):
        position_0, position_1, z = measure()
        gap = position_1 - position_0
        gaps.append(gap.tolist())
        print("run %d: T1 landed X%+.4f Y%+.4f from T0 (focus Z%.2f)" % (index + 1, gap[0], gap[1], z))
        if args.stored:
            new = np.array(args.stored) - gap
            print("        offset that closes the gap: X%.3f Y%.3f" % (new[0], new[1]))
    if len(gaps) > 1:
        a = np.array(gaps)
        print("spread over %d runs: X%.1f Y%.1f microns" % (len(gaps), a[:, 0].std() * 1000, a[:, 1].std() * 1000))
    json.dump(gaps, open(os.path.join(HERE, "last_measurement.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
