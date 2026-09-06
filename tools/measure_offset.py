"""Measure the XY offset error between the two nozzles.

The map is built from the nozzle circle itself: move the toolhead a known
distance and see how far the circle centre moves. That is direct and
unambiguous, unlike correlating a window, which mixes whatever else is in it.

Each nozzle is then steered onto the same pixel and the machine position read.
The difference between those two positions is what is left wrong in the stored
offset.
"""
import json, os, sys
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rig                       # noqa: E402
import nznozzle as nozzle        # noqa: E402

TARGET = (640.0, 400.0)
MIN_R, MAX_R = 60, 340


def locate(save_as=None):
    """The nozzle circle centre, refined to sub-pixel."""
    frame = rig.frame(count=4)
    if save_as:
        rig.save(frame, save_as)
    circles = nozzle.find_circles(frame, MIN_R, MAX_R, param2=55)
    height, width = frame.shape
    central = [c for c in circles
               if 0.12 * width < c["x"] < 0.88 * width
               and 0.12 * height < c["y"] < 0.88 * height]
    if not central:
        raise SystemExit("no nozzle circle in the middle of the frame")
    rough = central[0]
    try:
        fine = nozzle.refine_centre(frame, rough)
        if np.hypot(fine["x"] - rough["x"], fine["y"] - rough["y"]) > rough["r"] * 0.3:
            fine = rough                      # the fit wandered; keep the rough one
            fine["residual"] = float("nan")
    except nozzle.NozzleError:
        fine = dict(rough); fine["residual"] = float("nan")
    return fine, frame


def build_map(probe=2.0):
    """Pixels per millimetre, from how far the circle centre moves."""
    origin, _ = locate()
    rig.rel(dx=probe, settle=5)
    after_x, _ = locate()
    rig.rel(dx=-probe, settle=5)
    rig.rel(dy=probe, settle=5)
    after_y, _ = locate()
    rig.rel(dy=-probe, settle=5)
    column_x = np.array([after_x["x"] - origin["x"], after_x["y"] - origin["y"]]) / probe
    column_y = np.array([after_y["x"] - origin["x"], after_y["y"] - origin["y"]]) / probe
    matrix = np.column_stack([column_x, column_y])
    return matrix, origin


def steer(matrix, passes=6, tolerance_px=2.0):
    travelled = np.zeros(2)
    for attempt in range(passes):
        where, _ = locate()
        error = np.array(TARGET) - np.array([where["x"], where["y"]])
        print(f"    pass {attempt+1}: circle at ({where['x']:7.2f},{where['y']:7.2f}) "
              f"r={where['r']:5.1f} residual {where.get('residual', float('nan')):5.2f} "
              f"off by {np.hypot(*error):6.2f} px")
        if np.hypot(*error) <= tolerance_px:
            return travelled, where
        move = np.linalg.solve(matrix, error)
        move = np.clip(0.8 * move, -12.0, 12.0)
        rig.rel(dx=float(move[0]), dy=float(move[1]), settle=5)
        travelled += move
    return travelled, where


def main(x, y, z, stored=(25.20, 0.32)):
    rig.require_ready()
    rig.moveto(z=z, feed=700, settle=8)
    results = {}
    start = np.array([x, y], dtype=float)
    for index, tool in enumerate(("T0", "T1")):
        print(f"\n=== {tool}")
        rig.send([tool], settle=16)
        rig.moveto(x=float(start[0]), y=float(start[1]), settle=8)
        matrix, _ = build_map()
        scale = float(abs(np.linalg.det(matrix)) ** 0.5)
        angle = float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0])))
        print(f"  map {matrix.round(2).tolist()}  {scale:.2f} px/mm  turned {angle:.1f} deg")
        if scale < 3.0:
            raise SystemExit("the pixel map looks wrong; is the nozzle really in view?")
        travelled, where = steer(matrix)
        settled = start + travelled
        results[tool] = dict(settled=settled.tolist(), pixel=[where["x"], where["y"]],
                             scale=scale, angle=angle,
                             residual=float(where.get("residual", float("nan"))))
        print(f"  {tool} settled at X{settled[0]:.3f} Y{settled[1]:.3f}")
    a = np.array(results["T0"]["settled"]); b = np.array(results["T1"]["settled"])
    error = b - a
    print(f"\nnozzle 1 stood X{error[0]:+.3f} Y{error[1]:+.3f} mm from where nozzle 0 stood")
    print("that is what is left wrong in the stored offset")
    print(f"  stored now      X{stored[0]:7.3f} Y{stored[1]:7.3f}")
    print(f"  stored + error  X{stored[0]+error[0]:7.3f} Y{stored[1]+error[1]:7.3f}")
    print(f"  stored - error  X{stored[0]-error[0]:7.3f} Y{stored[1]-error[1]:7.3f}")
    results["error_mm"] = error.tolist()
    json.dump(results, open(os.path.join(HERE, "measure_offset.json"), "w"), indent=1)


if __name__ == "__main__":
    main(float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]))
