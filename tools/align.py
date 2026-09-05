"""Put a nozzle over the camera, then read where each nozzle lands.

The map is measured with a window over the toolhead rather than the whole
frame. Over the toolhead, a window is a single depth, so an X move (toolhead
only) and a Y move (camera only) both give a clean local displacement. A window
covering the whole frame mixes the toolhead with the room behind it, and the
answer then depends on how much of each it happened to catch.
"""
import sys, os, json
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rig, nzdepth as depth

S = os.path.dirname(os.path.abspath(__file__))
FOCAL = 8.65 * 154.8       # px, from the scale measured at Z150
TARGET = (640.0, 400.0)    # the middle of the frame


def find_nozzle(z, move=2.0, patch=96, step=32):
    """The nearest point on the toolhead, which is the tip of the live nozzle."""
    a = rig.frame(count=4)
    rig.rel(dy=move, settle=5)
    b = rig.frame(count=4)
    rig.rel(dy=-move, settle=5)
    rows = depth.patch_shifts(a, b, patch=patch, step=step,
                              min_response=0.30, min_texture=6.0)
    if len(rows) < 10:
        raise SystemExit("only %d patches" % len(rows))
    d, _ = depth.depths(rows, move, FOCAL)
    centre, nearest, confident = depth.nearest_region(rows, d, take=6)
    return centre, nearest, confident, a


def measure_map(half=150, move=2.0, centre=TARGET):
    """Pixels per millimetre, as a 2x2 matrix, measured where the toolhead is."""
    a = rig.frame(count=3)
    rig.rel(dx=move, settle=5)
    bx = rig.frame(count=3)
    rig.rel(dx=-move, settle=5)
    rig.rel(dy=move, settle=5)
    by = rig.frame(count=3)
    rig.rel(dy=-move, settle=5)
    (dxx, dxy), rx = rig.shift_in_window(a, bx, centre, half=half)
    (dyx, dyy), ry = rig.shift_in_window(a, by, centre, half=half)
    matrix = np.array([[dxx / move, dyx / move], [dxy / move, dyy / move]])
    return matrix, (rx, ry)


def steer(z, passes=6, tolerance_px=12.0):
    """Drive the live nozzle onto the target pixel, tracking how far we moved.

    The machine position is accumulated here rather than read back, because the
    whole point of the measurement is the difference between where each tool had
    to stand, and that difference is exactly the sum of these moves.
    """
    travelled = np.zeros(2)
    for attempt in range(passes):
        centre, nearest, confident, frame = find_nozzle(z)
        error = np.array(TARGET) - np.array(centre)
        print(f"  pass {attempt+1}: nozzle at ({centre[0]:.0f},{centre[1]:.0f}) "
              f"{nearest:.1f} mm away, {confident} confident, off by "
              f"{np.hypot(*error):.0f} px")
        if np.hypot(*error) <= tolerance_px:
            return centre, nearest, frame, travelled
        matrix, responses = measure_map(centre=centre)
        det = float(np.linalg.det(matrix))
        print(f"     map {matrix.round(2).tolist()} det {det:.1f} "
              f"responses {responses[0]:.2f},{responses[1]:.2f}")
        if abs(det) < 4.0:
            raise SystemExit("the pixel map is degenerate; cannot steer")
        move = np.linalg.solve(matrix, error)
        move = np.clip(move, -25.0, 25.0)
        print(f"     moving X{move[0]:+.2f} Y{move[1]:+.2f} mm")
        rig.rel(dx=float(move[0]), dy=float(move[1]), settle=6)
        travelled += move
    return centre, nearest, frame, travelled


if __name__ == "__main__":
    x, y, z = (float(v) for v in sys.argv[1:4])
    rig.require_ready()
    rig.moveto(z=z, feed=700, settle=8)
    results = {}
    start = np.array([x, y])
    for tool in ("T0", "T1"):
        print(f"=== {tool}")
        rig.send([tool], settle=16)
        rig.moveto(x=float(start[0]), y=float(start[1]), settle=8)
        centre, nearest, frame, travelled = steer(z)
        settled = start + travelled
        results[tool] = dict(pixel=list(centre), distance=nearest,
                             settled=[float(settled[0]), float(settled[1])],
                             travelled=[float(travelled[0]), float(travelled[1])])
        rig.save(frame, "align_%s.jpg" % tool)
        print(f"  {tool} settled at X{settled[0]:.3f} Y{settled[1]:.3f}, "
              f"pixel ({centre[0]:.1f},{centre[1]:.1f})")
    if "T0" in results and "T1" in results:
        a = np.array(results["T0"]["settled"]); b = np.array(results["T1"]["settled"])
        residual = b - a
        results["residual_mm"] = [float(residual[0]), float(residual[1])]
        print(f"\nnozzle 1 had to stand X{residual[0]:+.3f} Y{residual[1]:+.3f} mm "
              f"from where nozzle 0 stood")
        print("that difference is the error left in the stored offset")
    json.dump(results, open(os.path.join(S, "align.json"), "w"), indent=1)
