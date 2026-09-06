"""Measure the nozzle bore in millimetres, and the pixel scale with it.

The scale cannot be taken from the picture alone, because nothing in it has a
known size. It comes from the machine: move a known distance and see how far the
nozzle moves in the image. Everything else follows from that one number.

The dark circle at the centre of the cone is wider than the bore itself, because
it includes the counterbore and the shadow in it. So this reports the visible
opening, and says so, rather than claiming to have measured 0.4 against 0.8 to
three decimal places.
"""
import os, sys
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rig                       # noqa: E402
import nznozzle as nozzle        # noqa: E402


def orifice(save_as=None):
    frame = rig.frame(count=5)
    if save_as:
        rig.save(frame, save_as)
    circles = nozzle.find_circles(frame, 20, 70, param2=30, min_separation_px=80, take=3)
    height, width = frame.shape
    central = [c for c in circles
               if 0.15 * width < c["x"] < 0.85 * width
               and 0.15 * height < c["y"] < 0.85 * height]
    if not central:
        raise SystemExit("no orifice sized circle near the middle of the frame")
    return central[0], frame


def main(probe=0.5):
    rig.require_ready()
    first, frame = orifice("bore_before.jpg")
    print(f"orifice at ({first['x']:7.2f},{first['y']:7.2f}) r={first['r']:5.2f} px")
    moves = []
    for axis, dx, dy in (("X", probe, 0.0), ("Y", 0.0, probe)):
        rig.rel(dx=dx, dy=dy, settle=5)
        after, _ = orifice()
        rig.rel(dx=-dx, dy=-dy, settle=5)
        shift = np.hypot(after["x"] - first["x"], after["y"] - first["y"])
        print(f"  {axis} move {probe} mm -> the orifice moved {shift:6.2f} px "
              f"= {shift/probe:6.2f} px/mm")
        moves.append(shift / probe)
    scale = float(np.mean(moves))
    print(f"\nscale here: {scale:.2f} px/mm  "
          f"(field of view {1280/scale:.1f} x {800/scale:.1f} mm)")
    print(f"visible opening: {2*first['r']/scale:.3f} mm across")
    print("that includes the counterbore, so it reads wider than the bore itself")
    for size in (0.4, 0.6, 0.8, 1.0):
        print(f"   a {size:.1f} mm bore would be {size*scale:5.1f} px across")


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 0.5)
