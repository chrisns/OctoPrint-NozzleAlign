"""Depth from a bed move. The nozzle is simply the nearest thing.

Moving Y moves the bed, and the camera with it, while the toolhead and the room
stay put. That is a pure camera translation, so a point at distance D from the
lens shifts by f * dY / D pixels. Near things shift more. Nothing else about the
scene matters: no appearance model, no focus of expansion to fit, and no ratio
of two small numbers to go unstable.

Compare that with dropping Z, which expands the picture about the point above
the lens. That works in principle but every depth then depends on locating that
point, and patches near it have almost no radius to divide by.

f is the focal length in pixels. It comes from the scale calibration: at a known
height the toolhead is a known distance away, and its measured px/mm gives f.
"""
import os, sys, json
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT = os.path.dirname(os.path.abspath(__file__))
FOCAL_PX = 1243.0        # from the scale fit: 8.03 px/mm at 154.8 mm
LENS_Z = -4.8            # where 1/scale crosses zero, i.e. the lens plane


def patch_shifts(a, b, size=192, step=64, min_response=0.10, min_texture=4.0):
    """Local shift per patch.

    Keep the bed move small. Phase correlation over a tapered window stops being
    reliable much beyond a quarter of the window, and it then reports a shift
    that quietly saturates instead of failing, which looks like a plausible
    depth and is not one.
    """
    h, w = a.shape
    win = cv2.createHanningWindow((size, size), cv2.CV_32F)
    rows = []
    for y in range(0, h - size + 1, step):
        for x in range(0, w - size + 1, step):
            pa = np.ascontiguousarray(a[y:y + size, x:x + size], dtype=np.float32)
            pb = np.ascontiguousarray(b[y:y + size, x:x + size], dtype=np.float32)
            if pa.std() < min_texture:
                continue
            (dx, dy), response = cv2.phaseCorrelate(pa, pb, win)
            if response < min_response:
                continue
            rows.append((x + size / 2.0, y + size / 2.0, dx, dy, response))
    return np.array(rows) if rows else np.zeros((0, 5))


def depth(rows, dmm, focal=FOCAL_PX):
    speed = np.hypot(rows[:, 2], rows[:, 3])
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(speed > 0.3, focal * abs(dmm) / speed, np.inf), speed


def report(rows, dmm, z, tag):
    d, speed = depth(rows, dmm)
    order = np.argsort(d)
    print(f"{len(rows)} patches with a usable shift, bed moved {dmm} mm at Z{z}")
    print("nearest patches:")
    for i in order[:10]:
        print(f"   ({rows[i,0]:6.0f},{rows[i,1]:6.0f})  shift {speed[i]:6.2f} px  "
              f"D {d[i]:7.1f} mm  height {z - (d[i] - 0):6.1f}  response {rows[i,4]:.2f}")
    print("furthest patches:")
    for i in order[-4:]:
        print(f"   ({rows[i,0]:6.0f},{rows[i,1]:6.0f})  shift {speed[i]:6.2f} px  D {d[i]:9.1f} mm")
    expected = z - LENS_Z
    print(f"the toolhead should sit about {expected:.0f} mm from the lens at Z{z}")
    return d, speed, order


def draw(base, rows, d, order, tag):
    vis = cv2.cvtColor(np.clip(base, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    finite = d[np.isfinite(d)]
    lo, hi = (np.percentile(finite, 5), np.percentile(finite, 90)) if len(finite) else (0, 1)
    for i in range(len(rows)):
        if not np.isfinite(d[i]):
            continue
        t = float(np.clip((d[i] - lo) / max(hi - lo, 1e-6), 0, 1))
        cv2.circle(vis, (int(rows[i, 0]), int(rows[i, 1])), 10,
                   (int(255 * t), 50, int(255 * (1 - t))), -1)
    for i in order[:3]:
        cv2.circle(vis, (int(rows[i, 0]), int(rows[i, 1])), 30, (0, 255, 0), 3)
    cv2.imwrite(os.path.join(OUT, tag + "_depth.jpg"), vis)


if __name__ == "__main__":
    import rig
    x, y, z = (float(v) for v in sys.argv[1:4])
    dmm = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0
    tag = sys.argv[5] if len(sys.argv) > 5 else "depth"
    rig.require_ready()
    rig.moveto(z=z, feed=700, settle=8)
    rig.moveto(x=x, y=y, settle=8)
    A = rig.frame(count=4)
    rig.rel(dy=dmm, settle=6)
    B = rig.frame(count=4)
    rig.rel(dy=-dmm, settle=5)
    rig.save(A, tag + "_a.jpg")
    rows = patch_shifts(A, B)
    if len(rows) < 10:
        raise SystemExit("too few patches: %d" % len(rows))
    d, speed, order = report(rows, dmm, z, tag)
    ceiling = 0.25 * 192
    if float(np.median(speed[order[:10]])) > ceiling:
        print(f"WARNING: the nearest shifts are near the {ceiling:.0f} px limit "
              f"of the patch; use a smaller bed move")
    draw(A, rows, d, order, tag)
    json.dump(dict(z=z, dmm=dmm,
                   nearest=[[float(rows[i, 0]), float(rows[i, 1]), float(d[i])]
                            for i in order[:8]]),
              open(os.path.join(OUT, tag + ".json"), "w"), indent=1)
    print("saved " + tag + "_depth.jpg")
