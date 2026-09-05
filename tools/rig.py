"""Drive the printer and the camera from this machine, for experiments.

Everything here refuses to move unless OctoPrint reports the printer connected,
and refuses any Z below FLOOR_Z.  FLOOR_Z starts high on purpose: the camera
sits on the bed and its height is not yet known.
"""
import io, json, os, subprocess, sys, time
import numpy as np, cv2
from PIL import Image

S = os.path.dirname(os.path.abspath(__file__))
HOST = "http://192.168.0.22:5000"
CAM = "http://192.168.0.22:1984/api/frame.jpeg?src=nozzle_cam"
KEY = open(os.path.join(S, ".opkey")).read().strip()   # not committed
H = {"X-Api-Key": KEY, "Content-Type": "application/json"}


def _curl(args, tries=5, delay=4.0, what="request", binary=False):
    """All HTTP goes through curl.

    Python's own sockets cannot reach the print PC from this sandbox, but curl
    can, so curl is the transport.  Retrying matters: the print PC drops the odd
    connection and an overnight run must not die of one.
    """
    last = None
    for _ in range(tries):
        try:
            out = subprocess.run(["curl", "-s", "--max-time", "30"] + args,
                                 capture_output=True, check=True)
            if out.stdout or binary:
                return out.stdout
            last = "empty response"
        except subprocess.CalledProcessError as exception:
            last = exception
        time.sleep(delay)
    raise SystemExit("%s failed %d times: %s" % (what, tries, last))


def _api(path):
    return json.loads(_curl(["-H", "X-Api-Key: " + KEY, HOST + path],
                            what="GET " + path))


def _post(path, payload):
    _curl(["-X", "POST", "-H", "X-Api-Key: " + KEY,
           "-H", "Content-Type: application/json",
           "-d", json.dumps(payload), "-o", "/dev/null",
           "-w", "%{http_code}", HOST + path], what="POST " + path)

FLOOR_Z = 60.0     # the camera sits on the bed; 60 mm still gives 19 px/mm
CEIL_Z = 200.0


def state():
    return _api("/api/connection")["current"]["state"]


def require_ready():
    s = state()
    if s != "Operational":
        raise SystemExit("printer not ready: %s" % s)


def send(commands, settle=6.0):
    require_ready()
    _post("/api/printer/command", {"commands": commands})
    time.sleep(settle)


def moveto(x=None, y=None, z=None, feed=5000, settle=6.0):
    if z is not None:
        if z < FLOOR_Z or z > CEIL_Z:
            raise SystemExit("refusing Z%.2f: outside the safe band %.0f-%.0f"
                             % (z, FLOOR_Z, CEIL_Z))
    parts = ["G1"]
    if x is not None: parts.append("X%.4f" % x)
    if y is not None: parts.append("Y%.4f" % y)
    if z is not None: parts.append("Z%.4f" % z)
    parts.append("F%d" % feed)
    send(["G90", " ".join(parts), "M400"], settle)


def rel(dx=0.0, dy=0.0, dz=0.0, feed=1200, settle=5.0):
    if dz:
        raise SystemExit("use moveto for Z so the floor is checked")
    send(["G91", "G1 X%.4f Y%.4f F%d" % (dx, dy, feed), "G90", "M400"], settle)


def frame(count=3, settle=1):
    total = None
    kept = 0
    for i in range(count + settle):
        blob = _curl([CAM], what="camera frame", binary=True)
        if len(blob) < 1000:
            time.sleep(1.0); continue
        a = np.asarray(Image.open(io.BytesIO(blob)).convert("L"), dtype=np.float32)
        if i < settle:
            continue
        total = a if total is None else total + a
        kept += 1
    if total is None or kept == 0:
        raise SystemExit("no frames from the camera")
    return total / float(kept)


def save(img, name):
    Image.fromarray(np.clip(img, 0, 255).astype("uint8")).save(
        os.path.join(S, name), quality=90)


def shift_in_window(a, b, centre, half=180):
    """Sub-pixel displacement of the content in a window, a -> b."""
    x, y = int(centre[0]), int(centre[1])
    y0, y1 = max(0, y - half), min(a.shape[0], y + half)
    x0, x1 = max(0, x - half), min(a.shape[1], x + half)
    pa = np.ascontiguousarray(a[y0:y1, x0:x1], dtype=np.float32)
    pb = np.ascontiguousarray(b[y0:y1, x0:x1], dtype=np.float32)
    win = cv2.createHanningWindow((pa.shape[1], pa.shape[0]), cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(pa, pb, win)
    return (float(dx), float(dy)), float(response)


def moving_blob(a, b, min_area=300):
    """Where the thing that moved sits in frame a, and how far it went."""
    d = cv2.GaussianBlur(a - b, (9, 9), 0)
    noise = float(np.median(np.abs(d - np.median(d)))) * 1.4826
    level = max(4.0, 5.0 * noise)
    out = {}
    for name, mask in (("pos", d > level), ("neg", d < -level)):
        m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cs = [c for c in cs if cv2.contourArea(c) >= min_area]
        if not cs:
            out[name] = None; continue
        c = max(cs, key=cv2.contourArea)
        mo = cv2.moments(c)
        out[name] = dict(area=float(cv2.contourArea(c)),
                         centre=(mo["m10"] / mo["m00"], mo["m01"] / mo["m00"]))
    changed = int((np.abs(d) > level).sum())
    return out, changed, level
