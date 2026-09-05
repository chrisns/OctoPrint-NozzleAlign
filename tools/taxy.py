"""Run TAXY's nozzle detector.

TAXY (github.com/PrintStructor/TAXY, GPL-3.0) publishes a YOLOv8-nano trained on
one class, "Nozzle", for upward facing nozzle cameras. That is this exact
geometry, so it is the right thing to start from rather than training anything.

It loads in cv2.dnn, so the print PC needs no new dependency at all.
"""
import os
import numpy as np
import cv2

MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "taxy_nozzle.onnx")
_net = None


def net():
    global _net
    if _net is None:
        _net = cv2.dnn.readNetFromONNX(MODEL)
    return _net


def letterbox(image, size=640, pad=114):
    h, w = image.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), pad, dtype=np.uint8)
    ox, oy = (size - nw) // 2, (size - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized
    return canvas, scale, ox, oy


def detect(gray_or_bgr, conf=0.25, iou=0.45):
    image = gray_or_bgr
    if image.ndim == 2:
        image = cv2.cvtColor(np.clip(image, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    canvas, scale, ox, oy = letterbox(image)
    blob = cv2.dnn.blobFromImage(canvas, 1 / 255.0, (640, 640), swapRB=True, crop=False)
    n = net()
    n.setInput(blob)
    out = n.forward()                      # (1, 4 + classes, 8400)
    out = np.squeeze(out, 0).T             # (8400, 4 + classes)
    scores = out[:, 4:].max(axis=1)        # this model ships two classes
    keep = scores >= conf
    out, scores = out[keep], scores[keep]
    boxes = []
    for cx, cy, bw, bh in out[:, :4]:
        boxes.append([int(cx - bw / 2), int(cy - bh / 2), int(bw), int(bh)])
    if not boxes:
        return []
    idx = cv2.dnn.NMSBoxes(boxes, scores.tolist(), conf, iou)
    results = []
    for i in np.array(idx).flatten():
        cx, cy, bw, bh = out[i, :4]
        results.append(dict(
            x=float((cx - ox) / scale), y=float((cy - oy) / scale),
            w=float(bw / scale), h=float(bh / scale), score=float(scores[i])))
    results.sort(key=lambda r: -r["score"])
    return results
