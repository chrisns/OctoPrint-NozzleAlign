# Bench tools

These drive the real printer from a workstation, outside OctoPrint, so a method
can be tried and measured before it goes into the plugin. They were how the
camera model in the main README was measured.

All of them refuse to move unless OctoPrint reports the printer connected, and
`rig.moveto` refuses any Z outside a safe band. The camera sits on the bed and
its height is not known, so the floor starts high and stays there.

| File | What it does |
|---|---|
| `rig.py` | printer and camera access, plus the window shift and moving blob helpers |
| `depthmap.py` | depth from a bed move, drawn over the frame: red near, blue far |
| `align.py` | steers the live nozzle onto a target pixel and records where each tool had to stand |
| `taxy.py` | runs TAXY's YOLOv8 nozzle detector through `cv2.dnn` |

`rig.py` talks to the print PC through `curl` rather than `requests`. Python's
own sockets could not reach the host from the sandbox this was written in, and
`curl` could.

`taxy.py` needs `taxy_nozzle.onnx`, which is not committed. Fetch it with:

```
curl -L -o taxy_nozzle.onnx https://github.com/PrintStructor/TAXY/raw/main/server/best.onnx
```

That model is GPL-3.0, from https://github.com/PrintStructor/TAXY. It is a
YOLOv8-nano trained on one class for upward facing nozzle cameras. It found
nothing in this rig's frames, which is expected: it wants a close, focused view
of a nozzle, and this camera gives neither yet.
