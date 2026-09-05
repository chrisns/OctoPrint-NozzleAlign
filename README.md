# XY nozzle alignment for a Snapmaker A350 dual extruder

An OctoPrint plugin that measures the XY offset between nozzle 0 and nozzle 1
with an upward facing camera on the bed, then writes the result to the printer.

The camera is the Printables model
[XY Nozzle Alignment Camera](https://www.printables.com/model/1099576-xy-nozzle-alignment-camera)
(OV9726 module, four white LEDs, magnets, bed mounted). That model targets
Klipper and Axiscope and ships no software. This repository is the Marlin and
OctoPrint side.

## How it measures

The plugin never asks you for a pixel scale. It measures one.

1. It moves nozzle 0 over the camera.
2. It commands a known move in X, then a known move in Y, and watches how far
   the tip travels in the image.
3. Those two displacements give a 2x2 matrix of pixels per millimetre. The
   matrix captures scale, camera rotation and mirroring in one step.
4. It drives nozzle 0 onto the target pixel, then reads the machine position.
5. It selects nozzle 1, repeats, and reads the machine position again.
6. The difference between the two positions is the correction.

The loop repeats until the residual falls below the tolerance, which defaults to
0.005 mm.

## The offset sign

Marlin stores the tool offset in `hotend_offset[axis][tool]`. Snapmaker does not
document the sign for its dual extrusion toolhead, and the firmware already
compensates for the stored value while you measure. The plugin therefore reports
both candidates and asks you to write one.

Write a candidate, run the measurement again, and read the correction.

- The correction falls towards zero: the sign was right.
- The correction roughly doubles: write the other candidate.

Record which one worked. This is a property of the firmware, not of your
machine, so you only have to settle it once.

## Writing to the firmware

`Snapmaker2-Controller` implements `M218`. For tool 1 it calls
`ModuleCtrlSaveHotendOffset()`, which stores the value in the toolhead module
over CAN. That persists at once, without `M500`. The plugin sends `M500` as well
so the controller EEPROM agrees.

Snapmaker lists `M218` as unsupported and says it may be removed in a later
firmware. The plugin therefore always reads the value back and tells you whether
the read-back matches. If it does not, use the reported numbers in your slicer
instead.

The touchscreen "XY Offset Calibration" wizard writes the same array, so running
that wizard afterwards overwrites what this plugin stored.

## Safety

The routine lowers the nozzle onto a camera that sits on the bed.

- It refuses to start until the camera X, Y and Z are set.
- Every travel move goes through the configured safe Z.
- It refuses to write an offset further than `offset_limit_mm` from the
  expected value.
- The Stop button aborts between moves.

Jog the nozzle over the camera by hand first and copy the coordinates into the
settings. Do not guess them.

## Install

On the print PC, using the OctoPrint interpreter:

```
C:\OctoPrint\WPy64-31050\python-3.10.5.amd64\python.exe -m pip install numpy opencv-python-headless
C:\OctoPrint\WPy64-31050\python-3.10.5.amd64\python.exe -m pip install -e C:\OctoPrint\plugins-src\nozzlealign
net stop OctoPrint5000 & net start OctoPrint5000
```

## The camera stream

`printpc/go2rtc.yaml` and `printpc/nozzlecam.bat` are the working go2rtc
configuration. Read the comments in the yaml before you change it. Three
separate faults made the naive configuration produce a valid JPEG containing no
picture, and each fix is load bearing.

Check a stream like this. A working frame has a standard deviation well above
20. A broken one sits at mean 128 with a standard deviation near 4.

```
curl -s -o f.jpg "http://printpc.cns.me:1984/api/frame.jpeg?src=nozzle_cam"
```

## Tests

The tests run on any machine. They need no printer and no OctoPrint.

```
python3 -m venv .venv
.venv/bin/pip install numpy opencv-python-headless pillow requests pytest
.venv/bin/python -m pytest tests -q
```

`tests/fakes.py` simulates a printer and a rotated, mirrored camera, so
`tests/test_routine.py` exercises the whole closed loop and checks that a known
tool error comes back out.

## Prior art

Chen and Cai, "Automatic Spatial Calibration for Dual-nozzle Extrusion-based 3D
Printers", Manufacturing Letters 44 (2025) 884-892. They use HSV segmentation, a
Hough circle transform and a right-angle prism, with a fixed 0.045 mm per pixel
scale measured by hand in ImageJ. This plugin measures the scale instead, and
covers XY only, because the camera looks straight up with no prism.
