# XY nozzle alignment for a Snapmaker A350 dual extruder

An OctoPrint plugin that measures the XY offset between nozzle 0 and nozzle 1
with an upward facing camera on the bed, then writes the result to the printer.

The camera is the Printables model
[XY Nozzle Alignment Camera](https://www.printables.com/model/1099576-xy-nozzle-alignment-camera)
(OV9726 module, four white LEDs, magnets, bed mounted). That model targets
Klipper and Axiscope and ships no software. This repository is the Marlin and
OctoPrint side.

## Put the camera anywhere and press one button

You give the plugin no coordinates, no pixel scale and no camera orientation. It
measures all of them, on every run.

Nothing about the camera is remembered between runs. The position, the focus
height, the pixel scale and the camera rotation are measured afresh each time.
Move the camera, or put it down somewhere else, and just run it again. No
setting changes.

A remembered position would be worse than useless. The routine lowers the nozzle
onto the camera, so a stale height would drive it into a camera that is no
longer there. The settings do show the last position found, but nothing reads
those values back to drive the machine.

### The pixel map

The plugin commands a known move in X, then a known move in Y, and watches how
far the nozzle travels in the image. Those two displacements are the columns of
a 2x2 matrix of pixels per millimetre.

One matrix carries the scale, the camera rotation, any mirroring and any skew.
The camera does not have to be square to the machine axes, and no setting
describes how it sits. The tests cover rotations of 0, 37, 91.5, −128 and 179
degrees with a mirrored image.

The map is rebuilt at every working height, because the scale changes with
distance from the lens.

### Finding the nozzle

The plugin nudges the toolhead and subtracts the two frames. Nothing else in the
picture moves, so the difference isolates the machine. That needs no model of
what a nozzle looks like, and it works whether the nozzle is darker or brighter
than its background.

There are two regimes, and the code reports which one it is in.

**Close to the camera** the nozzle is the only thing in frame. Its two positions
account for nearly everything that changed, so a compact blob really is the
nozzle. Only in this regime is a position trusted as a nozzle position.

**Far from the camera** the whole toolhead is in view and moves as one piece. No
blob is the nozzle. A textured toolhead throws off plenty of small round
difference blobs, and any one of them would pass a roundness test while being
nothing to do with the nozzle, so roundness is not the test. Coverage is: when
the best two blobs account for only a small share of everything that moved,
the measurement stays coarse.

In the coarse regime the displacement comes from the difference image alone.
Correlating the two frames does not work, because most of the picture is a
static background that pins the answer at zero however far the toolhead moved.
For a rigid shift `s` the difference is `D(x) = A(x) - A(x - s)`, whose
autocorrelation carries a strong negative dip at `+s` and `-s` and nothing from
the static background. The dip gives the distance and the two difference lobes
give the direction.

The pixel map is right in both regimes, because it only depends on how far the
picture shifted. That is what lets the search steer from a long way off.

### Finding the camera

"Find the camera" on the Nozzle Align tab does this.

1. Home, then rise to the search height.
2. Sweep the whole bed, one frame per point. Every frame shares the same static
   background and the toolhead appears in only a few of them, so the per-pixel
   median across the sweep is the empty view. Scoring each frame against that
   median needs no parking position and no reference shot. The cut comes from
   the spread of the sweep itself, so it does not depend on the lens or the
   lighting.
3. Measure at the best few spots, then steer the toolhead to the image centre. This is coarse: it works on the whole
   toolhead, because at that range the nozzle has not separated yet.
4. Step down, re-measuring and re-centring at each height. As the view narrows,
   the nozzle separates and the measurement becomes fine.
5. Measure focus as the variance of the Laplacian. Only heights where the nozzle
   separated can win, so the answer always comes from the nozzle itself.
6. Cut a picture of the nozzle at that height and keep it as a template.

The descent is guarded three ways.

- The image scale of the nozzle is inversely proportional to its distance from
  the lens, so `1 / scale` falls linearly as Z falls. Where that line crosses
  zero is the lens. The plugin fits that line as it descends and stops a set
  clearance above the answer.
- A hard floor from the settings.
- A limit on how much of the frame the nozzle may fill.

### Measuring the offset

1. Move nozzle 0 over the camera and build the map.
2. Drive nozzle 0 onto the target pixel, then read the machine position.
3. Select nozzle 1, repeat, and read the machine position again.
4. The difference between the two positions is the correction.

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

- It refuses to measure until the camera X, Y and Z are known.
- Every travel move goes through the configured safe Z.
- It refuses to write an offset further than `offset_limit_mm` from the
  expected value.
- The Stop button aborts between moves.

The search height must clear everything on the bed. That is the one number worth
checking before the first run.

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
