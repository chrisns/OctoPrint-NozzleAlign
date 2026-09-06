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

### What each axis does to the picture

On an A350 the three axes are not equivalent through a camera that sits on the
bed, and the difference is what makes the toolhead findable at all.

| Axis | What moves | What the picture does |
|---|---|---|
| X | the toolhead | the toolhead slides; the background is still |
| Y | the bed, and the camera with it | everything slides, near things further |
| Z | the toolhead | the toolhead expands about the point above the lens |

Depth comes from the Y move. It is a pure camera translation, so a point at
distance `D` from the lens shifts by `focal * dY / D` pixels. That segments the
toolhead from the room with no appearance model at all, and it survives blur and
glare, which an appearance model does not.

The Z move looks equivalent and is not. Every depth then depends on locating the
point straight above the lens, and patches near that point have almost no radius
to divide by, so their ratios are noise. It was tried and abandoned.

Two measurement traps, both found the hard way:

- Phase correlation over a tapered window stops being reliable much past a
  quarter of the window width. Beyond that it does not fail; it returns a
  saturated shift that looks like a plausible depth and is not one. Keep bed
  moves small, and check the move against `depth.max_reliable_move_mm`.
- A window covering the whole frame mixes the toolhead with the room behind it,
  and the scale it reports depends on how much of each it caught. Measure over
  the toolhead, where the window is a single depth.

### Measured on the machine

For the OV9726 at 1280x800, on 2026-09-05:

- 8.65 px/mm at Z150, from high confidence patches over the toolhead.
- Focal length about 1339 px.
- `1 / scale` is a straight line in Z crossing zero near the bed, so the lens
  plane sits at about Z-5.
- At Z60 that gives 19 px/mm, which is 0.05 mm per pixel. There is no need to go
  near the camera to measure a nozzle offset well.

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

### Checking that it is really the toolhead

Something moves in the picture whenever the machine moves, and it is not always
the toolhead. On the first live run here it was the bowden tube swinging
overhead. Two probe moves happily produced a pixel map from it, because any two
displacements define a matrix, and the routine then steered confidently on a
swinging tube.

So the map is checked with a third move, in a direction the first two did not
use. Its displacement follows from the map, and only something rigidly attached
to the toolhead will match the prediction. A tube, a reflection or a shadow will
not, and the run stops and says so rather than reporting a confident wrong
answer.

### Finding the camera

"Find the camera" on the Nozzle Align tab does this.

1. Home, then rise to the search height.
2. Stand at the middle of the bed and nudge the toolhead. The camera's field of
   view is wide, about 117 mm at 80 mm distance on this machine, so the toolhead
   is usually already in it. If nothing moves in the picture, work across the
   bed in a serpentine sweep until something does.

   Comparing still frames against the median of a sweep looked cheaper and was
   tried. It does not work. This camera sees the toolhead from most of the bed,
   so the median is not an empty view and the score has no clear peak. One probe
   move is the only test that settles it.
3. Steer the toolhead to the image centre. This is coarse: it works on the whole
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

## The offset sign, settled on the machine

The firmware does apply the stored offset over the serial link. Measured on
2026-09-05, at Z150 with the camera watching the toolhead:

- Command T0 to X, then T1 to `X + stored_offset`, and the two toolhead bodies
  land in the same place to **0.07 mm**.
- Add 1.000 mm to the stored X offset with `M218`, and the point where the
  bodies line up moves with it, in the same direction.

So the convention is: **the stored offset is how much further along X you
command tool 1 to put its body where tool 0's body was.** The plugin still
reports both candidates and still asks you to confirm by re-measuring, because
that check costs one run and catches a firmware change.

Two caveats from the same measurements. The 1.000 mm change read back as about
1.35 mm through the camera, so the pixel scale at that spot is not yet good to
better than a third. And this test aligns the toolhead *bodies*, which measures
the offset the firmware is applying, not the true separation of the nozzles. The
true separation needs a view of the nozzle tips.

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

## Knowing when the answer is not the nozzle

Two checks decide whether a reading may be believed, and both were added
because the routine had already returned a confident wrong answer without them.

**The height must match the commanded Z.** Z is defined as the height of the
nozzle tip above the bed, so when the tip is the nearest thing the camera sees,
it must measure back as Z. Everything else on the toolhead sits higher and reads
further away. At Z150 the body reads 166 to 172 mm.

**The reading must repeat.** A single measurement cannot tell a feature from a
lucky patch of noise. One scan point returned 150.8 mm, which is exactly what a
nozzle tip would read. Five repeats at that same point returned 170.8, 170.6,
172.0, 169.4 and 171.9. The 150.8 was noise, and without repeats it would have
been reported as the answer.

Depth itself repeats to about 1 mm, so the measurement is precise. What it is
measuring is simply not the tip.

## The camera has to be aimed at the nozzle

Everything above about depth and parallax was built to find a nozzle in a
picture that did not really contain one. Once the camera points straight up at a
nozzle and is focused on it, the problem changes completely: a nozzle is then
the most circular thing in the frame, and its orifice is a clean dark circle at
the centre of a bright brass cone.

`nozzle.py` does that. It takes the strongest circles, refines each centre by
fitting the rim to sub-pixel, and uses the known spacing between the two nozzles
as a free check that it found the right two. The fit residual is a quality
score: a clean rim fits tightly, a rim caked in burnt filament does not.

Both the hex body and the orifice are detected, and they come out concentric.
The hex is the coarse target and the orifice the fine one.

Two other detectors were tried on the same frames and found nothing at all.
TAXY's YOLOv8 nozzle model wants a much closer view, even on upscaled crops. The
TAMV `SimpleBlobDetector` recipe wants a dark blob on a light field, which is
not what this lens and lighting produce.

### Setting the camera up

1. Open the live stream and put a nozzle over the camera.
2. Focus on the nozzle, not on the machine behind it.
3. Watch the clipping. A rim blown out to white cannot be located precisely
   however sharp the focus, so less light beats more focus once it saturates.
4. Fit matching nozzles. A 0.8 and a 0.4 image differently, so a detector tuned
   to one is worse on the other.
5. Save the working height. Nothing goes below it.

## The measurement, on the machine

Each nozzle is steered until its bore sits on the same pixel, and the machine
position is recorded. The difference between the two positions is the
separation. Nothing depends on knowing the pixel scale accurately, because both
nozzles finish on the *same* pixel, so lens distortion cancels out.

Three runs on 2026-09-06:

| run | X | Y |
|---|---|---|
| 1 | 25.0696 | 0.3540 |
| 2 | 25.0734 | 0.3489 |
| 3 | 25.0720 | 0.3559 |
| mean | **25.0717** | **0.3529** |

Repeatable to 1.6 microns in X and 3.0 in Y. The steering converged to under
three microns of the target pixel every time.

The machine stored X 25.2000 Y 0.3200, so it was out by −0.129 mm in X.

### Two things that had to be right

**Backlash.** A probe measured after a reversal comes up short, and the pixel
map then came out nearly singular: 12.6 px/mm with its two columns 0.98 aligned.
Steering with that asked for a 118 px move in the wrong direction. Approaching
every measurement from the same side gives 73 px/mm with square columns from the
identical probe. `geometry.backlash_free` and `geometry.validate_map` cover both
halves of that.

**Telling the bore from everything else.** Brightness alone picked a glint off
the heater block 260 px from the nozzle. Darkness alone picked the burnt cone.
The bore is a dark hole inside a bright collar, and only the pair identifies it.
`nozzle.find_bore` scores that pattern and repeats to 0.01 px.

## Status, 2026-09-06

Working and verified on the machine:

- Both camera streams, after three separate faults in the go2rtc path.
- The plugin loads, and its serial layer reads and writes the hotend offset.
  `M218` writes, persists and reads back on this firmware.
- The camera model: 8.65 px/mm at Z150, focal length about 1339 px, lens plane
  near bed level. Working at Z60 gives 0.05 mm per pixel.
- Depth from a bed move segments the toolhead from the room cleanly.
- Body alignment repeats to 0.07 mm, which sets the noise floor.

Blocked on the camera itself:

The camera views the toolhead from behind and below, not straight up at the
nozzles. The nozzle tips are largely hidden by the extruder body, the lens is
focused for long range so a close view is blurred, and about 39 percent of the
central frame clips at Z60. Every method tried finds the nearest part of the
toolhead, which is not a nozzle tip.

The camera is roughly under X151 Y304 on the bed. Aim it straight up at a
nozzle, focus it for about 40 mm, and the rest of the chain is ready.
