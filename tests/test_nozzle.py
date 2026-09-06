# coding=utf-8
"""The bore detector, which is what worked on the machine."""

import numpy as np
import pytest

from nozzlealign_pkg import nozzle

cv2 = pytest.importorskip("cv2")


# -- the bore detector, which is what worked on the machine -----------------


def bore_scene(centre=(594.0, 511.0), bore_r=17.0, collar_r=38.0,
               glint=(858.0, 535.0), size=(800, 1280)):
    """A dark bore in a bright collar, on a dirty cone, with a glint elsewhere.

    The glint is the trap. It is the brightest thing in the frame, and a
    brightness based detector picked it over the bore on the real machine.
    """
    import cv2
    height, width = size
    rng = np.random.default_rng(13)
    image = 90.0 + rng.normal(0.0, 35.0, (height, width)).astype(np.float32)
    ys, xs = np.mgrid[0:height, 0:width]
    cone = np.hypot(xs - centre[0], ys - centre[1]) < 230
    image[cone] = 110.0 + rng.normal(0.0, 45.0, int(cone.sum()))   # burnt cone
    radius = np.hypot(xs - centre[0], ys - centre[1])
    image[radius <= collar_r] = 235.0                              # bright collar
    image[radius <= bore_r] = 25.0                                 # the bore
    image[np.hypot(xs - glint[0], ys - glint[1]) <= 16] = 255.0    # the glint
    return cv2.GaussianBlur(image, (5, 5), 1)


def test_the_bore_is_found_and_not_the_glint():
    found = nozzle.find_bore(bore_scene(), centre=(640, 400))
    assert found["x"] == pytest.approx(594.0, abs=4.0)
    assert found["y"] == pytest.approx(511.0, abs=4.0)


def test_a_glint_alone_does_not_look_like_a_bore():
    """Brightness alone is not the signature; the dark core is half of it."""
    import cv2
    plain = np.full((800, 1280), 120.0, dtype=np.float32)
    ys, xs = np.mgrid[0:800, 0:1280]
    plain[np.hypot(xs - 700, ys - 400) <= 20] = 255.0
    plain = cv2.GaussianBlur(plain, (5, 5), 1)
    with_bore = bore_scene()
    assert nozzle.find_bore(with_bore, (640, 400))["score"] > \
           nozzle.find_bore(plain, (640, 400))["score"] * 2


def test_the_bore_is_found_wherever_it_sits():
    for want in ((400.0, 300.0), (800.0, 600.0), (640.0, 400.0)):
        found = nozzle.find_bore(bore_scene(centre=want, glint=(200.0, 700.0)),
                                 centre=(640, 400))
        assert found["x"] == pytest.approx(want[0], abs=5.0)
        assert found["y"] == pytest.approx(want[1], abs=5.0)


def test_the_search_radius_is_respected():
    scene = bore_scene(centre=(1150.0, 700.0), glint=(300.0, 200.0))
    found = nozzle.find_bore(scene, centre=(300, 200), search_radius=120)
    assert np.hypot(found["x"] - 1150.0, found["y"] - 700.0) > 200


def test_a_lit_bore_is_found_as_well_as_a_dark_one():
    """With the lens focused, the bore shows a bright spot in its middle."""
    scene = bore_scene(centre=(594.0, 511.0))
    ys, xs = np.mgrid[0:scene.shape[0], 0:scene.shape[1]]
    scene[np.hypot(xs - 594.0, ys - 511.0) <= 4] = 240.0
    found = nozzle.find_bore(scene, centre=(640, 400))
    assert abs(found["x"] - 594.0) < 1.5
    assert abs(found["y"] - 511.0) < 1.5
