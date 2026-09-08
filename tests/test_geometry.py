# coding=utf-8
import math
import numpy as np
import pytest

from nozzlealign_pkg import settings
from nozzlealign_pkg.settings import DEFAULTS

from nozzlealign_pkg.geometry import (
    GeometryError,
    build_pixel_map,
    pixel_error_to_mm,
    pixels_per_mm,
    rotation_degrees,
)


def synthetic(scale, rotation_deg, flip_y=True):
    """Pixels per mm matrix for a camera at a known scale and rotation."""
    angle = math.radians(rotation_deg)
    matrix = scale * np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )
    if flip_y:
        matrix = np.array([[1.0, 0.0], [0.0, -1.0]]).dot(matrix)
    return matrix


def test_build_map_recovers_scale_and_rotation():
    truth = synthetic(scale=80.0, rotation_deg=0.0, flip_y=False)
    origin = np.array([640.0, 400.0])
    distance = 1.0
    matrix = build_pixel_map(
        origin,
        origin + truth.dot([distance, 0.0]),
        origin + truth.dot([0.0, distance]),
        distance,
    )
    assert np.allclose(matrix, truth)
    assert pixels_per_mm(matrix) == pytest.approx(80.0)
    assert rotation_degrees(matrix) == pytest.approx(0.0)


def test_build_map_handles_rotation_and_mirroring():
    truth = synthetic(scale=52.5, rotation_deg=30.0, flip_y=True)
    origin = np.array([100.0, 200.0])
    matrix = build_pixel_map(
        origin, origin + truth.dot([2.0, 0.0]), origin + truth.dot([0.0, 2.0]), 2.0
    )
    assert np.allclose(matrix, truth)
    assert pixels_per_mm(matrix) == pytest.approx(52.5)


def test_pixel_error_round_trips():
    truth = synthetic(scale=64.0, rotation_deg=12.0, flip_y=True)
    origin = np.array([640.0, 400.0])
    matrix = build_pixel_map(
        origin, origin + truth.dot([1.0, 0.0]), origin + truth.dot([0.0, 1.0]), 1.0
    )
    target = np.array([640.0, 400.0])
    current = target + truth.dot([0.37, -0.22])
    dx, dy = pixel_error_to_mm(matrix, current, target)
    assert dx == pytest.approx(-0.37, abs=1e-9)
    assert dy == pytest.approx(0.22, abs=1e-9)


def test_zero_distance_is_rejected():
    with pytest.raises(GeometryError):
        build_pixel_map([0, 0], [1, 1], [2, 2], 0)


def test_parallel_moves_are_rejected():
    with pytest.raises(GeometryError):
        build_pixel_map([0.0, 0.0], [10.0, 0.0], [20.0, 0.0], 1.0)


# -- guarding against a map that only looks like a map ----------------------


def test_a_good_map_passes():
    from nozzlealign_pkg.geometry import validate_map
    # measured on the machine with the approach controlled for backlash
    ok, scale, cosine = validate_map([[73.34, -5.48], [7.65, 73.24]])
    assert ok
    assert scale == pytest.approx(73.6, abs=1.0)
    assert cosine < 0.05


def test_a_backlash_wrecked_map_is_rejected():
    """This one was measured for real, and steering with it moved 118 px wrong."""
    from nozzlealign_pkg.geometry import validate_map
    ok, scale, cosine = validate_map([[9.98, -35.58], [-11.66, 35.42]])
    assert not ok
    assert cosine > 0.9


def test_an_implausible_scale_is_rejected():
    from nozzlealign_pkg.geometry import validate_map
    assert not validate_map([[213.0, 0.0], [0.0, 213.0]])[0]
    assert not validate_map([[5.0, 0.0], [0.0, 5.0]])[0]


def test_the_backlash_free_pair_arrives_at_the_right_place():
    from nozzlealign_pkg.geometry import backlash_free
    first, second = backlash_free(0.4, -0.3, backoff=1.0)
    assert first[0] + second[0] == pytest.approx(0.4)
    assert first[1] + second[1] == pytest.approx(-0.3)
    # and the last leg is always the same direction, whatever the target
    for dx, dy in ((5.0, 5.0), (-5.0, -5.0), (0.0, 0.0)):
        assert backlash_free(dx, dy)[1] == (1.0, 1.0)


def _bed(**overrides):
    box = dict(bed_x_min=25.0, bed_x_max=295.0, bed_y_min=25.0, bed_y_max=325.0)
    box.update(overrides)
    return box


def test_a_search_area_inside_the_printer_profile_is_accepted():
    profile = dict(volume=dict(x=320.0, y=350.0, origin="lowerleft"))
    assert settings.validate_bed(_bed(), profile)["bed_x_max"] == 295.0


def test_a_search_area_past_the_bed_is_refused():
    """These four numbers go straight into G1 moves. Past the bed is a crash."""
    profile = dict(volume=dict(x=220.0, y=220.0, origin="lowerleft"))
    with pytest.raises(settings.SettingsError) as error:
        settings.validate_bed(_bed(), profile)
    assert "bed_x_max is 295.0" in str(error.value)
    assert "0.0 to 220.0" in str(error.value)


def test_a_centred_origin_allows_negative_coordinates():
    profile = dict(volume=dict(x=300.0, y=300.0, origin="center"))
    settings.validate_bed(_bed(bed_x_min=-100.0, bed_y_min=-100.0,
                               bed_x_max=100.0, bed_y_max=100.0), profile)
    with pytest.raises(settings.SettingsError):
        settings.validate_bed(_bed(bed_x_min=-200.0), profile)


def test_a_minimum_above_its_maximum_is_refused():
    with pytest.raises(settings.SettingsError) as error:
        settings.validate_bed(_bed(bed_x_min=300.0))
    assert "has to be the smaller one" in str(error.value)


def test_no_profile_still_checks_what_it_can():
    assert settings.validate_bed(_bed(), None)
    with pytest.raises(settings.SettingsError):
        settings.validate_bed(_bed(bed_y_max=0.0), None)


def test_the_defaults_pass_against_the_machine_this_was_written_for():
    profile = dict(volume=dict(x=320.0, y=350.0, origin="lowerleft"))
    settings.validate_bed(DEFAULTS, profile)
    settings.validate_heights(DEFAULTS)


def test_a_floor_below_the_lens_is_refused():
    """min_z is the hard floor. Below it the nozzle reaches the camera."""
    with pytest.raises(settings.SettingsError):
        settings.validate_heights(dict(DEFAULTS, min_z=0.0))
    with pytest.raises(settings.SettingsError) as error:
        settings.validate_heights(dict(DEFAULTS, safe_z=10.0))
    assert "has to be the higher one" in str(error.value)


def test_a_camera_height_outside_the_band_is_refused():
    with pytest.raises(settings.SettingsError) as error:
        settings.validate_heights(dict(DEFAULTS, camera_z=5.0))
    assert "outside" in str(error.value)


def test_every_setting_can_be_changed_from_the_interface():
    """A setting only reachable by editing config.yaml by hand is not a setting."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    pane = (root / "octoprint_nozzlealign" / "templates"
            / "nozzlealign_settings.jinja2").read_text()
    missing = sorted(name for name in DEFAULTS if name not in pane)
    assert missing == []


def test_no_setting_is_declared_but_never_read():
    """A setting nobody reads is a promise the plugin does not keep."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    package = root / "octoprint_nozzlealign"
    body = ""
    for path in package.glob("*.py"):
        if path.name != "settings.py":
            body += path.read_text()
    for folder in ("templates", "static/js"):
        for path in (package / folder).rglob("*"):
            if path.is_file():
                body += path.read_text()
    unread = sorted(name for name in DEFAULTS if name not in body)
    assert unread == []
