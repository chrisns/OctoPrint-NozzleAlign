# coding=utf-8
import numpy as np
import pytest

from nozzlealign_pkg import vision



def make_frame(cx, cy, radius=40.0, size=(800, 1280), background=210.0, ink=25.0):
    height, width = size
    image = np.full((height, width), background, dtype=np.float32)
    ys, xs = np.mgrid[0:height, 0:width]
    image[(xs - cx) ** 2 + (ys - cy) ** 2 <= radius ** 2] = ink
    return image


def test_frame_health_spots_a_dead_stream():
    dead = np.full((800, 1280), 128.0, dtype=np.float32)
    dead += np.random.default_rng(1).normal(0.0, 2.0, dead.shape)
    mean, deviation = vision.frame_health(dead)
    assert mean == pytest.approx(128.0, abs=1.0)
    assert deviation < 5.0
    assert vision.is_blank(dead)


def test_frame_health_accepts_a_real_picture():
    frame = make_frame(640, 400)
    assert not vision.is_blank(frame)


class _Reply(object):
    def __init__(self, content):
        self.content = content

    def raise_for_status(self):
        pass


def test_an_empty_answer_means_the_camera_is_unplugged(monkeypatch):
    """go2rtc answers a missing device with 200 and no body, which reads like success."""
    monkeypatch.setattr(vision.requests, "get", lambda *a, **k: _Reply(b""))
    present, why = vision.camera_present("http://camera/frame.jpeg")
    assert present is False
    assert "not plugged in" in why


def test_a_real_frame_says_the_camera_is_there(monkeypatch):
    monkeypatch.setattr(vision.requests, "get", lambda *a, **k: _Reply(b"x" * 40000))
    assert vision.camera_present("http://camera/frame.jpeg") == (True, "")


def test_a_missing_camera_raises_its_own_error(monkeypatch):
    """Told apart from a broken stream, because the fix is different."""
    monkeypatch.setattr(vision.requests, "get", lambda *a, **k: _Reply(b""))
    with pytest.raises(vision.CameraMissing) as error:
        vision.fetch_frame("http://camera/frame.jpeg", attempts=2, retry_delay=0.0)
    assert "not plugged in" in str(error.value)


def test_a_short_answer_is_a_broken_stream_not_a_missing_camera(monkeypatch):
    monkeypatch.setattr(vision.requests, "get", lambda *a, **k: _Reply(b"x" * 50))
    with pytest.raises(vision.CaptureError) as error:
        vision.fetch_frame("http://camera/frame.jpeg", attempts=2, retry_delay=0.0)
    assert not isinstance(error.value, vision.CameraMissing)


def test_an_unreachable_camera_is_reported_not_raised(monkeypatch):
    def boom(*args, **kwargs):
        raise vision.requests.RequestException("no route to host")
    monkeypatch.setattr(vision.requests, "get", boom)
    present, why = vision.camera_present("http://camera/frame.jpeg")
    assert present is False
    assert "did not answer" in why
