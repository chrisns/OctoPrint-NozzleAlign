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
