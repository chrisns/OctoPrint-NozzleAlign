# coding=utf-8
from nozzlealign_pkg.gcode import (
    format_offset_command,
    parse_hotend_offset,
    parse_position,
)


def test_parse_position_marlin():
    lines = [
        "X:150.00 Y:175.00 Z:10.00 E:0.00 Count X: 12000 Y:14000 Z:8000",
        "ok",
    ]
    assert parse_position(lines) == (150.0, 175.0, 10.0)


def test_parse_position_negative_and_spaces():
    lines = ["X: -1.25 Y:  0.50 Z:200.125 E:0.00"]
    assert parse_position(lines) == (-1.25, 0.5, 200.125)


def test_parse_position_missing():
    assert parse_position(["ok", "echo:busy: processing"]) is None


def test_parse_offset_from_m218_echo():
    lines = [
        "echo:Hotend offsets:",
        "  M218 T1 X26.000 Y0.100 Z-1.524",
        "ok",
    ]
    assert parse_hotend_offset(lines, tool=1) == (26.0, 0.1, -1.524)


def test_parse_offset_from_m503_dump():
    lines = [
        "echo:  G21    ; Units in mm",
        "echo:Hotend offsets:",
        "echo:  M218 T1 X25.900 Y-0.050 Z-1.500",
        "echo:Steps per unit:",
    ]
    assert parse_hotend_offset(lines, tool=1) == (25.9, -0.05, -1.5)


def test_parse_offset_ignores_other_tool():
    lines = ["  M218 T0 X0.000 Y0.000 Z0.000", "ok"]
    assert parse_hotend_offset(lines, tool=1) is None


def test_parse_offset_bare_triples():
    lines = [
        "Tool head offset:",
        "0.00,0.00,0.000",
        "26.00,0.00,-1.524",
        "ok",
    ]
    assert parse_hotend_offset(lines, tool=1) == (26.0, 0.0, -1.524)


def test_parse_offset_unreadable():
    assert parse_hotend_offset(["ok", "wait"], tool=1) is None


def test_format_offset_command():
    assert format_offset_command(1, 26.0, -0.125) == "M218 T1 X26.000 Y-0.125"
