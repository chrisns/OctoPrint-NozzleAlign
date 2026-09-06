# coding=utf-8
"""Every setting the plugin reads, with the value measured on the machine.

This lives apart from the plugin class so the tests can build a real config
without OctoPrint installed, and so a setting cannot be read by the routine
without being declared here.
"""

DEFAULTS = dict(
    # camera
    snapshot_url="http://127.0.0.1:1984/api/frame.jpeg?src=nozzle_cam",
    http_timeout=20.0,
    frame_average=4,
    # Where the active T0 nozzle last sat over the lens, and the height at
    # which its bore was sharpest. A starting hint for the next run, which
    # searches around it and sweeps the focus again. Measured 2026-09-06.
    camera_x=175.27,
    camera_y=284.91,
    camera_z=32.0,
    # Heights. min_z is a hard floor no move may cross; the lens sits at
    # about Z12 on this mount and the bore is sharp near Z32.
    min_z=20.0,
    safe_z=90.0,
    home_first=True,
    # motion
    feedrate=3000,
    fine_feedrate=600,
    z_feedrate=300,
    move_timeout=90.0,
    home_timeout=240.0,
    tool_settle_s=2.0,
    settle_s=1.0,
    backlash_backoff_mm=1.0,
    # finding the bore around the stored point, at the working height
    search_span_mm=12.0,
    search_step_mm=7.0,
    search_radius_px=600,
    # focus sweep
    focus_span_mm=4.0,
    focus_step_mm=1.0,
    focus_template_px=90,
    focus_window_px=80,
    focus_peak_ratio=0.6,
    # bore detector, sized for about 75 px/mm
    bore_search_radius_px=430,
    bore_inner_r=12,
    bore_ring_lo=18,
    bore_ring_hi=34,
    bore_core_r=5,
    bore_min_score=85.0,
    bore_track_score=55.0,
    track_search_px=260,
    track_lock_px=50,
    track_min_match=0.7,
    # pixel map
    probe_distance=1.5,
    map_attempts=3,
    map_min_scale=30.0,
    map_max_scale=130.0,
    map_max_cos=0.35,
    # closed loop
    tolerance_mm=0.008,
    max_passes=8,
    max_correction_mm=4.0,
    active_check_mm=3.0,
    target_x_px=None,
    target_y_px=None,
    # the firmware's own acceptance band: the toolhead resets an X offset
    # outside 26 +/- 1.2 mm, or a Y offset outside 0 +/- 1.2 mm, to default
    nominal_offset_x=26.0,
    nominal_offset_y=0.0,
    offset_limit_mm=1.2,
    save_to_eeprom=True,
)
