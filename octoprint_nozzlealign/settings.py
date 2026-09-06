# coding=utf-8
"""Every setting the plugin reads, with the value measured on the machine.

This lives apart from the plugin class so the tests can build a real config
without OctoPrint installed, and so a setting cannot be read by the routine
without being declared here.
"""

DEFAULTS = dict(
    # camera
    snapshot_url="http://127.0.0.1:1984/api/frame.jpeg?src=nozzle_cam",
    stream_url="http://127.0.0.1:1984/api/stream.mjpeg?src=nozzle_cam",
    http_timeout=20.0,
    frame_average=3,
    # Where the active T0 nozzle last sat over the lens, for information
    # only: every run sweeps the bed for the camera from scratch. camera_z
    # is the height at which the bore was last sharpest; the focus sweep
    # starts there. Measured 2026-09-06.
    camera_x=None,
    camera_y=None,
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
    settle_s=0.4,
    backlash_backoff_mm=1.0,
    # finding the camera: at the search height the field of view is about
    # 75 by 47 mm. The head sweeps X back and forth in steps, row by row in
    # Y, and each frame is compared with the one before; only the toolhead
    # can change the picture. Then a grid of X nudges around the first hit
    # finds the middle of the toolhead, and a ring at the working height
    # finds the bore.
    search_z=90.0,
    closein_z=90.0,
    bed_x_min=25.0,
    bed_x_max=295.0,
    bed_y_min=25.0,
    bed_y_max=325.0,
    bed_row_mm=50.0,
    sweep_step_mm=20.0,
    sweep_feedrate=4000,
    sweep_overrun_s=0.8,
    sweep_lag_s=0.25,
    closein_max_steps=4,
    closein_min_scale=6.0,
    closein_min_response=0.05,
    closein_done_mm=2.0,
    closein_max_move_mm=40.0,
    motion_nudge_mm=4.0,
    motion_threshold=8.0,
    motion_min_fraction=0.03,
    motion_min_blob=0.2,
    bed_search_span_mm=25.0,
    search_step_mm=7.0,
    search_radius_px=380,
    # focus sweep
    focus_span_mm=4.0,
    focus_step_mm=2.0,
    focus_fine_step_mm=0.5,
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
    max_correction_mm=6.0,
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
