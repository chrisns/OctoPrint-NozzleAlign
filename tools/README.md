# Bench tools

These drive the real printer from a workstation, outside OctoPrint, so a method
can be tried and measured before it goes into the plugin. They use the
plugin's own modules for the detector, the focus sweep and the pixel map.

All of them refuse to move unless OctoPrint reports the printer connected, and
`rig.moveto` refuses any Z outside `FLOOR_Z` to `CEIL_Z`. The floor is 20 mm.
The lens sits at about Z12 on the standard mount.

| File | What it does |
|---|---|
| `rig.py` | printer and camera access through the OctoPrint and go2rtc HTTP APIs |
| `measure.py` | the full offset measurement, with `--repeat N` for the spread; writes nothing |

`rig.py` talks to the print PC through `curl` rather than `requests`. Python's
own sockets could not reach the host from the sandbox this was written in, and
`curl` could. It reads the OctoPrint API key from `tools/.opkey`, which is not
committed.
