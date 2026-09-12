"""Launch descriptions importable as modules, so their choices can be tested.

A launch file under ``launch/`` is installed as data, not as a module, so a test
cannot import it. The capture launch's four decisions -- an unclaimed arm, a
gravity-free scene, the real driver in mock mode, and direct hand position --
are exactly the kind of thing that breaks silently, so the description lives
here and ``launch/sim_capture.launch.py`` is a one-line shim over it.
"""
