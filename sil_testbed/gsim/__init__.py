"""
gsim -- software-in-the-loop (SIL) governance testbed for AI-driven IoT.

A physics-based digital twin of a smart-building HVAC zone, closed around a real
model trained on real sensor data and a real declarative governance plane.

TERMINOLOGY. This is SOFTWARE-IN-THE-LOOP (SIL), not hardware-in-the-loop (HIL), and the
distinction is not pedantic. In HIL, real sensor and actuator hardware sits in the loop and
the plant is simulated in real time. In SIL, the real controller SOFTWARE runs against a
plant MODEL. We have the latter: real governance code, real recorded sensor data, a real
learned model, and a simulated room. No physical hardware was operated.

We called this "HIL" throughout its development, in filenames, in banners, and in prose. It
was wrong, and it was wrong in the direction that flatters us -- a reviewer from the
cyber-physical community would have read it as an overclaim, which is precisely the charge
the paper is trying to answer. The Raspberry Pi backend in gsim/hal.py is the HIL rung; it
is released, parity-checked, and NOT RUN.

Scope, stated once and honestly: no physical hardware was operated in this study.
Every physical quantity reported here is produced by the digital twin in `plant.py`,
whose model and parameters are given in full so they can be challenged. The hardware
backends in `hal.py` are released and implement the same interface, so the hardware
evaluation is an interface swap rather than a rewrite -- but it has not been run, and
nothing in this package should be read as claiming otherwise.
"""

__version__ = "1.0.0"
