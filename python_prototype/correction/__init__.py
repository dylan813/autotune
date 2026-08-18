"""Musical correction policy: scale snapping, smoothing, hysteresis, vibrato.

The correction stage turns a raw F0 contour into a *target* F0 trajectory.
The pitch-shift stage then maps detected -> target.
"""
