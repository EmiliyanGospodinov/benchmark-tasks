"""Overhead-camera check of the shaft flange: is the yellow collar still there, gone, or can it not be seen?

Pure NumPy on the native 320x176 ``top_cam`` RGB frame; no simulator state. The camera and the fixture are fixed,
so the flange of the shaft appears in a fixed image window (rows 97-110, columns 141-158, measured on the recorded
native frames). The window is read only when most of the grey flange is visible: an arm or the tool over the shaft
makes the window unreadable, and unreadable never counts as "gone".

Limits (measured on recorded frames): the overhead view is about 5.5 mm per pixel and both hands cover the shaft
while the screw is turned, so this check cannot see the collar's 28 mm rise off the shaft. It can see, once the
tool has been carried away, whether the collar left with it or stayed on the shaft.
"""
from __future__ import annotations

import numpy as np

WINDOW = (slice(97, 111), slice(141, 159))  # 14 x 18 px around the flange and shaft
MIN_GREY = 130  # of 252 window pixels: at least about half of the flange must be visible to read the window
MAX_ORANGE_GONE = 1
MIN_ORANGE_ON = 8


def classes(img):
    """Orange (collar, knob) and neutral grey (flange) pixel masks; thresholds measured on the native frames."""
    px = np.asarray(img).astype(int)
    r, g, b = px[..., 0], px[..., 1], px[..., 2]
    orange = (r >= 60) & (r > 1.05 * g) & (g > 1.4 * b) & (r - b > 40)
    hi, lo = px.max(axis=-1), px.min(axis=-1)
    grey = ((hi - lo) <= 18) & (lo >= 70) & (hi <= 220) & ~orange
    return orange, grey


def read(img):
    """'on', 'gone' or None (unreadable) for the collar on the shaft flange."""
    if img is None:
        return None
    img = np.asarray(img)
    if img.ndim != 3 or img.shape[0] < 111 or img.shape[1] < 159 or img.shape[2] < 3:
        return None
    orange, grey = classes(img[WINDOW][..., :3])
    n_orange, n_grey = int(orange.sum()), int(grey.sum())
    if n_orange >= MIN_ORANGE_ON:
        return "on"
    if n_orange <= MAX_ORANGE_GONE and n_grey >= MIN_GREY:
        return "gone"
    return None


class ClearConfirm:
    """The collar counts as gone from the shaft only after ``need`` consecutive readable 'gone' frames."""

    def __init__(self, need=5):
        self.need, self.run = need, 0

    def update(self, reading):
        self.run = self.run + 1 if reading == "gone" else 0
        return self.run >= self.need
