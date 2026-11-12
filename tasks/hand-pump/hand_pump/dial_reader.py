"""Read the test-chamber dial from the native 320x176 overhead RGB observation (NumPy only).

Pure image processing, no simulator state: the dial is located by its own printed markings (red target mark,
green acceptance-band ticks) and the white face; the black needle is read in an inner ring of the face (clear
of the black rim ticks and the red mark). The decision compares the needle's sweep from its resting (zero)
position with the sweep to the red target mark: the scale direction and the target come from the dial itself.
"""
from __future__ import annotations

import math

import numpy as np


def _masks(img):
    """Colour classes measured on the native frames (final-004): the dial face is neutral white (~234,235,236),
    the table warm (~232,220,191), the thin needle a neutral mid-grey (~173,174,176), the band ticks green."""
    px = img.astype(int)
    r, g, b = px[..., 0], px[..., 1], px[..., 2]
    hi, lo = px.max(axis=-1), px.min(axis=-1)
    neutral = (hi - lo) <= 14
    face = neutral & (lo >= 215)
    needle = neutral & (hi >= 110) & (hi <= 212)
    green = (g > 150) & (r < 120) & (g > r + 60) & (g > b + 50)
    red = (r > 140) & (g < 90) & (b < 90)
    return red, green, face, needle


def locate(img, search_r=16):
    """Dial centre and radius (px) from the green band ticks and the white face around them; None if not found."""
    _, green, white, _ = _masks(img)
    ys, xs = np.nonzero(green)
    if len(xs) < 2:
        return None
    mx, my = np.median(xs), np.median(ys)
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    near = (xx - mx) ** 2 + (yy - my) ** 2 <= search_r ** 2
    fy, fx = np.nonzero(white & near)
    if len(fx) < 20:
        return None
    # mean-shift on the white face from the band: re-centre on the white pixels within 12 px until it settles
    cx, cy = float(mx), float(my)
    wy, wx = np.nonzero(white)
    for _ in range(20):
        sel = (wx - cx) ** 2 + (wy - cy) ** 2 <= 10 ** 2
        if sel.sum() < 20:
            return None
        nx_, ny_ = wx[sel].mean(), wy[sel].mean()
        if abs(nx_ - cx) < 0.05 and abs(ny_ - cy) < 0.05:
            break
        cx, cy = nx_, ny_
    sel = (wx - cx) ** 2 + (wy - cy) ** 2 <= 11 ** 2
    radius = float(np.percentile(np.hypot(wx[sel] - cx, wy[sel] - cy), 95))
    return cx, cy, radius


def _angle(xs, ys, cx, cy):
    return math.atan2(-(np.mean(ys) - cy), np.mean(xs) - cx)  # image y down -> counter-clockwise positive


def read(img, dial, inner=(0.20, 0.85), min_face=0.80):
    """Needle angle and target (green band centre) angle in radians; None when unreadable or occluded."""
    cx, cy, rad = dial
    _, green, face, needle = _masks(img)
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    rr = np.hypot(xx - cx, yy - cy)
    disk = rr <= 0.85 * rad
    visible = (face | needle | green)[disk].mean()  # anything else on the face = occluded (gripper, shadow)
    gy, gx = np.nonzero(green & (rr <= 1.2 * rad))
    target = _angle(gx, gy, cx, cy) if len(gx) >= 2 else None
    if visible < min_face:
        return None, target
    ny, nx = np.nonzero(needle & (rr >= inner[0] * rad) & (rr <= inner[1] * rad))
    angle = _angle(nx, ny, cx, cy) if len(nx) >= 2 else None
    return angle, target


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def progress(needle, zero, target):
    """Fraction of the zero->target sweep covered by the needle, measured in the sweep's own direction."""
    full = _wrap(target - zero)
    if abs(full) < 1e-6:
        return None
    return _wrap(needle - zero) / full


class DialStop:
    """Stop decision from the observation stream: zero = the needle's resting angle in the first readable frames."""

    def __init__(self, margin=0.05):
        self.margin, self.dial, self.zero, self.target, self.last = margin, None, None, None, None

    def update(self, img):
        if self.dial is None:
            self.dial = locate(img)
            if self.dial is None:
                return False
        needle, target = read(img, self.dial)
        if target is not None:
            self.target = target if self.target is None else self.target
        if needle is None or self.target is None:
            return False  # occluded or unreadable: never stop on a missing reading
        if self.zero is None:
            self.zero = needle
            return False
        self.last = progress(needle, self.zero, self.target)
        return self.last is not None and self.last >= 1.0 - self.margin


# --- v2 (REV-A6): needle from the darkest ray through the hub, inner ring only -----------------------------
def _gray(img):
    return img.astype(float).mean(axis=-1)


def _bilinear(g, x, y):
    h, w = g.shape
    x = np.clip(x, 0, w - 1.001); y = np.clip(y, 0, h - 1.001)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    return (g[y0, x0] * (1 - fx) * (1 - fy) + g[y0, x0 + 1] * fx * (1 - fy)
            + g[y0 + 1, x0] * (1 - fx) * fy + g[y0 + 1, x0 + 1] * fx * fy)


def read_v2(img, dial, ring=(0.15, 0.5), min_face=0.80, min_contrast=6.0):
    """Needle angle (radians, counter-clockwise in image coordinates) from the darkest ray between ring radii
    (clear of the rim ticks, which start at 0.6 R, and the hub). Returns (angle, target) or (None, target) when
    occluded or when no ray is clearly darker than the face (unreadable is never a stop)."""
    cx, cy, rad = dial
    _, green, face, needle = _masks(img)
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    rr = np.hypot(xx - cx, yy - cy)
    disk = rr <= 0.85 * rad
    visible = (face | needle | green)[disk].mean()
    gy, gx = np.nonzero(green & (rr <= 1.2 * rad))
    target = _angle(gx, gy, cx, cy) if len(gx) >= 2 else None
    if visible < min_face:
        return None, target
    g = _gray(img)
    angs = np.radians(np.arange(0, 360, 1.0))
    rs = np.linspace(ring[0] * rad, ring[1] * rad, 9)
    A, Rr = np.meshgrid(angs, rs, indexing="ij")
    vals = _bilinear(g, cx + Rr * np.cos(A), cy - Rr * np.sin(A)).mean(axis=1)
    k = np.ones(5) / 5
    sm = np.convolve(np.r_[vals[-2:], vals, vals[:2]], k, mode="valid")
    i = int(np.argmin(sm))
    if np.median(sm) - sm[i] < min_contrast:
        return None, target
    # sub-degree refinement: darkness-weighted mean over +-6 deg
    w = np.clip(np.median(sm) - sm[[(i + j) % 360 for j in range(-6, 7)]], 0, None)
    off = float((w * np.arange(-6, 7)).sum() / max(w.sum(), 1e-9))
    return math.radians(i + off), target


GAUGE_DEG_PER_KPA = 270.0 / 120.0  # printed scale: 0-120 kPa over 270 deg; 60 kPa is the red mark
TARGET_KPA = 60.0


def kpa_from(needle, target):
    """Reading from the needle's clockwise offset relative to the dial's own target mark (green band centre)."""
    return TARGET_KPA + math.degrees(_wrap(target - needle)) / GAUGE_DEG_PER_KPA


STOP_LO_KPA, STOP_HI_KPA, STOP_FRAMES = 58.0, 75.0, 2


class PressureStop:
    """Shared runtime/gate rule; unreadable frames reset consecutive evidence."""

    def __init__(self):
        self.dial = None
        self.consecutive = 0

    def update(self, image, pumping):
        if self.dial is None:
            self.dial = locate(image)
        estimate = None
        if self.dial is not None:
            needle, target = read_v2(image, self.dial)
            if needle is not None and target is not None:
                estimate = kpa_from(needle, target)
        qualifies = pumping and estimate is not None and STOP_LO_KPA <= estimate <= STOP_HI_KPA
        self.consecutive = self.consecutive + 1 if qualifies else 0
        return self.consecutive >= STOP_FRAMES, estimate
