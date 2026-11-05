"""Seg/Script reused from A5 f7e6030; no sibling runtime imports."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
from yam_arena import ik
from yam_arena.constants import FINGER_OPEN, READY

@dataclass
class Seg:
    duration: float
    target: Callable  # sim -> (pos, rot)
    finger: float
    label: str
    freeze: bool = True  # evaluate the target once, on entering the segment


class Script:
    def __init__(self, segs, base):
        self.segs, self.base = segs, np.asarray(base, float)
        self.q = np.array(READY, float)
        self.prev = ik.tcp_pose(self.q, self.base)
        self.prev_finger = FINGER_OPEN
        self.index, self.goal = -1, None
        self.worst_mm = 0.0

    @property
    def duration(self):
        return sum(s.duration for s in self.segs)

    def act(self, sim, t):
        acc = 0.0
        for i, s in enumerate(self.segs):
            if t < acc + s.duration:
                break
            acc += s.duration
        else:
            i, s = len(self.segs), None
        if i != self.index:
            if self.goal is not None:
                self.prev, self.prev_finger = self.goal[:2], self.goal[2]
            self.index = i
            self.goal = None if s is None else (*(self.prev if s.target is None else s.target(sim)), s.finger)
        if s is None:
            pos, rot, finger, label = *self.prev, self.prev_finger, "done"
        else:
            if not s.freeze and s.target is not None:
                self.goal = (*s.target(sim), s.finger)
            u = ik.smoothstep((t - acc) / s.duration)
            pos = self.prev[0] + u * (self.goal[0] - self.prev[0])
            rot = ik.slerp_matrix(self.prev[1], self.goal[1], u)
            finger = self.prev_finger + u * (s.finger - self.prev_finger)
            label = s.label
        q, pe, _ = ik.solve(np.asarray(pos, float), ik.matrix_to_quat_xyzw(rot), self.q, base_pos=self.base, iters=60)
        self.q = q
        self.worst_mm = max(self.worst_mm, pe * 1000)
        return np.r_[q, float(np.clip(finger / FINGER_OPEN, 0.0, 1.0))], label


