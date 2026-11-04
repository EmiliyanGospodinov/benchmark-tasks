"""CPU checks of the mat-placement test (design.feet_on_mat)."""
import math

import numpy as np

from hand_pump import design as d


def pose(dx=0.0, dy=0.0, yaw=0.0):
    return np.r_[d.BARREL_POS + np.array([dx, dy, 0.0]), d.yaw_quat(yaw)]


def test_start_pose_is_on_mat():
    assert d.feet_on_mat(pose())


def test_shifted_sideways_off_the_mat():
    assert d.feet_on_mat(pose(dy=0.04))       # 4 cm aside: feet still on the 14 cm mat
    assert not d.feet_on_mat(pose(dy=0.06))   # a foot overhangs the mat edge


def test_pushed_back_off_the_mat():
    assert not d.feet_on_mat(pose(dx=-0.21))  # where the unheld pump ended (pump-006)


def test_turned_across_the_mat():
    assert not d.feet_on_mat(pose(yaw=math.pi / 2))
