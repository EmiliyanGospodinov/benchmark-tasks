import numpy as np
from retrieval_tool import design as d


def test_returned_control_under_rigid_transform():
    shaft = np.r_[[0.3, 0.1, 0.8], d.yaw_quat(0.7)]
    assert d.release_closed(shaft, shaft.copy()) is True


def test_retracted_control_not_locked():
    shaft = np.r_[[0.3, 0.1, 0.8], d.yaw_quat(0.7)]
    control = shaft.copy()
    control[:3] += d.quat_to_matrix(shaft[3:]) @ [0, d.RELEASE_TRAVEL, 0]
    assert not d.release_closed(shaft, control)


def test_broken_or_tilted_guide_not_locked():
    shaft = np.r_[[0.3, 0.1, 0.8], d.yaw_quat(0)]
    control = shaft.copy(); control[0] += 0.01
    assert not d.release_closed(shaft, control)
    control = shaft.copy(); control[3:] = d.yaw_quat(0.2)
    assert not d.release_closed(shaft, control)


def test_invalid_control_not_locked():
    shaft = np.r_[[0.3, 0.1, 0.8], d.yaw_quat(0)]
    control = shaft.copy(); control[1] = np.nan
    assert not d.release_closed(shaft, control)
    control = shaft.copy(); control[3:] = 0
    assert not d.release_closed(shaft, control)
