"""CPU checks of the retrieval-tool scorer and coupling geometry: python -m pytest tests (no simulator)."""
import math

import numpy as np

from retrieval_tool import design as d
from retrieval_tool.scoring import RetrievalScore, cart_in_tray, tooth_in_pocket

HZ = 20
OPEN, HELD = [1.0, 1.0], [0.25, 0.25]
FAR = [np.array([0.0, 0.3, 1.0]), np.array([0.0, -0.3, 1.0])]
SLOW = {"shaft": 0.0, "head": 0.0, "cartridge": 0.0}


def poses(head=None, cart=None, shaft=None):
    p = d.start_poses("assembled")
    if head is not None:
        p["head"] = head
    if cart is not None:
        p["cartridge"] = cart
    if shaft is not None:
        p["shaft"] = shaft
    return p


def run(states):
    s = RetrievalScore(HZ)
    return s, [s.update(*st) for st in states]


def locked_head(yaw=0.0, lift=0.0):
    return np.r_[d.PIN_XY, d.TOP + d.MAT_T + d.ROD_T + lift, d.yaw_quat(yaw)]


def cart_at(x, y=0.0, z=0.0):
    return np.r_[x, y, d.TOP + z, 0, 0, 0, 1.0]


def in_tray_cart():
    return cart_at(d.TRAY_CENTER[0] - d.CART_L / 2, d.TRAY_CENTER[1], 0.003)


def hooked(cart_x):
    """Head positioned so its tooth sits in the pocket of a cartridge whose front is at cart_x."""
    tip_x = cart_x + (d.POCKET_X0 + d.POCKET_X1) / 2
    pin_x = tip_x - (d.TOOTH_X0 + d.ARM_X1) / 2
    shaft = np.r_[pin_x, 0.0, d.TOP + 0.010, 0, 0, 0, 1.0]
    head = np.r_[pin_x, 0.0, d.TOP + 0.010 + d.ROD_T, 0, 0, 0, 1.0]
    return shaft, head


def test_locked_geometry():
    s = d.start_poses("assembled")
    assert d.locked(s["shaft"], s["head"])
    assert not d.locked(s["shaft"], locked_head(yaw=-math.pi / 2))  # insertion angle: lugs in the slots
    assert not d.locked(s["shaft"], locked_head(lift=0.008))  # resting on the lugs, not seated
    assert not d.locked(d.start_poses("loose")["shaft"], d.start_poses("loose")["head"])


def test_upside_down_head_and_invalid_rotation_cannot_be_locked():
    s = d.start_poses("assembled")
    upside_down = locked_head()
    upside_down[3:] = [1.0, 0.0, 0.0, 0.0]
    assert not d.locked(s['shaft'], upside_down)
    zero_rotation = locked_head()
    zero_rotation[3:] = 0.0
    assert not d.locked(s['shaft'], zero_rotation)


def test_untouched_start_scores_zero():
    p = d.start_poses("loose")
    s, out = run([(p, False, OPEN, FAR, SLOW)] * 100)
    assert max(out) == 0.0


def test_assembled_only_scores_0_2():
    s, out = run([(poses(), True, OPEN, FAR, SLOW)] * 60)
    assert out[-1] == 0.2 and not s.success


def _extraction(n=40):
    """Hooked pull from the start position to 0.30 m with the locked tool."""
    states = []
    for x in np.linspace(d.cart_start_x(), 0.30, n):
        sh, hd = hooked(x)
        states.append((poses(head=hd, cart=cart_at(x), shaft=sh), True, HELD, FAR, SLOW))
    return states


def test_hooked_extraction_scores_0_4():
    s, out = run(_extraction())
    assert out[-1] == 0.4 and s.extracted


def test_cartridge_moved_without_the_tool_gets_no_extraction_credit():
    states = [(poses(cart=cart_at(x)), True, HELD, FAR, SLOW) for x in np.linspace(d.cart_start_x(), 0.30, 40)]
    states += [(poses(cart=in_tray_cart()), True, OPEN, FAR, SLOW)] * 60
    s, out = run(states)
    assert out[-1] == 0.2 and not s.extracted  # e.g. pushed/teleported, not pulled by the locked hook


def test_extraction_with_unlocked_head_gets_no_credit():
    states = []
    for x in np.linspace(d.cart_start_x(), 0.30, 40):
        sh, hd = hooked(x)
        states.append((poses(head=hd, cart=cart_at(x), shaft=sh), False, HELD, FAR, SLOW))
    s, out = run(states)
    assert not s.extracted and out[-1] == 0.0


def test_full_success_after_release_hold():
    states = _extraction()
    states += [(poses(cart=in_tray_cart()), True, HELD, [np.array([0.2, 0.0, d.TOP + 0.03])] * 2, SLOW)] * 10
    states += [(poses(cart=in_tray_cart()), True, OPEN, FAR, SLOW)] * (2 * HZ)
    s, out = run(states)
    assert out[-1] == 1.0 and s.success
    assert max(out[:-1]) < 1.0


def test_wrong_placement_and_held_tool_are_partial():
    states = _extraction() + [(poses(cart=cart_at(0.30)), True, OPEN, FAR, SLOW)] * 60  # left on the table
    s, out = run(states)
    assert out[-1] == 0.4 and not s.success
    states = _extraction() + [(poses(cart=in_tray_cart()), True, HELD, [np.array([0.13, 0.0, d.TOP + 0.03])] * 2,
                               SLOW)] * 60  # still gripping the tool
    s, out = run(states)
    assert out[-1] == 0.8 and not s.success


def test_tool_not_at_rest_or_unlocked_at_end():
    away = np.r_[0.15, 0.10, d.TOP, 0, 0, 0, 1.0]
    states = _extraction() + [(poses(cart=in_tray_cart(), shaft=away, head=np.r_[0.15, 0.10, d.TOP + d.ROD_T, 0, 0, 0, 1]),
                               True, OPEN, FAR, SLOW)] * 60
    s, out = run(states)
    assert out[-1] == 0.6 and not s.success
    states = _extraction() + [(poses(cart=in_tray_cart()), False, OPEN, FAR, SLOW)] * 60
    s, out = run(states)
    assert out[-1] == 0.6 and not s.success


def test_nonfinite_is_invalid():
    p = poses()
    p["cartridge"] = np.r_[np.nan, 0, 0, 0, 0, 0, 1]
    s, out = run([(p, True, OPEN, FAR, SLOW)])
    assert out[-1] == 0.0 and s.reason == "invalid state"


def test_below_pocket_does_not_count_as_engagement():
    cart = cart_at(d.cart_start_x())
    _, head = hooked(d.cart_start_x())
    assert tooth_in_pocket(head, cart)
    head[2] -= 0.20
    assert not tooth_in_pocket(head, cart)


def test_tray_rejects_below_table_and_overhanging_corners():
    cart = in_tray_cart()
    assert cart_in_tray(cart)
    cart[2] -= 0.20
    assert not cart_in_tray(cart)
    cart = in_tray_cart()
    cart[0] += 0.045  # centre still inside, but the end overhangs the tray
    assert not cart_in_tray(cart)


def test_short_oscillations_do_not_bank_hooked_distance():
    states = []
    for x in [d.cart_start_x(), d.cart_start_x() - 0.02] * 10:
        sh, hd = hooked(x)
        states.append((poses(head=hd, cart=cart_at(x), shaft=sh), True, HELD, FAR, SLOW))
    states.append((poses(cart=cart_at(0.30)), False, OPEN, FAR, SLOW))
    s, out = run(states)
    assert s.hooked_travel < 0.021 and not s.extracted and out[-1] == 0.2


def test_unhooked_return_removes_unfinished_pull_credit():
    states = []
    for x in np.linspace(d.cart_start_x(), d.cart_start_x() - 0.11, 20):
        sh, hd = hooked(x)
        states.append((poses(head=hd, cart=cart_at(x), shaft=sh), True, HELD, FAR, SLOW))
    states += [(poses(cart=cart_at(d.cart_start_x())), False, OPEN, FAR, SLOW),
               (poses(cart=cart_at(0.30)), False, OPEN, FAR, SLOW)]
    s, out = run(states)
    assert s.hooked_travel == 0 and not s.extracted and out[-1] == 0.2


def test_invalid_speed_cannot_complete_hold():
    states = _extraction()
    states += [(poses(cart=in_tray_cart()), True, OPEN, FAR, dict(SLOW, cartridge=float('nan')))] * 60
    s, out = run(states)
    assert out[-1] == 0.0 and not s.success


def test_rotated_cartridge_requires_all_corners_outside():
    from retrieval_tool.scoring import cart_out
    pose = np.r_[[d.HOOD_MOUTH_X - 0.001, 0.0, d.TOP], d.yaw_quat(np.pi / 2)]
    assert not cart_out(pose)  # midpoint clear; far corner remains 19 mm inside
    pose[0] = d.HOOD_MOUTH_X - d.CART_W / 2 - 0.001
    assert cart_out(pose)
