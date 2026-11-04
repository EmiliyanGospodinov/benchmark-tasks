"""On the station table: a bench-held quick-release blade holder with a worn blade installed, a
replacement blade lying in a two-fork rack beside it, and a rest tray for the removed blade. The
embodiment ``yam_blade_change`` builds the holder, collar, pin and blades; this file lists the fixed
rack, tray and carriage rail."""

from yam_arena.task_file import Box

from blade_change import design as d

SCENE = "station"
_GREY, _DARK, _TRAY = (0.62, 0.64, 0.67), (0.25, 0.27, 0.30), (0.30, 0.55, 0.35)
_z = d.TOP
_rest_bottom = d.AXIS_Z - 0.03 - d.BODY_H / 2
_tx, _ty = d.TRAY_XY
_iw, _il, _wh = d.TRAY

OBJECTS = [
    # carriage rail under the holder
    Box("rail", pos=(d.AXIS_X, d.NOSE_Y + 0.02, _z + 0.010), size=(0.05, 0.12, 0.020), color=_DARK),
    # replacement rack: two forks (a base block and two side posts each) under the blade body
    *[Box(f"fork_{i}_{k}", pos=(d.NEW_REST_X + dx, y, z), size=size, color=_GREY)
      for i, y in enumerate((d.NOSE_Y - 0.01 - 0.04, d.NOSE_Y - 0.01 - 0.13))
      for k, (dx, z, size) in enumerate((
          (0.0, (_z + _rest_bottom) / 2, (0.02, 0.012, _rest_bottom - _z)),
          (+0.006, _rest_bottom + 0.006, (0.004, 0.012, 0.012)),
          (-0.006, _rest_bottom + 0.006, (0.004, 0.012, 0.012))))],
    # rest tray: floor and four low walls
    Box("tray_floor", pos=(_tx, _ty, _z + 0.002), size=(_iw + 0.008, _il + 0.008, 0.004), color=_TRAY),
    Box("tray_wall_xp", pos=(_tx + _iw / 2 + 0.002, _ty, _z + _wh / 2), size=(0.004, _il + 0.008, _wh), color=_TRAY),
    Box("tray_wall_xn", pos=(_tx - _iw / 2 - 0.002, _ty, _z + _wh / 2), size=(0.004, _il + 0.008, _wh), color=_TRAY),
    Box("tray_wall_yp", pos=(_tx, _ty + _il / 2 + 0.002, _z + _wh / 2), size=(_iw, 0.004, _wh), color=_TRAY),
    Box("tray_wall_yn", pos=(_tx, _ty - _il / 2 - 0.002, _z + _wh / 2), size=(_iw, 0.004, _wh), color=_TRAY),
]


def success(objects):
    return False  # scored by the embodiment (blade_change.scoring)
