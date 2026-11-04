"""Layout stub for yam_hand_pump. Plain yam_arena does not build this mechanism."""
from yam_arena.task_file import Box
from hand_pump import design as d

SCENE = "station"
OBJECTS = [Box("rest_mat", pos=(*d.MAT_CENTER, d.TOP + d.MAT_T / 2),  # on the table, not coplanar with it
               size=(*d.MAT_SIZE, d.MAT_T), color=(0.38, 0.40, 0.43))]


def success(objects):
    return False  # feasibility probe only; no task score yet
