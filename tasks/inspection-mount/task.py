"""On the station table: an inspection arm on a bench post, hanging on its lower stop with the
clamp knob loose. The embodiment ``yam_inspection_mount`` builds the post, arm and knob (and the
hinge and clamp) itself; this file lists the post's fixed foot plate."""

from yam_arena.task_file import Box

from mount_aim import design as d

SCENE = "station"
OBJECTS = [Box("post_foot", pos=(d.POST_XY[0], d.POST_XY[1], d.TOP + 0.003), size=(0.10, 0.08, 0.006),
               color=(0.25, 0.27, 0.30))]


def success(objects):
    return False  # scored by the embodiment (mount_aim.scoring)
