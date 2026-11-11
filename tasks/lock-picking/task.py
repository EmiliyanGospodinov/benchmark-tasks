"""On the station table: an enlarged clear pin-tumbler training lock with a tension wrench, and a hook pick.
The embodiment ``yam_lock_picking`` builds the trainer, the pick and their joints; this file lists a work mat."""

from yam_arena.task_file import Box

from lock_picking import design as d

SCENE = "station"
OBJECTS = [Box("work_mat", pos=(0.26, 0.05, d.TOP + 0.0005), size=(0.30, 0.40, 0.001), color=(0.30, 0.32, 0.35))]


def success(objects):
    return False  # scored by the embodiment (lock_picking.scoring)
