"""On the station table: a ratchet lowering trainer with its weighted lever raised and held by the pawl.
The embodiment ``yam_ratchet_lowering`` builds the trainer and its joints; this file lists a floor mat."""

from yam_arena.task_file import Box

from ratchet_lowering import design as d

SCENE = "station"
OBJECTS = [Box("trainer_mat", pos=(d.SHAFT[0] + 0.03, -0.01, d.TOP + 0.0005), size=(0.30, 0.38, 0.001),
               color=(0.30, 0.32, 0.35))]


def success(objects):
    return False  # scored by the embodiment (ratchet_lowering.scoring)
