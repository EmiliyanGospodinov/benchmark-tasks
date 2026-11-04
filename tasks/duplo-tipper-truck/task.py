"""On the station table: the loose parts of a DUPLO tipper truck, a payload block and a receiving tray.

- A white DUPLO car base 4 x 8 with its four wheels (it rolls freely along y), centred near
  (0.30, -0.035) with its tow pin toward +y (the rear).
- A medium-azure DUPLO 2 x 2 brick, a yellow truck cab body 136 x 64 x 86 mm, a black DUPLO
  hinge plate 2 x 2 with an extended hinge, and a medium-azure tipper bed 64 x 72 x 36 mm, each
  loose on the table.
- An orange 20 mm wooden cube (the payload).
- A grey tray fixed to the table behind the truck (toward +y): 90 x 100 mm inside, 12 mm walls.

Parts join when pressed into place at the position the instruction specifies. The bed turns
on its hinge from closed to about 65 degrees open. World frame: x forward (away from the
arms), y left, z up, metres. This file describes the scene; the embodiment
``yam_duplo_tipper`` builds it and scores it.
"""

from yam_arena.task_file import Box

from duplo_tipper import design as d

SCENE = "station"
# Placeholders that satisfy the task-file contract (names and free/fixed status). The
# embodiment spawns the real part bodies itself; object_jitter must stay 0.
OBJECTS = [Box("payload", pos=(0.16, 0.20, d.TOP + 0.0105), size=(0.02,) * 3, mass=d.MASS["payload"])]


def success(objects):
    return False  # scored by the embodiment (duplo_tipper.scoring)
