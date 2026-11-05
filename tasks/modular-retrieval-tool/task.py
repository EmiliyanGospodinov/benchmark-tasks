"""On the station table: a two-piece retrieval tool (shaft and hook head, loose), a fixed service hood with a
cartridge deep inside, a recovery tray and a head stand. The embodiment ``yam_retrieval_tool`` builds all of
them; this file lists only the fixed hood mounting bracket behind the hood."""
from yam_arena.task_file import Box

from retrieval_tool import design as d

SCENE = "station"
OBJECTS = [Box("hood_bracket", pos=(*d.BRACKET[0], d.TOP + d.BRACKET[1][2] / 2), size=d.BRACKET[1], color=(0.30, 0.32, 0.35))]


def success(objects):
    return False  # scored by the embodiment (retrieval_tool.scoring)
