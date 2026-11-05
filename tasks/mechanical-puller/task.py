"""Station table: a fixed shaft with a collar, a two-jaw puller on a two-pillar tool rest. The embodiment builds the parts."""
from yam_arena.task_file import Box

from mechanical_puller import design as d

SCENE = "station"
OBJECTS = [Box(f"rest_{n}", pos=(d.REST_XY[0] + s * d.PILLAR_X, d.REST_XY[1], d.TOP + (d.PARK_Z - d.CROSSBAR[2] / 2) / 2),
               size=(0.006, 0.016, d.PARK_Z - d.CROSSBAR[2] / 2), color=(0.3, 0.3, 0.32)) for n, s in (("a", -1), ("b", 1))]


def success(objects):
    return False  # scored by the embodiment

# Fixed shallow recovery tray; physical floor and walls.
tx,ty=d.TRAY_XY
OBJECTS += [Box("tray_floor",pos=(tx,ty,d.TOP+d.TRAY_FLOOR/2),
                size=(.120,.100,d.TRAY_FLOOR),color=(.15,.55,.3))]
for name,offset,size in [("front",(0,-.0475),(.120,.005,.014)),
                         ("rear",(0,.0475),(.120,.005,.014)),
                         ("left",(-.0575,0),(.005,.090,.014)),
                         ("right",(.0575,0),(.005,.090,.014))]:
    OBJECTS.append(Box("tray_"+name,pos=(tx+offset[0],ty+offset[1],d.TOP+.007),size=size,color=(.15,.55,.3)))
