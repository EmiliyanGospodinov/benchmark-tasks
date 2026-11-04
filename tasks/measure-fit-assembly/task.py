"""Station measurement diagnostic; complete fit task is in development."""
from yam_arena.task_file import Box
SCENE="station"
OBJECTS=[Box("parts_mat",pos=(.26,.18,.7495),size=(.32,.12,.004),color=(.38,.40,.43))]
def success(objects):
    return False
