from .body import _Body, GREY,BLUE,DARK,WHITE,STEEL,ORANGE
from . import design as d
# Original coarse pixel lettering: rendered geometry, no external graphics.
FONT={'A':['010','101','111','101','101'],'B':['110','101','110','101','110'],'C':['111','100','100','100','111'],'D':['110','101','101','101','110'],'O':['111','101','101','101','111'],'P':['110','101','110','100','100'],'X':['101','101','010','101','101'],'L':['100','100','100','100','111'],'1':['010','110','010','010','111'],'-':['000','000','111','000','000']}
def text(b,word,origin,pixel=.008,plane='top'):
 x,y,z=origin
 for j,ch in enumerate(word):
  for row,line in enumerate(FONT[ch]):
   for col,v in enumerate(line):
    if v=='1':
     if plane=='top':b.box((x+(2-row)*pixel,y-(j*4+col)*pixel,z),(.9*pixel,.9*pixel,.001),0,DARK,False)
     else:b.box((x,y-(j*4+col)*pixel,z+(2-row)*pixel),(.001,.9*pixel,.9*pixel),0,DARK,False)
def probe(name):
 b=_Body(name,.09,(.8,.6));b.box((0,0,0),(.045,.025,.046),0,ORANGE if name.endswith('l') else BLUE);b.box((.024,0,0),(.008,.015,.015),0,DARK);return b.save()
def tip(name):
 b=_Body(name,.015,(.4,.3));b.cylinder_x(.02,.065,.003,STEEL);return b.save()
def panel():
 b=_Body('panel',2,(.5,.4),kinematic=True);b.box((0,0,.10),(.01,.29,.20),0,WHITE)
 for name,pos in d.PADS.items():
  y,z=pos[1],pos[2]-d.TOP;b.box((-.0045,y,z),(.001,.026,.026),0,STEEL);text(b,name,(-.0052,y+.033,z),.010,'front')
 return b.save()
def meter():
 b=_Body('meter',1,(.5,.4),kinematic=True);b.box((0,0,0),(.16,.22,.012),0,WHITE);return b.save()
def display(name,word):
 b=_Body('display_'+name,.01,(.5,.4),kinematic=True);text(b,word,(0,.065,0),.022);return b.save()
def result_board():
 b=_Body('result_board',1,(.6,.5),kinematic=True)
 for x,label in [(-.1,'P'),(0,'X'),(.1,'O')]:
  b.box((x,0,.002),(.08,.08,.004),0,(.85,.85,.85));text(b,label,(x,.068,.005),.009)
 return b.save()
def pointer():
 b=_Body('pointer',.05,(.8,.6));b.box((0,0,0),(.025,.025,.036),0,BLUE);return b.save()
BUILDERS={'probe_l':lambda:probe('probe_l'),'probe_r':lambda:probe('probe_r'),'tip_l':lambda:tip('tip_l'),'tip_r':lambda:tip('tip_r'),'panel':panel,'meter':meter,'display_dash':lambda:display('dash','--'),'display_ol':lambda:display('ol','OL'),'display_one':lambda:display('one','1'),'pointer':pointer,'result_board':result_board}
