"""History of physical gauging, fitting and motion; no script-phase or arm-count inputs."""
import math
from collections import deque

# Tolerances (mm). The indicator display resolves 0.01 mm.
STOP_MM=.05        # shaft must be within this of a physical stop for a reading
STEADY_MM=.02      # reading spread over 0.25 s for a held reading
INITIAL_MM=12.     # nominal initial travel; accepted within INITIAL_TOL_MM
INITIAL_TOL_MM=.1
TARGET_MM=(2.9,3.1)  # accepted final end play
AGREE_MM=.05       # remeasurement must agree with physical clearance

class Gauging:
    def __init__(self,hz):
        self.n=max(2,round(.25*hz));self.buffers=[deque(maxlen=self.n),deque(maxlen=self.n)]
        self.readings=[None,None]
    def update(self,near,far,reading,valid):
        for i,at in enumerate((near,far)):
            buf=self.buffers[i]
            if not(valid and at): buf.clear();continue
            buf.append(reading)
            if len(buf)==self.n and max(buf)-min(buf)<STEADY_MM:
                self.readings[i]=sum(buf)/len(buf)
    @property
    def travel(self):
        return None if None in self.readings else abs(self.readings[1]-self.readings[0])

class FitScore:
    def __init__(self,hz=20):
        self.hz=hz;self.initial=Gauging(hz);self.final=Gauging(hz)
        self.initial_travel_mm=None;self.post_measured=False
        self.angle_prev=None;self.angle_sum=0.;self.angle_min=0.;self.angle_max=0.
        self.hold=0.;self.success=False;self.score=0.
    def update(self,*,shaft_mm,probe_mm,contact_valid,reference_valid,installed,
               clearance_mm,angle_rad,released,stable,valid=True):
        self.success=False
        if not valid or not all(math.isfinite(x) for x in (shaft_mm,probe_mm,clearance_mm,angle_rad)):
            self.hold=0.;self.score=0.;return self.score
        gauge_valid=contact_valid and reference_valid
        if not installed and self.initial_travel_mm is None:
            self.initial.update(abs(shaft_mm)<STOP_MM,abs(shaft_mm-INITIAL_MM)<STOP_MM,probe_mm,gauge_valid)
            if self.initial.travel is not None and abs(self.initial.travel-INITIAL_MM)<=INITIAL_TOL_MM:
                self.initial_travel_mm=self.initial.travel
        self.score=.25 if self.initial_travel_mm is not None else 0.
        fitted=installed and TARGET_MM[0]<=clearance_mm<=TARGET_MM[1]
        if not fitted:
            self.hold=0.; self.angle_prev=None
            self.final=Gauging(self.hz);self.post_measured=False
            self.angle_sum=0.;self.angle_min=0.;self.angle_max=0.
            return self.score
        if self.initial_travel_mm is None:return self.score
        self.score=.5
        self.final.update(abs(shaft_mm)<STOP_MM,abs(shaft_mm-clearance_mm)<STOP_MM,probe_mm,gauge_valid)
        if self.final.travel is not None and TARGET_MM[0]<=self.final.travel<=TARGET_MM[1] and abs(self.final.travel-clearance_mm)<=AGREE_MM:
            self.post_measured=True
        if self.angle_prev is not None:
            self.angle_sum+=math.atan2(math.sin(angle_rad-self.angle_prev),math.cos(angle_rad-self.angle_prev))
            self.angle_min=min(self.angle_min,self.angle_sum);self.angle_max=max(self.angle_max,self.angle_sum)
        self.angle_prev=angle_rad
        if self.post_measured:self.score=.75
        if self.post_measured and self.angle_max-self.angle_min>=math.radians(60) and released and stable:
            self.hold+=1/self.hz
        else:self.hold=0.
        self.success=self.hold>=2.
        if self.success:self.score=1.
        return self.score
