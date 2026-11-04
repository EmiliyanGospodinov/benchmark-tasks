"""Geometric contact-indicator readings, independent of script phase and spacer identity."""
import numpy as np


def rotation(q):
    q=np.asarray(q,float)
    if q.shape!=(4,) or not np.all(np.isfinite(q)) or np.linalg.norm(q)<1e-12:
        raise ValueError('Invalid orientation')
    x,y,z,w=q/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def read_indicator(gauge,probe,shaft):
    """Returns extension and tip/shaft geometry; proximity is not a force reading.

    Bodies use the adapter's xyzw orientation convention. The contact tip ends
    26 mm ahead of the probe origin, and the shaft ends 70 mm behind its origin.
    Reading is probe travel relative to its instrument, never shaft ground truth.
    """
    poses=[np.asarray(p,float) for p in (gauge,probe,shaft)]
    if any(p.shape!=(7,) or not np.all(np.isfinite(p)) for p in poses):
        raise ValueError('Invalid measured pose')
    gauge,probe,shaft=poses
    g,p,s=(rotation(x[3:]) for x in poses)
    extension=float((probe[:3]-gauge[:3])@g[:,0]-.05)
    tip=probe[:3]+p@np.array([.026,0.,0.])
    end=shaft[:3]+s@np.array([-.07,0.,0.])
    delta=end-tip
    axial=float(delta@s[:,0])
    lateral=float(np.linalg.norm(delta-axial*s[:,0]))
    aligned=float(p[:,0]@s[:,0])>.999
    return {'extension_mm':1000*extension,'axial_gap_mm':1000*axial,
            'lateral_error_mm':1000*lateral,
            'contact_geometry_valid':bool(aligned and abs(axial)<.0005 and lateral<.002)}
