"""Offline axisymmetric linear-elastic estimate for the integral polymer collar.

Includes the stepped hub/flange and relieved bore (no housing). Rigid steel shaft,
frictionless radial assembly contact, free outer surfaces. Contact normal load is
converted to axial sliding friction. Not a calibrated breakout/creep model.
"""
import json
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


def estimate(h=.0005, young=2.4e9, poisson=.35, interference=10e-6, contact_length=.002, cylinder=False):
    inner=.006-interference/2
    # Preserve every geometry transition while refining each interval.
    axis=lambda breaks: np.unique(np.concatenate([np.linspace(a,b,max(2,int(np.ceil((b-a)/h))+1)) for a,b in zip(breaks[:-1],breaks[1:])]))
    rs=axis([inner,.011] if cylinder else [inner,.0065,.011,.022]); zs=axis([0,.002] if cylinder else [0,.002,.004,.014])
    nodes=[]; index={}; tris=[]
    def node(i,j):
        if (i,j) not in index:index[i,j]=len(nodes);nodes.append([rs[i],zs[j]])
        return index[i,j]
    for j in range(len(zs)-1):
        z=(zs[j]+zs[j+1])/2
        for i in range(len(rs)-1):
            r=(rs[i]+rs[i+1])/2
            if not cylinder and (r < (inner if z<.002 else .0065) or r > (.011 if z<.004 else .022)):continue
            a,b,c,e=node(i,j),node(i+1,j),node(i+1,j+1),node(i,j+1)
            tris.extend([(a,b,c),(a,c,e)])
    xy=np.array(nodes); n=len(nodes)*2
    lam=young*poisson/((1+poisson)*(1-2*poisson)); shear=young/(2*(1+poisson))
    D=np.zeros((4,4));D[:3,:3]=lam;D[np.arange(3),np.arange(3)]+=2*shear;D[3,3]=shear
    rows=[];cols=[];values=[]
    for tri in tris:
        pt=xy[list(tri)]; M=np.c_[np.ones(3),pt]; area=abs(np.linalg.det(M))/2
        grad=np.linalg.inv(M)[1:,:]; radius=pt[:,0].mean(); B=np.zeros((4,6))
        B[0,0::2]=grad[0];B[1,1::2]=grad[1];B[2,0::2]=1/(3*radius)
        B[3,0::2]=grad[1];B[3,1::2]=grad[0]
        ke=B.T@D@B*(2*np.pi*radius*area); dofs=np.array([[2*k,2*k+1] for k in tri]).ravel()
        rows.extend(np.repeat(dofs,6));cols.extend(np.tile(dofs,6));values.extend(ke.ravel())
    K=coo_matrix((values,(rows,cols)),shape=(n,n)).tocsr()
    contact=np.flatnonzero(np.isclose(xy[:,0],inner,atol=1e-10)&(xy[:,1]<=contact_length+1e-10))
    active=contact.copy();u=np.zeros(n)
    for iteration in range(30):
        prescribed=np.r_[2*active,2*contact[0]+1];u[:]=0;u[2*active]=interference/2
        free=np.setdiff1d(np.arange(n),prescribed)
        u[free]=spsolve(K[free,:][:,free],-K[free,:][:,prescribed]@u[prescribed])
        reactions=K@u
        next_active=np.array([k for k in contact if (k in active and reactions[2*k]>-1e-7) or (k not in active and u[2*k]<interference/2-1e-12)])
        if np.array_equal(active,next_active):break
        active=next_active
    else:raise RuntimeError('contact active set did not converge')
    reaction=reactions[2*active];normal=float(reaction.sum())
    return {'mesh_m':h,'nodes':len(xy),'normal_N':normal,'force_mu_012_N':normal*.12,'min_contact_reaction_N':float(reaction.min()),'max_radial_displacement_m':float(u[0::2].max())}

if __name__=='__main__':
    print(json.dumps([estimate(h) for h in [.001,.0005,.00025]],indent=2))
