from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.special import logsumexp

FEATURES=['distance_m','heading_error','speed_mps','gap_s','segment_length_m','candidate_rank','projected_fraction']

def trip_files(folder,require_route=False):
    folder=Path(folder); name=folder.name
    track=np.loadtxt(folder/f'{name}.track',ndmin=2)
    nodes=np.loadtxt(folder/f'{name}.nodes',ndmin=2)
    arcs=np.loadtxt(folder/f'{name}.arcs',dtype=int,ndmin=2)
    routefile=folder/f'{name}.route'
    route=np.loadtxt(routefile,dtype=int,ndmin=1) if routefile.exists() else None
    if require_route and route is None: raise ValueError('Missing reference route')
    return track,nodes,arcs,route

def project(lonlat,lat0):
    x=lonlat[:,0]*111320*np.cos(np.deg2rad(lat0))
    y=lonlat[:,1]*110540
    return np.column_stack((x,y))

def candidate_features(track,nodes,arcs,k=6,search_neighbors=32):
    lat0=float(np.mean(track[:,1])); gps=project(track[:,:2],lat0); xy=project(nodes,lat0)
    starts=xy[arcs[:,0]]; ends=xy[arcs[:,1]]; vec=ends-starts
    lens2=np.sum(vec*vec,axis=1); lens=np.sqrt(lens2)
    tree=cKDTree((starts+ends)/2)
    # Midpoint-only search can miss long road segments. This is a fast baseline,
    # not an exact global nearest-segment index.
    n=min(len(arcs),max(search_neighbors,k)); _,indices=tree.query(gps,k=n)
    indices=np.asarray(indices).reshape(len(gps),n)
    aa=starts[indices]; vv=vec[indices]; ll=lens2[indices]
    frac=np.clip(np.sum((gps[:,None,:]-aa)*vv,axis=2)/np.maximum(ll,1e-9),0,1)
    nearest=aa+frac[:,:,None]*vv
    distances=np.linalg.norm(gps[:,None,:]-nearest,axis=2)
    order=np.argsort(distances,axis=1)[:,:k]
    chosen=np.take_along_axis(indices,order,axis=1)
    dist=np.take_along_axis(distances,order,axis=1)
    frac=np.take_along_axis(frac,order,axis=1)
    dt=np.diff(track[:,2],prepend=track[0,2]); dt[0]=1
    movement=np.diff(gps,axis=0,prepend=gps[:1]); speed=np.linalg.norm(movement,axis=1)/np.maximum(dt,1)
    direction=vec[chosen]; dot=np.sum(direction*movement[:,None,:],axis=2)
    norm=np.linalg.norm(direction,axis=2)*np.linalg.norm(movement,axis=1)[:,None]
    angle=np.arccos(np.clip(np.abs(dot)/np.maximum(norm,1e-9),-1,1))
    angle[np.linalg.norm(movement,axis=1)<1e-6]=0
    x=np.stack([dist,angle,np.broadcast_to(speed[:,None],dist.shape),np.broadcast_to(dt[:,None],dist.shape),lens[chosen],np.broadcast_to(np.arange(k),dist.shape),frac],axis=2)
    return chosen,x,gps,xy

def derived_labels(track,nodes,arcs,route,candidates):
    # Weak labels: align observations monotonically to the provided chronological route.
    # This does NOT establish independently verified per-observation truth.
    if route is None or not len(route): return None,None
    lat0=float(np.mean(track[:,1])); gps=project(track[:,:2],lat0); xy=project(nodes,lat0)
    valid=route[(route>=0)&(route<len(arcs))]
    if not len(valid): return None,None
    start=xy[arcs[valid,0]]; vec=xy[arcs[valid,1]]-start
    # Restrict candidate search to route geometry; monotone smoothing of index.
    tree=cKDTree((start+start+vec)/2)
    near=min(16,len(valid)); _,ix=tree.query(gps,k=near)
    ix=np.asarray(ix).reshape(len(gps),near)
    a=start[ix]; v=vec[ix]
    f=np.clip(np.sum((gps[:,None,:]-a)*v,axis=2)/np.maximum(np.sum(v*v,axis=2),1e-9),0,1)
    d=np.linalg.norm(gps[:,None,:]-(a+f[:,:,None]*v),axis=2)
    selected=ix[np.arange(len(gps)),np.argmin(d,axis=1)]
    # Monotone progression avoids jumping backwards through chronological route.
    selected=np.maximum.accumulate(selected)
    target=valid[selected]
    y=(candidates==target[:,None]).astype(int)
    hits=y.any(axis=1)
    return y,hits

def transition_cost(previous,current,arcs,xy,gap):
    a=arcs[previous]; b=arcs[current]
    # Connected arcs are rewarded; non-connected arcs penalized by endpoint gap.
    end=xy[a[1]]; begin=xy[b[0]]
    endpoint=np.linalg.norm(end[:,None,:]-begin[None,:,:],axis=2)
    linked=(a[:,None,1]==b[None,:,0]) | (previous[:,None]==current[None,:])
    penalty=np.where(linked,0.0,np.minimum(endpoint/35,12.0)+1.5)
    return penalty * (1 if gap<=5 else 0.35)

def viterbi(probs,candidates,arcs,xy,times):
    # Log emission and a graph-connectivity transition prior.
    logp=np.log(np.clip(probs,1e-6,1))
    t,k=logp.shape
    score=logp[0].copy(); back=np.zeros((t,k),dtype=np.int32)
    for i in range(1,t):
        cost=transition_cost(candidates[i-1],candidates[i],arcs,xy,times[i]-times[i-1])
        values=score[:,None]-cost
        back[i]=np.argmax(values,axis=0)
        score=values[back[i],np.arange(k)]+logp[i]
    path=np.empty(t,dtype=int); state=int(np.argmax(score))
    for i in range(t-1,-1,-1):
        path[i]=candidates[i,state]
        if i: state=back[i,state]
    return path

def predicted_geometry(path,nodes,arcs):
    return [[[float(nodes[arcs[a,0],1]),float(nodes[arcs[a,0],0])],[float(nodes[arcs[a,1],1]),float(nodes[arcs[a,1],0])]] for a in path]
