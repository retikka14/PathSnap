from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from src.core import trip_files, candidate_features, derived_labels, project

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'/'Student folder'/'student_test'
OUT=ROOT/'outputs'/'diagnostics'
OUT.mkdir(parents=True,exist_ok=True)

def point_segment_dist(points, starts, ends):
    v=ends-starts
    den=np.maximum(np.einsum('ij,ij->i',v,v),1e-12)
    f=np.clip(np.einsum('ijk,jk->ij',points[:,None,:]-starts[None,:,:],v)/den[None,:],0,1)
    return np.linalg.norm(points[:,None,:]-(starts[None,:,:]+f[:,:,None]*v[None,:,:]),axis=2)

def exact_nearest_distances(gps, starts, ends, batch_points=32, batch_arcs=1500):
    # Exact minimum distance to any segment, chunked to avoid enormous arrays.
    result=np.full(len(gps),np.inf)
    for i in range(0,len(gps),batch_points):
        pp=gps[i:i+batch_points]; best=np.full(len(pp),np.inf)
        for j in range(0,len(starts),batch_arcs):
            d=point_segment_dist(pp,starts[j:j+batch_arcs],ends[j:j+batch_arcs])
            best=np.minimum(best,d.min(axis=1))
        result[i:i+len(pp)]=best
    return result

def analyze(trip,k,exact_sample):
    folder=DATA/trip
    if not folder.is_dir(): raise FileNotFoundError(folder)
    track,nodes,arcs,route=trip_files(folder,True)
    cand,features,gps,xy=candidate_features(track,nodes,arcs,k=k)
    weak,hits=derived_labels(track,nodes,arcs,route,cand)
    lat0=float(np.mean(track[:,1])); gps=project(track[:,:2],lat0); xy=project(nodes,lat0)
    starts=xy[arcs[:,0]]; ends=xy[arcs[:,1]]
    # Independently measure proximity to the reference route (without monotonic alignment).
    ref=np.unique(route[(route>=0)&(route<len(arcs))])
    refdist=np.full(len(gps),np.inf)
    for j in range(0,len(ref),1000):
        idx=ref[j:j+1000]
        for i in range(0,len(gps),64):
            d=point_segment_dist(gps[i:i+64],starts[idx],ends[idx])
            refdist[i:i+64]=np.minimum(refdist[i:i+64],d.min(axis=1))
    closest_candidate=features[:,:,0].min(axis=1)
    # The monotonic heuristic can force later points onto distant route segments.
    # Compare geometric proximity separately from weak-label coverage.
    ix=np.unique(np.linspace(0,len(gps)-1,min(exact_sample,len(gps)),dtype=int))
    exact=exact_nearest_distances(gps[ix],starts,ends)
    approx=closest_candidate[ix]
    sampled_miss=(approx>exact+1e-4)
    result={
      'trip':trip,'observations':len(gps),'road_arcs':len(arcs),'reference_arcs':len(route),
      'candidate_coverage_proxy':float(hits.mean()),
      'median_nearest_candidate_m':float(np.median(closest_candidate)),
      'p95_nearest_candidate_m':float(np.percentile(closest_candidate,95)),
      'median_nearest_reference_m':float(np.median(refdist)),
      'p95_nearest_reference_m':float(np.percentile(refdist,95)),
      'reference_within_25m_fraction':float(np.mean(refdist<=25)),
      'reference_within_50m_fraction':float(np.mean(refdist<=50)),
      'sampled_exact_search_points':len(ix),
      'sampled_midpoint_search_miss_fraction':float(sampled_miss.mean()),
      'sampled_midpoint_excess_distance_m_p95':float(np.percentile(approx-exact,95)),
      'note':'Coverage uses heuristic monotonic labels; exact-search diagnostic is sampled, not a full replacement.'
    }
    df=pd.DataFrame({'gps_index':np.arange(len(gps)),'time':track[:,2], 'weak_label_in_candidates':hits.astype(int),'nearest_candidate_m':closest_candidate,'nearest_reference_m':refdist})
    df.to_csv(OUT/f'{trip}_observations.csv',index=False)
    (OUT/f'{trip}_summary.json').write_text(json.dumps(result,indent=2))
    return result,df

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--trips',nargs='+',default=['00000056','00000007'])
    ap.add_argument('--k',type=int,default=6)
    ap.add_argument('--exact-sample',type=int,default=120)
    args=ap.parse_args()
    summaries=[]; dfs={}
    for trip in args.trips:
        result,df=analyze(trip,args.k,args.exact_sample)
        summaries.append(result);dfs[trip]=df
        print(json.dumps(result,indent=2),flush=True)
    pd.DataFrame(summaries).to_csv(OUT/'comparison.csv',index=False)
    fig,axes=plt.subplots(2,1,figsize=(12,8),sharex=False)
    for trip,df in dfs.items():
        x=np.arange(len(df))/max(len(df)-1,1)*100
        axes[0].plot(x,df.nearest_reference_m.clip(upper=200),label=trip,alpha=.8)
        axes[1].plot(x,df.weak_label_in_candidates.rolling(100,min_periods=1).mean(),label=trip)
    axes[0].set(ylabel='Nearest reference segment (m; capped at 200)',title='Reference geometry proximity')
    axes[1].set(xlabel='Trip progress (%)',ylabel='Rolling weak-label coverage',title='Candidate coverage by trip progress',ylim=(-.05,1.05))
    for ax in axes:ax.grid(alpha=.25);ax.legend()
    fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=170);plt.close(fig)
    print('SAVED outputs/diagnostics/comparison.csv and comparison.png')
if __name__=='__main__':main()
