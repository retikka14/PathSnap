from pathlib import Path
import os
import numpy as np
from joblib import load
from fastapi import FastAPI,HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .core import trip_files,candidate_features,viterbi,predicted_geometry
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'/'Student folder'; ARTIFACT=ROOT/'outputs'/'model.joblib'
app=FastAPI(title='PathSnap API')
app.add_middleware(CORSMiddleware,allow_origins=['*'],allow_methods=['GET'],allow_headers=['*'])
@app.get('/health')
def health():return {'status':'ok','model_ready':ARTIFACT.exists()}
@app.get('/trips')
def trips():
    split=os.environ.get('PATHSNAP_EVAL_SPLIT','student_test')
    root=DATA/split
    if not root.exists():return {'trips':[],'partition':split}
    return {'partition':split,'trips':[p.name for p in sorted(root.iterdir()) if p.is_dir()][:10]}
@app.get('/predict/{trip_id}')
def predict(trip_id:str):
    if not trip_id.isdigit() or len(trip_id)!=8:raise HTTPException(400,'Invalid trip ID')
    if not ARTIFACT.exists():raise HTTPException(503,'Train a model first')
    split=os.environ.get('PATHSNAP_EVAL_SPLIT','student_test')
    root=(DATA/split).resolve();folder=(root/trip_id).resolve()
    if folder.parent!=root or not folder.is_dir():raise HTTPException(404,'Trip not found')
    try:
        artifact=load(ARTIFACT);model=artifact['model'];k=artifact['k']
        track,nodes,arcs,_=trip_files(folder)
        cand,x,_,xy=candidate_features(track,nodes,arcs,k)
        p=model.predict_proba(x.reshape(-1,x.shape[-1]))[:,1].reshape(len(track),k)
        # Normalize candidate binary scores into an emission distribution.
        p=p/np.maximum(p.sum(axis=1,keepdims=True),1e-9)
        raw=cand[np.arange(len(track)),np.argmax(p,axis=1)]
        decoded=viterbi(p,cand,arcs,xy,track[:,2])
        # The API deliberately never loads or returns reference labels.
        return {'trip_id':trip_id,'observations':len(track),'gps':[[float(r[1]),float(r[0])] for r in track], 'raw_arc_ids':raw.tolist(),'decoded_arc_ids':decoded.tolist(),'predicted_segments':predicted_geometry(decoded,nodes,arcs)}
    except Exception as e:raise HTTPException(500,f'Inference failed: {type(e).__name__}: {e}')
