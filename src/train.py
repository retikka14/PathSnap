import argparse,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score,log_loss
from sklearn.model_selection import GroupKFold
from .core import trip_files,candidate_features,derived_labels,FEATURES

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'/'Student folder'
OUT=ROOT/'outputs'; OUT.mkdir(exist_ok=True)

def dataset(split,limit,k):
    folders=sorted(p for p in (DATA/split).iterdir() if p.is_dir())[:limit or None]
    xs=[];ys=[];gs=[];miss=[]
    for folder in folders:
        track,nodes,arcs,route=trip_files(folder,True)
        candidates,x,_,_=candidate_features(track,nodes,arcs,k)
        y,hits=derived_labels(track,nodes,arcs,route,candidates)
        if y is None: continue
        xs.append(x.reshape(-1,x.shape[-1]));ys.append(y.ravel());gs.extend([folder.name]*y.size)
        miss.append(1-float(hits.mean()))
        print(f'{split} {folder.name}: {len(track)} observations, weak-label candidate miss {miss[-1]:.3f}',flush=True)
    if not xs: raise ValueError(f'No trips in {split}')
    return np.concatenate(xs),np.concatenate(ys),np.array(gs),float(np.mean(miss))

def models():
    return {
      'logistic':LogisticRegression(max_iter=250,class_weight='balanced'),
      'decision_tree':DecisionTreeClassifier(max_depth=12,min_samples_leaf=12,random_state=42),
      'random_forest':RandomForestClassifier(n_estimators=70,max_depth=16,min_samples_leaf=5,n_jobs=-1,random_state=42),
      'extra_trees':ExtraTreesClassifier(n_estimators=70,max_depth=16,min_samples_leaf=5,n_jobs=-1,random_state=42),
      'hist_gradient_boosting':HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=20,random_state=42)
    }

def metrics(model,x,y,k):
    start=time.perf_counter(); p=model.predict_proba(x)[:,1]; elapsed=time.perf_counter()-start
    pred=(p>=0.5).astype(int)
    usable=(len(y)//k)*k
    top=np.argmax(p[:usable].reshape(-1,k),axis=1)
    yy=y[:usable].reshape(-1,k)
    return dict(macro_f1=float(f1_score(y,pred,average='macro',zero_division=0)),log_loss=float(log_loss(y,np.clip(p,1e-6,1-1e-6),labels=[0,1])),top1_accuracy=float(np.mean(yy[np.arange(len(top)),top])),inference_seconds=elapsed)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--train-trips',type=int,default=0);ap.add_argument('--test-trips',type=int,default=0);ap.add_argument('--k',type=int,default=6);ap.add_argument('--tune-trials',type=int,default=12)
    args=ap.parse_args();k=args.k
    if k<2: raise ValueError('k must be >=2')
    X,y,g,miss=dataset('train',args.train_trips,k)
    Xt,yt,gt,miss_test=dataset('student_test',args.test_trips,k)
    if len(np.unique(y))<2: raise ValueError('Training labels have only one class; inspect alignment and candidates')
    results=[];fitted={}
    # Trip-grouped CV on training trips only; never split observations from a trip across folds.
    n_groups=len(np.unique(g)); n_splits=min(3,n_groups)
    if n_splits<2: raise ValueError('At least two training trips needed for CV')
    for name,model in models().items():
        scores=[]
        for a,b in GroupKFold(n_splits=n_splits).split(X,y,g):
            m=models()[name];m.fit(X[a],y[a]);scores.append(metrics(m,X[b],y[b],k)['macro_f1'])
        start=time.perf_counter();model.fit(X,y);training=time.perf_counter()-start
        result=dict(model=name,cv_macro_f1=float(np.mean(scores)),train_seconds=training,train_candidate_miss=miss,student_candidate_miss=miss_test,**metrics(model,Xt,yt,k))
        results.append(result);fitted[name]=model;print('RESULT',json.dumps(result),flush=True)
    df=pd.DataFrame(results).sort_values(['cv_macro_f1','top1_accuracy'],ascending=False)
    df.to_csv(OUT/'model_comparison.csv',index=False)
    best_name=str(df.iloc[0]['model']);best=fitted[best_name]
    # Bayesian optimization: optimize only on training-trip group validation.
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    from sklearn.model_selection import GroupShuffleSplit
    a,b=next(GroupShuffleSplit(n_splits=1,test_size=0.2,random_state=42).split(X,y,g))
    def objective(trial):
        kind=trial.suggest_categorical('family',['random_forest','extra_trees','hist_gradient_boosting'])
        if kind=='hist_gradient_boosting':
            m=HistGradientBoostingClassifier(max_iter=trial.suggest_int('iterations',50,180),max_leaf_nodes=trial.suggest_int('leaf_nodes',10,40),learning_rate=trial.suggest_float('learning_rate',0.02,0.2,log=True),random_state=42)
        else:
            cls=RandomForestClassifier if kind=='random_forest' else ExtraTreesClassifier
            m=cls(n_estimators=trial.suggest_int('trees',40,120),max_depth=trial.suggest_int('depth',7,22),min_samples_leaf=trial.suggest_int('min_leaf',2,15),n_jobs=-1,random_state=42)
        m.fit(X[a],y[a]);return metrics(m,X[b],y[b],k)['macro_f1']
    study=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective,n_trials=args.tune_trials)
    pd.DataFrame([dict(number=t.number,value=t.value,**t.params) for t in study.trials]).to_csv(OUT/'bayesian_trials.csv',index=False)
    params=study.best_params;kind=params['family']
    if kind=='hist_gradient_boosting': tuned=HistGradientBoostingClassifier(max_iter=params['iterations'],max_leaf_nodes=params['leaf_nodes'],learning_rate=params['learning_rate'],random_state=42)
    else:
        cls=RandomForestClassifier if kind=='random_forest' else ExtraTreesClassifier
        tuned=cls(n_estimators=params['trees'],max_depth=params['depth'],min_samples_leaf=params['min_leaf'],n_jobs=-1,random_state=42)
    tuned.fit(X,y)
    tuned_metrics=metrics(tuned,Xt,yt,k)
    dump({'model':tuned,'features':FEATURES,'k':k,'best_baseline':best_name,'params':params},OUT/'model.joblib')
    with open(OUT/'training_summary.json','w') as f:json.dump({'best_baseline':best_name,'best_bayesian_params':params,'bayesian_validation_macro_f1':study.best_value,'tuned_student_metrics':tuned_metrics,'warning':'Labels derived heuristically from chronological reference route; metrics are proxy estimates, not verified observation truth.'},f,indent=2)
    print('SAVED',OUT/'model.joblib',OUT/'model_comparison.csv',OUT/'bayesian_trials.csv',flush=True)
if __name__=='__main__':main()
