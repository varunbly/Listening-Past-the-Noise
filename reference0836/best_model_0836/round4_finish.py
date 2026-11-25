"""Freeze round-four selection before new resampling; deliver validated CSV."""
from course_input import course_data
import argparse
import json
import hashlib
import zipfile
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from advance import BASE,R3,metrics,distance,predict_graph
from advance_finish import labels,baseline,assemble as assemble3,row_priors,calibrated
from deliver import temper
from refine import graph_from_distance
from round4 import R4

def previous(seed):
    out=R4/f'seed{seed}'; out.mkdir(exist_ok=True)
    path=out/'previous.npz'
    if path.exists(): return
    train,test,cls,y,ci,ui=labels()
    if not (R3/f'seed{seed}/r2_ensemble.npz').exists(): baseline(seed)
    selection=json.loads((R3/'selection.json').read_text())
    for c in selection['configs']:
        target=R3/f'seed{seed}'/f'{c["name"]}.npz'
        if target.exists(): continue
        d=c['weight']*distance(c['view'])+(1-c['weight'])*distance('original')
        np.fill_diagonal(d,np.inf); g=graph_from_distance(d,c['k']); del d
        p=np.zeros((len(ci),41))
        for tr,va in StratifiedKFold(5,shuffle=True,random_state=seed).split(ci,y[ci]):
            p[va]=predict_graph(c,g,y,ci[tr],ui)[ci[va]]
        np.savez_compressed(target,oof=p)
    p=assemble3(selection,seed)
    np.savez_compressed(path,oof=p)
    (out/'previous.json').write_text(json.dumps(dict(seed=seed,**metrics(y[ci],p)),indent=2))
    print('Previous round-three baseline',seed,metrics(y[ci],p),flush=True)

def choose():
    path=R4/'selection.json'
    if path.exists(): raise ValueError('Round-four selection already frozen')
    train,test,cls,y,ci,ui=labels(); prior=row_priors(y,ci,73)
    records=[('previous',None,np.load(R3/'selected_oof.npz')['pred'])]
    for p in sorted((R4/'seed73').glob('*.json')):
        r=json.loads(p.read_text())
        if 'config' in r and r['macro_map3']>=.78: records.append((p.stem,r['config'],np.load(p.with_suffix('.npz'))['oof']))
    candidates=[]
    for power in [.5,1.,2.]:
        arrays=[temper(r[2],power) for r in records]
        for beta in [0.,.125,.25]:
            p=arrays[0]; weights=np.zeros(len(records)); weights[0]=1.
            best=metrics(y[ci],calibrated(p,prior,beta))['macro_map3']
            for _ in range(2):
                improvement=None
                for j,a in enumerate(arrays[1:],1):
                    if weights[j]>0: continue
                    for w in [.15,.3,.5]:
                        q=(1-w)*p+w*a; m=metrics(y[ci],calibrated(q,prior,beta))
                        if m['micro_map3']<.827: continue
                        if m['macro_map3']>best+.001 and (improvement is None or m['macro_map3']>improvement[0]): improvement=(m['macro_map3'],j,w,q)
                if improvement is None: break
                best,j,w,p=improvement; weights*=1-w; weights[j]=w
            if metrics(y[ci],calibrated(p,prior,beta))['micro_map3']>=.827:
                candidates.append((best,power,beta,weights,p))
    best,power,beta,weights,p=max(candidates,key=lambda c:c[0])
    final=calibrated(p,prior,beta)
    s={'selection_seed':73,'confirmation_seed':419,'power':power,'beta':beta,'objective':'macro_MAP@3_with_micro_floor',
       'selection_metrics':metrics(y[ci],final),
       'models':[dict(name=records[i][0],weight=float(w),config=records[i][1]) for i,w in enumerate(weights) if w],
       'configs':[records[i][1] for i,w in enumerate(weights) if w and i>0],
       'note':'Frozen before seed-419 confirmation. Public score 0.83227 is recorded only as the previous submission outcome, not used to select parameters.'}
    path.write_text(json.dumps(s,indent=2)); np.savez_compressed(R4/'selection_oof.npz',pred=final,y=y[ci])
    print(json.dumps(s,indent=2),flush=True)

def blend(s,seed=None):
    train,test,cls,y,ci,ui=labels(); p=None
    for m in s['models']:
        if m['name']=='previous':
            path=R3/'test_predictions.npz' if seed is None else R4/f'seed{seed}/previous.npz'
            key='probabilities' if seed is None else 'oof'
        else: path=R4/f'{m["name"]}_full.npz' if seed is None else R4/f'seed{seed}'/f'{m["name"]}.npz'; key='pred' if seed is None else 'oof'
        with np.load(path) as data:
            if seed is None: assert data['ids'].tolist()==test.id.tolist()
            a=data[key]
        q=m['weight']*temper(a,s['power']); p=q if p is None else p+q
    prior=np.bincount(y[ci],minlength=41)/len(ci) if seed is None else row_priors(y,ci,seed)
    return calibrated(p,prior,s['beta'])

def confirm():
    s=json.loads((R4/'selection.json').read_text()); seed=419
    train,test,cls,y,ci,ui=labels(); a=np.load(R4/'seed419/previous.npz')['oof']; b=blend(s,seed)
    old=metrics(y[ci],a); new=metrics(y[ci],b)
    av=((np.argsort(-a,axis=1)[:,:3]==y[ci,None])/np.arange(1,4)).sum(1)
    bv=((np.argsort(-b,axis=1)[:,:3]==y[ci,None])/np.arange(1,4)).sum(1)
    rng=np.random.default_rng(419); bootstrap=[]
    for _ in range(3000): bootstrap.append(np.mean([rng.choice((bv-av)[y[ci]==c],int((y[ci]==c).sum())).mean() for c in range(41)]))
    r={'previous':old,'candidate':new,'paired_macro_gain':new['macro_map3']-old['macro_map3'],
       'paired_stratified_bootstrap_95':np.quantile(bootstrap,[.025,.975]).tolist(),
       'accepted':new['macro_map3']>old['macro_map3']+.001 and new['micro_map3']>=old['micro_map3']-.01,
       'seed':seed,'folds':5,'selection':s,'previous_public_score':.83227,
       'note':'Repeated CV on known checked training data, not a fresh independent test. New public score is unknown; 0.9 is not guaranteed.'}
    (R4/'confirmation.json').write_text(json.dumps(r,indent=2)); np.savez_compressed(R4/'confirmation_oof.npz',previous=a,pred=b,y=y[ci])
    print(json.dumps(r,indent=2),flush=True)

def final():
    r=json.loads((R4/'confirmation.json').read_text()); s=r['selection']
    if not r['accepted']:
        raise ValueError('No confirmed gain: keep the previous submission rather than generating an identical ZIP.')
    p=blend(s)
    train,test,cls,y,ci,ui=labels(); assert p.shape==(1592,41) and np.isfinite(p).all()
    sub=pd.DataFrame({'id':test.id,'label':[' '.join(cls[row]) for row in np.argsort(-p,axis=1)[:,:3]]})
    assert sub.id.is_unique and all(len(v.split())==len(set(v.split()))==3 for v in sub.label)
    path=BASE/'submission_round4.csv'; sub.to_csv(path,index=False)
    assert pd.read_csv(path,dtype=str).id.tolist()==test.id.tolist()
    np.savez_compressed(R4/'test_predictions.npz',ids=test.id.values.astype(str),probabilities=p,classes=cls)
    r['submission_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    r['train_csv_sha256']=hashlib.sha256((course_data(BASE) / 'train.csv').read_bytes()).hexdigest()
    r['test_csv_sha256']=hashlib.sha256((course_data(BASE) / 'test.csv').read_bytes()).hexdigest()
    (R4/'final_report.json').write_text(json.dumps(r,indent=2))
    target=BASE.parent/'Kaggle_Submission_Round4_Only.zip'
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z: z.write(path,'submission.csv')
    print('Validated',target,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('stage',choices=['baseline','select','confirm','final']); ap.add_argument('--seed',type=int,default=419); a=ap.parse_args()
    if a.stage=='baseline': previous(a.seed)
    elif a.stage=='select': choose()
    elif a.stage=='confirm': confirm()
    else: final()
