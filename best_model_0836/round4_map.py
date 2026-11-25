"""Checked-label MAP adaptation of the competition-trained patch GMMs."""
import argparse
import json
import numpy as np
from scipy.special import softmax
from sklearn.model_selection import StratifiedKFold
from joblib import load
from threadpoolctl import threadpool_limits
from advance import BASE,R3,metrics
from advance_finish import labels
from advance_gmm import latent
from round4 import R4

def statistics(view,train,test,ci):
    target=R4/f'mapstats_{view}.npz'
    if target.exists():
        with np.load(target) as a: return a['q0'],a['q1']
    kind='log' if view.startswith('log') else 'pcen'; width=int(view[len(kind):])
    frames=latent(kind,width,train,test,ci); gmm=load(R3/f'dictionary_{view}_k64.joblib')[2]
    q0=np.zeros((len(train)+len(test),64)); q1=np.zeros((len(train)+len(test),64,64))
    with threadpool_limits(limits=2):
        for i,frame in frames.items():
            q=gmm.predict_proba(frame); q0[i]=q.mean(0); q1[i]=q.T@frame/len(frame)
    np.savez_compressed(target,q0=q0,q1=q1)
    return q0,q1

def map_predict(q0,q1,gmm,y,training,query,relevance):
    coeff=[]; intercept=[]
    for c in range(41):
        idx=training[y[training]==c]
        n=100*q0[idx].sum(0); first=100*q1[idx].sum(0)
        alpha=n/(n+relevance)
        means=alpha[:,None]*first/np.maximum(n[:,None],1e-8)+(1-alpha[:,None])*gmm.means_
        weights=(n+200*gmm.weights_)/(n.sum()+200)
        delta=(means-gmm.means_)/gmm.covariances_
        bias=np.log(np.maximum(weights,1e-12)/gmm.weights_)-.5*((means**2-gmm.means_**2)/gmm.covariances_).sum(1)
        coeff.append(delta.ravel()); intercept.append(bias)
    with threadpool_limits(limits=2):
        logits=q1[query].reshape(len(query),-1)@np.asarray(coeff).T+q0[query]@np.asarray(intercept).T
    return softmax(4*logits,axis=1)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--stage',choices=['search','confirm','final'],default='search'); ap.add_argument('--seed',type=int,default=73); args=ap.parse_args()
    train,test,cls,y,ci,ui=labels()
    configs=[dict(name=f'map_{v}_r{r}',kind='map',view=v,relevance=r) for v in ['log12','pcen24'] for r in [8.,32.]]
    if args.stage in ['confirm','final']: configs=[c for c in json.loads((R4/'selection.json').read_text())['configs'] if c['kind']=='map']
    out=R4/f'seed{args.seed}'; out.mkdir(exist_ok=True)
    for view in sorted(set(c['view'] for c in configs)):
        q0,q1=statistics(view,train,test,ci); gmm=load(R3/f'dictionary_{view}_k64.joblib')[2]
        for c in [a for a in configs if a['view']==view]:
            path=out/f'{c["name"]}.json'
            if path.exists() and args.stage!='final': continue
            if args.stage=='final':
                p=map_predict(q0,q1,gmm,y,ci,np.arange(len(train),len(train)+len(test)),c['relevance'])
                np.savez_compressed(R4/f'{c["name"]}_full.npz',pred=p,ids=test.id.values.astype(str)); continue
            p=np.zeros((len(ci),41))
            for tr,va in StratifiedKFold(5,shuffle=True,random_state=args.seed).split(ci,y[ci]):
                p[va]=map_predict(q0,q1,gmm,y,ci[tr],ci[va],c['relevance'])
            r=dict(config=c,seed=args.seed,folds=5,**metrics(y[ci],p))
            path.write_text(json.dumps(r,indent=2)); np.savez_compressed(path.with_suffix('.npz'),oof=p)
            print(c['name'],r['micro_map3'],r['macro_map3'],flush=True)

if __name__=='__main__': main()
