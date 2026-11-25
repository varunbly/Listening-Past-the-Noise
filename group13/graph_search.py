"""Optional classical graph label propagation; no held-out labels are seeded."""
from course_input import course_data
import json
import argparse
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from improve import BASE, prepare, score

def graph(x,k=30):
    d=euclidean_distances(x,squared=True)
    np.fill_diagonal(d,np.inf)
    near=np.stack([np.argpartition(row,k)[:k].copy() for row in d])
    rows=np.repeat(np.arange(len(x)),k)
    dd=d[rows,near.ravel()]
    scale=np.sqrt(np.max(dd.reshape(len(x),k),axis=1)).clip(1e-5)
    weights=np.exp(-dd/(scale[rows]*scale[near.ravel()]))
    w=csr_matrix((weights,(rows,near.ravel())),shape=d.shape)
    w=(w+w.T)*.5
    return w.multiply(1/np.maximum(np.asarray(w.sum(1)).ravel(),1e-12)[:,None]).tocsr()

def propagate(g,y,checked_idx,unchecked_idx,noise_alpha=.95,prior=0.,target_power=0.):
    n=g.shape[0]
    alpha=np.ones(n,dtype='float32')
    alpha[checked_idx]=.05
    alpha[unchecked_idx]=noise_alpha
    seed=np.zeros((n,int(y.max())+1),dtype='float32')
    idx=np.r_[checked_idx,unchecked_idx]
    seed[idx,y[idx]]=1.
    source=(1-alpha[:,None])*seed
    f=source.copy()
    for _ in range(250):
        nxt=alpha[:,None]*(g@f)+source
        if np.max(np.abs(nxt-f))<1e-6:
            f=nxt
            break
        f=nxt
    if prior:
        f/=np.maximum(f.mean(0),1e-8)**prior
    if target_power:
        target=np.bincount(y[checked_idx],minlength=seed.shape[1]).astype(float)
        target/=target.sum()
        f*=target[None,:]**target_power
    return f/np.maximum(f.sum(1,keepdims=True),1e-12)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--jobs',type=int,default=2)
    ap.add_argument('--scope',choices=['dev','full'],default='dev')
    ap.add_argument('--folds',type=int,default=4)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--views',default='')
    ap.add_argument('--neighbors',default='15,40')
    ap.add_argument('--alphas',default='.8,.95')
    ap.add_argument('--priors',default='.5')
    ap.add_argument('--target-powers',default='0')
    args=ap.parse_args()
    train,test,x,caches=prepare(course_data(BASE),args.jobs)
    cls=np.array(sorted(train.label.unique()))
    y=np.searchsorted(cls,train.label)
    ci=np.where(train.checked.values==1)[0]
    ui=np.where(train.checked.values==0)[0]
    eval_idx=ci
    if args.scope=='dev':
        eval_idx,hold=train_test_split(ci,test_size=.2,stratify=y[ci],random_state=2026)
        eval_idx=np.sort(eval_idx)
    views=['stats']
    for k in [16,32,64]:
        p=BASE/'cache'/f'fisher_audio-v1_{k}.npz'
        if p.exists(): views.append(f'fisher{k}')
    for p in sorted((BASE/'cache').glob('fisher_audio-v1_*t*.npz')):
        suffix=p.stem.removeprefix('fisher_audio-v1_')
        views.append(f'fisher{suffix}')
    if 'fisher64' in views: views.append('combined')
    if args.views: views=[v for v in views if v in args.views.split(',')]
    for name in views:
        if name=='stats':
            view=normalize(np.clip(StandardScaler().fit(x[ui]).transform(x),-6,6))
        elif name=='combined':
            statsx=normalize(np.clip(StandardScaler().fit(x[ui]).transform(x),-6,6))
            fish=np.load(BASE/'cache'/'fisher_audio-v1_64.npz')['x']
            view=np.concatenate([statsx/np.sqrt(2),fish/np.sqrt(2)],axis=1)
            del statsx,fish
        else:
            view=np.load(BASE/'cache'/f'fisher_audio-v1_{name[6:]}.npz')['x']
        for neighbors in map(int,args.neighbors.split(',')):
            print('Graph',name,neighbors,flush=True)
            g=graph(view,neighbors)
            for alpha in map(float,args.alphas.split(',')):
                for prior in map(float,args.priors.split(',')):
                    for tp in map(float,args.target_powers.split(',')):
                        extra=f'_tp{tp}' if tp else ''
                        tag=f'graph_{name}_k{neighbors}_a{alpha}_p{prior}{extra}_s{args.seed}_{args.folds}fold'
                        out=BASE/'results'/args.scope/f'{tag}.json'
                        out.parent.mkdir(parents=True,exist_ok=True)
                        if out.exists(): continue
                        oof=np.zeros((len(eval_idx),len(cls)))
                        for tr,va in StratifiedKFold(args.folds,shuffle=True,random_state=args.seed).split(eval_idx,y[eval_idx]):
                            f=propagate(g,y,eval_idx[tr],ui,alpha,prior,tp)
                            oof[va]=f[eval_idx[va]]
                        res={'config':dict(name=tag,kind='graph',view=name,k=neighbors,noise_alpha=alpha,prior=prior,target_power=tp),'seed':args.seed,'folds':args.folds,'scope':args.scope,'map3':score(y[eval_idx],oof)}
                        out.write_text(json.dumps(res,indent=2))
                        np.savez_compressed(out.with_suffix('.npz'),oof=oof)
                        print(tag,res['map3'],flush=True)

if __name__=='__main__': main()
