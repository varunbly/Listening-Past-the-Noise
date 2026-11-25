"""Optional classical graph label propagation; no held-out labels are seeded."""
import json
import argparse
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from threadpoolctl import threadpool_limits
from improve import BASE, prepare, score

def graph(x,k=30):
    with threadpool_limits(limits=4):
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
    ap.add_argument('--views',default='')
    ap.add_argument('--neighbors',default='15,40')
    ap.add_argument('--alphas',default='.8,.95')
    ap.add_argument('--priors',default='.5')
    ap.add_argument('--target-powers',default='0')
    args=ap.parse_args()
    train,test,x,caches=prepare(BASE/'project4',args.jobs)
    cls=np.array(sorted(train.label.unique()))
    y=np.searchsorted(cls,train.label)
    ci=np.where(train.checked.values==1)[0]
    ui=np.where(train.checked.values==0)[0]
    dev,hold=train_test_split(ci,test_size=.2,stratify=y[ci],random_state=2026)
    dev=np.sort(dev)
    views=['stats']
    for k in [16,32,64]:
        p=BASE/'cache'/f'fisher_audio-v1_{k}.npz'
        if p.exists(): views.append(f'fisher{k}')
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
                        tag=f'graph_{name}_k{neighbors}_a{alpha}_p{prior}{extra}_s42_4fold'
                        out=BASE/'results'/'dev'/f'{tag}.json'
                        if out.exists(): continue
                        oof=np.zeros((len(dev),len(cls)))
                        for tr,va in StratifiedKFold(4,shuffle=True,random_state=42).split(dev,y[dev]):
                            f=propagate(g,y,dev[tr],ui,alpha,prior,tp)
                            oof[va]=f[dev[va]]
                        res={'config':dict(name=tag,kind='graph',view=name,k=neighbors,noise_alpha=alpha,prior=prior,target_power=tp),'seed':42,'folds':4,'map3':score(y[dev],oof)}
                        out.write_text(json.dumps(res,indent=2))
                        np.savez_compressed(out.with_suffix('.npz'),oof=oof)
                        print(tag,res['map3'],flush=True)

if __name__=='__main__': main()
