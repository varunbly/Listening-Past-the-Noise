"""Round three: unsupervised spectrotemporal patch Fisher representations.

No external audio, pretrained model, neural network, or leaderboard label is used.
Model selection tracks macro MAP@3 to expose unequal checked-class support.
"""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:
    os.environ.setdefault(key,'1')
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from scipy.special import softmax
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.svm import SVC
from joblib import Parallel,delayed,dump,load
from threadpoolctl import threadpool_limits
from improve import BASE,prepare,score
from refine import graph_from_distance,robust_propagate

R3=BASE/'round3'; R3.mkdir(exist_ok=True)

def metrics(y,p):
    hit=(np.argsort(-p,axis=1)[:,:3]==y[:,None])/np.arange(1,4)
    values=hit.sum(1)
    per=np.array([values[y==c].mean() for c in range(p.shape[1])])
    return {'micro_map3':float(values.mean()),'macro_map3':float(per.mean()),'per_class':per.tolist()}

def spectrogram(path,target):
    if Path(target).exists(): return
    with threadpool_limits(limits=1):
        y,sr=sf.read(path,dtype='float32')
        if y.ndim==2: y=y.mean(1)
        if sr!=22050: y=librosa.resample(y,orig_sr=sr,target_sr=22050); sr=22050
        y=np.pad(y,(0,max(0,4096-len(y))))
        y/=max(float(np.abs(y).max()),1e-8)
        power=np.abs(librosa.stft(y,n_fft=1024,hop_length=256))**2
        m=librosa.feature.melspectrogram(S=power,sr=sr,n_mels=48,fmin=20)
        log=librosa.power_to_db(m,ref=np.max,top_db=80)
        pcen=np.log1p(librosa.pcen(m*2**20,sr=sr,hop_length=256,gain=.98,bias=2,power=.5,time_constant=.4))
        energy=m.sum(0)
        target=Path(target); tmp=target.with_suffix('.tmp.npz')
        np.savez_compressed(tmp,log=log.astype('float16'),pcen=pcen.astype('float16'),energy=energy.astype('float32'))
        tmp.replace(target)

def cache_spectrograms(train,test,jobs):
    folder=R3/'spectrograms'; folder.mkdir(exist_ok=True)
    files=[]; todo=[]
    for split,df in [('train',train),('test',test)]:
        for ident in df.id:
            cache=folder/f'{split}_{ident}.npz'; files.append(cache)
            if not cache.exists(): todo.append((BASE/'project4/audio/audio'/split/f'{ident}.wav',cache))
    start=time.time()
    for i in range(0,len(todo),256):
        Parallel(n_jobs=jobs)(delayed(spectrogram)(p,c) for p,c in todo[i:i+256])
        print('Patch spectrograms',min(i+256,len(todo)),'/',len(todo),'seconds',round(time.time()-start),flush=True)
    return files

def patches(file,kind,width,maximum=180):
    with np.load(file) as item:
        a=item[kind].astype('float32'); energy=item['energy']
    if a.shape[1]<width:
        a=np.pad(a,((0,0),(0,width-a.shape[1])),mode='edge')
        energy=np.pad(energy,(0,width-len(energy)),mode='edge')
    windows=np.lib.stride_tricks.sliding_window_view(a,width,axis=1)
    centers=np.arange(0,windows.shape[1],3)
    active=energy[np.minimum(centers+width//2,len(energy)-1)]>energy.max()*.0025
    if active.sum()>2: centers=centers[active]
    if len(centers)>maximum: centers=centers[np.linspace(0,len(centers)-1,maximum).astype(int)]
    return windows[:,centers,:].transpose(1,0,2).reshape(len(centers),-1).copy()

def fit_dictionary(files,ui,kind,width,k):
    target=R3/f'dictionary_{kind}{width}_k{k}.joblib'
    previous=load(target) if target.exists() else None
    if previous is not None and previous[2].converged_: return previous
    rng=np.random.default_rng(912)
    pool=[]
    for j,i in enumerate(ui):
        a=patches(files[i],kind,width,maximum=30)
        pool.append(a[rng.choice(len(a),min(12,len(a)),replace=False)])
        if j%1024==0: print('Dictionary patches',kind,width,j,'/',len(ui),flush=True)
    pool=np.concatenate(pool)
    if len(pool)>50000: pool=pool[rng.choice(len(pool),50000,replace=False)]
    scaler=StandardScaler().fit(pool) if previous is None else previous[0]
    with threadpool_limits(limits=4):
        pca=PCA(64,whiten=True,svd_solver='randomized',random_state=912).fit(scaler.transform(pool)) if previous is None else previous[1]
        z=np.clip(pca.transform(scaler.transform(pool)),-8,8).astype('float32')
        gmm=GaussianMixture(k,covariance_type='diag',max_iter=80,reg_covar=.01,random_state=912) if previous is None else previous[2]
        if previous is not None: gmm.set_params(warm_start=True,max_iter=120)
        gmm.fit(z)
    bundle=(scaler,pca,gmm)
    dump(bundle,target)
    print('Dictionary fitted',kind,width,k,'converged',gmm.converged_,'iterations',gmm.n_iter_,flush=True)
    return bundle

def encode_batch(files,kind,width,bundle):
    scaler,pca,gmm=bundle
    result=[]
    with threadpool_limits(limits=1):
        for file in files:
            a=np.clip(pca.transform(scaler.transform(patches(file,kind,width))),-8,8)
            q=gmm.predict_proba(a); n=len(a)
            q0=q.sum(0)[:,None]/n; q1=q.T@a/n; q2=q.T@(a*a)/n
            means=gmm.means_; var=gmm.covariances_; w=gmm.weights_[:,None]
            u=(q1-q0*means)/np.sqrt(w*var)
            v=(q2-2*q1*means+q0*(means**2-var))/np.sqrt(2*w*var**2)
            f=np.r_[u.ravel(),v.ravel()]
            f=np.sign(f)*np.sqrt(np.abs(f))
            result.append((f/max(np.linalg.norm(f),1e-12)).astype('float32'))
    return np.stack(result)

def bow_batch(files,kind,width,bundle):
    scaler,pca,gmm=bundle; result=[]
    with threadpool_limits(limits=1):
        for file in files:
            a=np.clip(pca.transform(scaler.transform(patches(file,kind,width))),-8,8)
            q=gmm.predict_proba(a)
            parts=[]
            for count in [1,2,4]:
                for chunk in np.array_split(q,count):
                    hist=chunk.mean(0) if len(chunk) else q.mean(0)
                    parts.append(np.sqrt(np.maximum(hist,0)))
            f=np.concatenate(parts); result.append((f/max(np.linalg.norm(f),1e-12)).astype('float32'))
    return np.stack(result)

def bags(train,test,jobs):
    files=cache_spectrograms(train,test,jobs)
    for kind,width,k in [('log',12,64),('pcen',12,64),('pcen',24,64)]:
        name=f'bow_{kind}{width}_k{k}'; target=R3/f'{name}.npy'
        if target.exists(): continue
        bundle=load(R3/f'dictionary_{kind}{width}_k{k}.joblib')
        blocks=[]
        for i in range(0,len(files),512):
            batch=files[i:i+512]; chunks=[batch[j:j+64] for j in range(0,len(batch),64)]
            blocks.extend(Parallel(n_jobs=jobs)(delayed(bow_batch)(c,kind,width,bundle) for c in chunks))
            print('Acoustic words',name,min(i+512,len(files)),flush=True)
        x=np.concatenate(blocks); assert x.shape==(7196,448) and np.isfinite(x).all(); np.save(target,x)

def representations(train,test,jobs,selected=None):
    files=cache_spectrograms(train,test,jobs)
    ui=np.where(train.checked.values==0)[0]
    for kind,width,k in [('log',12,64),('pcen',12,64),('pcen',24,64)]:
        name=f'{kind}{width}_k{k}'; target=R3/f'{name}.npy'
        if selected is not None and name not in selected: continue
        if target.exists(): continue
        bundle=fit_dictionary(files,ui,kind,width,k)
        blocks=[]; start=time.time()
        for i in range(0,len(files),512):
            batch=files[i:i+512]
            chunks=[batch[j:j+64] for j in range(0,len(batch),64)]
            blocks.extend(Parallel(n_jobs=jobs)(delayed(encode_batch)(c,kind,width,bundle) for c in chunks))
            print('Encode',name,min(i+512,len(files)),'/',len(files),'seconds',round(time.time()-start),flush=True)
        x=np.concatenate(blocks)
        assert x.shape==(7196,8192) and np.isfinite(x).all()
        np.save(target,x)
        del x,blocks

def distance(name):
    if name=='original' or name.startswith('r2_'):
        key='original' if name=='original' else name[3:]
        path=BASE/'round2/distances'/f'{key}.npy'
        if not path.exists():
            from refine import extra_features,feature_views
            train,test,x,caches=prepare(BASE/'project4',2)
            ui=np.where(train.checked.values==0)[0]
            extras={} if key=='original' else extra_features(train,test,2)
            view=feature_views(x,ui,extras)[key]
            with threadpool_limits(limits=4): d=euclidean_distances(view,squared=True)
            np.fill_diagonal(d,np.inf); path.parent.mkdir(parents=True,exist_ok=True); np.save(path,d)
        return np.load(path,mmap_mode='r')
    folder=R3/'distances'; folder.mkdir(exist_ok=True)
    target=folder/f'{name}.npy'
    if not target.exists():
        x=np.load(R3/f'{name}.npy',mmap_mode='r')
        with threadpool_limits(limits=4): d=euclidean_distances(x,squared=True)
        np.fill_diagonal(d,np.inf); np.save(target,d)
    return np.load(target,mmap_mode='r')

def noise_em(g,y,tr,ui,prior=.5,target_power=0.,channel_power=.5,iterations=4):
    """Estimate a class-conditional noisy-label channel and soft true labels.

    Only tr checked labels and observed ui labels enter the update. Validation
    and test rows have no supervised seed, including in the channel estimates.
    """
    n=g.shape[0]; classes=int(y.max())+1
    alpha=np.ones(n,dtype='float32'); alpha[tr]=.05; alpha[ui]=.5
    seeds=np.zeros((n,classes),dtype='float32'); seeds[tr,y[tr]]=1.; seeds[ui,y[ui]]=1.
    def solve(initial):
        source=(1-alpha[:,None])*seeds; f=initial.copy()
        for _ in range(100):
            nxt=alpha[:,None]*(g@f)+source
            if np.max(np.abs(nxt-f))<1e-6: return nxt
            f=nxt
        return f
    f=solve(seeds)
    prior_channel=.5*np.eye(classes)+.5/classes
    channel=prior_channel.copy()
    observed=np.eye(classes,dtype='float32')[y[ui]]
    for _ in range(iterations):
        neighbor=g@f
        q=neighbor[ui]*np.maximum(channel[:,y[ui]].T,1e-5)**channel_power
        q/=np.maximum(q.sum(1,keepdims=True),1e-12)
        q=.9*q+.1*observed
        counts=q.T@observed+20*prior_channel
        channel=counts/np.maximum(counts.sum(1,keepdims=True),1e-12)
        seeds[ui]=q
        f=solve(f)
    if prior: f/=np.maximum(f.mean(0),1e-8)**prior
    if target_power:
        p=np.bincount(y[tr],minlength=classes).astype(float); p/=p.sum(); f*=p[None,:]**target_power
    return f/np.maximum(f.sum(1,keepdims=True),1e-12)

def predict_graph(c,g,y,tr,ui):
    if c.get('kind')=='em':
        return noise_em(g,y,tr,ui,prior=c['prior'],target_power=c['target_power'],channel_power=c['channel_power'])
    return robust_propagate(g,y,tr,ui,alpha=.5,prior=c['prior'],target_power=c['target_power'])

def predict_svm(c,kernel,y,tr,ui,predict):
    fit=np.r_[tr,ui]; weight=np.r_[np.ones(len(tr)),np.full(len(ui),.25)]
    model=SVC(C=c['C'],kernel='precomputed',class_weight='balanced',decision_function_shape='ovr')
    model.fit(kernel[np.ix_(fit,fit)],y[fit],sample_weight=weight)
    return softmax(model.decision_function(kernel[np.ix_(predict,fit)]),axis=1)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--stage',choices=['features','bows','search','bow-search','em','svm','confirm','final'],default='features')
    ap.add_argument('--jobs',type=int,default=3)
    ap.add_argument('--seed',type=int,default=73)
    ap.add_argument('--selected-only',action='store_true')
    args=ap.parse_args()
    train,test,x,caches=prepare(BASE/'project4',args.jobs)
    if args.stage=='features':
        selected={c['view'] for c in json.loads((R3/'selection.json').read_text())['configs']} if args.selected_only else None
        representations(train,test,args.jobs,selected); return
    if args.stage=='bows': bags(train,test,args.jobs); return
    cls=np.array(sorted(train.label.unique())); y=np.searchsorted(cls,train.label)
    ci=np.where(train.checked.values==1)[0]; ui=np.where(train.checked.values==0)[0]
    out=R3/f'seed{args.seed}'; out.mkdir(exist_ok=True)
    splits=list(StratifiedKFold(5,shuffle=True,random_state=args.seed).split(ci,y[ci]))
    configs=[]
    for name in ['log12_k64','pcen12_k64','pcen24_k64']:
        if not (R3/f'{name}.npy').exists(): continue
        for weight in [.3,.6,1.]:
            for prior,target in [(.5,0.),(.5,.5),(1.,0.)]:
                configs.append(dict(name=f'{name}_w{weight}_p{prior}_t{target}',view=name,weight=weight,k=10,prior=prior,target_power=target))
    if args.stage=='em':
        configs=[dict(name=f'em_{v}_cp{cp}_t{t}',kind='em',view=v,weight=1.,k=10,prior=.5,target_power=t,channel_power=cp) for v in ['original','r2_augmented0.2'] for cp in [.5,1.] for t in [0.,.5]]
    if args.stage=='bow-search':
        configs=[dict(name=f'{v}_w{w}_t{t}',view=v,weight=w,k=10,prior=.5,target_power=t) for v in ['bow_log12_k64','bow_pcen12_k64','bow_pcen24_k64'] if (R3/f'{v}.npy').exists() for w in [.5,1.] for t in [0.,.5]]
    if args.stage=='svm':
        configs=[dict(name=f'svm_{v}_g{gamma}',kind='svm',view=v,gamma=gamma,C=C) for v in ['log12_k64','pcen12_k64','pcen24_k64'] if (R3/f'{v}.npy').exists() for gamma,C in [(.01,100.),(.1,20.)]]
    if args.stage in ['confirm','final']: configs=[c for c in json.loads((R3/'selection.json').read_text())['configs'] if c.get('kind')!='classgmm']
    graphs={}
    for c in configs:
        path=out/f'{c["name"]}.json'
        if path.exists() and args.stage!='final': continue
        if c.get('kind')=='svm':
            kernel=np.exp(-c['gamma']*distance(c['view'])); np.fill_diagonal(kernel,1.)
            if args.stage=='final':
                p=predict_svm(c,kernel,y,ci,ui,np.arange(len(train),len(train)+len(test)))
                np.savez_compressed(R3/f'{c["name"]}_full.npz',pred=p,ids=test.id.values.astype(str)); continue
            oof=np.zeros((len(ci),41))
            for tr,va in splits: oof[va]=predict_svm(c,kernel,y,ci[tr],ui,ci[va])
            result=dict(config=c,seed=args.seed,folds=5,**metrics(y[ci],oof))
            np.savez_compressed(out/f'{c["name"]}.npz',oof=oof); path.write_text(json.dumps(result,indent=2)); del kernel
            print(c['name'],'micro',result['micro_map3'],'macro',result['macro_map3'],flush=True)
            continue
        key=(c['view'],c['weight'],c['k'])
        if key not in graphs:
            d=np.array(distance(c['view']),copy=True) if c['weight']==1. else c['weight']*distance(c['view'])+(1-c['weight'])*distance('original')
            np.fill_diagonal(d,np.inf)
            graphs[key]=graph_from_distance(d,c['k']); del d
            print('Built patch graph',key,flush=True)
        g=graphs[key]
        if args.stage=='final':
            p=predict_graph(c,g,y,ci,ui)
            np.savez_compressed(R3/f'{c["name"]}_full.npz',pred=p[len(train):],ids=test.id.values.astype(str)); continue
        oof=np.zeros((len(ci),41))
        for tr,va in splits:
            p=predict_graph(c,g,y,ci[tr],ui)
            oof[va]=p[ci[va]]
        result=dict(config=c,seed=args.seed,folds=5,**metrics(y[ci],oof))
        np.savez_compressed(out/f'{c["name"]}.npz',oof=oof); path.write_text(json.dumps(result,indent=2))
        print(c['name'],'micro',result['micro_map3'],'macro',result['macro_map3'],flush=True)

if __name__=='__main__': main()
