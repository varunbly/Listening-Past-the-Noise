"""Round two: temporal audio textures and robust, adaptive label diffusion.

Old holdout results are never read here. All comparisons use the same new
stratified folds; these are resampling estimates, not a new independent test.
"""
from course_input import course_data
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:
    os.environ.setdefault(k,'1')
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from scipy import ndimage
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics.pairwise import euclidean_distances
from joblib import Parallel,delayed
from threadpoolctl import threadpool_limits
from improve import BASE,prepare,score
from graph_search import graph,propagate
from deliver import predict_one,temper

R2=BASE/'round2'
R2.mkdir(exist_ok=True)

def graph_from_distance(d,k):
    near=np.stack([np.argpartition(row,k)[:k].copy() for row in d])
    rows=np.repeat(np.arange(len(d)),k); cols=near.ravel()
    dd=d[rows,cols]
    scale=np.sqrt(dd.reshape(len(d),k).max(1)).clip(1e-5)
    w=csr_matrix((np.exp(-dd/(scale[rows]*scale[cols])),(rows,cols)),shape=d.shape)
    w=(w+w.T)*.5
    return w.multiply(1/np.maximum(np.asarray(w.sum(1)).ravel(),1e-12)[:,None]).tocsr()

def distances(view,name):
    folder=R2/'distances'; folder.mkdir(exist_ok=True)
    path=folder/f'{name}.npy'
    if not path.exists():
        with threadpool_limits(limits=4): d=euclidean_distances(view,squared=True)
        np.fill_diagonal(d,np.inf); np.save(path,d)
    return np.load(path,mmap_mode='r')

def learned_graph(c,views,x,ui,y,tr,projected):
    dim=c['metric_dim']
    if dim not in projected:
        z=np.clip(StandardScaler().fit(x[ui]).transform(x),-6,6)
        with threadpool_limits(limits=4):
            projected[dim]=PCA(dim,whiten=True,svd_solver='randomized',random_state=42).fit(z[ui]).transform(z)
    z=projected[dim]
    lda=LinearDiscriminantAnalysis(solver='eigen',shrinkage='auto')
    with threadpool_limits(limits=4):
        lda.fit(z[tr],y[tr]); embedding=normalize(lda.transform(z)).astype('float32')
        d=euclidean_distances(embedding,squared=True)
    w=c['metric_weight']; d=w*d+(1-w)*distances(views['original'],'original')
    np.fill_diagonal(d,np.inf)
    return graph_from_distance(d,c['k'])

def moments(x):
    return np.r_[x.mean(1),x.std(1),np.quantile(x,[.1,.5,.9],axis=1).ravel()]

def temporal_features(path,target):
    target=Path(target)
    if target.exists(): return
    with threadpool_limits(limits=1):
        y,sr=sf.read(path,dtype='float32')
        if y.ndim==2: y=y.mean(1)
        if sr!=22050: y=librosa.resample(y,orig_sr=sr,target_sr=22050); sr=22050
        y=np.pad(y,(0,max(0,4096-len(y))))
        y=y/max(np.abs(y).max(),1e-8)
        power=np.abs(librosa.stft(y,n_fft=1024,hop_length=256))**2
        mel=librosa.feature.melspectrogram(S=power,sr=sr,n_mels=96,fmin=20)
        logmel=librosa.power_to_db(mel,ref=np.max,top_db=80)
        pcen=librosa.pcen(mel*2**20,sr=sr,hop_length=256,gain=.98,bias=2,power=.5,time_constant=.4)
        pcenlog=np.log1p(pcen).astype('float32')
        mf=librosa.feature.mfcc(S=logmel,n_mfcc=30)
        pmf=librosa.feature.mfcc(S=pcenlog,n_mfcc=30)
        # Temporal pyramids retain event order lost by whole-clip statistics.
        pyramid=[]
        for a in [mf,pmf]:
            for count in [1,2,4]:
                for chunk in np.array_split(a,count,axis=1):
                    pyramid.extend([chunk.mean(1),chunk.std(1)])
        pcenstats=np.r_[moments(pcenlog),moments(pmf)]
        # Envelope modulation: energy in slow/fast temporal fluctuations.
        modulation=[]
        bands=np.array_split(pcenlog,16,axis=0)
        limits=[0,.5,1,2,4,8,16,32,64,np.inf]
        freq=np.fft.rfftfreq(pcenlog.shape[1],256/sr)
        for band in bands:
            envelope=band.mean(0)
            spec=np.abs(np.fft.rfft(envelope-envelope.mean()))**2
            energy=np.array([spec[(freq>=lo)&(freq<hi)].sum() for lo,hi in zip(limits[:-1],limits[1:])])
            modulation.append(np.sqrt(energy/max(energy.sum(),1e-12)))
        # Uniform local binary patterns on spectrograms, not learned images.
        mapping=np.full(256,58,dtype=int)
        position=0
        for code in range(256):
            bits=[(code>>i)&1 for i in range(8)]
            transitions=sum(bits[i]!=bits[(i+1)%8] for i in range(8))
            if transitions<=2: mapping[code]=position; position+=1
        texture=[]
        for radius in [1,2,4]:
            a=ndimage.gaussian_filter(logmel,sigma=(.5,.5))
            code=np.zeros(a.shape,dtype='uint8')
            for bit,(df,dt) in enumerate([(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]):
                shifted=ndimage.shift(a,(df*radius,dt*radius),order=0,mode='nearest',prefilter=False)
                code|=((shifted>=a).astype('uint8')<<bit)
            code=mapping[code]
            for band in np.array_split(code,8,axis=0):
                hist=np.bincount(band.ravel(),minlength=59).astype(float)
                texture.append(np.sqrt(hist/hist.sum()))
        # Oriented gradients distinguish horizontal harmonics and transients.
        gradients=[]
        dy,dx=np.gradient(logmel)
        magnitude=np.hypot(dx,dy)
        orientation=((np.arctan2(dy,dx)+np.pi)*8/(2*np.pi)).astype(int)%8
        for bands_n in [4,8]:
            for freq_idx in np.array_split(np.arange(96),bands_n):
                for time_idx in np.array_split(np.arange(logmel.shape[1]),2):
                    w=magnitude[np.ix_(freq_idx,time_idx)]
                    orient=orientation[np.ix_(freq_idx,time_idx)]
                    hist=np.bincount(orient.ravel(),weights=w.ravel(),minlength=8)
                    gradients.append(np.sqrt(hist/max(hist.sum(),1e-12)))
        groups={'temporal':np.concatenate(pyramid),'pcen':pcenstats,
                'modulation':np.concatenate(modulation),'texture':np.r_[np.concatenate(texture),np.concatenate(gradients)]}
        tmp=target.with_suffix('.tmp.npz')
        np.savez_compressed(tmp,**{k:np.nan_to_num(v).astype('float32') for k,v in groups.items()})
        tmp.replace(target)

def extra_features(train,test,jobs):
    folder=R2/'feature_cache'; folder.mkdir(exist_ok=True)
    files=[]; todo=[]
    for split,df in [('train',train),('test',test)]:
        for ident in df.id:
            f=folder/f'{split}_{ident}.npz'; files.append(f)
            if not f.exists(): todo.append((course_data(BASE) / 'audio/audio'/split/f'{ident}.wav',f))
    start=time.time()
    for i in range(0,len(todo),128):
        Parallel(n_jobs=jobs)(delayed(temporal_features)(p,c) for p,c in todo[i:i+128])
        print('Temporal features',min(i+128,len(todo)),'/',len(todo),'seconds',round(time.time()-start),flush=True)
    keys=['temporal','pcen','modulation','texture']
    missing=[k for k in keys if not (R2/f'{k}.npy').exists() or todo]
    if missing:
        with np.load(files[0]) as first:
            buffers={k:np.empty((len(files),len(first[k])),dtype='float32') for k in missing}
        for i,path in enumerate(files):
            with np.load(path) as item:
                for k in missing: buffers[k][i]=item[k]
            if i%512==0: print('Assembling cached temporal features',i,'/',len(files),flush=True)
        for k,a in buffers.items(): np.save(R2/f'{k}.npy',a)
    result={k:np.load(R2/f'{k}.npy',mmap_mode='r') for k in keys}
    return result

def robust_propagate(g,y,tr,ui,alpha=.5,prior=.5,target_power=.5,adaptive=0.,clean_clamp=.05):
    labels=int(y.max())+1
    seed=np.zeros((len(g.indptr)-1,labels),dtype='float32')
    seed[tr,y[tr]]=1.; seed[ui,y[ui]]=1.
    a=np.ones(len(seed),dtype='float32'); a[tr]=clean_clamp; a[ui]=alpha
    def solve():
        source=(1-a[:,None])*seed
        f=source.copy()
        for _ in range(200):
            nxt=a[:,None]*(g@f)+source
            if np.max(np.abs(nxt-f))<1e-6: return nxt
            f=nxt
        return f
    f=solve()
    if adaptive:
        neighbor=g@f
        confidence=neighbor[ui,y[ui]]/np.maximum(neighbor[ui].max(1),1e-8)
        # Neighbor agreement excludes the recording's direct self label.
        proposed=np.clip(1-.8*confidence,.2,.98)
        a[ui]=(1-adaptive)*alpha+adaptive*proposed
        f=solve()
    if prior: f/=np.maximum(f.mean(0),1e-8)**prior
    if target_power:
        p=np.bincount(y[tr],minlength=labels).astype(float); p/=p.sum()
        f*=p[None,:]**target_power
    return f/np.maximum(f.sum(1,keepdims=True),1e-12)

def feature_views(x,ui,extras):
    views={}
    stats=normalize(np.clip(StandardScaler().fit(x[ui]).transform(x),-6,6))
    fish=np.load(BASE/'cache/fisher_audio-v1_64.npz')['x']
    views['original']=(np.concatenate([stats,fish],axis=1)/np.sqrt(2)).astype('float32')
    for key,a in extras.items():
        z=np.clip(StandardScaler().fit(a[ui]).transform(a),-5,5)
        with threadpool_limits(limits=4):
            pca=PCA(min(96,a.shape[1]-1),whiten=False,random_state=42,svd_solver='randomized').fit(z[ui])
            views[key]=normalize(pca.transform(z)).astype('float32')
    if extras:
        new=normalize(np.concatenate([views[k]/2 for k in extras],axis=1))
        views['new']=new
        for weight in [.2,.4]:
            views[f'augmented{weight}']=np.concatenate([views['original']*np.sqrt(1-weight),new*np.sqrt(weight)],axis=1).astype('float32')
    return views

def get_view(views,name):
    if name.startswith('mix_'):
        return np.concatenate([views['original']*np.sqrt(.8),views[name[4:]]*np.sqrt(.2)],axis=1).astype('float32')
    return views[name]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--stage',choices=['features','baseline','search','ablation','metric','confirm','final'],default='search')
    ap.add_argument('--jobs',type=int,default=3)
    ap.add_argument('--seed',type=int,default=73)
    args=ap.parse_args()
    train,test,x,caches=prepare(course_data(BASE),args.jobs)
    y=np.searchsorted(sorted(train.label.unique()),train.label)
    ci=np.where(train.checked.values==1)[0]; ui=np.where(train.checked.values==0)[0]
    if args.stage=='features': extra_features(train,test,args.jobs); return
    splits=list(StratifiedKFold(5,shuffle=True,random_state=args.seed).split(ci,y[ci]))
    output=R2/f'seed{args.seed}'; output.mkdir(exist_ok=True)
    if args.stage=='baseline':
        selection=json.loads((BASE/'selection.json').read_text()); oof=np.zeros((len(ci),41)); graph_cache={}
        for fold,(tr,va) in enumerate(splits):
            for m in selection['models']:
                p=predict_one(m['config'],x,train,y,ci[tr],ci[va],caches,graph_cache)
                oof[va]+=m['weight']*temper(p,selection['probability_power'])
            print('Previous ensemble fold',fold+1,score(y[ci[va]],oof[va]),flush=True)
        np.savez_compressed(output/'previous_ensemble.npz',oof=oof)
        (output/'previous_ensemble.json').write_text(json.dumps({'name':'previous_ensemble','map3':score(y[ci],oof),'seed':args.seed,'folds':5},indent=2))
        print('Previous ensemble matched CV',score(y[ci],oof),flush=True); return
    extras={k:np.load(R2/f'{k}.npy',mmap_mode='r') for k in ['temporal','pcen','modulation','texture'] if (R2/f'{k}.npy').exists()}
    if args.stage=='metric': extras={}
    views=feature_views(x,ui,extras)
    if args.stage=='metric':
        original=views['original']; base_d=distances(original,'original')
        # The supervised projection is refitted on checked training labels in
        # each fold. No validation label enters PCA, LDA, or the graph seeds.
        z=np.clip(StandardScaler().fit(x[ui]).transform(x),-6,6)
        for dim in [64,128]:
            with threadpool_limits(limits=4):
                zz=PCA(dim,whiten=True,svd_solver='randomized',random_state=42).fit(z[ui]).transform(z)
            outputs={w:np.zeros((len(ci),41)) for w in [.2,.5,.8]}
            for fold,(tr,va) in enumerate(splits):
                lda=LinearDiscriminantAnalysis(solver='eigen',shrinkage='auto')
                with threadpool_limits(limits=4):
                    lda.fit(zz[ci[tr]],y[ci[tr]])
                    embedding=normalize(lda.transform(zz)).astype('float32')
                    ld=euclidean_distances(embedding,squared=True)
                for w in outputs:
                    combined=(1-w)*base_d+w*ld
                    np.fill_diagonal(combined,np.inf)
                    g=graph_from_distance(combined,10)
                    p=robust_propagate(g,y,ci[tr],ui,alpha=.5,prior=.5,target_power=.5)
                    outputs[w][va]=p[ci[va]]
                print('LDA metric',dim,'fold',fold+1,flush=True)
            for w,oof in outputs.items():
                name=f'lda{dim}_w{w}'
                conf=dict(name=name,view='original',k=10,alpha=.5,prior=.5,target_power=.5,adaptive=0.,metric_dim=dim,metric_weight=w)
                np.savez_compressed(output/f'{name}.npz',oof=oof)
                (output/f'{name}.json').write_text(json.dumps(dict(config=conf,map3=score(y[ci],oof),seed=args.seed,folds=5),indent=2))
                print(name,score(y[ci],oof),flush=True)
        return
    configs=[]
    for view in ['original','augmented0.2','augmented0.4','new']:
        if view not in views: continue
        for k in [5,10]:
            for adaptive in [0.,1.]:
                configs.append(dict(name=f'{view}_k{k}_ad{adaptive}',view=view,k=k,alpha=.5,prior=.5,target_power=.5,adaptive=adaptive))
    if args.stage=='ablation':
        configs=[dict(name=f'mix_{key}',view=f'mix_{key}',k=10,alpha=.5,prior=.5,target_power=.5,adaptive=0.) for key in extras]
    if args.stage in ['confirm','final']:
        configs=json.loads((R2/'selection.json').read_text())['configs']
    graphs={}
    projected={}
    for c in configs:
        tag=c['name']; target=output/f'{tag}.json'
        if target.exists() and args.stage!='final': continue
        key=(c['view'],c['k'])
        if not c.get('metric_dim') and key not in graphs:
            print('Building graph',key,flush=True); graphs[key]=graph_from_distance(distances(get_view(views,c['view']),c['view']),c['k'])
        g=graphs.get(key)
        if args.stage=='final':
            if c.get('metric_dim'): g=learned_graph(c,views,x,ui,y,ci,projected)
            p=robust_propagate(g,y,ci,ui,**{a:c[a] for a in ['alpha','prior','target_power','adaptive']})
            np.savez_compressed(R2/f'{tag}_full.npz',test=p[len(train):],ids=test.id.values.astype(str)); continue
        oof=np.zeros((len(ci),41)); fold_scores=[]
        for f,(tr,va) in enumerate(splits):
            if c.get('metric_dim'): g=learned_graph(c,views,x,ui,y,ci[tr],projected)
            p=robust_propagate(g,y,ci[tr],ui,**{a:c[a] for a in ['alpha','prior','target_power','adaptive']})
            oof[va]=p[ci[va]]; fold_scores.append(score(y[ci[va]],oof[va]))
        res=dict(config=c,map3=score(y[ci],oof),fold_scores=fold_scores,seed=args.seed,folds=5)
        np.savez_compressed(output/f'{tag}.npz',oof=oof); target.write_text(json.dumps(res,indent=2))
        print(tag,res['map3'],flush=True)

if __name__=='__main__': main()
