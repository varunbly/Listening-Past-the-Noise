"""Round four: spectrotemporal modulations and supervised linear NCA metrics.

Classical signal processing and regularized linear metric learning only.
"""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:
    os.environ.setdefault(key,'1')
import argparse
import json
from pathlib import Path
import time
from contextlib import nullcontext
import numpy as np
import librosa
from scipy import ndimage,signal,fft
from scipy.special import softmax
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler,normalize
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics.pairwise import euclidean_distances
from threadpoolctl import threadpool_limits
from joblib import Parallel,delayed
from improve import BASE,prepare
from advance import R3,metrics,distance
from advance_finish import labels
from refine import graph_from_distance,robust_propagate

R4=BASE/'round4'; R4.mkdir(exist_ok=True)

def moments(a):
    return np.r_[a.mean(1),a.std(1),np.quantile(a,[.1,.5,.9],axis=1).ravel()]

def extra_one(path,target,control_threads=True):
    if Path(target).exists(): return
    with np.load(path) as item: m=item['log'].astype('float32'); pcen=item['pcen'].astype('float32')
    with (threadpool_limits(limits=1) if control_threads else nullcontext()):
        # Median-filter soft masks separate stationary and transient structures.
        h=ndimage.median_filter(m,size=(1,31),mode='nearest')
        p=ndimage.median_filter(m,size=(11,1),mode='nearest')
        hm=1/(1+10**np.clip((p-h)/10,-8,8))
        hp=[]
        for mask in [hm,1-hm]:
            a=m+10*np.log10(np.maximum(mask,1e-6))
            mf=librosa.feature.mfcc(S=a,n_mfcc=24)
            delta=librosa.feature.delta(mf,width=min(9,(mf.shape[1]-1)//2*2+1))
            hp.extend([moments(a),moments(mf),moments(delta),mask.mean(1),mask.std(1)])
        hp=np.concatenate(hp)
        # Separable complex Gabor filters describe sweeps and modulation rates.
        modulation=[]
        a=np.clip((m+80)/80,0,1)
        wavelets=[]
        for rate in [.5,1.,2.,4.,8.,16.]:
            frequency=rate*256/22050
            sigma_t=min(40.,1/(2*np.pi*frequency)); rt=int(np.ceil(3*sigma_t)); t=np.arange(-rt,rt+1)
            kt=np.exp(-.5*(t/sigma_t)**2)*np.exp(2j*np.pi*frequency*t)
            kt-=kt.mean(); kt/=max(np.abs(kt).sum(),1e-12)
            wavelets.append((rt,kt))
        max_rt=max(rt for rt,kt in wavelets)
        nfft=fft.next_fast_len(m.shape[1]+4*max_rt)
        transforms=[(rt,fft.fft(kt,nfft)) for rt,kt in wavelets]
        spectral=[(0.,1.,1)]+[(freq,max(1.,1/(2*np.pi*freq)),sign_) for freq in [.0625,.125,.25] for sign_ in [-1,1]]
        for freq,sigma,sign_ in spectral:
            r=int(np.ceil(3*sigma)); axis=np.arange(-r,r+1)
            kernel=np.exp(-.5*(axis/sigma)**2)*np.exp(2j*np.pi*freq*sign_*axis)
            if freq: kernel-=kernel.mean()
            kernel/=max(np.abs(kernel).sum(),1e-12)
            padded=np.pad(a,((r,r),(0,0)),mode='reflect')
            q=signal.fftconvolve(padded,kernel[:,None],mode='same',axes=0)[r:r+len(a)]
            padded=np.pad(q,((0,0),(max_rt,max_rt)),mode='reflect')
            Q=fft.fft(padded,nfft,axis=1)
            for rt,transform in transforms:
                response=np.abs(fft.ifft(Q*transform[None,:],axis=1)[:,max_rt+rt:max_rt+rt+m.shape[1]])
                v=response.reshape(8,6,m.shape[1]).mean(1)
                quantiles=np.quantile(v,[.5,.9],axis=1)
                modulation.extend(np.stack([v.mean(1),v.std(1),quantiles[0],quantiles[1]],axis=1).ravel())
        modulation=np.asarray(modulation)
        modulation[np.abs(modulation)<1e-10]=0
        # Modulation summaries for PCEN, at a few temporal scales.
        for sigma in [1,4,16]:
            smooth=ndimage.gaussian_filter1d(pcen,sigma,axis=1,mode='nearest')
            modulation=np.r_[modulation,moments(np.abs(pcen-smooth))]
    target=Path(target); temp=target.with_suffix('.tmp.npz')
    np.savez_compressed(temp,hpss=np.nan_to_num(hp).astype('float32'),gabor=np.nan_to_num(modulation).astype('float32'))
    temp.replace(target)

def extra_batch(items):
    with threadpool_limits(limits=1):
        for source,target in items: extra_one(source,target,control_threads=False)

def read_feature_batch(files):
    values={'hpss':[],'gabor':[]}
    for path in files:
        with np.load(path) as item:
            for key in values: values[key].append(item[key])
    return {key:np.stack(value) for key,value in values.items()}

def extract(jobs):
    train,test,cls,y,ci,ui=labels()
    from advance import cache_spectrograms
    cache_spectrograms(train,test,jobs)
    folder=R4/'feature_cache_v2'; folder.mkdir(exist_ok=True)
    files=[]; pending=[]
    for split,df in [('train',train),('test',test)]:
        for ident in df.id:
            source=R3/'spectrograms'/f'{split}_{ident}.npz'; target=folder/source.name
            files.append(target)
            if not target.exists(): pending.append((source,target))
    start=time.time()
    for i in range(0,len(pending),128):
        batch=pending[i:i+128]
        Parallel(n_jobs=jobs)(delayed(extra_batch)(batch[j:j+32]) for j in range(0,len(batch),32))
        print('Modulation features',min(i+128,len(pending)),'/',len(pending),'seconds',round(time.time()-start),flush=True)
    with np.load(files[0]) as first: groups={k:np.empty((len(files),len(first[k])),dtype='float32') for k in first.files}
    for i in range(0,len(files),512):
        batch=files[i:i+512]
        blocks=Parallel(n_jobs=jobs)(delayed(read_feature_batch)(batch[j:j+128]) for j in range(0,len(batch),128))
        pos=i
        for block in blocks:
            length=len(block['hpss'])
            for key in groups: groups[key][pos:pos+length]=block[key]
            pos+=length
        print('Assembling feature matrices',min(i+512,len(files)),'/',len(files),flush=True)
    for key,a in groups.items():
        assert np.isfinite(a).all(); np.save(R4/f'{key}.npy',a)
        print(key,a.shape,flush=True)

def inputs():
    train,test,x,caches=prepare(BASE/'project4',2)
    _,_,cls,y,ci,ui=labels(); result={}
    for key,a in [('old',x)]+[(k,np.load(R4/f'{k}.npy',mmap_mode='r')) for k in ['hpss','gabor'] if (R4/f'{k}.npy').exists()]:
        scaled=np.clip(StandardScaler().fit(a[ui]).transform(a),-6,6)
        with threadpool_limits(limits=3):
            pca=PCA(64,whiten=True,svd_solver='randomized',random_state=604).fit(scaled[ui])
            projected=np.clip(pca.transform(scaled),-8,8)
            result[key]=normalize(projected).astype('float32')
            result[key+'_nw']=normalize(projected*np.sqrt(pca.explained_variance_)).astype('float32')
    if 'hpss' in result and 'gabor' in result:
        result['new']=normalize(np.concatenate([result['hpss'],result['gabor']],axis=1))
        result['new_nw']=normalize(np.concatenate([result['hpss_nw'],result['gabor_nw']],axis=1))
        scaled=np.concatenate([result['old'],result['new']],axis=1)
        with threadpool_limits(limits=3):
            pca=PCA(64,whiten=True,svd_solver='randomized',random_state=604).fit(scaled[ui])
            result['combined']=normalize(pca.transform(scaled)).astype('float32')
    return result

def strong_distance():
    result=np.array(distance('original'),dtype='float32',copy=True); result*=.4
    for name,w in [('log12_k64',.2666666667),('pcen24_k64',.3333333333)]:
        other=distance(name)
        for i in range(0,len(result),256): result[i:i+256]+=w*other[i:i+256]
    return result

def mix_distance(d,base,weight):
    d*=weight
    for i in range(0,len(d),256): d[i:i+256]+=(1-weight)*base[i:i+256]
    np.fill_diagonal(d,np.inf)
    return d

def context_graph(base,k,reciprocal=.2,shared=True):
    near=np.stack([np.argpartition(row,k)[:k].copy() for row in base])
    rows=np.repeat(np.arange(len(base)),k); cols=near.ravel(); d=base[rows,cols]
    scale=np.sqrt(d.reshape(len(base),k).max(1)).clip(1e-5)
    values=np.exp(-d/(scale[rows]*scale[cols]))
    adjacency=csr_matrix((np.ones(len(rows)),(rows,cols)),shape=base.shape)
    reverse=np.asarray(adjacency.T[rows,cols]).ravel()
    values*=np.where(reverse>0,1.,reciprocal)
    if shared:
        norms=np.sqrt(np.asarray(adjacency.multiply(adjacency).sum(1)).ravel())
        normalized=adjacency.multiply(1/np.maximum(norms,1e-8)[:,None]).tocsr()
        common=normalized@normalized.T
        similarity=np.asarray(common[rows,cols]).ravel()
        values*=.2+.8*similarity
    w=csr_matrix((values,(rows,cols)),shape=base.shape); w=(w+w.T)*.5
    return w.multiply(1/np.maximum(np.asarray(w.sum(1)).ravel(),1e-12)[:,None]).tocsr()

def nca_loss(flat,X,same,weights,initial,reg,trust=None):
    A=flat.reshape(initial.shape); z=X@A.T
    norm=(z*z).sum(1); d=np.maximum(norm[:,None]+norm[None,:]-2*z@z.T,0)
    np.fill_diagonal(d,np.inf)
    logits=-d if trust is None else -d+np.log(np.maximum(trust,1e-8))[None,:]
    prob=softmax(logits,axis=1)
    masked=prob*same; correct=masked.sum(1,keepdims=True)
    objective=float((weights[:,None]*correct).sum())
    W=weights[:,None]*(masked-prob*correct)
    S=W+W.T; np.fill_diagonal(S,-W.sum(0))
    gradient=-2*z.T@S@X+2*reg*(A-initial)
    loss=-objective+reg*np.sum((A-initial)**2)
    return float(loss),gradient.ravel()

def learn_metric(X,y,reg,maxiter=60,trust=None):
    X=np.asarray(X,dtype='float64')*np.sqrt(10.)
    initial=np.eye(X.shape[1]); same=y[:,None]==y[None,:]
    trust=np.ones(len(y)) if trust is None else trust
    weights=trust/np.bincount(y,weights=trust)[y]; weights/=weights.sum()
    with threadpool_limits(limits=3):
        result=minimize(nca_loss,initial.ravel(),args=(X,same,weights,initial,reg,trust),method='L-BFGS-B',jac=True,options={'maxiter':maxiter,'ftol':1e-8,'maxls':25})
    return result.x.reshape(initial.shape),{'iterations':int(result.nit),'success':bool(result.success),'loss':float(result.fun),'training_points':len(y)}

def semi_training(g,y,tr,ui):
    p=robust_propagate(g,y,tr,ui,alpha=.5,prior=.5,target_power=0.)
    neighbor=g@p
    probability=neighbor/np.maximum(neighbor.sum(1,keepdims=True),1e-12)
    ranked=np.sort(probability[ui],axis=1)
    confidence=probability[ui,y[ui]]
    good=(probability[ui].argmax(1)==y[ui])&(confidence>=.45)&((ranked[:,-1]-ranked[:,-2])>=.15)
    chosen=[]
    for c in range(41):
        candidates=np.where(good&(y[ui]==c))[0]
        candidates=candidates[np.argsort(-confidence[candidates])[:12]]
        chosen.extend(ui[candidates])
    chosen=np.array(chosen,dtype=int)
    return np.r_[tr,chosen],np.r_[np.ones(len(tr)),np.full(len(chosen),.1)]

def graph_for(c,views,y,training,base):
    if c['kind']=='context': return context_graph(base,c['k'],c['reciprocal'],c['shared'])
    if c['kind']=='nca':
        A,info=learn_metric(views[c['view']][training],y[training],c['reg'])
        embedding=normalize((views[c['view']]*np.sqrt(10.))@A.T).astype('float32')
        print('NCA fitted',c['view'],c['reg'],info,flush=True)
    else: embedding=views[c['view']]
    with threadpool_limits(limits=3): d=euclidean_distances(embedding,squared=True)
    d=mix_distance(d,base,c['weight'])
    return graph_from_distance(d,c['k'])

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--stage',choices=['features','graph','context','nca','nca-semi','confirm','final'],default='features'); ap.add_argument('--seed',type=int,default=73); ap.add_argument('--jobs',type=int,default=2); args=ap.parse_args()
    if args.stage=='features': extract(args.jobs); return
    train,test,cls,y,ci,ui=labels(); views=inputs(); base=strong_distance()
    out=R4/f'seed{args.seed}'; out.mkdir(exist_ok=True)
    configs=[]
    if args.stage=='graph':
        configs=[dict(name=f'{v}_w{w}_k{k}',kind='graph',view=v,weight=w,k=k,prior=1.) for v in ['hpss','gabor','new','hpss_nw','gabor_nw','new_nw'] if v in views for w in [.2,.5] for k in [5,10]]
    if args.stage=='context':
        configs=[dict(name=f'context_k{k}_r{r}_s{s}',kind='context',view='base',k=k,prior=1.,reciprocal=r,shared=s) for k in [10,25] for r in [.2,1.] for s in [False,True]]
    if args.stage=='nca':
        configs=[dict(name=f'nca_{v}_r{reg}_w{w}',kind='nca',view=v,weight=w,k=10,prior=1.,reg=reg) for v in ['old','combined'] if v in views for reg in [.003,.01] for w in [.25,.5]]
    if args.stage=='nca-semi':
        configs=[dict(name=f'ncasemi_{v}_w{w}',kind='nca',view=v,weight=w,k=10,prior=1.,reg=.01,semi=True) for v in ['old','combined'] if v in views for w in [.25,.5]]
    if args.stage in ['confirm','final']: configs=[c for c in json.loads((R4/'selection.json').read_text())['configs'] if c['kind'] not in ['stack','map']]
    learned={}; graphs={}; teacher_graph=None
    if any(c.get('semi') for c in configs): teacher_graph=graph_from_distance(base,10)
    for c in configs:
        path=out/f'{c["name"]}.json'
        if path.exists() and args.stage!='final': continue
        splits=[(np.arange(len(ci)),np.array([],dtype=int))] if args.stage=='final' else list(StratifiedKFold(5,shuffle=True,random_state=args.seed).split(ci,y[ci]))
        oof=np.zeros((len(ci),41)); diagnostics=[]
        for fold,(tr,va) in enumerate(splits):
            if c['kind']=='nca':
                key=(c['view'],c['reg'],c.get('semi',False),fold)
                if key not in learned:
                    training,trust=semi_training(teacher_graph,y,ci[tr],ui) if c.get('semi') else (ci[tr],None)
                    A,info=learn_metric(views[c['view']][training],y[training],c['reg'],maxiter=40 if c.get('semi') else 60,trust=trust); learned[key]=(A,info)
                    print('NCA',key,info,flush=True)
                A,info=learned[key]; diagnostics.append(info)
                embedding=normalize((views[c['view']]*np.sqrt(10.))@A.T).astype('float32')
                with threadpool_limits(limits=3): d=euclidean_distances(embedding,squared=True)
                d=mix_distance(d,base,c['weight']); g=graph_from_distance(d,c['k'])
            else:
                key=c['name']
                if key not in graphs: graphs[key]=graph_for(c,views,y,ci[tr],base)
                g=graphs[key]
            p=robust_propagate(g,y,ci[tr],ui,alpha=.5,prior=c['prior'],target_power=0.)
            if args.stage=='final':
                np.savez_compressed(R4/f'{c["name"]}_full.npz',pred=p[len(train):],ids=test.id.values.astype(str)); continue
            oof[va]=p[ci[va]]
        if args.stage!='final':
            result=dict(config=c,seed=args.seed,folds=5,diagnostics=diagnostics,**metrics(y[ci],oof))
            path.write_text(json.dumps(result,indent=2)); np.savez_compressed(path.with_suffix('.npz'),oof=oof)
            print(c['name'],result['micro_map3'],result['macro_map3'],flush=True)

if __name__=='__main__': main()
