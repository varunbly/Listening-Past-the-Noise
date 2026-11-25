"""Class-conditional patch-density models using checked training labels only."""
import argparse
import json
import numpy as np
from joblib import load,Parallel,delayed
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
from scipy.special import softmax
from threadpoolctl import threadpool_limits
from advance import BASE,R3,patches,metrics
from advance_finish import labels

def fit_model(frames):
    with threadpool_limits(limits=1):
        return GaussianMixture(4,covariance_type='diag',reg_covar=.1,max_iter=80,random_state=722).fit(frames)

def latent(kind,width,train,test,ci):
    path=R3/f'latent_{kind}{width}.joblib'
    if path.exists(): return load(path)
    scaler,pca,ubm=load(R3/f'dictionary_{kind}{width}_k64.joblib')
    result={}
    idx=np.r_[ci,np.arange(len(train),len(train)+len(test))]
    with threadpool_limits(limits=4):
        for n,i in enumerate(idx):
            split='train' if i<len(train) else 'test'
            ident=train.id.iloc[i] if i<len(train) else test.id.iloc[i-len(train)]
            a=patches(R3/'spectrograms'/f'{split}_{ident}.npz',kind,width,maximum=120)
            result[int(i)]=np.clip(pca.transform(scaler.transform(a)),-8,8).astype('float32')
            if n%512==0: print('Class density latent',kind,width,n,'/',len(idx),flush=True)
    from joblib import dump
    dump(result,path,compress=3)
    return result

def predict(models,frames,indices,background):
    lens=[len(frames[int(i)]) for i in indices]; offsets=np.r_[0,np.cumsum(lens)]
    pooled=np.concatenate([frames[int(i)] for i in indices])
    output={a:np.zeros((len(indices),len(models))) for a in ['mean','top25','top50']}
    with threadpool_limits(limits=1):
        bg=background.score_samples(pooled)
        for c,model in enumerate(models):
            likelihood=model.score_samples(pooled)-bg
            for j,(start,stop) in enumerate(zip(offsets[:-1],offsets[1:])):
                v=likelihood[start:stop]
                output['mean'][j,c]=v.mean()
                for name,fraction in [('top25',.25),('top50',.5)]:
                    k=max(1,int(len(v)*fraction))
                    output[name][j,c]=np.partition(v,len(v)-k)[-k:].mean()
    return {key:softmax(value/16.,axis=1) for key,value in output.items()}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--stage',choices=['search','confirm','final'],default='search'); ap.add_argument('--seed',type=int,default=73); args=ap.parse_args()
    train,test,classes,y,ci,ui=labels()
    configs=[dict(name=f'classgmm_{v}_{a}',kind='classgmm',view=v,aggregate=a) for v in ['log12','pcen24'] for a in ['mean','top25','top50']]
    if args.stage in ['confirm','final']:
        configs=[c for c in json.loads((R3/'selection.json').read_text())['configs'] if c.get('kind')=='classgmm']
    out=R3/f'seed{args.seed}'; out.mkdir(exist_ok=True)
    for view in sorted(set(c['view'] for c in configs)):
        active=[c for c in configs if c['view']==view and (args.stage=='final' or not (out/f'{c["name"]}.json').exists())]
        if not active: continue
        kind='log' if view.startswith('log') else 'pcen'; width=int(view[len(kind):])
        frames=latent(kind,width,train,test,ci)
        background=load(R3/f'dictionary_{view}_k64.joblib')[2]
        splits=[(np.arange(len(ci)),np.array([],dtype=int))] if args.stage=='final' else list(StratifiedKFold(5,shuffle=True,random_state=args.seed).split(ci,y[ci]))
        outputs={c['aggregate']:np.zeros((len(ci),41)) for c in active}
        for fold,(tr,va) in enumerate(splits):
            models=Parallel(n_jobs=3)(delayed(fit_model)(np.concatenate([frames[int(i)] for i in ci[tr] if y[i]==c])) for c in range(41))
            ids=np.arange(len(train),len(train)+len(test)) if args.stage=='final' else ci[va]
            p=predict(models,frames,ids,background)
            if args.stage=='final':
                for c in active: np.savez_compressed(R3/f'{c["name"]}_full.npz',pred=p[c['aggregate']],ids=test.id.values.astype(str))
            else:
                for a in outputs: outputs[a][va]=p[a]
            print('Class-conditional GMM',view,'fold',fold+1,flush=True)
        if args.stage!='final':
            for c in active:
                p=outputs[c['aggregate']]; result=dict(config=c,seed=args.seed,folds=5,**metrics(y[ci],p))
                np.savez_compressed(out/f'{c["name"]}.npz',oof=p); (out/f'{c["name"]}.json').write_text(json.dumps(result,indent=2))
                print(c['name'],result['micro_map3'],result['macro_map3'],flush=True)

if __name__=='__main__': main()
