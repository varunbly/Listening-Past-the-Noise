"""Class-balanced selection, frozen repeated-CV confirmation, and delivery."""
import argparse
import hashlib
import json
import zipfile
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from improve import BASE,prepare
from deliver import predict_one,temper
from refine import graph_from_distance,robust_propagate
from advance import R3,metrics,distance

def labels():
    train=pd.read_csv(BASE/'project4/train.csv'); test=pd.read_csv(BASE/'project4/test.csv')
    classes=np.array(sorted(train.label.unique())); y=np.searchsorted(classes,train.label)
    ci=np.where(train.checked.values==1)[0]; ui=np.where(train.checked.values==0)[0]
    return train,test,classes,y,ci,ui

def row_priors(y,ci,seed):
    result=np.zeros((len(ci),41))
    for tr,va in StratifiedKFold(5,shuffle=True,random_state=seed).split(ci,y[ci]):
        counts=np.bincount(y[ci[tr]],minlength=41).astype(float)
        result[va]=counts/counts.sum()
    return result

def calibrated(p,priors,beta,target=None,eta=0.):
    q=p/np.maximum(priors,1e-8)**beta
    if eta and target is not None: q*=np.maximum(target,1e-8)**eta
    return q/np.maximum(q.sum(1,keepdims=True),1e-12)

def baseline(seed):
    train,test,x,caches=prepare(BASE/'project4',2)
    _,_,classes,y,ci,ui=labels()
    out=R3/f'seed{seed}'; out.mkdir(exist_ok=True)
    old=np.zeros((len(ci),41)); graph_cache={}
    oldconf=json.loads((BASE/'selection.json').read_text())
    components=[np.zeros_like(old) for _ in oldconf['models']]
    r2conf=json.loads((BASE/'round2/selection.json').read_text())
    # Reuse the equivalent cached squared-distance representation instead of
    # materializing a 600 MB concatenated float64 feature matrix per fold.
    for m in oldconf['models']:
        c=m['config']
        if c['kind']=='graph' and c['view']=='combined':
            d=np.load(BASE/'round2/distances/original.npy',mmap_mode='r')
            graph_cache[('combined',c['k'])]=graph_from_distance(d,c['k'])
    new=np.zeros_like(old); graphs={}
    for m in r2conf['models']:
        c=m['config']
        if c is not None:
            d=np.load(BASE/'round2/distances'/f'{c["view"]}.npy',mmap_mode='r')
            graphs[c['name']]=graph_from_distance(d,c['k'])
    for f,(tr,va) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(ci,y[ci])):
        for j,m in enumerate(oldconf['models']):
            p=predict_one(m['config'],x,train,y,ci[tr],ci[va],caches,graph_cache)
            components[j][va]=p
            old[va]+=m['weight']*temper(p,oldconf['probability_power'])
        for m in r2conf['models']:
            c=m['config']
            if c is None: p=old[va]
            else:
                p=robust_propagate(graphs[c['name']],y,ci[tr],ui,**{k:c[k] for k in ['alpha','prior','target_power','adaptive']})[ci[va]]
            new[va]+=m['weight']*temper(p,r2conf['power'])
        print('Matched baselines fold',f+1,metrics(y[ci[va]],old[va])['micro_map3'],flush=True)
    core=sum(m['weight']*temper(p,oldconf['probability_power']) for m,p in zip(oldconf['models'],components) if m['config']['view']!='fisher32')
    core/=sum(m['weight'] for m in oldconf['models'] if m['config']['view']!='fisher32')
    for name,p in [('old_ensemble',old),('r2_ensemble',new),('old_core',core)]:
        np.savez_compressed(out/f'{name}.npz',oof=p)
        (out/f'{name}.json').write_text(json.dumps(dict(name=name,seed=seed,**metrics(y[ci],p)),indent=2))
        print(name,metrics(y[ci],p),flush=True)

def records73():
    records=[('old_ensemble',None,np.load(BASE/'round2/seed73/previous_ensemble.npz')['oof']),
             ('r2_ensemble',None,np.load(BASE/'round2/selection_oof.npz')['pred'])]
    if (R3/'seed73/old_core.npz').exists(): records.append(('old_core',None,np.load(R3/'seed73/old_core.npz')['oof']))
    for path in sorted((R3/'seed73').glob('*.json')):
        r=json.loads(path.read_text())
        if 'config' in r and r['macro_map3']>=.75: records.append((path.stem,r['config'],np.load(path.with_suffix('.npz'))['oof']))
    return records

def select():
    path=R3/'selection.json'
    if path.exists(): raise ValueError('Selection already frozen')
    train,test,cls,y,ci,ui=labels(); priors=row_priors(y,ci,73)
    target=np.load(R3/'seed73/target_priors.npy')
    records=records73(); choices=[]
    for power in [.5,1.,2.]:
        arrays=[temper(r[2],power) for r in records]
        for beta,eta in [(b,e) for b in [0.,.25,.5,.75] for e in [0.,.25,.5]]:
            first=max([i for i,r in enumerate(records) if r[1] is None],key=lambda i:metrics(y[ci],calibrated(arrays[i],priors,beta,target,eta))['macro_map3'])
            weights=np.zeros(len(records)); weights[first]=1.
            p=arrays[first]; best=metrics(y[ci],calibrated(p,priors,beta,target,eta))['macro_map3']
            for _ in range(2):
                candidate=None
                for j,a in enumerate(arrays):
                    if weights[j]: continue
                    for w in [.2,.4,.6]:
                        q=(1-w)*p+w*a
                        s=metrics(y[ci],calibrated(q,priors,beta,target,eta))['macro_map3']
                        if s>best+.001 and (candidate is None or s>candidate[0]): candidate=(s,j,w,q)
                if candidate is None: break
                best,j,w,p=candidate; weights*=1-w; weights[j]=w
            choices.append((best,power,beta,eta,weights,p))
    best,power,beta,eta,weights,p=max(choices,key=lambda c:c[0])
    prediction=calibrated(p,priors,beta,target,eta)
    selection={'selection_seed':73,'confirmation_seed':277,'objective':'macro_MAP@3','power':power,'prior_beta':beta,'target_prior_power':eta,
               'selection_scores':metrics(y[ci],prediction),
               'models':[dict(name=records[i][0],weight=float(w),config=records[i][1]) for i,w in enumerate(weights) if w],
               'configs':[records[i][1] for i,w in enumerate(weights) if w and records[i][1] is not None],
               'note':'Public leaderboard labels are unavailable and its single aggregate score was not optimized. Macro MAP@3 counteracts unequal checked support; it is not the competition metric unless test classes are balanced.'}
    path.write_text(json.dumps(selection,indent=2))
    np.savez_compressed(R3/'selected_oof.npz',y=y[ci],pred=prediction)
    print(json.dumps(selection,indent=2),flush=True)

def assemble(selection,seed=None):
    full=seed is None; result=None
    expected_ids=pd.read_csv(BASE/'project4/test.csv',dtype=str).id.tolist() if full else None
    def full_prediction(path,key):
        with np.load(path) as data:
            assert data['ids'].tolist()==expected_ids,'Prediction/test ID mismatch'
            return data[key]
    for m in selection['models']:
        if m['name']=='old_ensemble':
            p=full_prediction(BASE/'test_predictions.npz','probabilities') if full else np.load(R3/f'seed{seed}/old_ensemble.npz')['oof']
        elif m['name']=='r2_ensemble':
            p=full_prediction(BASE/'round2/test_predictions.npz','probabilities') if full else np.load(R3/f'seed{seed}/r2_ensemble.npz')['oof']
        elif m['name']=='old_core':
            if full:
                path=R3/'old_core_full.npz'
                if not path.exists():
                    train,test,x,caches=prepare(BASE/'project4',2); _,_,cls,y,ci,ui=labels()
                    conf=json.loads((BASE/'selection.json').read_text()); cache={('combined',10):graph_from_distance(distance('original'),10)}
                    p=np.zeros((len(test),41)); total=0.
                    for a in conf['models']:
                        if a['config']['view']=='fisher32': continue
                        q=predict_one(a['config'],x,train,y,ci,np.arange(len(train),len(train)+len(test)),caches,cache)
                        p+=a['weight']*temper(q,conf['probability_power']); total+=a['weight']
                    p/=total; np.savez_compressed(path,pred=p,ids=test.id.values.astype(str))
                p=full_prediction(path,'pred')
            else: p=np.load(R3/f'seed{seed}/old_core.npz')['oof']
        else:
            p=full_prediction(R3/f'{m["name"]}_full.npz','pred') if full else np.load(R3/f'seed{seed}'/f'{m["name"]}.npz')['oof']
        q=m['weight']*temper(p,selection['power']); result=q if result is None else result+q
    train,test,cls,y,ci,ui=labels()
    prior=np.bincount(y[ci],minlength=41)/len(ci) if full else row_priors(y,ci,seed)
    eta=selection.get('target_prior_power',0.)
    target=(np.load(R3/'target_prior_final.npy') if full else np.load(R3/f'seed{seed}/target_priors.npy')) if eta else None
    return calibrated(result,prior,selection['prior_beta'],target,eta)

def confirm():
    selection=json.loads((R3/'selection.json').read_text()); seed=277
    train,test,cls,y,ci,ui=labels(); p=assemble(selection,seed)
    old=np.load(R3/'seed277/old_ensemble.npz')['oof']; r2=np.load(R3/'seed277/r2_ensemble.npz')['oof']
    report={'previous':metrics(y[ci],old),'round2':metrics(y[ci],r2),'candidate':metrics(y[ci],p),
            'selection':selection,'validation_seed':seed,'folds':5,'public_score_previous':.7834,
            'note':'Repeated cross-validation, not an untouched holdout. Actual public/private score of the new candidate is unknown.'}
    old_values=((np.argsort(-old,axis=1)[:,:3]==y[ci,None])/np.arange(1,4)).sum(1)
    new_values=((np.argsort(-p,axis=1)[:,:3]==y[ci,None])/np.arange(1,4)).sum(1)
    rng=np.random.default_rng(277); deltas=[]
    for _ in range(3000):
        per=[]
        for c in range(41):
            d=(new_values-old_values)[y[ci]==c]; per.append(rng.choice(d,len(d)).mean())
        deltas.append(np.mean(per))
    report['paired_macro_gain']=float(np.mean(report['candidate']['per_class'])-np.mean(report['previous']['per_class']))
    report['paired_stratified_bootstrap_95']=np.quantile(deltas,[.025,.975]).tolist()
    report['accepted']=report['candidate']['macro_map3']>report['previous']['macro_map3']
    (R3/'confirmation.json').write_text(json.dumps(report,indent=2))
    np.savez_compressed(R3/'confirmation_predictions.npz',old=old,new=p,y=y[ci])
    print(json.dumps({k:v for k,v in report.items() if k not in ['selection']},indent=2),flush=True)

def final():
    selection=json.loads((R3/'selection.json').read_text()); report=json.loads((R3/'confirmation.json').read_text())
    p=assemble(selection) if report['accepted'] else np.load(BASE/'round2/test_predictions.npz')['probabilities']
    train,test,cls,y,ci,ui=labels()
    assert p.shape==(1592,41) and np.isfinite(p).all()
    sub=pd.DataFrame({'id':test.id,'label':[' '.join(cls[row]) for row in np.argsort(-p,axis=1)[:,:3]]})
    assert sub.id.is_unique and all(len(set(v.split()))==len(v.split())==3 for v in sub.label)
    target=BASE/'submission_round3.csv'; sub.to_csv(target,index=False)
    assert pd.read_csv(target,dtype=str).id.tolist()==test.id.tolist()
    np.savez_compressed(R3/'test_predictions.npz',ids=test.id.values.astype(str),probabilities=p,classes=cls)
    report['submission_sha256']=hashlib.sha256(target.read_bytes()).hexdigest()
    report['train_csv_sha256']=hashlib.sha256((BASE/'project4/train.csv').read_bytes()).hexdigest()
    report['test_csv_sha256']=hashlib.sha256((BASE/'project4/test.csv').read_bytes()).hexdigest()
    (R3/'final_report.json').write_text(json.dumps(report,indent=2))
    with zipfile.ZipFile(BASE.parent/'Kaggle_Submission_Round3_Only.zip','w',zipfile.ZIP_DEFLATED) as z: z.write(target,'submission.csv')
    print('Validated',target,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('stage',choices=['baseline','select','confirm','final']); ap.add_argument('--seed',type=int,default=277); args=ap.parse_args()
    if args.stage=='baseline': baseline(args.seed)
    elif args.stage=='select': select()
    elif args.stage=='confirm': confirm()
    else: final()
