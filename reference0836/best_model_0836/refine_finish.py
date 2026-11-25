"""Freeze round-two blending on seed 73, compare once on seed 131, submit."""
from course_input import course_data
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from refine import BASE,R2
from improve import score
from deliver import temper

def choose():
    folder=R2/'seed73'
    train=pd.read_csv(course_data(BASE) / 'train.csv'); cls=sorted(train.label.unique())
    y=np.searchsorted(cls,train.loc[train.checked==1,'label'])
    records=[('previous_ensemble',None,np.load(folder/'previous_ensemble.npz')['oof'])]
    for p in sorted(folder.glob('*.json')):
        r=json.loads(p.read_text())
        if 'config' in r: records.append((p.stem,r['config'],np.load(p.with_suffix('.npz'))['oof']))
    alternatives=[]
    for power in [1.,.5,2.]:
        weights=np.zeros(len(records)); weights[0]=1.
        p=temper(records[0][2],power); best=score(y,p)
        for _ in range(2):
            addition=None
            for j,(name,c,a) in enumerate(records[1:],1):
                if weights[j]: continue
                for w in [.1,.25,.5,.75]:
                    q=(1-w)*p+w*temper(a,power); s=score(y,q)
                    if s>best+.001 and (addition is None or s>addition[0]): addition=(s,j,w,q)
            if addition is None: break
            best,j,w,p=addition; weights*=1-w; weights[j]=w
        alternatives.append((best,power,weights,p))
    best,power,weights,p=max(alternatives,key=lambda a:a[0])
    selection={'selection_seed':73,'confirmation_seed':131,'folds':5,'selection_cv_map3':best,
               'previous_selection_cv_map3':score(y,records[0][2]),'power':power,
               'models':[{'name':records[i][0],'weight':float(w),'config':records[i][1]} for i,w in enumerate(weights) if w],
               'configs':[records[i][1] for i,w in enumerate(weights) if w and i>0],
               'note':'Frozen before seed-131 confirmation. Repeated CV, not an untouched test set.'}
    path=R2/'selection.json'
    if path.exists(): raise ValueError('Round-two selection already frozen')
    path.write_text(json.dumps(selection,indent=2))
    np.savez_compressed(R2/'selection_oof.npz',y=y,pred=p)
    print(json.dumps(selection,indent=2),flush=True)

def assemble(selection,folder,full=False):
    p=None
    for m in selection['models']:
        if m['config'] is None:
            if full:
                with np.load(BASE/'test_predictions.npz') as data:
                    assert data['ids'].tolist()==pd.read_csv(course_data(BASE) / 'test.csv',dtype=str).id.tolist()
                    a=data['probabilities']
            else: a=np.load(folder/'previous_ensemble.npz')['oof']
        else:
            if full:
                with np.load(R2/f'{m["name"]}_full.npz') as data:
                    assert data['ids'].tolist()==pd.read_csv(course_data(BASE) / 'test.csv',dtype=str).id.tolist()
                    a=data['test']
            else: a=np.load(folder/f'{m["name"]}.npz')['oof']
        q=m['weight']*temper(a,selection['power'])
        p=q if p is None else p+q
    return p

def finish(final=False):
    selection=json.loads((R2/'selection.json').read_text())
    train=pd.read_csv(course_data(BASE) / 'train.csv'); test=pd.read_csv(course_data(BASE) / 'test.csv')
    cls=np.array(sorted(train.label.unique())); y=np.searchsorted(cls,train.loc[train.checked==1,'label'])
    if not final:
        folder=R2/'seed131'; previous=np.load(folder/'previous_ensemble.npz')['oof']; p=assemble(selection,folder)
        old=((np.argsort(-previous,axis=1)[:,:3]==y[:,None])/np.arange(1,4)).sum(1)
        new=((np.argsort(-p,axis=1)[:,:3]==y[:,None])/np.arange(1,4)).sum(1)
        rng=np.random.default_rng(131)
        delta=new-old; boot=np.array([rng.choice(delta,len(delta)).mean() for _ in range(5000)])
        report={'previous_map3':score(y,previous),'new_map3':score(y,p),'paired_improvement':float(delta.mean()),
                'paired_bootstrap_95_interval':np.quantile(boot,[.025,.975]).tolist(),
                'selection':selection,'accepted':bool(delta.mean()>0),
                'validation_note':'Both models evaluated on the same 5-fold checked-label splits, seed 131. Labels have been used in earlier development; this is not a new independent holdout. The original 0.806 holdout score cannot validate this new pipeline.'}
        (R2/'confirmation.json').write_text(json.dumps(report,indent=2)); np.savez_compressed(R2/'confirmation_oof.npz',y=y,pred=p,previous=previous)
        print(json.dumps(report,indent=2),flush=True); return
    report=json.loads((R2/'confirmation.json').read_text())
    p=assemble(selection,None,True) if report['accepted'] else np.load(BASE/'test_predictions.npz')['probabilities']
    assert p.shape==(len(test),len(cls)) and np.isfinite(p).all()
    sub=pd.DataFrame({'id':test.id,'label':[' '.join(cls[row]) for row in np.argsort(-p,axis=1)[:,:3]]})
    target=BASE/'submission_round2.csv'; sub.to_csv(target,index=False)
    assert pd.read_csv(target,dtype=str).id.tolist()==test.id.tolist()
    assert all(len(v.split())==len(set(v.split()))==3 and set(v.split())<=set(cls) for v in sub.label)
    np.savez_compressed(R2/'test_predictions.npz',ids=test.id.values.astype(str),probabilities=p,classes=cls)
    report['submission_sha256']=hashlib.sha256(target.read_bytes()).hexdigest()
    report['train_csv_sha256']=hashlib.sha256((course_data(BASE) / 'train.csv').read_bytes()).hexdigest()
    report['test_csv_sha256']=hashlib.sha256((course_data(BASE) / 'test.csv').read_bytes()).hexdigest()
    report['training_rows']=len(train); report['checked_rows']=int(train.checked.sum()); report['test_rows']=len(test)
    (R2/'final_report.json').write_text(json.dumps(report,indent=2)); print('Validated',target,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('stage',choices=['choose','confirm','final']); args=ap.parse_args()
    if args.stage=='choose': choose()
    else: finish(args.stage=='final')
