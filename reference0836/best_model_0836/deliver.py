"""Select on development CV, freeze selection, evaluate holdout, refit, submit."""
from course_input import course_data
import argparse
import json
import hashlib
import sys
from importlib.metadata import version
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits
from improve import BASE, prepare, score, svm_prob, stats_view, class_weights

def temper(p,power):
    p=np.maximum(p,1e-12)**power
    return p/p.sum(1,keepdims=True)

def choose(y):
    records=[]
    for path in sorted((BASE/'results'/'dev').glob('*_s42_4fold.json')):
        record=json.loads(path.read_text())
        p=np.load(path.with_suffix('.npz'))['oof']
        if p.shape[0]!=len(y): raise ValueError('Development split mismatch')
        records.append((path.stem,record,p))
    if not records: raise ValueError('Run development search first')
    records.sort(key=lambda a:a[1]['map3'],reverse=True)
    candidates=[]
    for power in [1.,.5,2.]:
        weights=np.zeros(len(records)); weights[0]=1.
        pred=temper(records[0][2],power)
        best=score(y,pred)
        for _ in range(2):
            improvement=None
            for j,(_,record,p) in enumerate(records):
                if weights[j]>0: continue
                for a in [.2,.35,.5]:
                    pp=(1-a)*pred+a*temper(p,power)
                    s=score(y,pp)
                    if s>best+.001 and (improvement is None or s>improvement[0]):
                        improvement=(s,j,a,pp)
            if improvement is None: break
            best,j,a,pred=improvement
            weights*=1-a; weights[j]=a
        candidates.append((best,power,weights,pred))
    best,power,weights,pred=max(candidates,key=lambda a:a[0])
    selection={'development_cv_map3':best,'probability_power':power,'models':[
        {'tag':records[j][0],'weight':float(w),'config':records[j][1]['config'],'single_cv_map3':records[j][1]['map3']}
        for j,w in enumerate(weights) if w>0]}
    np.savez_compressed(BASE/'selected_development_oof.npz',y=y,pred=pred)
    return selection

def predict_one(conf, x, train, y, checked_train, predict_idx, caches, graph_cache):
    chk=train.checked.values==1
    ui=np.where(~chk)[0]
    fit_idx=np.r_[checked_train,ui] if conf.get('unchecked_weight',1)>0 else checked_train
    weights=np.where(chk[fit_idx],1.,conf.get('unchecked_weight',1.))
    view=conf['view']
    if view.startswith(('stats','pca')):
        xx=stats_view(x,view,ui)
    elif view.startswith('fisher'):
        xx=np.load(BASE/'cache'/f'fisher_audio-v1_{view[6:]}.npz')['x']
    elif view=='combined':
        xx=None
    else: raise ValueError(view)
    if conf['kind']=='graph':
        from graph_search import graph, propagate
        key=(view,conf['k'])
        if key not in graph_cache:
            if view=='stats': xx=normalize(np.clip(StandardScaler().fit(x[ui]).transform(x),-6,6))
            if view=='combined':
                statsx=normalize(np.clip(StandardScaler().fit(x[ui]).transform(x),-6,6))
                fish=np.load(BASE/'cache'/'fisher_audio-v1_64.npz')['x']
                xx=np.concatenate([statsx/np.sqrt(2),fish/np.sqrt(2)],axis=1)
            graph_cache[key]=graph(xx,conf['k'])
        p=propagate(graph_cache[key],y,checked_train,ui,conf['noise_alpha'],conf['prior'],conf.get('target_power',0.))
        return p[predict_idx]
    if conf['kind']=='svm':
        if conf.get('scale',True):
            xx=np.clip(StandardScaler().fit(xx[ui]).transform(xx),-6,6)
        if conf.get('l2'): xx=normalize(xx)
        gamma=conf.get('gamma_factor',1.)/(xx.shape[1]*xx.var())
        if conf.get('cleaner'):
            with threadpool_limits(limits=4):
                k=rbf_kernel(xx[checked_train],gamma=gamma)
                ku=rbf_kernel(xx[ui],xx[checked_train],gamma=gamma)
            teacher=SVC(C=20,kernel='precomputed',decision_function_shape='ovr').fit(k,y[checked_train])
            rank=np.argmax(np.argsort(-teacher.decision_function(ku),axis=1)==y[ui,None],axis=1)
            weights[len(checked_train):]*=np.select([rank==0,rank==1,rank==2],[1.,.5,.25],default=.05)
            del k,ku,teacher
        with threadpool_limits(limits=4):
            kernel=rbf_kernel(xx[fit_idx],gamma=gamma)
        model=SVC(C=conf['C'],kernel='precomputed',class_weight=class_weights(conf,y,fit_idx,checked_train,weights),decision_function_shape='ovr')
        model.fit(kernel,y[fit_idx],sample_weight=weights)
        del kernel
        result=[]
        for start in range(0,len(predict_idx),256):
            with threadpool_limits(limits=4):
                k=rbf_kernel(xx[predict_idx[start:start+256]],xx[fit_idx],gamma=gamma)
            result.append(svm_prob(model,k))
        return np.concatenate(result)
    if conf['kind']=='trees':
        model=ExtraTreesClassifier(n_estimators=conf.get('n_estimators',350),max_features=conf.get('max_features','sqrt'),min_samples_leaf=conf.get('leaf',2),n_jobs=4,class_weight=conf.get('class_weight','balanced'),random_state=42)
    elif conf['kind']=='lgb':
        from lightgbm import LGBMClassifier
        model=LGBMClassifier(n_estimators=conf.get('n_estimators',250),num_leaves=conf.get('leaves',15),learning_rate=.06,colsample_bytree=.6,subsample=.8,reg_lambda=5.,max_bin=conf.get('max_bin',63),min_child_samples=10,n_jobs=4,verbosity=-1,random_state=42)
    else: raise ValueError(conf)
    model.fit(xx[fit_idx],y[fit_idx],sample_weight=weights)
    return model.predict_proba(xx[predict_idx])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--jobs',type=int,default=2)
    ap.add_argument('--reselect',action='store_true',help='Select using development results only; do not tune on the locked holdout')
    args=ap.parse_args()
    if args.reselect and (BASE/'holdout_results.json').exists():
        raise ValueError('Selection is frozen because the locked holdout has already been evaluated. Do not retune against this holdout.')
    train,test,x,caches=prepare(course_data(BASE),args.jobs)
    classes=np.array(sorted(train.label.unique()))
    y=np.searchsorted(classes,train.label)
    ci=np.where(train.checked.values==1)[0]
    dev,hold=train_test_split(ci,test_size=.2,stratify=y[ci],random_state=2026)
    dev=np.sort(dev)
    frozen=BASE/'selection.json'
    if frozen.exists() and not args.reselect:
        selection=json.loads(frozen.read_text())
    else:
        selection=choose(y[dev])
        frozen.write_text(json.dumps(selection,indent=2))
    print('Frozen selection:',json.dumps(selection,indent=2),flush=True)
    graph_cache={}
    if not (BASE/'holdout_results.json').exists():
        ph=np.zeros((len(hold),len(classes)))
        for m in selection['models']:
            print('Holdout fit:',m['tag'],flush=True)
            p=predict_one(m['config'],x,train,y,dev,hold,caches,graph_cache)
            ph+=m['weight']*temper(p,selection['probability_power'])
        contributions=((np.argsort(-ph,axis=1)[:,:3]==y[hold,None])/np.arange(1,4)).sum(1)
        rng=np.random.default_rng(42)
        boot=np.array([rng.choice(contributions,len(hold)).mean() for _ in range(5000)])
        report={'map3':score(y[hold],ph),'n_checked_holdout':len(hold),'bootstrap_95_percentile_interval':np.quantile(boot,[.025,.975]).tolist(),'selection':selection,'note':'Holdout evaluated after freezing selection. Interval describes this small holdout, not leaderboard uncertainty.'}
        (BASE/'holdout_results.json').write_text(json.dumps(report,indent=2))
        np.savez_compressed(BASE/'locked_holdout_predictions.npz',ids=train.id.iloc[hold].values.astype(str),y=y[hold],pred=ph)
        print('LOCKED HOLDOUT MAP@3',report['map3'],flush=True)
    pt=np.zeros((len(test),len(classes)))
    for m in selection['models']:
        print('Full training:',m['tag'],flush=True)
        p=predict_one(m['config'],x,train,y,ci,np.arange(len(train),len(x)),caches,graph_cache)
        pt+=m['weight']*temper(p,selection['probability_power'])
    labels=[' '.join(classes[row]) for row in np.argsort(-pt,axis=1)[:,:3]]
    sub=pd.DataFrame({'id':test.id,'label':labels})
    assert sub.id.is_unique and sub.id.tolist()==test.id.tolist()
    assert len(sub)==1592 and pt.shape==(1592,41) and np.isfinite(pt).all()
    assert all(len(v.split())==len(set(v.split()))==3 and set(v.split())<=set(classes) for v in sub.label)
    sub.to_csv(BASE/'submission_improved.csv',index=False)
    reread=pd.read_csv(BASE/'submission_improved.csv',dtype={'id':str})
    assert reread.equals(sub) and list(reread.columns)==['id','label']
    np.savez_compressed(BASE/'test_predictions.npz',ids=test.id.values.astype(str),classes=classes,probabilities=pt)
    manifest={'train_csv_sha256':hashlib.sha256((course_data(BASE)/'train.csv').read_bytes()).hexdigest(),'test_csv_sha256':hashlib.sha256((course_data(BASE)/'test.csv').read_bytes()).hexdigest(),'training_rows':len(train),'test_rows':len(test),'checked_rows':len(ci),'classes':classes.tolist(),'selection':selection,'holdout':json.loads((BASE/'holdout_results.json').read_text())}
    manifest['python']=sys.version
    manifest['package_versions']={m:version(m) for m in ['numpy','pandas','scipy','scikit-learn','librosa','soundfile','joblib','lightgbm']}
    (BASE/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
    print('Validated submission:',BASE/'submission_improved.csv',flush=True)

if __name__=='__main__': main()
