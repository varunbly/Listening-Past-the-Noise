"""Reproduce the previous ensemble's 0.833871 MAP@3 five-fold CV result.

No round-two temporal features or updated ensemble are used by this evaluator.
Every held-out checked label is excluded from supervised fits and graph seeds.
"""
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from improve import BASE, prepare, score
from deliver import predict_one, temper

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seed',type=int,default=131)
    ap.add_argument('--jobs',type=int,default=2)
    args=ap.parse_args()
    train,test,x,caches=prepare(BASE/'project4',args.jobs)
    classes=np.array(sorted(train.label.unique()))
    y=np.searchsorted(classes,train.label)
    checked=np.where(train.checked.values==1)[0]
    selection=json.loads((BASE/'selection.json').read_text())
    oof=np.zeros((len(checked),len(classes)))
    graph_cache={}
    fold_scores=[]
    splitter=StratifiedKFold(5,shuffle=True,random_state=args.seed)
    for fold,(tr,va) in enumerate(splitter.split(checked,y[checked])):
        for component in selection['models']:
            p=predict_one(component['config'],x,train,y,checked[tr],checked[va],caches,graph_cache)
            oof[va]+=component['weight']*temper(p,selection['probability_power'])
        fold_scores.append(score(y[checked[va]],oof[va]))
        print(f'Fold {fold+1}: {fold_scores[-1]:.8f}',flush=True)
    report={'model':'previous_frozen_ensemble','map3':score(y[checked],oof),
            'fold_scores':fold_scores,'seed':args.seed,'folds':5,
            'checked_recordings':len(checked),
            'note':'Repeated checked-label cross-validation, not test/leaderboard accuracy. The frozen model was chosen previously; this does not establish a new independent holdout.'}
    (BASE/'validation_0834.json').write_text(json.dumps(report,indent=2))
    np.savez_compressed(BASE/'frozen_validation_predictions.npz',ids=train.id.iloc[checked].values.astype(str),y=y[checked],predictions=oof)
    print('MAP@3:',report['map3'],flush=True)

if __name__=='__main__': main()
