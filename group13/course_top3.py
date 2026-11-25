"""Keep the public model's first guess and train a conditional ranker for the other two."""
from course_input import course_data
import json
import hashlib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from compare_course_ensembles import ROOT,REFERENCE,metrics,credit,ranks,paired_interval
from course_local_graph import OUT
from diagnose_course_shift import course_weights
from select_course_graph import matrices,fit,apply


def train_lower(b,x,y,counts,importance):
    # When the first guess is right, the other guesses cannot change MAP@3.
    # Only top-one errors supply targets for the conditional fit.
    first=b.argmax(1)
    use=first!=y
    conditional=b[use].copy()
    conditional[np.arange(len(conditional)),first[use]]=-100
    return fit(conditional,x[use],y[use],counts,1.,np.sqrt(importance[use]))


def rank_lower(b,x,param):
    out=apply(b,x,param)
    out[np.arange(len(b)),b.argmax(1)]=out.max(1)+1
    return out


def base_choices(stage):
    if stage=="test":
        sources=[(REFERENCE/"test_predictions.npz","probabilities"),(ROOT/"test_predictions.npz","probabilities")]
    else:
        name="confirmation" if stage==419 else "selection"
        sources=[(REFERENCE/f"{name}_oof.npz","pred"),
                 (ROOT/f"results/course_comparison/previous_seed{stage}.npz","pred")]
    choices=[]
    for path,key in sources:
        with np.load(path) as data:
            choices.append(data[key].argmax(1))
    return np.column_stack(choices)


def arbitrate(lower,b,x,param,primary):
    # Revisit a first guess only for a close disagreement between the two
    # original models. Independent evidence must support the runner-up.
    initial=ranks(b)
    old,alt=initial[:,0],initial[:,1]
    contested=((primary.min(1)==np.minimum(old,alt)) & (primary.max(1)==np.maximum(old,alt)))
    row=np.arange(len(b))
    learned=apply(b,x,param)
    supports=(x[row,alt][:,[3,5,6]]>x[row,old][:,[3,5,6]]).sum(1)
    change=(contested & (b[row,old]-b[row,alt]<np.log(2)) & (supports>=2)
            & (learned[row,alt]-learned[row,old]>np.log(1.25)))
    result=lower.copy()
    result[row[change],alt[change]]=lower.max(1)[change]+1
    return result


def main():
    train=pd.read_csv(course_data(ROOT) / 'train.csv',dtype={"id":str})
    checked=train[train.checked==1]
    classes=np.sort(train.label.unique())
    y=np.searchsorted(classes,checked.label)
    counts=np.bincount(y)
    ids,importance,_=course_weights()
    if not np.array_equal(checked.id,ids):
        raise ValueError("Incorrect ID order for calibration")
    samples=[];results={}; arbitration={}
    for seed in (73,419):
        b,x=matrices(seed,ids,counts)
        samples.append((b,x))
        oof=np.empty_like(b); second=np.empty_like(b)
        primary=base_choices(seed)
        for _,fold in StratifiedKFold(5,shuffle=True,random_state=seed).split(ids,y):
            for tr,va in StratifiedKFold(2,shuffle=True,random_state=877).split(fold,y[fold]):
                a,v=fold[tr],fold[va]
                param=train_lower(b[a],x[a],y[a],counts,importance[a])
                oof[v]=rank_lower(b[v],x[v],param)
                full=fit(b[a],x[a],y[a],counts,.5,np.sqrt(importance[a]))
                second[v]=arbitrate(oof[v],b[v],x[v],full,primary[v])
        old=credit(b,y);new=credit(oof,y)
        results[seed]=dict(previous=metrics(b,y),candidate=metrics(oof,y),
                           weighted_previous=np.average(old,weights=importance),
                           weighted_candidate=np.average(new,weights=importance),
                           improved=int((new>old).sum()),worse=int((new<old).sum()),
                           paired95=paired_interval(new-old,y))
        other=credit(second,y)
        arbitration[seed]=dict(candidate=metrics(second,y),weighted_candidate=np.average(other,weights=importance),
                               paired95=paired_interval(other-old,y),first_changes=int((oof.argmax(1)!=second.argmax(1)).sum()))
    param=train_lower(np.concatenate([a for a,_ in samples]),np.concatenate([a for _,a in samples]),
                      np.tile(y,2),counts,np.tile(importance,2))
    sample=pd.read_csv(course_data(ROOT) / 'sample_submission.csv',dtype=str)
    b,x=matrices("test",sample.id.to_numpy(),counts)
    top=ranks(rank_lower(b,x,param)); prior=ranks(b)
    sample.label=[" ".join(a) for a in classes[top]]
    path=OUT/"submission_top3.csv"
    sample.to_csv(path,index=False)
    if (not pd.read_csv(path,dtype=str).equals(sample) or not np.array_equal(top[:,0],prior[:,0])
            or len(sample)!=1592 or not sample.id.is_unique):
        raise ValueError("Unexpected final submission")
    result=dict(validation=results,changed_rankings=int((top!=prior).any(1).sum()),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (OUT/"top3_results.json").write_text(json.dumps(result,indent=2))
    full=fit(np.concatenate([a for a,_ in samples]),np.concatenate([a for _,a in samples]),
             np.tile(y,2),counts,.5,np.tile(np.sqrt(importance),2))
    np.savez_compressed(OUT/"final_rankers.npz", lower=param, arbitration=full, classes=classes.astype(str))
    pred=arbitrate(rank_lower(b,x,param),b,x,full,base_choices("test"))
    idx=ranks(pred)
    sample.label=[" ".join(row) for row in classes[idx]]
    second_file=OUT/"submission_arbitrated.csv"
    sample.to_csv(second_file,index=False)
    if not pd.read_csv(second_file,dtype=str).equals(sample):
        raise ValueError("Invalid arbitration file")
    extra=dict(validation=arbitration,changed_first=int((idx[:,0]!=prior[:,0]).sum()),
               changed_rankings=int((idx!=prior).any(1).sum()))
    (OUT/"arbitration.json").write_text(json.dumps(extra,indent=2))
    print(json.dumps(extra,indent=2))


if __name__=="__main__":
    main()
