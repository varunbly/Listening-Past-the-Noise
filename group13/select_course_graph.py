"""Cross-fit a regularized correction to the 0.83931 submission on checked audio."""
from course_input import course_data
import json
import hashlib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax
from sklearn.model_selection import StratifiedKFold

from blend_checked_prior import pool
from compare_course_ensembles import ROOT, REFERENCE, metrics, credit, paired_interval, ranks
from course_local_graph import OUT


def matrices(stage,ids,counts):
    source=ROOT/"results/native_upgrade/final_course"
    recipe=json.loads((source/"metrics.json").read_text())
    entries={}
    if stage=="test":
        parts=[]
        for path in (REFERENCE/"test_predictions.npz",ROOT/"test_predictions.npz"):
            with np.load(path,allow_pickle=True) as data:
                if not np.array_equal(ids,data["ids"]):
                    raise ValueError("Baseline submission order mismatch")
                parts.append(data["probabilities"])
        for_scores="test"
    else:
        seed=stage
        name="selection" if seed==73 else "confirmation"
        with np.load(REFERENCE/f"{name}_oof.npz") as git, np.load(
            ROOT/f"results/course_comparison/previous_seed{seed}.npz") as old:
            if not np.array_equal(old["ids"],ids) or not np.array_equal(old["y"],git["y"]):
                raise ValueError("Baseline checked predictions misaligned")
            parts=[git["pred"],old["pred"]]
        for_scores=f"oof{seed}"
    base=pool(parts,recipe["powers"],recipe["weight"],recipe["exponent"],counts)
    for prefix,components in [("noise_",["ranker","graph"]),("mixture_",["density"]),
                              ("pairs_",["pairs","verified"]),("envelope_",["envelope"]),
                              ("segment_",["segment"]),
                              ("forest_",["forest","leaf","temporal_leaf"])]:
        with np.load(OUT/f"{prefix}{for_scores}.npz") as data:
            if not np.array_equal(data["ids"],ids):
                raise ValueError(f"Misaligned {prefix} features")
            entries.update({k:data[k] for k in components})
    def logits(p):
        a=np.log(p+.001)
        return a-a.max(1,keepdims=True)
    b=np.log(base.clip(1e-12))
    b-=b.max(1,keepdims=True)
    features=[logits(p) for p in entries.values()]
    # Let a diffuse auxiliary vote have less effect when the baseline is decisive.
    confidence=np.max(base,axis=1)[:,None]
    features.extend([logits(entries[k])*(1-confidence) for k in ("ranker","pairs","envelope")])
    return b,np.stack(features,axis=-1)


def fit(b,x,y,counts,penalty,importance=None):
    # Class-balanced and ordinary risk receive comparable weight. Per-class
    # intercepts are strongly shrunk to limit the influence of tiny classes.
    nfeat=x.shape[-1]
    weights=counts[y]**-.5
    if importance is not None:
        weights*=importance
    weights/=weights.sum()
    def cost(param):
        w=param[:nfeat]
        bias=param[nfeat:]
        scores=b+(x@w)+bias
        loss=weights @ (logsumexp(scores,axis=1)-scores[np.arange(len(y)),y])
        err=softmax(scores,axis=1)
        err[np.arange(len(y)),y]-=1
        err*=weights[:,None]
        loss+=penalty*(w@w)+.015*(bias@bias)
        return loss,np.r_[np.einsum("ij,ijk->k",err,x)+2*penalty*w,err.sum(0)+.03*bias]
    sol=minimize(cost,np.zeros(nfeat+len(counts)),jac=True,method="L-BFGS-B",
                 bounds=[(0,.8)]*nfeat+[(-1.1,1.1)]*len(counts))
    if not sol.success:
        raise ValueError(sol.message)
    return sol.x


def apply(b,x,param):
    n=x.shape[-1]
    return b+x@ param[:n]+param[n:]


def foldscores(b,x,y,counts,seed,penalty):
    out=np.empty(b.shape)
    # Both halves share the same base-model fit; all base models excluded
    # the entire outer fold. Calibration labels never trained a model used to
    # generate evidence on the other half of that fold.
    for _,val in StratifiedKFold(5,shuffle=True,random_state=seed).split(y,y):
        for a,v in StratifiedKFold(2,shuffle=True,random_state=877).split(val,y[val]):
            tr,va=val[a],val[v]
            param=fit(b[tr],x[tr],y[tr],counts,penalty)
            out[va]=apply(b[va],x[va],param)
    return out


def main():
    train=pd.read_csv(course_data(ROOT) / 'train.csv',dtype={"id":str})
    checked=train[train.checked==1]
    classes=np.sort(train.label.unique())
    ids=checked.id.to_numpy()
    y=np.searchsorted(classes,checked.label)
    counts=np.bincount(y)
    b,x=matrices(73,ids,counts)
    grid=[]
    for penalty in (.1,.5,2):
        pred=foldscores(b,x,y,counts,73,penalty)
        grid.append(dict(method="stack",penalty=penalty,**metrics(pred,y)))
    # Event duration is weak as a standalone classifier, but can distinguish
    # otherwise similar sounds. Compare this simple specialization to stacking.
    for method in ("envelope","temporal"):
        for strength in (.12,.25,.5,.8):
            aux=x[:,:,5] if method=="envelope" else .5*x[:,:,5]+.5*x[:,:,9]
            grid.append(dict(method=method,penalty=strength,**metrics(b+strength*aux,y)))
    choice=max(grid,key=lambda a:a["map3"]+a["macro_map3"])
    penalty=choice["penalty"]
    b2,x2=matrices(419,ids,counts)
    def specialist(base,features):
        aux=features[:,:,5] if choice["method"]=="envelope" else .5*features[:,:,5]+.5*features[:,:,9]
        return base+penalty*aux
    confirmed=(foldscores(b2,x2,y,counts,419,penalty) if choice["method"]=="stack"
               else specialist(b2,x2))
    delta=credit(confirmed,y)-credit(b2,y)
    baseline=metrics(b2,y)
    fold_delta=[float(delta[v].mean()) for _,v in StratifiedKFold(5,shuffle=True,random_state=419).split(ids,y)]
    report=dict(previous_public=.83931,selection_baseline=metrics(b,y),selection=grid,
                method=choice["method"],penalty=penalty,baseline=baseline,confirmation=metrics(confirmed,y),
                difference=float(delta.mean()),paired_95=paired_interval(delta,y),fold_gains=fold_delta,
                better=int((delta>0).sum()),worse=int((delta<0).sum()))
    # Fit the shared coefficients for deployment using all checked OOF
    # predictions. Each clip appears once in each seed and has equal total weight.
    param=fit(np.concatenate([b,b2]),np.concatenate([x,x2]),np.tile(y,2),counts,penalty)
    test=pd.read_csv(course_data(ROOT) / 'sample_submission.csv',dtype=str)
    tb,tx=matrices("test",test.id.to_numpy(),counts)
    old=pd.read_csv(ROOT/"results/native_upgrade/final_course/submission.csv",dtype=str)
    if not np.array_equal(test.id,old.id) or old.label.tolist()!=[" ".join(r) for r in classes[ranks(tb)]]:
        raise ValueError("Previous submission rankings were not reproduced")
    prediction=apply(tb,tx,param) if choice["method"]=="stack" else specialist(tb,tx)
    top=ranks(prediction)
    test.label=[" ".join(a) for a in classes[top]]
    if len(test)!=1592 or not test.id.is_unique or not np.isfinite(prediction).all():
        raise ValueError("Incomplete submission")
    dest=OUT/"submission.csv"
    test.to_csv(dest,index=False)
    if not pd.read_csv(dest,dtype=str).equals(test):
        raise ValueError("CSV round trip failed")
    report.update(test_changed_first=int((top[:,0]!=ranks(tb)[:,0]).sum()),
                  test_changed_ranked=int((test.label!=old.label).sum()),
                  sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
                  coefficients=param.tolist() if choice["method"]=="stack" else None)
    pd.DataFrame({"id":ids,"label":classes[y],"previous":credit(b2,y),"candidate":credit(confirmed,y),
                  "first":classes[ranks(confirmed)[:,0]]}).to_csv(OUT/"checked_errors.csv",index=False)
    (OUT/"evaluation.json").write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!="coefficients"},indent=2))


if __name__=="__main__":
    main()
