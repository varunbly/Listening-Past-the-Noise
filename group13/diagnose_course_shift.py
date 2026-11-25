"""Check whether checked and test recordings are distinguishable using audio alone."""
from course_input import course_data
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from course_local_graph import ROOT,OUT


def course_weights():
    data=pd.read_csv(course_data(ROOT) / 'train.csv',dtype={"id":str})
    clean=np.flatnonzero(data.checked.eq(1))
    test=pd.read_csv(course_data(ROOT) / 'test.csv',dtype=str)
    rows=np.r_[clean, np.arange(len(data),len(data)+len(test))]
    parts=[np.load(OUT/"envelope.npy")[rows]]
    with np.load(OUT/"mixture_spaces.npz") as z:
        parts.extend(z[k][rows,:15] for k in z.files[:-1])
    x=np.column_stack(parts)
    domain=np.r_[np.zeros(len(clean)),np.ones(len(test))]
    logit=np.zeros(len(x))
    for tr,va in StratifiedKFold(5,shuffle=True,random_state=1103).split(x,domain):
        scaler=StandardScaler().fit(x[tr])
        model=LogisticRegression(C=.008,class_weight="balanced",max_iter=500)
        model.fit(scaler.transform(x[tr]),domain[tr])
        logit[va]=model.decision_function(scaler.transform(x[va]))
    odds=np.exp(logit[:len(clean)]).clip(.33,3.)
    return data.id.to_numpy()[clean],odds,roc_auc_score(domain,logit)


def main():
    ids,odds,auc=course_weights()
    errors=pd.read_csv(OUT/"checked_errors.csv",dtype={"id":str})
    if not np.array_equal(errors.id,ids):
        raise ValueError("Misaligned diagnostic rows")
    grouped=errors.groupby("label").agg(previous=("previous","mean"),candidate=("candidate","mean"),
                                        checked_count=("id","size"))
    result=dict(checked_vs_test_auc=auc,
                reweighted_checked={k:np.average(errors[k],weights=odds) for k in ["previous","candidate"]},
                weight_effective_size=odds.sum()**2/(odds@odds),
                class_groups=grouped.groupby("checked_count")[["previous","candidate"]].mean().to_dict("index"),
                weakest=grouped.sort_values("candidate").head(5).to_dict("index"))
    (OUT/"shift.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
