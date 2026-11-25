"""Learn acoustic partitions on unchecked labels, then fit reliable labels in leaf space."""
from course_input import course_data
from course_local_graph import ROOT, OUT
from course_noise_mixture import acoustic_spaces
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.linalg import solve
from scipy.special import softmax
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize


def partitions(noisy,clean,y):
    env=np.load(OUT/"envelope.npy")
    summary=np.load(ROOT/"cache/audio-v1/matrix.npy",mmap_mode="r")
    spaces=acoustic_spaces(noisy)
    views=[env,np.column_stack([env,summary]),np.column_stack([env,*spaces])]
    acoustic=[env,*spaces]
    q=np.column_stack(acoustic).astype("float32")
    kernels=[]
    source=[]
    for i,x in enumerate(views):
        model=ExtraTreesClassifier(n_estimators=280,min_samples_leaf=3,max_features="sqrt",n_jobs=4,
                                   random_state=825+i,class_weight="balanced")
        model.fit(x[noisy],y[noisy])
        proba=model.predict_proba(x)
        out=np.full((len(x),41),.0001)
        out[:,model.classes_]+=proba
        source.append(normalize(out,norm="l1"))
        leaf=model.apply(x)
        offset=np.r_[0,np.cumsum(leaf.max(0)[:-1]+1)]
        col=(leaf+offset).ravel()
        L=sparse.csr_matrix((np.ones(len(col),np.float32),col,np.arange(0,len(col)+1,leaf.shape[1])),
                            shape=(len(leaf),int(col.max())+1))
        # Weight fine partitions more heavily than broad, frequently shared leaves.
        freq=np.asarray(L[noisy].sum(0)).ravel().clip(1)
        L=normalize(L.multiply(freq**-.25).tocsr())
        kernels.append((L @ L[clean].T).toarray())
    return q,np.mean(kernels[1:],axis=0),np.mean(source[1:],axis=0),kernels[0]


def predict(q,k,g,temporal,y,clean,tr,noisy,query):
    known=clean[tr]
    eye=np.eye(41,dtype=np.float32)
    gram=k[:,tr]
    target=eye[y[known]]
    system=gram[known].copy()
    system.flat[::len(known)+1]+=.12
    beta=solve(system,target,assume_a="pos")
    margins=gram @ beta
    # The unchecked-label forest supplies partitions; the clean fit estimates
    # their association with true classes independently from those raw labels.
    prior=target.mean(0)
    density=(gram**2) @ target/(prior**.6)
    via_leaf=softmax(14*margins+.3*np.log(density+.003),axis=1)
    noise=via_leaf[noisy]
    observed=y[noisy]
    rank=(noise>noise[np.arange(len(noisy)),observed,None]).sum(1)
    weight=np.select([rank==0,rank<3,rank<8],[.85,.55,.32],default=.12)
    model=ExtraTreesClassifier(n_estimators=320,max_features=.16,min_samples_leaf=2,class_weight="balanced",
                               n_jobs=4,random_state=730)
    model.fit(q[np.r_[known,noisy]],y[np.r_[known,noisy]],sample_weight=np.r_[np.ones(len(known)),weight])
    p=model.predict_proba(q[query])
    time_density=temporal[query][:,tr]**2 @ target/(prior**.6)
    return dict(forest=p,leaf=via_leaf[query],
                temporal_leaf=normalize(time_density+.001,norm="l1"),supplied=g[query])


def main():
    train=pd.read_csv(course_data(ROOT) / 'train.csv',dtype={"id":str})
    test=pd.read_csv(course_data(ROOT) / 'test.csv',dtype=str)
    clean=np.flatnonzero(train.checked.eq(1)); noisy=np.flatnonzero(train.checked.eq(0))
    y=np.searchsorted(np.sort(train.label.unique()),train.label)
    q,k,g,temporal=partitions(noisy,clean,y)
    for seed in (73,419):
        data={}
        for tr,va in StratifiedKFold(5,shuffle=True,random_state=seed).split(clean,y[clean]):
            for key,p in predict(q,k,g,temporal,y,clean,tr,noisy,clean[va]).items():
                if key not in data:
                    data[key]=np.zeros((len(clean),41))
                data[key][va]=p
        np.savez(OUT/f"forest_oof{seed}.npz",ids=train.id.to_numpy(str)[clean],**data)
    query=np.arange(len(train),len(train)+len(test))
    np.savez(OUT/"forest_test.npz",ids=test.id.to_numpy(str),
             **predict(q,k,g,temporal,y,clean,np.arange(len(clean)),noisy,query))


if __name__=="__main__":
    main()
