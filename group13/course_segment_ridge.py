"""Multiple-instance kernel ridge on overlapping acoustic windows of each clip."""
from course_input import course_data
from course_local_graph import ROOT,OUT
import numpy as np
import pandas as pd
from joblib import load
from scipy.linalg import solve
from scipy.special import softmax
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize


def segments(path, scaler, gmm):
    with np.load(path) as z:
        x=scaler.transform(z["frames"])
    q=gmm.predict_proba(x)
    n=len(x)
    # Whole bag, quarters and overlapping half bags cover short and sustained events.
    indices=[np.arange(n)]+[np.arange(int(start*n),min(n,max(int(start*n)+1,int((start+width)*n))))
                             for width,starts in [(.25,[0,.25,.5,.75]),(.5,[0,.25,.5])] for start in starts]
    values=[]
    for ids in indices:
        a,b=x[ids],q[ids]
        mass=b.mean(0)[:,None]
        mean=b.T @ a/len(a)
        second=b.T @ (a*a)/len(a)
        u=(mean-mass*gmm.means_)/np.sqrt(gmm.weights_[:,None]*gmm.covariances_)
        v=(second-2*mean*gmm.means_+mass*(gmm.means_**2-gmm.covariances_))/(np.sqrt(2*gmm.weights_[:,None])*gmm.covariances_)
        f=np.r_[u.ravel(),v.ravel()]
        f=np.sign(f)*np.sqrt(np.abs(f))
        moment=np.r_[a.mean(0),a.std(0),np.quantile(a,[.2,.8],axis=0).ravel()]
        values.append(np.r_[.85*f/max(np.linalg.norm(f),1e-8),.52*moment/max(np.linalg.norm(moment),1e-8)])
    return np.array(values,dtype="float32")


def representation(train,test,noisy):
    dest=OUT/"segment_basis.npy"
    if dest.exists():
        return np.load(dest)
    scaler,gmm=load(ROOT/"cache/fisher_audio-v1_8.joblib")
    x=np.stack([segments(ROOT/"cache/audio-v1"/f"{s}_{ident}.npz",scaler,gmm)
                for s,data in [("train",train),("test",test)] for ident in data.id])
    rng=np.random.default_rng(472)
    ix=rng.choice(noisy,1152,replace=False)
    landmarks=x[ix,rng.integers(8,size=len(ix))]
    def kernel(a,b):
        sim=a @ b.T
        return .5*np.exp(3*(sim-1))+.5*np.exp(7*(sim-1))
    eig,u=np.linalg.eigh(kernel(landmarks,landmarks))
    transform=u[:,-768:]/np.sqrt(np.maximum(eig[-768:],.02))
    basis=np.concatenate([kernel(b,landmarks) @ transform for b in np.array_split(x,64)])
    np.save(dest,basis)
    return basis


def classifier(x,w,t,penalty):
    a=x.reshape(-1,x.shape[-1])
    weights=w.reshape(-1)
    v=(x*w[:,:,None]).sum(1).T @ t
    gram=a.T @ (a*weights[:,None])
    gram.flat[::len(gram)+1]+=penalty
    return solve(gram,v,assume_a="pos")


def predict(x,y,clean,noisy,query):
    eye=np.eye(41,dtype="float32")
    uniform=np.array([.3,.1,.1,.1,.1,.1,.1,.1],np.float32)
    initial=classifier(x[clean],np.tile(uniform,(len(clean),1)),eye[y[clean]],.06)
    teacher=x[noisy] @ initial
    evidence=.55*teacher[:,0]+.45*teacher.max(1)
    score=evidence[np.arange(len(noisy)),y[noisy]]
    rank=(evidence>score[:,None]).sum(1)
    rel=np.select([rank==0,rank<3,rank<8],[.85,.62,.38],default=.16).astype("float32")
    idx=np.r_[clean,noisy]
    # Select informative time windows with a separate fit on checked clips.
    support=teacher[np.arange(len(noisy)),:,y[noisy]]
    attention=.45*uniform+.55*softmax(12*support,axis=1)
    w=np.vstack([np.tile(uniform,(len(clean),1)),attention*rel[:,None]])
    target=np.vstack([eye[y[clean]],rel[:,None]*eye[y[noisy]]+(1-rel[:,None])*softmax(14*evidence,axis=1)])
    beta=classifier(x[idx],w,target,.13)
    windows=x[query] @ beta
    result=.40*windows[:,0]+.23*windows.mean(1)+.37*np.sort(windows,axis=1)[:,-2:].mean(1)
    return softmax(18*result,axis=1)


def main():
    train=pd.read_csv(course_data(ROOT) / 'train.csv',dtype={"id":str})
    test=pd.read_csv(course_data(ROOT) / 'test.csv',dtype=str)
    clean=np.flatnonzero(train.checked.eq(1)); noisy=np.flatnonzero(train.checked.eq(0))
    y=np.searchsorted(np.sort(train.label.unique()),train.label)
    x=representation(train,test,noisy)
    for seed in (73,419):
        p=np.zeros((len(clean),41))
        for tr,va in StratifiedKFold(5,shuffle=True,random_state=seed).split(clean,y[clean]):
            p[va]=predict(x,y,clean[tr],noisy,clean[va])
        np.savez(OUT/f"segment_oof{seed}.npz",ids=train.id.to_numpy(str)[clean],segment=p)
    np.savez(OUT/"segment_test.npz",ids=test.id.to_numpy(str),
             segment=predict(x,y,clean,noisy,np.arange(len(train),len(train)+len(test))))


if __name__=="__main__":
    main()
