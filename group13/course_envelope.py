"""Classical auxiliary classifier for event duration, attack, decay and repetition."""
from course_input import course_data
from course_local_graph import ROOT, OUT
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize


def envelope(path):
    x,rate=sf.read(path,dtype="float32")
    duration=len(x)/rate
    if len(x)<1024:
        x=np.pad(x,(0,1024-len(x)))
    frames=np.lib.stride_tricks.sliding_window_view(x,512)[::256]
    rms=np.sqrt((frames**2).mean(1)+1e-12)
    level=np.log(rms/rms.max()+1e-6)
    duration_features=[np.log(duration+1e-4)]
    for db in [-1.,-2.,-3.5,-5.]:
        active=np.flatnonzero(level>=db)
        gaps=np.diff(active)
        runs=np.split(active,np.flatnonzero(gaps>3)+1)
        duration_features.extend([np.log1p(len(active)),len(active)/len(level),
                                  (active[-1]-active[0]+1)/len(level),
                                  np.log1p(len(runs)), np.log1p(max(len(a) for a in runs)),
                                  np.log1p(active[0]), np.log1p(len(level)-1-active[-1])])
    position=np.linspace(0,1,len(level))
    for signal in (rms,np.mean(np.abs(np.diff(frames,axis=1)),axis=1)):
        signal=signal/max(signal.max(),1e-10)
        cdf=signal.cumsum()/max(signal.sum(),1e-12)
        duration_features.extend(np.interp([.05,.15,.35,.5,.65,.85,.95],cdf,position))
        for h in [2,8,32]:
            sm=uniform_filter1d(signal,size=h,mode="nearest")
            diff=np.diff(sm)
            peaks,_=find_peaks(sm,prominence=.12,distance=max(1,h//2))
            duration_features.extend([np.log1p(len(peaks)),len(peaks)/max(.1,duration),
                                      np.quantile(diff,.95),np.quantile(diff,.05),
                                      np.sqrt(np.mean(diff**2)),position[sm.argmax()]])
    return duration_features


def main():
    train=pd.read_csv(course_data(ROOT) / 'train.csv',dtype={"id":str})
    test=pd.read_csv(course_data(ROOT) / 'test.csv',dtype=str)
    clean=np.flatnonzero(train.checked.eq(1))
    noisy=np.flatnonzero(train.checked.eq(0))
    y=np.searchsorted(np.sort(train.label.unique()),train.label)
    cache=OUT/"envelope.npy"
    if not cache.exists():
        x=np.array([envelope(course_data(ROOT) / 'audio/audio'/s/(k+".wav"))
                    for s,frame in [("train",train),("test",test)] for k in frame.id],dtype="float32")
        if not np.isfinite(x).all():
            raise ValueError("Invalid envelope")
        np.save(cache,x)
    x=np.load(cache)
    def fit(known,query):
        ix=np.r_[known,noisy]
        model=ExtraTreesClassifier(n_estimators=320,min_samples_leaf=3,max_features=.65,n_jobs=4,
                                   random_state=253,class_weight="balanced")
        model.fit(x[ix],y[ix],sample_weight=np.r_[np.ones(len(known)),np.full(len(noisy),.22)])
        return normalize(model.predict_proba(x[query])+.001,norm="l1")
    for seed in (73,419):
        p=np.zeros((len(clean),41))
        for tr,va in StratifiedKFold(5,shuffle=True,random_state=seed).split(clean,y[clean]):
            p[va]=fit(clean[tr],clean[va])
        np.savez(OUT/f"envelope_oof{seed}.npz",ids=train.id.to_numpy(str)[clean],envelope=p)
    np.savez(OUT/"envelope_test.npz",ids=test.id.to_numpy(str),
             envelope=fit(clean,np.arange(len(train),len(train)+len(test))))


if __name__=="__main__":
    main()
