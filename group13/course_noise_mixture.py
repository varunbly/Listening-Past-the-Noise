"""Estimate latent clip categories and class-specific label noise with EM."""
from course_input import course_data
from course_local_graph import ROOT, OUT, clip_view
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize
from compare_course_ensembles import metrics


def acoustic_spaces(unchecked):
    cache=OUT/"mixture_spaces.npz"
    if cache.exists():
        with np.load(cache) as z:
            return [z[k] for k in z]
    rng=np.random.default_rng(844)
    sample=rng.choice(unchecked, min(3200,len(unchecked)),replace=False)
    spaces=[]
    for feature in ["summary","fisher8","fisher64","dictionary","texture","covariance"]:
        x=clip_view(feature,unchecked)
        pca=PCA(n_components=48,random_state=472,svd_solver="randomized")
        pca.fit(x[sample])
        spaces.append((pca.transform(x)/pca.explained_variance_**.5).clip(-6,6))
    with np.load(OUT/"diffusion.npz") as z:
        x=normalize(z["vec"][:,-50:-1]*z["eig"][-50:-1]**2)
    spaces.append(x*np.sqrt(x.shape[1]))
    np.savez(cache,*spaces)
    return spaces


def density(x, responsibility, fit):
    # A pooled covariance stabilizes estimates for classes with few checked
    # examples. The Student-t score limits the effect of acoustically atypical clips.
    mass=responsibility.sum(0).clip(1e-5)
    xx=x[fit]
    means=responsibility.T @ xx / mass[:,None]
    pooled=(xx.T @ (xx*responsibility.sum(1)[:,None])-
            (means.T*mass) @ means)/mass.sum()
    pooled=.85*pooled+.15*np.eye(x.shape[1],dtype="float32")
    scores=np.empty((len(x),41),dtype=np.float32)
    for c in range(41):
        dx=xx-means[c]
        cov=((dx.T*responsibility[:,c]) @ dx + 70*pooled)/(mass[c]+70)
        v,u=np.linalg.eigh(cov)
        v=v.clip(.04)
        norm=((x-means[c]) @ (u/np.sqrt(v)))**2
        scores[:,c]=-.5*np.log(v).sum()-.5*(x.shape[1]+8)*np.log1p(norm.sum(1)/8)
    return scores


def infer(views, labels, clean, noisy, query):
    eye=np.eye(41,dtype="float32")
    fit=np.r_[clean,noisy]
    # A checked clip has more influence on density estimation than a single
    # unchecked clip. The E step can change any unchecked clip's latent class.
    supplied=eye[labels[noisy]]
    post=supplied.copy()
    models=[]
    for iteration in range(6):
        resp=np.vstack([eye[labels[clean]]*5,post])
        scores=np.mean([density(view,resp,fit) for view in views], axis=0)
        models.append(softmax(scores[query]/2.6,axis=1))
        # Learn the confusion probabilities independently for each latent class.
        # A weak diagonal prior encodes that supplied labels carry information.
        counts=post.T @ supplied + 4*eye + .30
        trans=normalize(counts,norm="l1")
        acoustic=softmax(scores[noisy]/3.5,axis=1)
        if iteration==0:
            # Seed off-diagonal estimates from audio, avoiding an absorbing
            # identity transition matrix when the initial labels are all hard.
            trans=normalize(acoustic.T @ supplied+4*eye+.30,norm="l1")
        evidence=np.log(acoustic+1e-8)+.72*np.log(trans[:,labels[noisy]].T+1e-8)
        post=.15*post+.85*softmax(evidence,axis=1)
    return {"density":np.mean(models[2:],axis=0)}


def main():
    train=pd.read_csv(course_data(ROOT) / 'train.csv',dtype={"id":str})
    test=pd.read_csv(course_data(ROOT) / 'test.csv',dtype=str)
    classes,labels=np.unique(train.label.to_numpy(str),return_inverse=True)
    clean=np.flatnonzero(train.checked.eq(1))
    noisy=np.flatnonzero(train.checked.eq(0))
    views=acoustic_spaces(noisy)
    for seed in [73,419]:
        out=OUT/f"mixture_oof{seed}.npz"
        if out.exists():
            continue
        probs=np.zeros((len(clean),len(classes)))
        for i,(tr,va) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(clean,labels[clean])):
            probs[va]=infer(views,labels,clean[tr],noisy,clean[va])["density"]
            print(seed,i,round(metrics(probs[va],labels[clean[va]])["map3"],4),flush=True)
        np.savez(out,ids=train.id.to_numpy(str)[clean],density=probs)
    scores=infer(views,labels,clean,noisy,np.arange(len(train),len(train)+len(test)))
    np.savez(OUT/"mixture_test.npz",ids=test.id.to_numpy(str),**scores)


if __name__=="__main__":
    main()
