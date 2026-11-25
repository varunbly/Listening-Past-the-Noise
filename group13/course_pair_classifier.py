"""Learn acoustic similarity from verified clip pairs and propagate weighted votes."""
from course_input import course_data
from course_local_graph import ROOT, OUT
from course_noise_mixture import acoustic_spaces
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import softmax
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize
from compare_course_ensembles import metrics


def top(a,k):
    return np.argpartition(-a,k-1,axis=1)[:,:k]


class AcousticPairs:
    def __init__(self,train):
        self.clean=np.flatnonzero(train.checked.eq(1))
        self.noisy=np.flatnonzero(train.checked.eq(0))
        self.y=np.searchsorted(np.sort(train.label.unique()),train.label)
        self.spaces=acoustic_spaces(self.noisy)
        self.unit=[normalize(x) for x in self.spaces]
        stat=np.load(ROOT/"cache/audio-v1/matrix.npy",mmap_mode="r")
        self.level=stat[:,:3].copy()
        n=len(stat)
        graph=sparse.load_npz(OUT/"graph.npz").tocsr()
        # Distances and walk signatures are label-free. Anchor candidates include
        # every checked clip when fitting the pair classifier.
        self.edges=[]
        for name in ["fisher8","fisher64","summary","texture","segments","dictionary"]:
            with np.load(OUT/f"near_{name}.npz") as data:
                ix,d=data["idx"],data["d"]
            scale=np.sort(d,axis=1)[:,9:10].clip(.001)
            self.edges.append(sparse.csr_matrix((np.exp(-d/scale).ravel(),ix.ravel(),
                                    np.arange(0,(n+1)*ix.shape[1],ix.shape[1])),shape=(n,n)))
        self.edges.append(graph)
        self.edges.append((graph @ graph).tocsr())
        proposal=[]
        for a,b in zip(graph.indptr[:-1],graph.indptr[1:]):
            ix=graph.indices[a:b]
            proposal.append(ix[np.argsort(-graph.data[a:b])[:45]])
        self.rows=np.repeat(np.arange(n),[len(x) for x in proposal])
        self.cols=np.concatenate(proposal)
        self.edge_features=self.features(self.rows,self.cols)
        # Keep a broader set of candidate anchors than the short graph edges.
        similar=sum(x @ x[self.clean].T for x in self.unit)/len(self.unit)
        nearest=top(similar,64)
        self.qa=np.repeat(np.arange(n),nearest.shape[1])
        self.qb=self.clean[nearest.ravel()]
        self.anchor_features=self.features(self.qa,self.qb)
        left,right=np.triu_indices(len(self.clean),1)
        self.ca,self.cb=self.clean[left],self.clean[right]
        self.train_features=self.features(self.ca,self.cb)
        self.n=n

    def features(self,rows,cols):
        batches=[]
        for start in range(0,len(rows),12000):
            a,b=rows[start:start+12000],cols[start:start+12000]
            feat=[np.abs(self.level[a]-self.level[b])]
            for x,xx in zip(self.spaces,self.unit):
                diff=np.abs(x[a]-x[b])
                feat.extend([(xx[a]*xx[b]).sum(1)[:,None],diff.mean(1)[:,None],
                             diff[:,:8],.5*(x[a,:3]+x[b,:3])])
            for e in self.edges:
                s=np.asarray(e[a,b]).reshape(-1,1)
                t=np.asarray(e[b,a]).reshape(-1,1)
                feat.extend([s+t,np.minimum(s,t)])
            batches.append(np.column_stack(feat).astype("float32"))
        return np.concatenate(batches)

    def predict(self,known,query):
        valid=np.isin(self.ca,known)&np.isin(self.cb,known)
        a,b=self.ca[valid],self.cb[valid]
        same=self.y[a]==self.y[b]
        count=np.bincount(self.y[known])
        # Reduce dominance of classes with many more possible positive pairs.
        weight=np.where(same, 12/np.sqrt(count[self.y[a]]*count[self.y[b]]),1.)
        rng=np.random.default_rng(8441)
        keep=same | (rng.random(len(a))<.35)
        model=LGBMClassifier(n_estimators=135,num_leaves=15,min_child_samples=45,learning_rate=.055,
                             colsample_bytree=.85,reg_lambda=6,n_jobs=4,verbosity=-1,random_state=8441)
        model.fit(self.train_features[valid][keep],same[keep],sample_weight=weight[keep])
        def probs(x):
            return model.booster_.predict(x,num_threads=4).astype("float32")
        good=np.isin(self.qb,known)&(self.qa!=self.qb)
        p=probs(self.anchor_features[good])
        label=self.y[self.qb[good]]
        a=self.qa[good]
        # The strongest independently verified acoustic match for each class.
        maxima=np.zeros((self.n,41),np.float32)
        np.maximum.at(maxima,(a,label),p)
        total=np.zeros_like(maxima)
        np.add.at(total,(a,label),p**2)
        direct=normalize(maxima**2+.35*total/np.sqrt(count),norm="l1")
        p=probs(self.edge_features)**1.5
        graph=sparse.csr_matrix((p,(self.rows,self.cols)),shape=(self.n,self.n))
        graph=normalize((graph+graph.T).tocsr(),norm="l1")
        noisy=self.noisy
        one=np.eye(41,dtype="float32")
        counts=direct[noisy].T @ one[self.y[noisy]] + 5*one+.25
        channel=normalize(counts,norm="l1")
        local=normalize(.5*direct + .5*graph @ direct,norm="l1")
        post=normalize((local[noisy]+.002)**.5 * channel[:,self.y[noisy]].T**.85,norm="l1")
        src=direct.copy()
        src[known]=one[self.y[known]]
        src[noisy]=post
        restart=np.full((self.n,1),.04,np.float32)
        restart[known]=.9
        restart[noisy]=.28
        belief=src.copy()
        for _ in range(45):
            belief=restart*src+(1-restart)*(graph @ belief)
        return dict(pairs=normalize(belief[query]+.0001,norm="l1"),
                    verified=normalize(direct[query]+.0001,norm="l1"))


def main():
    train=pd.read_csv(course_data(ROOT) / 'train.csv',dtype={"id":str})
    test=pd.read_csv(course_data(ROOT) / 'test.csv',dtype=str)
    model=AcousticPairs(train)
    clean=model.clean
    for seed in [73,419]:
        path=OUT/f"pairs_oof{seed}.npz"
        if path.exists():
            continue
        result={}
        for i,(tr,va) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(clean,model.y[clean])):
            for key,p in model.predict(clean[tr],clean[va]).items():
                if key not in result:
                    result[key]=np.zeros((len(clean),41))
                result[key][va]=p
            print(seed,i,round(metrics(result["pairs"][va],model.y[clean[va]])["map3"],4),flush=True)
        np.savez(path,ids=train.id.to_numpy(str)[clean],**result)
    pred=model.predict(clean,np.arange(len(train),len(train)+len(test)))
    np.savez(OUT/"pairs_test.npz",ids=test.id.to_numpy(str),**pred)


if __name__=="__main__":
    main()
