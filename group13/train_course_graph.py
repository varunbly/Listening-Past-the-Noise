"""Train a course-only local graph classifier with checked anchors and a learned noise channel."""
from course_input import course_data
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import softmax
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
from lightgbm import LGBMClassifier

from course_local_graph import ROOT, OUT, affinity
from compare_course_ensembles import metrics


def walk(operator, values, steps, restart):
    result = values.copy()
    for _ in range(steps):
        result = (1-restart)*(operator @ result) + restart*values
    return result


class GraphModel:
    def __init__(self, train):
        self.y = np.searchsorted(np.sort(train.label.unique()),train.label)
        self.clean = np.flatnonzero(train.checked.eq(1))
        self.noisy = np.flatnonzero(train.checked.eq(0))
        graph = sparse.load_npz(OUT/"graph.npz").astype("float32")
        self.p = normalize(graph,norm="l1")
        self.n = graph.shape[0]
        self.eye = np.eye(41,dtype="float32")
        anchors = np.zeros((self.n,len(self.clean)),np.float32)
        anchors[self.clean,np.arange(len(self.clean))] = 1
        self.static = []
        self.weights = []
        for r in (.08,.5):
            flow = walk(self.p,anchors,32,r)
            flow[self.clean,np.arange(len(self.clean))]=0
            self.weights.append(flow)
            self.static.append(self.noisy_votes(self.p,32,r))
        for name in ["summary","fisher8","fisher64","segments","dictionary","texture","covariance"]:
            with np.load(OUT/f"near_{name}.npz") as z:
                layer = normalize(affinity(z["idx"],z["d"],16),norm="l1")
            self.static.append(self.noisy_votes(layer,8,.18))
        with np.load(OUT/"diffusion.npz") as z:
            self.coords = normalize(z["vec"][:,:-1] * z["eig"][:-1]**4)
        sim = self.coords @ self.coords[self.clean].T
        sim[self.clean,np.arange(len(self.clean))]=-2
        self.weights.append(np.exp(7*(sim-1)))
        self.label_features = self.make_features(self.static)

    def sources(self, idx):
        y = np.zeros((self.n,41),np.float32)
        y[idx] = self.eye[self.y[idx]]
        return y

    def noisy_votes(self, operator, steps, restart):
        # Split the noisy anchors themselves. A recording's supplied label cannot
        # then confirm itself via either the restart term or a return walk.
        parts=np.array_split(np.random.default_rng(812).permutation(self.noisy),4)
        total=np.zeros((self.n,41),np.float32)
        own=np.zeros_like(total)
        for part in parts:
            votes=walk(operator,self.sources(part),steps,restart)
            total+=votes
            own[part]=votes[part]
        total[self.noisy]=np.maximum(total[self.noisy]-own[self.noisy],0)*4/3
        return total

    @staticmethod
    def make_features(views):
        items = []
        for view in views:
            a = np.log(view + 1e-8)
            items.extend([a,a-a.max(axis=1,keepdims=True)])
        return np.stack(items,axis=-1)

    def evidence(self, known, rows):
        anchors=np.searchsorted(self.clean,known)
        features=[]
        for mat in self.weights:
            sim=mat[np.ix_(rows,anchors)]
            total, maximum = [], []
            for c in range(41):
                a = sim[:, self.y[known]==c]
                total.append(a.sum(axis=1))
                maximum.append(a.max(axis=1))
            features.extend([np.column_stack(total),np.column_stack(maximum)])
        return np.concatenate([self.label_features[rows],self.make_features(features)],axis=2)

    def predict(self, known, rows):
        # Each checked example has zero affinity to its own anchor in all views.
        # Only other known labels can contribute evidence for the training pair.
        x=self.evidence(known,known)
        truth=self.eye[self.y[known]].reshape(-1)
        ranker=LGBMClassifier(n_estimators=170,learning_rate=.04,num_leaves=9,min_child_samples=42,
                             reg_lambda=5,verbosity=-1,n_jobs=3,random_state=140)
        ranker.fit(x.reshape(-1,x.shape[-1]),truth)
        def score(indices):
            a=self.evidence(known,indices)
            return softmax(ranker.booster_.predict(a.reshape(-1,a.shape[-1]),raw_score=True,
                                                   num_threads=3).reshape(len(indices),41),axis=1)
        noisy=self.noisy
        pred=score(np.r_[noisy,rows])
        pair=pred[len(noisy):]
        acoustic=pred[:len(noisy)]
        # Estimate P(supplied label | latent class) from acoustic votes. Shrink
        # toward the identity to prevent weak acoustic evidence undoing good labels.
        counts = acoustic.T @ self.eye[self.y[noisy]] + 8*self.eye + .15
        channel = normalize(counts,norm="l1")
        posterior = normalize((acoustic+.001)**.6 * channel[:,self.y[noisy]].T**.75,norm="l1")
        init=self.sources(known)
        init[noisy]=posterior
        strength=np.full((self.n,1),.002,np.float32)
        strength[known]=.9
        strength[noisy]=.20
        belief=init.copy()
        for _ in range(50):
            belief=strength*init+(1-strength)*(self.p @ belief)
        graph=normalize(belief[rows]+1e-8,norm="l1")
        # Discriminative boundary in graph diffusion coordinates. Confidence
        # weights reduce influence from doubtful supplied labels.
        fit=np.r_[known,noisy]
        labels=np.r_[self.y[known],posterior.argmax(1)]
        weight=np.r_[np.ones(len(known)),.55*posterior.max(1)]
        svm=SVC(C=8,gamma=1.4,class_weight="balanced",decision_function_shape="ovr")
        svm.fit(self.coords[fit],labels,sample_weight=weight)
        votes=svm.decision_function(self.coords[rows])
        svm_p=softmax(.7*(votes-votes.max(axis=1,keepdims=True)),axis=1)
        mixed=softmax(.5*np.log(pair+1e-6)+.30*np.log(graph+1e-6)+.20*np.log(svm_p+1e-6),axis=1)
        return dict(ranker=pair, graph=graph, svm=svm_p, mixed=mixed)


def main():
    train=pd.read_csv(course_data(ROOT) / 'train.csv',dtype={"id":str})
    test=pd.read_csv(course_data(ROOT) / 'test.csv',dtype=str)
    model=GraphModel(train)
    clean=model.clean
    for seed in (73,419):
        path=OUT/f"noise_oof{seed}.npz"
        if path.exists():
            continue
        result={}
        for i,(tr,va) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(clean,model.y[clean])):
            pred=model.predict(clean[tr],clean[va])
            for k,p in pred.items():
                if k not in result:
                    result[k]=np.empty((len(clean),41))
                result[k][va]=p
            print(seed,i,round(metrics(pred["mixed"],model.y[clean[va]])["map3"],4),flush=True)
        np.savez(path,ids=train.id.to_numpy(str)[clean],**result)
    pred=model.predict(clean,np.arange(len(train),len(train)+len(test)))
    np.savez(OUT/"noise_test.npz",ids=test.id.to_numpy(str),**pred)


if __name__=="__main__":
    main()
