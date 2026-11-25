"""Nested training-label-only label-shift estimates from unlabelled test audio."""
import argparse
import json
import numpy as np
from scipy.optimize import lsq_linear
from sklearn.model_selection import StratifiedKFold
from advance import BASE,R3,distance
from advance_finish import labels
from refine import graph_from_distance,robust_propagate

def estimate(g,y,training,ui,test_indices,folds=2,seed=911):
    source=np.zeros((len(training),41)); target=np.zeros((len(test_indices),41))
    for tr,va in StratifiedKFold(folds,shuffle=True,random_state=seed).split(training,y[training]):
        p=robust_propagate(g,y,training[tr],ui,alpha=.5,prior=.5,target_power=.5)
        source[va]=p[training[va]]; target+=p[test_indices]/folds
    C=np.stack([source[y[training]==c].mean(0) for c in range(41)],axis=1)
    mean=target.mean(0)
    matrix=np.vstack([C,np.sqrt(.01)*np.eye(41),10*np.ones((1,41))])
    rhs=np.r_[mean,np.sqrt(.01)*np.full(41,1/41),10]
    prior=lsq_linear(matrix,rhs,bounds=(0,1)).x
    prior=np.maximum(prior,1e-5); prior/=prior.sum()
    # Shrink noisy small-class estimates halfway toward a uniform distribution.
    prior=.5*prior+.5/41
    return prior,float(np.linalg.norm(C@prior-mean))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--seed',type=int,default=73); ap.add_argument('--final',action='store_true'); args=ap.parse_args()
    train,test,cls,y,ci,ui=labels(); test_ids=np.arange(len(train),len(train)+len(test))
    g=graph_from_distance(distance('r2_augmented0.2'),10)
    if args.final:
        p,residual=estimate(g,y,ci,ui,test_ids,folds=5)
        np.save(R3/'target_prior_final.npy',p)
        (R3/'target_prior_final.json').write_text(json.dumps({'classes':cls.tolist(),'estimated_prior':p.tolist(),'residual':residual,'note':'Estimated using training labels and unlabelled test features only; not test ground truth.'},indent=2))
        print('Final unlabelled-test prior estimated',flush=True); return
    out=R3/f'seed{args.seed}'; out.mkdir(exist_ok=True)
    priors=np.zeros((len(ci),41)); residuals=[]
    for f,(tr,va) in enumerate(StratifiedKFold(5,shuffle=True,random_state=args.seed).split(ci,y[ci])):
        p,residual=estimate(g,y,ci[tr],ui,test_ids)
        priors[va]=p; residuals.append(residual)
        print('Nested prior estimate fold',f+1,'seed',args.seed,flush=True)
    np.save(out/'target_priors.npy',priors)
    (out/'target_priors.json').write_text(json.dumps({'seed':args.seed,'outer_folds':5,'inner_folds':2,'residuals':residuals,'note':'Each outer validation label is excluded from inner teacher fits, confusion estimates and graph seeds. Test labels unavailable.'},indent=2))

if __name__=='__main__': main()
