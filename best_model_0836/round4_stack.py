"""Nested class-specific log-linear stacking of classical graph predictions."""
import argparse
import json
import numpy as np
from scipy.special import logsumexp,softmax
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from advance import BASE,R3,distance,metrics
from advance_finish import labels
from refine import graph_from_distance,robust_propagate
from round4 import R4,strong_distance,inputs,graph_for

def stack_loss(flat,X,y,reg):
    models,classes=X.shape[1:]
    weights=flat[:models*classes].reshape(models,classes); bias=flat[models*classes:]
    logits=np.einsum('nmc,mc->nc',X,weights)+bias
    counts=np.bincount(y,minlength=classes); sw=1/np.maximum(counts[y],1); sw/=sw.sum()
    loss=float((sw*(logsumexp(logits,axis=1)-logits[np.arange(len(y)),y])).sum())
    loss+=reg/classes*(np.sum((weights-1/models)**2)+np.sum(bias*bias))
    grad=softmax(logits,axis=1); grad[np.arange(len(y)),y]-=1.; grad*=sw[:,None]
    gw=np.einsum('nc,nmc->mc',grad,X)+2*reg/classes*(weights-1/models)
    gb=grad.sum(0)+2*reg/classes*bias
    return loss,np.r_[gw.ravel(),gb]

def full_loss(flat,Z,offset,y,reg):
    features=Z.shape[1]; classes=offset.shape[1]
    W=flat[:features*classes].reshape(features,classes); b=flat[features*classes:]
    logits=offset+Z@W+b
    counts=np.bincount(y,minlength=classes); sw=1/np.maximum(counts[y],1); sw/=sw.sum()
    loss=float((sw*(logsumexp(logits,axis=1)-logits[np.arange(len(y)),y])).sum())
    loss+=reg/classes*(np.sum(W*W)+np.sum(b*b))
    grad=softmax(logits,axis=1); grad[np.arange(len(y)),y]-=1.; grad*=sw[:,None]
    gw=Z.T@grad+2*reg/classes*W; gb=grad.sum(0)+2*reg/classes*b
    return loss,np.r_[gw.ravel(),gb]

def train_stack(p,y,reg,full=False):
    X=np.log(np.maximum(p,1e-8)); models,classes=X.shape[1:]
    if full:
        raw=X.reshape(len(X),-1); mean=raw.mean(0); scale=raw.std(0).clip(1e-4)
        Z=np.clip((raw-mean)/scale,-6,6); offset=X.mean(1)
        start=np.zeros(Z.shape[1]*classes+classes)
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=3):
            result=minimize(full_loss,start,args=(Z,offset,y,reg),jac=True,method='L-BFGS-B',options={'maxiter':200,'ftol':1e-10})
        return {'full':True,'coef':result.x,'mean':mean,'scale':scale},{'success':bool(result.success),'iterations':int(result.nit),'loss':float(result.fun)}
    start=np.r_[np.full(models*classes,1/models),np.zeros(classes)]
    result=minimize(stack_loss,start,args=(X,y,reg),jac=True,method='L-BFGS-B',bounds=[(0,2.)]*(models*classes)+[(-2.,2.)]*classes,options={'maxiter':200,'ftol':1e-10})
    return result.x,{'success':bool(result.success),'iterations':int(result.nit),'loss':float(result.fun)}

def predict_stack(p,coef):
    if isinstance(coef,dict):
        X=np.log(np.maximum(p,1e-8)); raw=X.reshape(len(X),-1)
        Z=np.clip((raw-coef['mean'])/coef['scale'],-6,6); classes=p.shape[2]
        flat=coef['coef']; W=flat[:Z.shape[1]*classes].reshape(Z.shape[1],classes); b=flat[Z.shape[1]*classes:]
        return softmax(X.mean(1)+Z@W+b,axis=1)
    models,classes=p.shape[1:]; w=coef[:models*classes].reshape(models,classes); b=coef[models*classes:]
    return softmax(np.einsum('nmc,mc->nc',np.log(np.maximum(p,1e-8)),w)+b,axis=1)

def source_graphs(variant):
    base=strong_distance(); graphs=[]
    graphs.append((graph_from_distance(distance('r2_augmented0.2'),10),.5,.5))
    graphs.append((graph_from_distance(distance('original'),10),.5,.5))
    for name in ['log12_k64','pcen24_k64']:
        d=.6*distance(name)+.4*distance('original'); np.fill_diagonal(d,np.inf)
        graphs.append((graph_from_distance(d,10),1.,0.)); del d
    if variant=='all':
        views=inputs()
        c=dict(kind='graph',view='gabor',weight=.2,k=10)
        graphs.append((graph_for(c,views,None,None,base),1.,0.))
    return graphs

def source_predictions(graphs,y,tr,ui,query):
    result=[]
    for graph,prior,target in graphs:
        p=robust_propagate(graph,y,tr,ui,alpha=.5,prior=prior,target_power=target)
        result.append(p[query])
    return np.stack(result,axis=1)

def run():
    ap=argparse.ArgumentParser(); ap.add_argument('--stage',choices=['search','confirm','final'],default='search'); ap.add_argument('--seed',type=int,default=73); ap.add_argument('--variant',choices=['old','all'],default='old'); ap.add_argument('--form',choices=['diag','full'],default='diag'); args=ap.parse_args()
    train,test,cls,y,ci,ui=labels(); out=R4/f'seed{args.seed}'; out.mkdir(exist_ok=True)
    configs=[dict(name=f'stack_{args.variant}_r{r}',kind='stack',variant=args.variant,reg=r) for r in [1.,5.]]
    if args.form=='full': configs=[dict(name=f'stack_{args.variant}_full_r{r}',kind='stack',variant=args.variant,reg=r,full=True) for r in [5.,20.]]
    if args.stage in ['confirm','final']: configs=[c for c in json.loads((R4/'selection.json').read_text())['configs'] if c['kind']=='stack']
    for variant in sorted(set(c['variant'] for c in configs)):
        active=[c for c in configs if c['variant']==variant and (args.stage=='final' or not (out/f'{c["name"]}.json').exists())]
        if not active: continue
        graphs=source_graphs(variant); outputs={c['name']:np.zeros((len(ci),41)) for c in active}
        outer=[(np.arange(len(ci)),None)] if args.stage=='final' else list(StratifiedKFold(5,shuffle=True,random_state=args.seed).split(ci,y[ci]))
        for fold,(tr,va) in enumerate(outer):
            training=ci[tr]; meta=np.zeros((len(training),len(graphs),41))
            for inner,(itr,iva) in enumerate(StratifiedKFold(2,shuffle=True,random_state=2029).split(training,y[training])):
                meta[iva]=source_predictions(graphs,y,training[itr],ui,training[iva])
            query=np.arange(len(train),len(train)+len(test)) if args.stage=='final' else ci[va]
            prediction=source_predictions(graphs,y,training,ui,query)
            for c in active:
                coef,info=train_stack(meta,y[training],c['reg'],c.get('full',False)); p=predict_stack(prediction,coef)
                if args.stage=='final':
                    extra={'coef':coef['coef'],'mean':coef['mean'],'scale':coef['scale']} if isinstance(coef,dict) else {'coef':coef}
                    np.savez_compressed(R4/f'{c["name"]}_full.npz',pred=p,ids=test.id.values.astype(str),**extra)
                else: outputs[c['name']][va]=p
                print(c['name'],'outer fold',fold+1,info,flush=True)
        if args.stage!='final':
            for c in active:
                p=outputs[c['name']]; r=dict(config=c,seed=args.seed,folds=5,inner_folds=2,**metrics(y[ci],p))
                (out/f'{c["name"]}.json').write_text(json.dumps(r,indent=2)); np.savez_compressed(out/f'{c["name"]}.npz',oof=p)
                print(c['name'],r['micro_map3'],r['macro_map3'],flush=True)

if __name__=='__main__': run()
