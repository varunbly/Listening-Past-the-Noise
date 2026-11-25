"""Classical audio classification with honest checked-label validation.

Only competition audio/labels are used. No neural or pretrained features.
Run with the supplied requirements, from any working directory.
"""
from __future__ import annotations
from course_input import course_data
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ.setdefault(_k, '1')
import argparse
import hashlib
import json
from pathlib import Path
import time
import warnings
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.special import softmax
import soundfile as sf
import librosa
from joblib import Parallel, delayed, dump
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import rbf_kernel

BASE = Path(__file__).resolve().parent
VERSION = 'audio-v1'
warnings.filterwarnings('ignore', category=UserWarning)

def score(y, p):
    top = np.argsort(-p, axis=1)[:, :3]
    return float(np.mean(np.sum((top == np.asarray(y)[:, None]) / np.arange(1, 4), axis=1)))

def summarize(a):
    a = np.asarray(a)
    return np.concatenate([a.mean(1), a.std(1), np.quantile(a, [.1, .5, .9], axis=1).ravel(),
                           stats.skew(a, axis=1, bias=False), stats.kurtosis(a, axis=1, bias=False)])

def extract_one(path, cache):
    cache = Path(cache)
    if cache.exists():
        return
    y, sr = sf.read(path, dtype='float32')
    if y.ndim == 2:
        y = y.mean(1)
    duration = len(y) / sr
    if sr != 22050:
        y = librosa.resample(y, orig_sr=sr, target_sr=22050)
        sr = 22050
    y = np.pad(y, (0, max(0, 4096 - len(y))))
    peak = max(float(np.max(np.abs(y))), 1e-8)
    raw_rms = float(np.sqrt(np.mean(y ** 2)))
    y = y / peak
    parts = [np.array([np.log1p(duration), np.log(peak), np.log(raw_rms + 1e-8)])]
    frame = None
    for n_fft, hop, n_mels in [(2048, 512, 96), (512, 256, 64)]:
        spec = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
        power = spec ** 2
        melpower = librosa.feature.melspectrogram(S=power, sr=sr, n_mels=n_mels, fmin=20)
        mel = librosa.power_to_db(melpower, ref=np.max, top_db=80)
        mfcc = librosa.feature.mfcc(S=mel, n_mfcc=40)
        delta = librosa.feature.delta(mfcc, width=min(9, (mfcc.shape[1]-1)//2*2+1))
        delta2 = librosa.feature.delta(mfcc, order=2, width=min(9, (mfcc.shape[1]-1)//2*2+1))
        rms = np.sqrt(np.mean(power, axis=0))
        active = rms > rms.max() * .05
        if active.sum() < 3:
            active[:] = True
        parts.extend([summarize(mel), summarize(mfcc), summarize(delta), summarize(delta2),
                      mel[:, active].mean(1), mel[:, active].std(1)])
        if n_fft == 2048:
            chroma = librosa.feature.chroma_stft(S=power, sr=sr, n_fft=n_fft, hop_length=hop)
            contrast = librosa.feature.spectral_contrast(S=spec, sr=sr)
            spectral = np.vstack([
                librosa.feature.spectral_centroid(S=spec, sr=sr),
                librosa.feature.spectral_bandwidth(S=spec, sr=sr),
                librosa.feature.spectral_rolloff(S=spec, sr=sr),
                librosa.feature.spectral_flatness(S=spec),
                librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop),
                librosa.feature.rms(S=spec, frame_length=n_fft, hop_length=hop)])
            parts.extend([summarize(chroma), summarize(contrast), summarize(spectral)])
            # Frame descriptor for an unsupervised Fisher representation.
            frame = np.vstack([mfcc[:30], delta[:15], delta2[:15]])[:, active].T
            if len(frame) > 240:
                frame = frame[np.linspace(0, len(frame)-1, 240).astype(int)]
    x = np.nan_to_num(np.concatenate(parts), nan=0., posinf=0., neginf=0.).astype('float32')
    temporary=cache.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, x=x, frames=frame.astype('float32'))
    temporary.replace(cache)

def prepare(root, jobs):
    # The cached matrices use the exact row order of this Project 4 release.
    manifest_hashes = {
        'train.csv': '2c2d73e7d58193d788240a03b7df87d13d14b351ab862236515b81a1dcc0477e',
        'test.csv': 'f19d969eedd6efe107a21688c31427ce41e2464d6a0e5e4358c74067bd1880ad',
    }
    for name, expected in manifest_hashes.items():
        manifest = root / name
        if not manifest.is_file() or hashlib.sha256(manifest.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Use the original {name} extracted from the course project4.zip.')
    train, test = pd.read_csv(root/'train.csv'), pd.read_csv(root/'test.csv')
    audio = root/'audio'/'audio'
    if not (audio/'train').exists():
        audio = root/'audio'
    cache = BASE/'cache'/VERSION
    cache.mkdir(parents=True, exist_ok=True)
    paths, caches = [], []
    for split, df in [('train', train), ('test', test)]:
        for ident in df.id:
            p = audio/split/f'{ident}.wav'
            if not p.exists():
                raise FileNotFoundError(p)
            paths.append(p)
            caches.append(cache/f'{split}_{ident}.npz')
    missing = [(p, c) for p, c in zip(paths, caches) if not c.exists()]
    print(f'Feature extraction: {len(missing)} remaining / {len(paths)} clips', flush=True)
    start = time.time()
    # Batches provide visible progress and allow interrupted runs to resume.
    for j in range(0, len(missing), 128):
        Parallel(n_jobs=jobs)(delayed(extract_one)(p, c) for p, c in missing[j:j+128])
        print(f'Features {min(j+128,len(missing))}/{len(missing)} elapsed={time.time()-start:.0f}s', flush=True)
    matrix=cache/'matrix.npy'
    if matrix.exists() and not missing:
        x=np.load(matrix,mmap_mode='r')
    else:
        x = np.stack([np.load(c)['x'] for c in caches])
        np.save(matrix,x)
    print('Feature matrix', x.shape, flush=True)
    return train, test, x, caches

def fisher_features(caches, checked, n_train, k, jobs, cache_dir=None):
    """Fit UBM only on unchecked training clips; no validation examples used."""
    cache_dir = BASE/'cache' if cache_dir is None else cache_dir
    out = cache_dir/f'fisher_{VERSION}_{k}.npz'
    if out.exists():
        return np.load(out)['x']
    rng = np.random.default_rng(42)
    pool = []
    for i in np.where(~checked)[0]:
        a = np.load(caches[i])['frames']
        pool.append(a[rng.choice(len(a), min(35, len(a)), replace=False)])
    pool = np.concatenate(pool)
    if len(pool) > 65000:
        pool = pool[rng.choice(len(pool), 65000, replace=False)]
    scaler = StandardScaler().fit(pool)
    pool = scaler.transform(pool)
    gmm = GaussianMixture(k, covariance_type='diag', max_iter=80, reg_covar=1e-3, random_state=42)
    print(f'Fitting unchecked-only {k}-component GMM on {pool.shape}', flush=True)
    gmm.fit(pool)
    print('GMM converged:', gmm.converged_, 'iterations:', gmm.n_iter_, flush=True)
    x = np.stack(Parallel(n_jobs=jobs)(delayed(fisher_one)(c, scaler, gmm) for c in caches))
    np.savez_compressed(out, x=x)
    dump((scaler, gmm), out.with_suffix('.joblib'))
    return x

def fisher_one(cache, scaler, gmm):
    a = scaler.transform(np.load(cache)['frames'])
    q = gmm.predict_proba(a)
    n = len(a)
    q0 = q.sum(0)[:, None] / n
    q1 = q.T @ a / n
    q2 = q.T @ (a*a) / n
    s = np.sqrt(gmm.covariances_)
    u = (q1 - q0*gmm.means_) / (np.sqrt(gmm.weights_[:,None])*s)
    v = (q2-2*q1*gmm.means_+q0*(gmm.means_**2-gmm.covariances_)) / (np.sqrt(2*gmm.weights_[:,None])*gmm.covariances_)
    f = np.concatenate([u.ravel(), v.ravel()])
    f = np.sign(f)*np.sqrt(np.abs(f))
    return (f / max(np.linalg.norm(f), 1e-12)).astype('float32')

def svm_prob(model, x):
    # Decision values are deterministic; internal probability CV is unnecessary.
    return softmax(model.decision_function(x), axis=1)

def stats_view(x,name,ui=None):
    if name=='stats': return x
    if name.startswith('pca'):
        if ui is None: raise ValueError('Unchecked-only fit indices required')
        xx=np.clip(StandardScaler().fit(x[ui]).transform(x),-6,6)
        pca=PCA(int(name[3:]),whiten=True,svd_solver='randomized',random_state=42).fit(xx[ui])
        return np.clip(pca.transform(xx),-8,8)
    if name=='stats_basic':
        keep=list(range(3))
        for start,d in [(3,96),(675,40),(955,40),(1235,40),(1707,12),(1791,7),(1840,6),(1882,64),(2330,40),(2610,40),(2890,40)]:
            keep.extend(range(start,start+5*d))
        keep.extend(range(1515,1707)); keep.extend(range(3170,3298))
        return x[:,keep]
    if name=='stats_compact':
        keep=list(range(3))
        for start,d in [(3,96),(1882,64),(1707,12),(1791,7),(1840,6)]:
            keep.extend(range(start,start+2*d))
        for start in [675,2330]:
            keep.extend(range(start,start+20)); keep.extend(range(start+40,start+60))
        for start in [955,1235,2610,2890]:
            keep.extend(range(start,start+10)); keep.extend(range(start+40,start+50))
        return x[:,keep]
    raise ValueError(name)

def class_weights(config,y,fit_idx,tr_c,weights):
    cw=config.get('class_weight')
    if cw=='clean_prior':
        observed=np.bincount(y[fit_idx],weights=weights)
        desired=np.bincount(y[tr_c],minlength=len(observed)).astype(float)
        ratio=(desired/desired.sum())/(observed/observed.sum()).clip(1e-8)
        return dict(enumerate(ratio))
    return cw

def cleaner_reliability(rank, config):
    values=np.asarray(config.get('cleaner_rank_weights',[1.,.5,.25,.05]),dtype=float)
    if values.ndim!=1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('cleaner_rank_weights must be a non-empty finite list')
    return values[np.clip(np.asarray(rank,dtype=int),0,len(values)-1)]

def evaluate(train, test, features, config, folds=5, seed=42, final=False):
    classes = np.array(sorted(train.label.unique()))
    y = np.searchsorted(classes, train.label)
    chk = train.checked.values == 1
    ci, ui = np.where(chk)[0], np.where(~chk)[0]
    oof = np.zeros((len(ci),len(classes)))
    test_probs = []
    vals = []
    for fold, (tr, va) in enumerate(StratifiedKFold(folds, shuffle=True, random_state=seed).split(ci, y[ci])):
        tr_c, va_c = ci[tr], ci[va]
        x = features
        # Every supervised fit and scaler excludes held-out checked rows.
        fit_idx = np.r_[tr_c, ui] if config['unchecked_weight'] > 0 else tr_c
        weights = np.where(chk[fit_idx], 1., config['unchecked_weight'])
        if config['kind'] == 'svm':
            if config.get('cleaner'):
                # This teacher sees only checked TRAINING examples in this fold.
                # It never sees held-out checked labels (the original leak).
                teacher=SVC(C=20,kernel='precomputed',decision_function_shape='ovr')
                teacher.fit(x[np.ix_(tr_c,tr_c)],y[tr_c])
                decisions=teacher.decision_function(x[np.ix_(ui,tr_c)])
                order=np.argsort(-decisions,axis=1)
                rank=np.argmax(order==y[ui,None],axis=1)
                reliability=cleaner_reliability(rank,config)
                weights[len(tr_c):]*=reliability
            model = SVC(C=config.get('C',10), kernel='precomputed', class_weight=class_weights(config,y,fit_idx,tr_c,weights), decision_function_shape='ovr', random_state=seed)
            model.fit(x[np.ix_(fit_idx,fit_idx)], y[fit_idx], sample_weight=weights)
            p = svm_prob(model, x[np.ix_(va_c,fit_idx)])
            xx = x
        elif config['kind'] == 'trees':
            model = ExtraTreesClassifier(n_estimators=config.get('n_estimators',350), max_features=config.get('max_features','sqrt'), min_samples_leaf=config.get('leaf',2), n_jobs=4, class_weight=config.get('class_weight','balanced'), random_state=seed+fold)
            xx = x
            model.fit(xx[fit_idx], y[fit_idx], sample_weight=weights)
            p = model.predict_proba(xx[va_c])
        elif config['kind'] == 'lgb':
            from lightgbm import LGBMClassifier
            model = LGBMClassifier(n_estimators=config.get('n_estimators',250), num_leaves=config.get('leaves',15), learning_rate=.06, colsample_bytree=.6, subsample=.8, reg_lambda=5., max_bin=config.get('max_bin',63), min_child_samples=10, n_jobs=4, verbosity=-1, random_state=seed+fold)
            xx = x
            model.fit(xx[fit_idx], y[fit_idx], sample_weight=weights)
            p = model.predict_proba(xx[va_c])
        else:
            raise ValueError(config)
        oof[va] = p
        vals.append(score(y[va_c],p))
        if final:
            test_probs.append(svm_prob(model, x[np.ix_(np.arange(len(train),len(x)),fit_idx)]) if config['kind']=='svm' else model.predict_proba(xx[len(train):]))
        print(f'  fold {fold+1}: {vals[-1]:.5f}', flush=True)
    result = dict(config=config, seed=seed, folds=folds, map3=score(y[ci],oof), fold_scores=vals)
    return result, oof, np.mean(test_probs,axis=0) if test_probs else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, default=course_data(BASE))
    ap.add_argument('--jobs', type=int, default=2)
    ap.add_argument('--stage', choices=['features','search','fisher','final'], default='search')
    ap.add_argument('--folds', type=int, default=4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--scope', choices=['dev','full'], default='dev')
    ap.add_argument('--models', default='', help='Comma-separated exact model names; empty runs all')
    ap.add_argument('--svm-only', action='store_true')
    args = ap.parse_args()
    train, test, x, caches = prepare(args.data, args.jobs)
    if args.stage=='features': return
    chk = train.checked.values==1
    if args.stage=='fisher':
        for k in [4,8,16,32,64]:
            fisher_features(caches, chk, len(train), k, args.jobs)
        return
    order = np.arange(len(x))
    if args.scope=='dev':
        ci = np.where(chk)[0]
        _, hold = train_test_split(ci, test_size=.2, stratify=train.label.iloc[ci], random_state=2026)
        keep = np.setdiff1d(np.arange(len(train)), hold)
        order = np.r_[keep, hold, np.arange(len(train),len(x))]
        (BASE/'holdout_ids.json').write_text(json.dumps(train.id.iloc[hold].tolist(),indent=2))
        test = pd.concat([train.iloc[hold][['id']],test],ignore_index=True)
        train = train.iloc[keep].reset_index(drop=True)
        x = x[order]
        chk = train.checked.values==1
        print('Development scope: 124 checked clips held out from all model selection',flush=True)
    configs = []
    for w in [0.,.15,.4,1.]:
        for C in [3.,20.]:
            configs.append(dict(name=f'svm_C{C}_w{w}',kind='svm',C=C,unchecked_weight=w,view='stats'))
    for w in [.15, .5, 1.]:
        configs.append(dict(name=f'trees_w{w}',kind='trees',unchecked_weight=w,view='stats',n_estimators=350,leaf=2,max_features='sqrt'))
    for w in [.25,1.]:
        configs.append(dict(name=f'lgb_w{w}',kind='lgb',unchecked_weight=w,view='stats',n_estimators=250))
    configs.append(dict(name='lgb_compact',kind='lgb',unchecked_weight=.25,view='stats_compact',n_estimators=150,max_bin=31))
    configs.append(dict(name='lgb_pca128',kind='lgb',unchecked_weight=.25,view='pca128',n_estimators=200,max_bin=31))
    for w in [.5,1.]:
        configs.append(dict(name=f'filter_stats_w{w}',kind='svm',C=20,unchecked_weight=w,view='stats',cleaner=True))
    for view in ['stats','stats_basic']:
        for factor in [.25,1.,4.]:
            configs.append(dict(name=f'{view}_g{factor}',kind='svm',C=20,unchecked_weight=.4,view=view,gamma_factor=factor))
        configs.append(dict(name=f'{view}_l2',kind='svm',C=20,unchecked_weight=.4,view=view,l2=True))
        for C in [20.,100.]:
            for factor in [.05,.1,.25]:
                configs.append(dict(name=f'{view}_C{C}_g{factor}',kind='svm',C=C,unchecked_weight=.4,view=view,gamma_factor=factor))
        configs.append(dict(name=f'{view}_balanced',kind='svm',C=20,unchecked_weight=.4,view=view,gamma_factor=.25,class_weight='balanced'))
        for weight in ['balanced','clean_prior']:
            configs.append(dict(name=f'{view}_C100_broad_{weight}',kind='svm',C=100,unchecked_weight=.4,view=view,gamma_factor=.05,class_weight=weight))
        configs.append(dict(name=f'{view}_prior',kind='svm',C=20,unchecked_weight=.4,view=view,gamma_factor=.25,class_weight='clean_prior'))
    for k in [4,8,16,32,64]:
        f = BASE/'cache'/f'fisher_{VERSION}_{k}.npz'
        if f.exists():
            for w in [0.,.15,.5,1.]:
                for C in [3.,20.]:
                    configs.append(dict(name=f'fisher{k}_C{C}_w{w}',kind='svm',C=C,unchecked_weight=w,view=f'fisher{k}',scale=False))
            configs.append(dict(name=f'filter_fisher{k}_w1',kind='svm',C=20,unchecked_weight=1.,view=f'fisher{k}',scale=False,cleaner=True))
            for weight in ['balanced','clean_prior']:
                for factor in [.1,1.]:
                    configs.append(dict(name=f'fisher{k}_g{factor}_{weight}',kind='svm',C=20,unchecked_weight=.5,view=f'fisher{k}',scale=False,class_weight=weight,gamma_factor=factor))
                configs.append(dict(name=f'fisher{k}_broad_{weight}',kind='svm',C=100,unchecked_weight=.5,view=f'fisher{k}',scale=False,class_weight=weight,gamma_factor=.01))
        if k==8:
            configs.append(dict(name='fisher8_g0.1_clean_prior_cleaner',kind='svm',C=20,unchecked_weight=.5,view='fisher8',scale=False,class_weight='clean_prior',gamma_factor=.1,cleaner=True))
            for name,rank_weights in [('filter_strict',[1.,.15,.03,.01]),('filter_mild',[1.,.75,.4,.1])]:
                configs.append(dict(name=f'fisher8_g0.1_clean_prior_{name}',kind='svm',C=20,unchecked_weight=.5,view='fisher8',scale=False,class_weight='clean_prior',gamma_factor=.1,cleaner=True,cleaner_rank_weights=rank_weights))
            for C in [3.,20.,100.]:
                for factor in [.05,.1,.2]:
                    for weight in ['balanced','clean_prior']:
                        configs.append(dict(name=f'fisher{k}_C{C}_g{factor}_{weight}',kind='svm',C=C,unchecked_weight=.5,view=f'fisher{k}',scale=False,class_weight=weight,gamma_factor=factor))
    for f in sorted((BASE/'cache').glob(f'fisher_{VERSION}_*t*.npz')):
        suffix=f.stem.removeprefix(f'fisher_{VERSION}_')
        view=f'fisher{suffix}'
        for w in [0.,.15,.5,1.]:
            for C in [3.,20.]:
                configs.append(dict(name=f'{view}_C{C}_w{w}',kind='svm',C=C,unchecked_weight=w,view=view,scale=False))
    for dim in [64,128,256]:
        for factor in [.1,1.]:
            for cw in [None,'balanced']:
                configs.append(dict(name=f'pca{dim}_g{factor}_cw{cw}',kind='svm',C=20,unchecked_weight=.4,view=f'pca{dim}',scale=False,class_weight=cw,gamma_factor=factor))
    if args.models:
        names=set(args.models.split(','))
        configs=[c for c in configs if c['name'] in names]
        if not configs: raise ValueError('No matching model names')
    if args.svm_only:
        configs=[c for c in configs if c['kind']=='svm']
    configs.sort(key=lambda c: 0 if c['view'].startswith('pca') else 1)
    output = BASE/'results'/args.scope
    output.mkdir(parents=True,exist_ok=True)
    kernels = {}
    projections = {}
    for conf in configs:
        tag = f'{conf["name"]}_s{args.seed}_{args.folds}fold'
        target = output/f'{tag}.json'
        if target.exists() and (args.stage!='final' or 'test' in np.load(output/f'{tag}.npz')):
            continue
        if conf['view'].startswith(('stats','pca')):
            if conf['view'] not in projections:
                projections[conf['view']]=stats_view(x,conf['view'],np.where(~chk)[0])
            view=projections[conf['view']]
        else:
            if conf['view'] not in projections:
                projections[conf['view']]=np.load(BASE/'cache'/f'fisher_{VERSION}_{conf["view"][6:]}.npz')['x'][order]
            view=projections[conf['view']]
        if conf['kind']=='svm':
            key = f'{conf["view"]}_g{conf.get("gamma_factor",1.)}_l2{conf.get("l2",False)}'
            # Feature scaling uses unchecked clips only, which belong to every
            # training fold. Held-out checked clips cannot affect it.
            if key not in kernels:
                xx = np.clip(StandardScaler().fit(view[:len(train)][~chk]).transform(view),-6,6) if conf.get('scale',True) else view
                if conf.get('l2'): xx=normalize(xx)
                kernel = rbf_kernel(xx, gamma=conf.get('gamma_factor',1.)/(xx.shape[1]*xx.var()))
                kernel_path = BASE/'cache'/f'kernel_{key}.npy'
                np.save(kernel_path,kernel)
                kernels[key]=kernel_path
                del kernel
            view = np.load(kernels[key],mmap_mode='r')
        print('\nEvaluating',tag,flush=True)
        result,oof,pt = evaluate(train,test,view,conf,args.folds,args.seed,args.stage=='final')
        target.write_text(json.dumps(result,indent=2))
        np.savez_compressed(output/f'{tag}.npz',oof=oof,**({'test':pt} if pt is not None else {}))
        print('MAP@3',result['map3'],flush=True)

if __name__=='__main__':
    main()
