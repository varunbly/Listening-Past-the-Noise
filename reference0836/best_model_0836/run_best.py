"""Replay the exact public-best ensemble or refit every stage from audio."""
from course_input import course_data
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile
import numpy as np

BASE=Path(__file__).resolve().parent
def replay():
    manifest=json.loads((BASE/'artifact_manifest.json').read_text())
    for name,digest in manifest.items():
        if hashlib.sha256((BASE/name).read_bytes()).hexdigest()!=digest:
            raise ValueError(f'Artifact changed: {name}. Use the original package to replay.')
    selection=json.loads((BASE/'round4/selection.json').read_text())
    with np.load(BASE/'round4/test_predictions.npz') as data:
        ids=data['ids']; classes=data['classes']; expected=data['probabilities']
    result=np.zeros_like(expected)
    for component in selection['models']:
        if component['name']=='previous': path=BASE/'round3/test_predictions.npz'; key='probabilities'
        else: path=BASE/'round4'/f'{component["name"]}_full.npz'; key='pred'
        with np.load(path) as data:
            assert np.array_equal(ids,data['ids'])
            p=np.maximum(data[key],1e-12)**selection['power']; p/=p.sum(1,keepdims=True)
            result+=component['weight']*p
    assert selection['beta']==0.,'This replay supports the frozen public-best selection'
    result/=result.sum(1,keepdims=True)
    np.testing.assert_allclose(result,expected,rtol=1e-6,atol=1e-8)
    assert result.shape==(1592,41) and np.isfinite(result).all() and len(set(ids))==1592
    text=io.StringIO(newline=''); writer=csv.writer(text,lineterminator='\r\n'); writer.writerow(['id','label'])
    for ident,row in zip(ids,np.argsort(-result,axis=1)[:,:3]):
        codes=classes[row]; assert len(set(codes))==3; writer.writerow([ident,' '.join(codes)])
    payload=text.getvalue().encode('utf-8')
    assert hashlib.sha256(payload).hexdigest()==manifest['submission_round4.csv']
    out=BASE/'output'; out.mkdir(exist_ok=True); (out/'submission.csv').write_bytes(payload)
    with zipfile.ZipFile(out/'Kaggle_Best_Public_0836.zip','w',zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('submission.csv',payload)
    print('Verified all artifacts and reproduced all 1,592 rows:',out/'Kaggle_Best_Public_0836.zip')

def retrain(jobs):
    for name in ['train.csv','test.csv']:
        if not (course_data(BASE)/name).is_file(): raise FileNotFoundError(f'Extract competition data to {course_data(BASE)}')
    commands=[('improve.py','--stage','features','--jobs',str(jobs)),
              ('improve.py','--stage','fisher','--jobs',str(jobs)),('deliver.py','--jobs',str(jobs)),
              ('refine.py','--stage','features','--jobs',str(jobs)),('refine.py','--stage','final','--jobs',str(jobs)),('refine_finish.py','final'),
              ('advance.py','--stage','features','--selected-only','--jobs',str(jobs)),('advance.py','--stage','final'),('advance_finish.py','final'),
              ('round4.py','--stage','features','--jobs',str(jobs)),('round4.py','--stage','final'),('round4_finish.py','final')]
    for command in commands:
        subprocess.run([sys.executable,'-u',str(BASE/command[0]),*command[1:]],cwd=BASE,check=True)
    print('Full pipeline refitted. Output:',BASE/'submission_round4.csv')

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--retrain',action='store_true'); parser.add_argument('--jobs',type=int,default=2); args=parser.parse_args()
    if args.retrain: retrain(args.jobs)
    else: replay()
