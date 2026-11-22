"""Build the notebook entry point for the frozen 0.834 model."""
from pathlib import Path
import ast
import nbformat as nbf

root=Path(__file__).resolve().parent
nb=nbf.v4.new_notebook()
nb.metadata.kernelspec={'name':'python3','display_name':'Python 3','language':'python'}
nb.cells=[
    nbf.v4.new_markdown_cell('# Listening Past the Noise: frozen 0.834 ensemble\n\nThis notebook runs the previous frozen model. MAP@3 is 0.833871 on five-fold checked-label CV with seed 131; this is not accuracy or a competition test score. Earlier reserved holdout MAP@3 was 0.806452. See README.md for the distinction between evaluations.\n\nInstall requirements-tested.txt in this notebook environment and extract the competition data into project4/. Keep this notebook next to the Python scripts.'),
    nbf.v4.new_code_cell('from pathlib import Path\nimport sys,subprocess,json\nimport pandas as pd\nBASE=Path.cwd()\nassert (BASE/"improve.py").exists(), "Open this notebook from the repository folder"\nJOBS=2\ndef run(script,*args):\n    subprocess.run([sys.executable,"-u",str(BASE/script),*map(str,args)],cwd=BASE,check=True)'),
    nbf.v4.new_code_cell('train=pd.read_csv(BASE/"project4/train.csv")\ntest=pd.read_csv(BASE/"project4/test.csv")\nprint("Training:",len(train),"Test:",len(test),"Checked:",train.checked.sum())\ndisplay(train.groupby("label").agg(total=("id","size"),checked=("checked","sum")))'),
    nbf.v4.new_markdown_cell('## Cache deterministic audio features and GMM Fisher vectors\n\nThese stages resume from cached results and can take several minutes. No external or pretrained model is used.'),
    nbf.v4.new_code_cell('run("improve.py","--stage","features","--jobs",JOBS)\nrun("improve.py","--stage","fisher","--jobs",JOBS)'),
    nbf.v4.new_markdown_cell('## Reproduce the five-fold score\n\nEach fold excludes its checked validation labels from model fitting and propagation seeds. Feature-only transductive steps are described in README.md.'),
    nbf.v4.new_code_cell('run("evaluate_frozen.py","--seed",131,"--jobs",JOBS)\nprint(json.loads((BASE/"validation_0834.json").read_text()))'),
    nbf.v4.new_markdown_cell('## Train using all training data and validate the submission\n\nThe checked-in selection.json specifies the frozen three-component ensemble. This does not run or select the later round-two model.'),
    nbf.v4.new_code_cell('run("deliver.py","--jobs",JOBS)\nsubmission=pd.read_csv(BASE/"submission_improved.csv",dtype=str)\nassert submission.id.tolist()==test.id.tolist()\nassert submission.id.is_unique\nassert all(len(v.split())==len(set(v.split()))==3 and set(v.split())<=set(train.label) for v in submission.label)\ndisplay(submission.head())\nprint("Validated rows:",len(submission))'),
    nbf.v4.new_code_cell('run("test_pipeline.py")')]
nbf.validate(nb)
for c in nb.cells:
    if c.cell_type=='code': ast.parse(c.source)
nbf.write(nb,root/'Listening_Past_the_Noise_0834.ipynb')
print('Notebook generated and validated')
