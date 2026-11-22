# Listening Past the Noise - Group 13

Classical audio classification with unchecked training labels. This repository contains the original EDA and the previous frozen ensemble that scored **0.833871 MAP@3** in the matched five-fold evaluation. It does not contain the later round-two ensemble.

## Results

| Evaluation of this frozen ensemble | MAP@3 |
|---|---:|
| Five-fold checked-label CV, seed 131 (620 checked clips) | **0.833871** |
| Earlier development CV (496 checked clips; used for selection) | 0.837702 |
| Earlier reserved holdout (124 checked clips) | 0.806452 |

These use different splits and training sizes. The 0.834 figure is MAP@3, not classification accuracy or a leaderboard score. Later repeated cross-validation is not a new independent holdout. Test performance is unknown until a competition submission is scored. `validation_0834.json` records the five-fold result.

## Setup and data

Python 3.13 and the exact versions in `requirements-tested.txt` were used for the measured run.

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements-tested.txt
```

Download the competition data separately and extract `project4.zip` into `project4/`:

```text
project4/
  train.csv
  test.csv
  sample_submission.csv
  audio/audio/train/<id>.wav
  audio/audio/test/<id>.wav
```

The code also supports `project4/audio/train` and `project4/audio/test`. The supplied dataset has 5,604 training clips, 1,592 test clips, 41 categories, and 620 checked labels. Actual checked counts range from 10 to 40 per class.

**Do not commit the dataset, WAV files, environments, or feature caches.** They are excluded by `.gitignore`.

## Reproduce the 0.834 model

```bash
python improve.py --stage features --jobs 2
python improve.py --stage fisher --jobs 2
python evaluate_frozen.py --seed 131 --jobs 2
python deliver.py --jobs 2
```

The first two stages cache features and fit GMM representations. They take several minutes and resume from caches. The evaluator recreates the five-fold score. `deliver.py` loads the checked-in `selection.json`, trains on all available training labels with their configured weights, and creates **`submission_improved.csv`**. It verifies all 1,592 test IDs and three distinct valid category codes per recording. It also writes a run manifest. The frozen selection is used directly; there is no need to rerun the full model search.

Alternatively, open **`Listening_Past_the_Noise_0834.ipynb`** from the repository folder. The original **`eda_listening_past_the_noise.ipynb`** is preserved for reference.

## Frozen pipeline

- 64% graph label propagation on combined clip statistics and 64-component Fisher vectors, with class-prior calibration.
- 16% weighted RBF SVM on 128 PCA-whitened clip features.
- 20% checked-only RBF SVM on 32-component Fisher vectors.

Clip features include two time resolutions of log-mel, MFCC, delta, chroma, and spectral statistics. GMMs and PCA/scaling fit unchecked training features only. Checked validation labels are excluded from each supervised training fold and from graph seeds. Graph construction and RBF bandwidth use unlabelled validation/test features, so the protocol is transductive; their labels are never used.

Only supplied competition recordings/labels and classical models are used. There are no neural networks, pretrained audio embeddings, external audio, or manually assigned sound labels.

## Checks

```bash
python -m unittest -v test_pipeline.py
```

Tests cover MAP@3, silent/short audio, and Fisher-vector normalization. `graph_search.py` and the search options in `improve.py` are research utilities; running them is optional and does not replace the published frozen selection.
