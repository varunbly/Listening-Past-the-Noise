# Highest public-scoring model: 0.836 MAP@3

This is the complete selected round-four pipeline, whose submission scored **0.836 publicly**, as reported by the team. Round three scored 0.83227 and round eight scored 0.83434. No public result of 0.9 was achieved. The older root-level 0.834 description refers to local CV, not the highest public score.

## Contents

- Feature extraction, training, inference, validation and tests for the original ensemble and rounds two through four.
- Frozen configurations and component probability checkpoints for every selected ensemble stage.
- Three trained MFCC Gaussian-mixture dictionaries and two trained spectrogram-patch dictionaries.
- Exact 1,592-row submission, final probabilities, validation results and public score history.
- SHA-256 manifest, pinned dependencies and one-command replay/retraining entry point.

This is a transductive classical ensemble: graph predictions depend on the supplied training and test audio together. The probability checkpoints reproduce the competition test set; they are not a general-purpose classifier for arbitrary new recordings. The fitted SVMs and graphs are reconstructed by the training code. Raw competition audio, environment files and bulky feature/distance caches are not committed. Obtain the competition data separately for full retraining.

## Exact submission replay (no audio needed)

Use Python 3.13. From this directory:

```bash
python -m pip install -r requirements-tested.txt
python run_best.py
```

The command verifies the artifact manifest, recombines the frozen component predictions, checks the result against the saved final probabilities, and verifies the CSV hash. Upload `output/Kaggle_Best_Public_0836.zip`. This is the same submission that scored 0.836, not a new candidate.

## Full retraining from competition audio

Extract the original dataset into `best_model_0836/project4/`, with `train.csv`, `test.csv` and `audio/audio/train/*.wav`, `audio/audio/test/*.wav`. Then run:

```bash
python run_best.py --retrain --jobs 2
```

This executes all selected stages, including the original SVM ensemble, round-two features/graphs, round-three patch graphs and round-four modulation/context graphs. It does not search for new settings. Existing feature caches are reused. Use a fresh package to repeat an uncached run. Training updates checkpoints, so manifest-based exact replay should use an unmodified package. Numerical differences across dependency versions/platforms are possible; the included checkpoint replay reproduces the measured submission exactly.

## Model and measured validation

Round four blends 72.25% of round three, 15% reciprocal-neighbour diffusion and 12.75% Gabor/acoustic diffusion, after square-root probability tempering. All models use only competition data; no neural network, external recording or pretrained audio embedding is used.

On matched seed-419 five-fold checked-label validation, overall MAP@3 increased from 0.842742 to 0.862097; class-balanced MAP@3 increased from 0.853049 to 0.871037. These are repeated CV measurements on training labels and are not independent test or public leaderboard scores. Public MAP@3 remains 0.836.

```bash
python -m unittest -v test_pipeline.py test_refine.py test_advance.py test_round4.py
```

Submission SHA-256: `b7c1d9df2ab959fc63dc08f525f075c92034025bc175aa4ce54118229a197157`.
