Group 13 | Project 4 | Listening Past the Noise

This is the model for the accompanying submission.csv. It uses the course WAVs and checked and unchecked course labels. The four-stage acoustic model and the local graph/SVM baseline supply class scores. Acoustic similarity, time-envelope, segment and tree models provide additional evidence. Two fitted rankers order the three guesses. The first guess can change when the baseline models disagree and independent acoustic cues support the alternative.

To refit the base components and generate predictions, extract the model ZIP to a writable folder. Use Python 3.11 and install the dependencies from requirements.txt. Pass the extracted course dataset directory to predict.py. It should contain train.csv, test.csv, and the audio/audio/train and audio/audio/test directories.

    python -m pip install -r requirements.txt
    python predict.py --data "/path/to/project4" --output submission.csv

The fit uses the supplied recordings together to build its neighbour graphs. Allow time and local disk space for feature extraction and training. Computations can vary slightly with numerical library versions and hardware. The two final rankers use the coefficients in ranker_weights.npz. Source for those rankers is in group13/course_top3.py. The group baseline source is from varunbly/Listening-Past-the-Noise-Group13 at commit 320fc9164d2104f9a57d688df781b50278e688d3. Some reusable waveform feature modules also support other audio corpora; the entry point selects the course dataset only.
