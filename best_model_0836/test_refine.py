import tempfile
import unittest
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.sparse import csr_matrix
from refine import temporal_features,robust_propagate

class RefinementTests(unittest.TestCase):
    def test_unknown_labels_cannot_change_predictions(self):
        g=csr_matrix(np.array([[0,.5,.5,0],[.5,0,0,.5],[.5,0,0,.5],[0,.5,.5,0]],dtype='float32'))
        a=robust_propagate(g,np.array([0,1,0,1]),np.array([0]),np.array([1]),adaptive=1.)
        b=robust_propagate(g,np.array([0,1,1,0]),np.array([0]),np.array([1]),adaptive=1.)
        np.testing.assert_allclose(a,b)
        self.assertTrue(np.isfinite(a).all())
        np.testing.assert_allclose(a.sum(1),1.,atol=1e-6)

    def test_temporal_silent_short_recording(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            wav=folder/'silent.wav'; target=folder/'features.npz'
            sf.write(wav,np.zeros(100),22050)
            temporal_features(wav,target)
            with np.load(target) as data:
                self.assertEqual(set(data.files),{'temporal','pcen','texture','modulation'})
                for name in data.files:
                    self.assertTrue(np.isfinite(data[name]).all())
                    self.assertGreater(len(data[name]),100)

if __name__=='__main__': unittest.main()
