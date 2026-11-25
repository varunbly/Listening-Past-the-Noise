import unittest
import tempfile
from pathlib import Path
import numpy as np
import soundfile as sf
from improve import extract_one, score, fisher_one
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

class PipelineTests(unittest.TestCase):
    def test_rank_metric(self):
        p=np.array([[.7,.2,.1,0],[.5,.3,.2,0],[.5,.3,.2,0],[.5,.3,.2,0]])
        self.assertAlmostEqual(score(np.arange(4),p),(1+.5+1/3)/4)

    def test_silent_and_short_audio(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for name,y in [('silent',np.zeros(100)),('tone',np.sin(np.arange(1000)*.17))]:
                wav=root/f'{name}.wav'
                out=root/f'{name}.npz'
                sf.write(wav,y,22050)
                extract_one(wav,out)
                with np.load(out) as a:
                    self.assertTrue(np.isfinite(a['x']).all())
                    self.assertTrue(np.isfinite(a['frames']).all())
                    self.assertEqual(a['frames'].shape[1],60)
                    self.assertGreater(a['x'].size,2000)

    def test_fisher_is_finite_and_normalized(self):
        rng=np.random.default_rng(12)
        x=rng.normal(size=(100,60)).astype('float32')
        scaler=StandardScaler().fit(x)
        gmm=GaussianMixture(3,covariance_type='diag',random_state=42).fit(scaler.transform(x))
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'frames.npz'
            np.savez(p,frames=x)
            f=fisher_one(p,scaler,gmm)
            self.assertEqual(f.shape,(360,))
            self.assertTrue(np.isfinite(f).all())
            self.assertAlmostEqual(np.linalg.norm(f),1.,places=5)

if __name__=='__main__': unittest.main()
