import unittest
import tempfile
from pathlib import Path
import numpy as np
import soundfile as sf
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from advance import spectrogram,patches,encode_batch,metrics,noise_em
from scipy.sparse import csr_matrix

class PatchTests(unittest.TestCase):
    def test_em_ignores_validation_labels(self):
        g=csr_matrix(np.array([[0,.5,.5,0],[.5,0,0,.5],[.5,0,0,.5],[0,.5,.5,0]],dtype='float32'))
        p=noise_em(g,np.array([0,1,0,1]),np.array([0]),np.array([1]))
        q=noise_em(g,np.array([0,1,1,0]),np.array([0]),np.array([1]))
        np.testing.assert_allclose(p,q)
        self.assertTrue(np.isfinite(p).all())
        np.testing.assert_allclose(p.sum(1),1.,atol=1e-6)

    def test_macro_metric_exposes_class_imbalance(self):
        y=np.array([0,0,0,1,2,3])
        p=np.eye(4)[[0,0,0,1,2,0]]+.001
        p[-1]=[.7,.2,.1,0.]
        m=metrics(y,p)
        self.assertAlmostEqual(m['micro_map3'],5/6)
        self.assertAlmostEqual(m['macro_map3'],.75)

    def test_silent_short_patch_and_encoding(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory); wav=folder/'silent.wav'; out=folder/'spec.npz'
            sf.write(wav,np.zeros(100),22050)
            spectrogram(wav,out)
            for kind in ['log','pcen']:
                for width in [12,24]:
                    a=patches(out,kind,width)
                    self.assertGreater(len(a),0)
                    self.assertEqual(a.shape[1],48*width)
                    self.assertTrue(np.isfinite(a).all())
            rng=np.random.default_rng(4)
            training=rng.normal(size=(150,48*12)).astype('float32')
            scaler=StandardScaler().fit(training)
            pca=PCA(64,random_state=4).fit(scaler.transform(training))
            gmm=GaussianMixture(4,covariance_type='diag',random_state=4).fit(pca.transform(scaler.transform(training)))
            f=encode_batch([out],'log',12,(scaler,pca,gmm))
            self.assertEqual(f.shape,(1,512))
            self.assertTrue(np.isfinite(f).all())
            self.assertAlmostEqual(float(np.linalg.norm(f)),1.,places=5)

if __name__=='__main__': unittest.main()
