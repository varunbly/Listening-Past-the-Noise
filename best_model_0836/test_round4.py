import unittest
import tempfile
from pathlib import Path
import numpy as np
from scipy.optimize import check_grad
from round4 import nca_loss,extra_one,context_graph,semi_training
from round4_stack import stack_loss,full_loss
from scipy.sparse import csr_matrix

class RoundFourTests(unittest.TestCase):
    def test_semisupervised_teacher_excludes_validation_labels(self):
        dense=np.zeros((123,123),dtype='float32')
        for c in range(41):
            group=[c,c+41,c+82]
            for i in group:
                for j in group:
                    if i!=j: dense[i,j]=.5
        graph=csr_matrix(dense); y=np.tile(np.arange(41),3)
        tr=np.arange(41); ui=np.arange(41,82)
        a,w=semi_training(graph,y,tr,ui)
        y2=y.copy(); y2[82:]=y2[82:][::-1]
        b,v=semi_training(graph,y2,tr,ui)
        np.testing.assert_array_equal(a,b); np.testing.assert_allclose(w,v)
        self.assertTrue((a<82).all())

    def test_stacking_gradient(self):
        rng=np.random.default_rng(11)
        p=rng.uniform(.1,1.,size=(12,3,4)); p/=p.sum(2,keepdims=True)
        X=np.log(p); y=np.tile(np.arange(4),3)
        coef=np.r_[np.full(12,1/3)+rng.normal(0,.03,12),rng.normal(0,.03,4)]
        f=lambda a:stack_loss(a,X,y,1.)[0]
        g=lambda a:stack_loss(a,X,y,1.)[1]
        self.assertLess(check_grad(f,g,coef),1e-6)
        Z=X.reshape(12,-1); offset=X.mean(1); coef=rng.normal(0,.01,Z.shape[1]*4+4)
        f=lambda a:full_loss(a,Z,offset,y,5.)[0]
        g=lambda a:full_loss(a,Z,offset,y,5.)[1]
        self.assertLess(check_grad(f,g,coef),1e-6)

    def test_context_graph_normalization(self):
        x=np.arange(8)[:,None]
        d=(x-x.T).astype('float32')**2
        np.fill_diagonal(d,np.inf)
        for shared in [True,False]:
            graph=context_graph(d,2,.2,shared)
            self.assertTrue(np.isfinite(graph.data).all())
            np.testing.assert_allclose(np.asarray(graph.sum(1)).ravel(),1.,atol=1e-6)
    def test_weighted_nca_gradient(self):
        rng=np.random.default_rng(9)
        x=rng.normal(size=(12,4)); y=np.repeat(np.arange(3),4)
        A=np.eye(4)+rng.normal(scale=.05,size=(4,4)); initial=np.eye(4)
        weights=np.linspace(1,2,12); weights/=weights.sum()
        same=y[:,None]==y[None,:]
        def f(a): return nca_loss(a,x,same,weights,initial,.01)[0]
        def g(a): return nca_loss(a,x,same,weights,initial,.01)[1]
        error=check_grad(f,g,A.ravel())
        self.assertLess(error,1e-6)
        trust=np.linspace(.1,1.,12)
        f2=lambda a:nca_loss(a,x,same,weights,initial,.01,trust)[0]
        g2=lambda a:nca_loss(a,x,same,weights,initial,.01,trust)[1]
        self.assertLess(check_grad(f2,g2,A.ravel()),1e-6)

    def test_constant_spectrogram_has_no_temporal_modulation(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'spec.npz'; target=Path(directory)/'feature.npz'
            np.savez(p,log=np.zeros((48,17),dtype='float16'),pcen=np.zeros((48,17),dtype='float16'))
            extra_one(p,target)
            with np.load(target) as item:
                self.assertTrue(np.isfinite(item['hpss']).all())
                self.assertTrue(np.isfinite(item['gabor']).all())
                np.testing.assert_allclose(item['gabor'],0.,atol=1e-8)

if __name__=='__main__': unittest.main()
