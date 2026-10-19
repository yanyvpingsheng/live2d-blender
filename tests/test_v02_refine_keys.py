import unittest,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent));sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'addon'))
from live2d_blender.skinning import SkinningFit,predict
from live2d_blender.refine_keys import refine_affine_keys
class RefinementTests(unittest.TestCase):
    def setUp(self):
        self.rest=np.array([[0.,0.,.1],[1.,0.,.2],[0.,1.,.3]])
        self.affines=np.array([[[[1.,0.,0.],[0.,1.,0.]]],[[[.7,.2,.1],[-.1,-.5,.2]]],[[[.1,-.4,0.],[.3,.2,-.3]]]])
        self.fit=SkinningFit(self.rest,np.ones((3,1)),self.affines,np.array([[0.],[.1],[-.1]]),np.array([[.2,.3,.1]]),{})
    def test_preserve_original_keys(self):
        fit,frames=refine_affine_keys(self.fit,[1,5,9],9)
        np.testing.assert_array_equal(frames[::9],[1,5,9]);np.testing.assert_array_equal(fit.affines[::9],self.affines)
    def test_linear_matrix_geometry(self):
        fit,frames=refine_affine_keys(self.fit,[1,5,9],9);source=predict(self.fit);got=predict(fit)
        for i,t in enumerate(frames):
            j=min(int((t-1)//4),1);u=(t-(1+j*4))/4;np.testing.assert_allclose(got[i],(1-u)*source[j]+u*source[j+1],atol=1e-14)
    def test_midpoints_and_quarters_are_not_keys(self):
        _,frames=refine_affine_keys(self.fit,[1,5,9],9)
        for heldout in [2,3,4,6,7,8]:self.assertGreater(float(abs(frames-heldout).min()),1e-8)
    def test_no_input_mutation(self):
        before=self.affines.copy();fit,_=refine_affine_keys(self.fit,[1,5,9],9);np.testing.assert_array_equal(self.affines,before);np.testing.assert_array_equal(fit.weights,self.fit.weights)
    def test_odd_subdivisions_required(self):
        for n in [0,2,8,-1]:
            with self.assertRaises(ValueError):refine_affine_keys(self.fit,[1,5,9],n)
if __name__=='__main__':unittest.main()
