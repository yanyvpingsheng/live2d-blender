import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'addon'))
from live2d_blender.skinning import fit_skinning,fit_pose_transforms,predict,project_simplex

class SkinningTests(unittest.TestCase):
    def setUp(self):
        self.rest=np.array([(x,y,0) for x in np.linspace(-1,1,7) for y in [-.3,0,.3]])
    def test_simplex(self):
        p=project_simplex(np.array([[-3,2,1],[1,1,1],[0,0,0]],float))
        self.assertTrue(np.all(p>=0));np.testing.assert_allclose(p.sum(axis=1),1)
    def test_one_bone_affine_and_depth(self):
        poses=[]
        for t in [0,.3,.8,1]:
            xy=self.rest[:,:2]@np.array([[1+t*.2,t*.1],[0,1-t*.1]]).T+np.array([t*.1,-t*.2])
            poses.append(np.column_stack((xy,np.full(len(xy),t*.05))))
        fit=fit_skinning(self.rest,poses,bone_count=1,regularization=1e-10)
        self.assertLess(np.max(np.abs(predict(fit)-poses)),1e-7)
        self.assertTrue(np.all(fit.weights==1))
    def test_fixed_weight_refit_needs_target(self):
        fit=fit_skinning(self.rest,[self.rest],bone_count=1)
        targets=np.array([self.rest+[.2,.5,0],self.rest+[-.1,0,.1]])
        a,d=fit_pose_transforms(fit,targets)
        np.testing.assert_allclose(predict(fit,a,d),targets,atol=1e-7)
    def test_single_influence_cap(self):
        poses=np.array([self.rest,self.rest+np.column_stack((self.rest[:,0]**2*.1,np.zeros((len(self.rest),2))))])
        fit=fit_skinning(self.rest,poses,bone_count=3,max_influences=1,iterations=3)
        self.assertLessEqual(int((fit.weights>1e-10).sum(axis=1).max()),1)
        np.testing.assert_allclose(fit.weights.sum(axis=1),1)
    def test_sparse_limit_and_no_mutation(self):
        rest=self.rest.copy();poses=np.array([rest,rest+np.column_stack((rest[:,0]**2*.1,rest[:,1]**2*.2,np.zeros(len(rest))))]);before=poses.copy()
        fit=fit_skinning(rest,poses,bone_count=5,max_influences=4,iterations=3,weight_iterations=20)
        self.assertLessEqual(int((fit.weights>1e-10).sum(axis=1).max()),4)
        self.assertGreaterEqual(float(fit.weights.min()),0)
        np.testing.assert_array_equal(rest,self.rest);np.testing.assert_array_equal(poses,before)
    def test_reject_invalid(self):
        with self.assertRaises(ValueError):fit_skinning(self.rest,[self.rest],bone_count=len(self.rest))
        with self.assertRaises(ValueError):fit_skinning(self.rest,[self.rest],bone_count=1,max_influences=5)
        invalid=self.rest.copy();invalid[0,0]=np.nan
        with self.assertRaises(ValueError):fit_skinning(invalid,[invalid],bone_count=1)
    def test_two_dimensional_input(self):
        fit=fit_skinning(self.rest[:,:2],[self.rest[:,:2]],bone_count=1)
        self.assertEqual(fit.rest.shape[1],3);self.assertLess(fit.metrics['max'],1e-10)

if __name__=='__main__':unittest.main()
