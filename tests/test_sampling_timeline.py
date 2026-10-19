import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'addon'))
from live2d_blender.sampling import sample_sequence
from live2d_blender.timeline import replay_physics
from live2d_blender.fixture import SyntheticBackend
from live2d_blender.data import Live2DError

class FakePhysics:
    def reset(self,parameters,parts,stabilize=True):
        self.value=parameters['P'];self.steps=[]
        return {'P':self.value},parts
    def evaluate(self,parameters,parts,dt):
        self.value+=parameters['P']*dt;self.steps.append(dt)
        return {'P':self.value},parts

class ReplayTests(unittest.TestCase):
    def test_seek_is_deterministic(self):
        solver=FakePhysics();sample=lambda t:({'P':t+1},{},1)
        a=replay_physics(solver,sample,.5)
        replay_physics(solver,sample,.1)
        c=replay_physics(solver,sample,.5)
        self.assertEqual(a,c);self.assertEqual(len(solver.steps),30)
    def test_fractional_step(self):
        solver=FakePhysics();replay_physics(solver,lambda t:({'P':1},{},1),.025)
        self.assertEqual(len(solver.steps),2);self.assertAlmostEqual(sum(solver.steps),.025)
    def test_limit(self):
        with self.assertRaises(Live2DError):replay_physics(FakePhysics(),lambda t:({'P':1},{},1),2,max_steps=1)

class SamplingTests(unittest.TestCase):
    def test_flatten_for_skinning(self):
        data=sample_sequence(SyntheticBackend(),[{}, {'ParamAngleX':30}, {'ParamAngleX':-30}],fps=24)
        self.assertEqual(data.positions.shape,(3,264,3))
        self.assertEqual(data.triangles.shape,(256,3))
        self.assertEqual(data.uvs.shape,(264,2))
        self.assertEqual(len(data.pieces),8)
        self.assertFalse(np.array_equal(data.positions[0],data.positions[1]))
    def test_nongeometric_change_reported(self):
        data=sample_sequence(SyntheticBackend(),[{}, {'ParamOpacity':0}])
        self.assertEqual(len(data.non_geometric_changes['opacity']),8)
    def test_effect_channels_are_aligned_and_typed(self):
        data=sample_sequence(SyntheticBackend(),[{}, {'ParamOpacity':0}])
        self.assertEqual(data.channels['opacity'].shape,(2,8))
        self.assertEqual(data.channels['multiply_color'].shape,(2,8,4))
        self.assertEqual(data.channels['visible'].dtype,np.bool_)
        self.assertEqual(data.channels['render_order'].dtype,np.int32)
        np.testing.assert_allclose(data.channels['opacity'][0],1)
        np.testing.assert_allclose(data.channels['opacity'][1],0)
    def test_empty_rejected(self):
        with self.assertRaises(Live2DError):sample_sequence(SyntheticBackend(),[])
    def test_input_counts(self):
        with self.assertRaises(Live2DError):sample_sequence(SyntheticBackend(),[{},{}],[{}])

if __name__=='__main__':unittest.main(verbosity=2)
