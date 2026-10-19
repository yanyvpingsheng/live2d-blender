import copy
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'addon'))
from live2d_blender.native_effects import validate_channels
from live2d_blender.data import Live2DError


def original_channels():
    return {'opacity':np.array([[1,.5],[.5,1]]),'visible':np.ones((2,2),bool),
            'multiply_color':np.ones((2,2,4)),'screen_color':np.zeros((2,2,4)),
            'render_order':np.array([[0,1],[1,0]])}


class NativeEffectValidationTests(unittest.TestCase):
    pieces=[{'id':'A'},{'id':'B'}]

    def test_valid_input_copied(self):
        original=original_channels();frames,channels=validate_channels(self.pieces,[1,3],original)
        self.assertEqual(frames.tolist(),[1,3])
        for key in original:
            np.testing.assert_array_equal(channels[key],original[key])
            self.assertFalse(np.shares_memory(channels[key],original[key]))

    def test_frame_validation(self):
        for frames in ([],[2,1],[1,1],[1,np.nan],[[1,2]]):
            with self.subTest(frames=frames),self.assertRaises(Live2DError):
                validate_channels(self.pieces,frames,original_channels())

    def test_missing_channel(self):
        for key in original_channels():
            channels=original_channels();del channels[key]
            with self.subTest(key=key),self.assertRaises(Live2DError):
                validate_channels(self.pieces,[1,3],channels)

    def test_shapes(self):
        for key in original_channels():
            channels=original_channels();channels[key]=np.zeros((1,2))
            with self.subTest(key=key),self.assertRaises(Live2DError):
                validate_channels(self.pieces,[1,3],channels)

    def test_nonfinite(self):
        for key in original_channels():
            channels=original_channels();channels[key]=channels[key].astype(float);channels[key].flat[0]=np.inf
            with self.subTest(key=key),self.assertRaises(Live2DError):
                validate_channels(self.pieces,[1,3],channels)

    def test_color_opacity_ranges(self):
        for key in ('opacity','multiply_color','screen_color'):
            channels=original_channels();channels[key].flat[0]=1.1
            with self.subTest(key=key),self.assertRaises(Live2DError):
                validate_channels(self.pieces,[1,3],channels)

    def test_core_float32_ulp_overshoot(self):
        channels=original_channels();channels['opacity'][0,0]=1.000000238418579
        _,normalized=validate_channels(self.pieces,[1,3],channels)
        self.assertEqual(normalized['opacity'][0,0],1.0)

    def test_boolean_visibility(self):
        channels=original_channels();channels['visible']=np.full((2,2),.5)
        with self.assertRaises(Live2DError):validate_channels(self.pieces,[1,3],channels)

    def test_integer_order(self):
        channels=original_channels();channels['render_order']=np.full((2,2),.5)
        with self.assertRaises(Live2DError):validate_channels(self.pieces,[1,3],channels)

    def test_duplicate_ids(self):
        with self.assertRaises(Live2DError):validate_channels([{'id':'A'},{'id':'A'}],[1,3],original_channels())

if __name__=='__main__':unittest.main(verbosity=2)
