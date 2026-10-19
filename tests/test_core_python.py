import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'addon'))
from live2d_blender.assets import load_model3, read_json, resolve_asset
from live2d_blender.core import AlignedBuffer, CoreLibrary
from live2d_blender.data import Drawable, Live2DError, Parameter, Snapshot, validate_snapshot
from live2d_blender.fixture import SyntheticBackend
from live2d_blender.motion import Motion, apply_expression, parse_curve

class AssetTests(unittest.TestCase):
    def test_package_manifest_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'x.model3.json'
            data={'Version':3,'FileReferences':{'Moc':'x.moc3','Textures':['a.png'],'Physics':'x.physics3.json'},'FutureField':{'Keep':True}}
            p.write_text(json.dumps(data))
            package=load_model3(p)
            self.assertEqual(package.raw,data)
            self.assertEqual(len(package.missing),3)
            with self.assertRaises(Live2DError): package.require_model()
    def test_path_escape(self):
        for name in ('../secret','/etc/passwd','C:\\secret','https://example.com/x','..\\secret','\\\\server\\file'):
            with self.subTest(name=name), self.assertRaises(Live2DError): resolve_asset('/tmp/model',name)
    def test_symlink_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/'escape').symlink_to('/etc')
            with self.assertRaises(Live2DError): resolve_asset(tmp,'escape/passwd')
    def test_invalid_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'a.json';p.write_text('{"Version":2}')
            with self.assertRaises(Live2DError):load_model3(p)
    def test_nan(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'a.json';p.write_text('{"x":NaN}')
            with self.assertRaises(Live2DError):read_json(p)
    def test_utf8_bom(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'a.json';p.write_text('{"x":1}',encoding='utf-8-sig')
            self.assertEqual(read_json(p),{'x':1})

class MotionTests(unittest.TestCase):
    def curve(self,segments):return parse_curve({'Id':'ParamX','Target':'Parameter','Segments':segments})
    def test_linear(self):
        c=self.curve([0,0,0,2,10]);self.assertEqual(c.value(1),5);self.assertEqual(c.value(4),10)
    def test_step(self):
        c=self.curve([0,2,2,1,8,0,2,10]);self.assertEqual(c.value(0.9),2);self.assertEqual(c.value(1),8)
    def test_inverse_step(self):
        c=self.curve([0,2,3,1,8]);self.assertEqual(c.value(0.1),8);self.assertEqual(c.value(1),8)
    def test_unrestricted_bezier(self):
        c=self.curve([0,0,1,0.1,0,0.2,1,1,1])
        self.assertGreater(c.value(.5),.7)
        self.assertLess(c.value(.5),1)
    def test_truncated(self):
        with self.assertRaises(Live2DError):self.curve([0,0,1,1,2])
    def test_nonmonotonic(self):
        with self.assertRaises(Live2DError):self.curve([0,0,1,.8,0,.2,1,1,1])
    def test_loop(self):
        m=Motion.from_json({'Version':3,'Meta':{'Duration':1,'Loop':True},'Curves':[{'Id':'P','Segments':[0,0,0,1,1]}]})
        self.assertAlmostEqual(m.sample(1.25)['Parameter']['P'],.25)
    def test_expression_blends(self):
        e={'Type':'Live2D Expression','Parameters':[{'Id':'A','Value':2,'Blend':'Add'},{'Id':'B','Value':2,'Blend':'Multiply'},{'Id':'C','Value':4,'Blend':'Overwrite'}]}
        self.assertEqual(apply_expression({'A':1,'B':2,'C':2},e,.5),{'A':2,'B':3,'C':3})

class DataTests(unittest.TestCase):
    def test_alignment(self):
        import ctypes
        for alignment in (16,64):
            b=AlignedBuffer(9,alignment,b'123456789')
            self.assertEqual(b.address % alignment,0)
            self.assertEqual(ctypes.string_at(b.address,9),b'123456789')
    def test_bad_native_path(self):
        with self.assertRaises(Live2DError):CoreLibrary('/not/existing/Core.so')
    def test_parameter_clamp_repeat(self):
        self.assertEqual(Parameter('p',-1,1,0).constrain(3),1)
        self.assertEqual(Parameter('p',-1,1,0,repeat=True).constrain(1.5),-.5)
        with self.assertRaises(Live2DError):Parameter('p',-1,1,0).constrain(math.nan)
    def test_bad_geometry(self):
        d=Drawable('broken',0,[(0,0)],[(0,0)],[0,1,0])
        with self.assertRaises(Live2DError):d.validate()
    def test_fixture_parameter_effects(self):
        b=SyntheticBackend();a=validate_snapshot(b.evaluate({}));z=validate_snapshot(b.evaluate({'ParamEyeOpen':0,'ParamAngleX':30}))
        self.assertEqual(len(a.drawables),8)
        self.assertNotEqual(a.drawables[3].vertices,z.drawables[3].vertices)
        self.assertTrue(all(p.is_file() for p in b.textures))
    def test_fixture_zero_opacity(self):
        self.assertTrue(all(d.opacity==0 for d in SyntheticBackend().evaluate({'ParamOpacity':0}).drawables))

if __name__=='__main__':unittest.main(verbosity=2)
