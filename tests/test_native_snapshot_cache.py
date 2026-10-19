"""Core snapshot regression tests using owned ctypes arrays, no SDK/model needed."""
import ctypes as C
import math
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'addon'))
from live2d_blender.core import NativeBackend,Vec2,Vec4,I,F,PI
from live2d_blender.data import Live2DError

class SnapshotCacheTests(unittest.TestCase):
    def setUp(self):
        self.b=NativeBackend.__new__(NativeBackend)
        self.b._model=1;self.b._static_drawables=None;self.b.warnings=[]
        self.vertices=(Vec2*3)(Vec2(0,0),Vec2(1,0),Vec2(0,1))
        self.uvs=(Vec2*3)(Vec2(0,0),Vec2(1,0),Vec2(0,1))
        self.indices=(C.c_uint16*3)(0,1,2)
        self.fields={
            'DrawableCount':1,'DrawableIds':(C.c_char_p*1)(b'triangle'),
            'DrawableTextureIndices':(I*1)(0),'DrawableVertexCounts':(I*1)(3),
            'DrawableIndexCounts':(I*1)(3),'DrawableMaskCounts':(I*1)(0),
            'DrawableVertexPositions':(C.POINTER(Vec2)*1)(self.vertices),
            'DrawableVertexUvs':(C.POINTER(Vec2)*1)(self.uvs),
            'DrawableIndices':(C.POINTER(C.c_uint16)*1)(self.indices),
            'DrawableMasks':(PI*1)(),
            'DrawableConstantFlags':(C.c_uint8*1)(4),'DrawableDynamicFlags':(C.c_uint8*1)(1),
            'DrawableOpacities':(F*1)(1),'DrawableMultiplyColors':(Vec4*1)(Vec4(1,1,1,1)),
            'DrawableScreenColors':(Vec4*1)(Vec4(0,0,0,0)),
            'DrawableParentPartIndices':(I*1)(-1),'DrawableBlendModes':(I*1)(0),
            'RenderOrders':(I*1)(0),
        }
        self.calls=[]
        def call(suffix,fallback=None):
            self.calls.append(suffix);return self.fields.get(suffix,fallback)
        self.b._call=call
        self.b.api={'csmResetDrawableDynamicFlags':lambda model:None}
    def test_static_fields_read_once_and_dynamic_updated(self):
        first=self.b.snapshot();self.calls.clear();self.vertices[1].x=2
        second=self.b.snapshot()
        self.assertEqual(first.drawables[0].vertices[1],(1,0))
        self.assertEqual(second.drawables[0].vertices[1],(2,0))
        for key in ['DrawableIds','DrawableVertexUvs','DrawableIndices','DrawableMasks','DrawableTextureIndices']:
            self.assertNotIn(key,self.calls)
    def test_snapshots_cannot_mutate_cache(self):
        first=self.b.snapshot();first.drawables[0].uvs[0]=(9,9);first.drawables[0].indices[0]=99;first.drawables[0].masks.append(99)
        second=self.b.snapshot();self.assertEqual(second.drawables[0].uvs[0],(0,0));self.assertEqual(second.drawables[0].indices,[0,1,2]);self.assertEqual(second.drawables[0].masks,[])
    def test_first_snapshot_full_validation(self):
        self.indices[0]=99
        with self.assertRaises(Live2DError):self.b.snapshot()
        self.assertIsNone(self.b._static_drawables)
    def test_dynamic_nan_vertex_rejected(self):
        self.b.snapshot();self.vertices[1].x=math.nan
        with self.assertRaises(Live2DError):self.b.snapshot()
    def test_dynamic_nan_opacity_rejected(self):
        self.b.snapshot();self.fields['DrawableOpacities'][0]=math.nan
        with self.assertRaises(Live2DError):self.b.snapshot()
    def test_dynamic_nan_color_rejected(self):
        self.b.snapshot();self.fields['DrawableMultiplyColors'][0].z=math.inf
        with self.assertRaises(Live2DError):self.b.snapshot()
    def test_geometry_count_change_rejected(self):
        self.b.snapshot();self.fields['DrawableVertexCounts'][0]=2
        with self.assertRaises(Live2DError):self.b.snapshot()
    def test_drawable_count_change_rejected(self):
        self.b.snapshot();self.fields['DrawableCount']=0
        with self.assertRaises(Live2DError):self.b.snapshot()
    def test_empty_geometry_null_pointer_allowed(self):
        for suffix in ['VertexCounts','IndexCounts']:self.fields['Drawable'+suffix][0]=0
        self.fields['DrawableVertexPositions'][0]=C.POINTER(Vec2)();self.fields['DrawableVertexUvs'][0]=C.POINTER(Vec2)();self.fields['DrawableIndices'][0]=C.POINTER(C.c_uint16)()
        self.assertEqual(self.b.snapshot().drawables[0].vertices,[])
    def test_closed_access_rejected_and_cache_released(self):
        self.b.snapshot();self.b.close();self.assertIsNone(self.b._static_drawables)
        with self.assertRaises(Live2DError):self.b.snapshot()
        with self.assertRaises(Live2DError):self.b.evaluate({})

if __name__=='__main__':unittest.main()
