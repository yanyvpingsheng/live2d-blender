"""Exact-RGBA tests using original tiny fixtures; no SDK/model assets needed."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'addon'))
from live2d_blender.cpu_renderer import fit_bounds, prepare_textures, render
from live2d_blender.data import Drawable, Live2DError, Snapshot


def texture(rgba):
    return np.array([[rgba]], dtype=np.float32)


def quad(ident='quad', tex=0, x0=0, x1=1, y0=0, y1=1, **kwargs):
    return Drawable(ident, tex, [(x0,y0),(x1,y0),(x1,y1),(x0,y1)],
                    [(0,0),(1,0),(1,1),(0,1)], [0,1,2,0,2,3], **kwargs)


def frame(drawables, textures, width=4, height=4, **kwargs):
    return render(Snapshot(drawables), textures, width, height, (0,0,1,1), **kwargs)


class RGBAAssertions:
    def assertRGBA(self, result, expected):
        np.testing.assert_allclose(result, np.broadcast_to(expected, result.shape), atol=2e-7, rtol=1e-6)


class ExactColorTests(RGBAAssertions, unittest.TestCase):
    def test_normal_premultiplied(self):
        self.assertRGBA(frame([quad(opacity=.5)], [texture((.8,.4,.2,.5))]), (.2,.1,.05,.25))

    def test_normal_source_over(self):
        a, b = quad('back', 0), quad('front', 1)
        self.assertRGBA(frame([a,b], [texture((.2,.4,.6,.5)), texture((.8,.1,.2,.25))]), (.275,.175,.275,.625))

    def test_additive_keeps_destination_alpha(self):
        a, b = quad('back', 0), quad('front', 1, blend_mode=1)
        self.assertRGBA(frame([a,b], [texture((.2,.4,.6,.5)), texture((.8,.1,.2,.25))]), (.3,.225,.35,.5))

    def test_multiply_keeps_destination_alpha(self):
        a, b = quad('back', 0), quad('front', 1, blend_mode=2)
        self.assertRGBA(frame([a,b], [texture((.2,.4,.6,.5)), texture((.8,.1,.2,.25))]), (.095,.155,.24,.5))

    def test_pma_glow_at_zero_alpha_is_retained(self):
        self.assertRGBA(frame([quad(blend_mode=1)], [texture((1,.5,0,.5))]), (.5,.25,0,0))
        with self.assertRaisesRegex(Live2DError, 'zero alpha'):
            frame([quad(blend_mode=1)], [texture((1,.5,0,.5))], alpha_mode='straight')

    def test_multiply_on_transparent_is_transparent(self):
        self.assertRGBA(frame([quad(blend_mode=2)], [texture((1,.5,0,.5))]), (0,0,0,0))

    def test_multiply_then_screen_and_ignored_color_alpha(self):
        d = quad(opacity=.5, multiply_color=(.5,1,.25,0), screen_color=(.2,.3,.4,1))
        self.assertRGBA(frame([d], [texture((.8,.4,.2,.5))]), (.13,.145,.1075,.25))
        self.assertRGBA(frame([d], [texture((.4,.2,.1,.5))], texture_alpha_mode='premultiplied'), (.13,.145,.1075,.25))

    def test_straight_output_and_background(self):
        self.assertRGBA(frame([quad(opacity=.5)], [texture((.8,.4,.2,.5))], alpha_mode='straight'), (.8,.4,.2,.25))
        self.assertRGBA(frame([], [], background=(.8,.4,.2,.5)), (.4,.2,.1,.5))

    def test_render_order_and_stable_ties(self):
        first, second = quad('red',0,render_order=10), quad('green',1,render_order=-2)
        textures = [texture((1,0,0,1)),texture((0,1,0,1))]
        self.assertRGBA(frame([first,second], textures), (1,0,0,1))
        first.render_order = -2
        self.assertRGBA(frame([first,second], textures), (0,1,0,1))

    def test_visibility_and_zero_opacity(self):
        ds = [quad('hidden',0,visible=False), quad('zero',0,opacity=0)]
        self.assertRGBA(frame(ds, [texture((1,0,0,1))]), (0,0,0,0))

    def test_legacy_unorm_clamp_and_hdr(self):
        ds = [quad('first'),quad('add',blend_mode=1)]
        self.assertRGBA(frame(ds,[texture((1,.7,.2,1))]), (1,1,.4,1))
        self.assertRGBA(frame(ds,[texture((1,.7,.2,1))],clamp=False), (2,1.4,.4,1))


class MaskTests(RGBAAssertions, unittest.TestCase):
    def make_masks(self, inverted=False):
        return [quad('mask1',0,opacity=0,visible=False),quad('mask2',1,opacity=.1,visible=False),
                quad('subject',2,masks=[0,1],inverted_mask=inverted,opacity=.4)]

    def test_alpha_union_not_max_or_product(self):
        ts = [texture((0,0,0,.5)),texture((0,0,0,.5)),texture((1,.5,0,1))]
        self.assertRGBA(frame(self.make_masks(),ts), (.3,.15,0,.3))
        self.assertRGBA(frame(self.make_masks(True),ts), (.1,.05,0,.1))

    def test_masks_ignore_source_own_mask_color_and_visibility(self):
        ds = self.make_masks()
        ds[0].masks = [1]
        ds[0].inverted_mask = True
        ds[0].multiply_color = (0,0,0,0)
        ds[0].screen_color = (1,1,1,1)
        ds[0].blend_mode = 2
        self.assertRGBA(frame(ds,[texture((1,1,1,.5)),texture((1,1,1,.5)),texture((1,.5,0,1))]), (.3,.15,0,.3))

    def test_disjoint_masks_union_spatially(self):
        ds = [quad('left',0,x1=.5,visible=False),quad('right',0,x0=.5,visible=False),quad('subject',0,masks=[0,1])]
        self.assertRGBA(frame(ds,[texture((1,1,1,1))]), (1,1,1,1))

    def test_mask_outside_geometry_and_inversion(self):
        ds = [quad('left',0,x1=.5,visible=False),quad('subject',0,masks=[0])]
        result = frame(ds,[texture((1,1,1,1))])
        self.assertRGBA(result[:,:2],(1,1,1,1)); self.assertRGBA(result[:,2:],(0,0,0,0))
        ds[-1].inverted_mask = True
        result = frame(ds,[texture((1,1,1,1))])
        self.assertRGBA(result[:,:2],(0,0,0,0)); self.assertRGBA(result[:,2:],(1,1,1,1))

    def test_empty_source_and_no_mask_inverted_flag(self):
        empty=Drawable('empty',0,[],[],[],visible=False)
        self.assertRGBA(frame([empty,quad(masks=[0])],[texture((1,1,1,1))]), (0,0,0,0))
        self.assertRGBA(frame([empty,quad(masks=[0],inverted_mask=True)],[texture((1,1,1,1))]), (1,1,1,1))
        self.assertRGBA(frame([quad(inverted_mask=True)],[texture((1,1,1,1))]), (1,1,1,1))

    def test_mask_cache_eviction_does_not_change_pixels(self):
        ds, ts = self.make_masks(), [texture((0,0,0,.5)),texture((0,0,0,.5)),texture((1,.5,0,1))]
        np.testing.assert_array_equal(frame(ds,ts),frame(ds,ts,mask_cache_bytes=0))
        np.testing.assert_array_equal(frame(ds,ts),frame(ds,ts,mask_cache_bytes=64))

    def test_mask_culling(self):
        ds = [quad('backfacing',0,visible=False,double_sided=False),quad('subject',0,masks=[0])]
        ds[0].indices=[0,2,1,0,3,2]
        self.assertRGBA(frame(ds,[texture((1,1,1,1))]), (0,0,0,0))


class GeometrySamplingTests(RGBAAssertions, unittest.TestCase):
    def test_shared_edge_is_not_double_blended(self):
        for size in (1,2,3,8,16):
            self.assertRGBA(frame([quad()],[texture((1,0,0,.5))],size,size),(.5,0,0,.5))

    def test_triangle_has_no_rectangle_leak(self):
        d=quad(); d.indices=[0,1,3]
        image=frame([d],[texture((1,1,1,1))])
        self.assertRGBA(image[0,0],(1,1,1,1)); self.assertRGBA(image[-1,-1],(0,0,0,0))

    def test_double_sided_and_ccw_front(self):
        d=quad(double_sided=False)
        self.assertRGBA(frame([d],[texture((1,0,0,.5))]),(.5,0,0,.5))
        d.indices=[0,2,1,0,3,2]
        self.assertRGBA(frame([d],[texture((1,0,0,.5))]),(0,0,0,0))
        d.double_sided=True
        self.assertRGBA(frame([d],[texture((1,0,0,.5))]),(.5,0,0,.5))

    def test_degenerate_and_offscreen_triangles(self):
        d=quad(); d.vertices=[(0,0)]*4
        self.assertRGBA(frame([d],[texture((1,1,1,1))]),(0,0,0,0))
        self.assertRGBA(frame([quad(x0=10,x1=11)],[texture((1,1,1,1))]),(0,0,0,0))

    def test_uv_and_image_row_origins(self):
        top=np.array([[[1,0,0,1],[0,1,0,1]],[[0,0,1,1],[1,1,1,1]]],dtype=np.float32)
        bottom=frame([quad()],[top],2,2)
        np.testing.assert_array_equal(bottom,top[::-1])
        np.testing.assert_array_equal(frame([quad()],[top],2,2,output_origin='top-left'),top)
        np.testing.assert_array_equal(frame([quad()],[top[::-1]],2,2,texture_origin='bottom-left'),bottom)

    def test_bilinear_center_and_nearest(self):
        ts=np.array([[[1,0,0,1],[0,1,0,1]],[[0,0,1,1],[1,1,1,1]]],dtype=np.float32)
        self.assertRGBA(frame([quad()],[ts],1,1),(.5,.5,.5,1))
        self.assertRGBA(frame([quad()],[ts],1,1,filtering='nearest'),(1,1,1,1))

    def test_bilinear_samples_before_premultiplication(self):
        # Straight and premultiplied input filtering are intentionally different.
        straight=np.array([[[1,0,0,1],[0,0,0,0]]],dtype=np.float32)
        self.assertRGBA(frame([quad()],[straight],1,1),(.25,0,0,.5))
        self.assertRGBA(frame([quad()],[straight],1,1,texture_alpha_mode='premultiplied'),(.5,0,0,.5))

    def test_uv_clamp_to_edge(self):
        d=quad(); d.uvs=[(-10,10)]*4
        ts=np.array([[[1,0,0,1],[0,1,0,1]],[[0,0,1,1],[1,1,1,1]]],dtype=np.float32)
        self.assertRGBA(frame([d],[ts]),(1,0,0,1))

    def test_inputs_are_not_mutated(self):
        ds=[quad()]; ts=[texture((1,.5,.25,.5))]
        before_d,before_t=copy.deepcopy(ds),ts[0].copy()
        frame(ds,ts)
        self.assertEqual(ds,before_d); np.testing.assert_array_equal(ts[0],before_t)


class ValidationTests(unittest.TestCase):
    def test_output_dtype_and_empty_scene(self):
        result=frame([],[])
        self.assertEqual(result.dtype,np.float32); self.assertEqual(result.shape,(4,4,4))
        self.assertFalse(result.any())

    def test_reject_unknown_blend(self):
        with self.assertRaisesRegex(Live2DError,'unsupported blend'):
            frame([quad(blend_mode=3)],[texture((1,1,1,1))])

    def test_reject_invalid_texture_and_bounds(self):
        for ts in ([np.zeros((0,1,4))],[np.ones((1,1,4))*2],[np.ones((1,1,4))*np.nan],[np.ones((1,1,4),dtype=int)]):
            with self.subTest(ts=ts), self.assertRaises(Live2DError): frame([quad()],ts)
        for bounds in ((0,0,0,1),(0,0,1,float('inf'))):
            with self.assertRaises(Live2DError): render(Snapshot([]),[],1,1,bounds)
        for size in (0,8193,1.5,True):
            with self.assertRaises(Live2DError): render(Snapshot([]),[],size,1)

    def test_invalid_texture_index_and_mask_index(self):
        with self.assertRaises(Live2DError): frame([quad(tex=9)],[texture((1,1,1,1))])
        with self.assertRaises(Live2DError): frame([quad(masks=[9])],[texture((1,1,1,1))])

    def test_uint8_rgb_and_path_loading(self):
        rgba=prepare_textures([np.array([[[255,0,128]]],dtype=np.uint8)])[0]
        np.testing.assert_allclose(rgba[0,0],[1,0,128/255,1])
        try:
            from PIL import Image
        except ImportError:
            self.skipTest('Optional Pillow not installed')
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'original_fixture.png'
            Image.fromarray(np.array([[[255,0,128,255]]],dtype=np.uint8)).save(path)
            np.testing.assert_array_equal(prepare_textures([path])[0],rgba)

    def test_fit_bounds(self):
        self.assertEqual(fit_bounds(Snapshot([])),(-1,-1,1,1))
        self.assertEqual(fit_bounds(Snapshot([quad()]),padding=0),(0,0,1,1))
        self.assertEqual(fit_bounds(Snapshot([quad()]),padding=.5),(-.5,-.5,1.5,1.5))


if __name__=='__main__':
    unittest.main(verbosity=2)
