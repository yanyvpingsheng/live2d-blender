"""Original synthetic alpha-mask tests; no external models, SDK or Pillow.

Pure tests: python -m unittest discover -s tests -p test_mask_bake.py
Blender tests: blender -b --factory-startup --python tests/test_mask_bake.py -- --blender
"""
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import zlib

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'addon'))
from live2d_blender import mask_bake as masks
from live2d_blender.data import Drawable, Live2DError


def quad(name='source', **kwargs):
    return Drawable(name, 0, [(0, 0), (1, 0), (1, 1), (0, 1)],
                    [(0, 0), (1, 0), (1, 1), (0, 1)], [0, 1, 2, 0, 2, 3], **kwargs)


def decode_png(path):
    """Tiny independent decoder for our unfiltered grayscale test output."""
    data = Path(path).read_bytes()
    assert data[:8] == b'\x89PNG\r\n\x1a\n'
    offset, compressed, shape = 8, b'', None
    while offset < len(data):
        size = struct.unpack('>I', data[offset:offset + 4])[0]
        kind, payload = data[offset + 4:offset + 8], data[offset + 8:offset + 8 + size]
        crc = struct.unpack('>I', data[offset + 8 + size:offset + 12 + size])[0]
        assert zlib.crc32(kind + payload) & 0xffffffff == crc
        if kind == b'IHDR':
            width, height, depth, color, compression, filtering, interlace = struct.unpack('>IIBBBBB', payload)
            assert (depth, color, compression, filtering, interlace) == (8, 0, 0, 0, 0)
            shape = (height, width)
        elif kind == b'IDAT':
            compressed += payload
        offset += size + 12
    rows = np.frombuffer(zlib.decompress(compressed), np.uint8).reshape(shape[0], shape[1] + 1)
    assert not rows[:, 0].any()
    return rows[:, 1:][::-1].copy()


class PureMaskTests(unittest.TestCase):
    def test_groups_are_canonical_original_indices(self):
        pieces = [{'id': 'a', 'masks': []}, {'id': 'b', 'masks': [3, 0, 3]},
                  {'id': 'c', 'masks': [0, 3], 'inverted_mask': True}, {'id': 'd', 'masks': []}]
        self.assertEqual(masks.mask_groups(pieces), [{'sources': [0, 3], 'targets': [1, 2]}])

    def test_invalid_source_indices_and_duplicates_fail(self):
        for index in (True, -1, 1, 0.0):
            with self.subTest(index=index), self.assertRaises(Live2DError):
                masks.mask_groups([{'id': 'a', 'masks': [index]}])
        with self.assertRaises(Live2DError):
            masks.mask_groups([{'id': 'a'}, {'id': 'a'}])

    def test_bounds_include_targets_with_fractional_and_pixel_padding(self):
        bounds = np.asarray(masks.fit_projection_bounds([[(0, 0), (1, 1)], [(-3, -2), (5, 4)]], 64))
        span = bounds[2:] - bounds[:2]
        np.testing.assert_allclose((span - [8, 6]) / 2, np.array([8, 6]) * .04 + 3 * span / 64)
        self.assertLess(bounds[0], -3)
        self.assertGreater(bounds[2], 5)

    def test_empty_and_degenerate_bounds_are_valid(self):
        for points in ([], [np.zeros((3, 2))], [np.zeros((0, 2))]):
            box = masks.fit_projection_bounds(points)
            self.assertTrue(np.isfinite(box).all())
            self.assertGreater(box[2], box[0]); self.assertGreater(box[3], box[1])
        for points in ([[[float('nan'), 0]]], [[[0, 0, 1]]]):
            with self.assertRaises(Live2DError):
                masks.fit_projection_bounds(points)

    def test_budget_checked_without_creating_files(self):
        self.assertEqual(masks.validate_bake_request(-2, 2, 64, 3), 5)
        for args in ((1, 2001, 32, 1), (1, 300, 2048, 1), (0, -1, 64, 1),
                     (True, 2, 64, 1), (0, 1, 0, 1), (0, 1, 64.0, 1)):
            with self.subTest(args=args), self.assertRaises(Live2DError):
                masks.validate_bake_request(*args)

    def test_union_ignores_source_appearance_and_own_mask(self):
        a = quad(opacity=0, visible=False, masks=[1], inverted_mask=True, blend_mode=2)
        b = quad('b', opacity=.01, multiply_color=(0, 0, 0, 0))
        b.texture = 1
        result = masks.rasterize_mask_group([a, b], [a.vertices, b.vertices],
                    [np.array([[[1, 0, 1, .5]]]), np.array([[[0, 1, 0, .25]]])],
                    [1, 0, 1], (0, 0, 1, 1), 32)
        np.testing.assert_allclose(result, .625)
        np.testing.assert_allclose(1 - result, .375)

    def test_empty_union_and_empty_geometry_are_zero(self):
        np.testing.assert_array_equal(masks.rasterize_mask_group([], [], [], [], (0, 0, 1, 1), 32), 0)
        d = Drawable('empty', 0, [], [], [])
        np.testing.assert_array_equal(masks.rasterize_mask_group([d], [[]], [np.ones((1, 1, 4))],
                                                               [0], (0, 0, 1, 1), 32), 0)

    def test_texture_orientation_and_only_alpha(self):
        d = quad()
        texture = np.zeros((2, 1, 4), np.float32)
        texture[0, 0] = (1, .2, .9, 0)
        texture[1, 0] = (.1, .4, .1, 1)
        result = masks.rasterize_mask_group([d], [d.vertices], [texture], [0], (0, 0, 1, 1), 32)
        np.testing.assert_array_equal(result[:8], 0)
        np.testing.assert_array_equal(result[-8:], 1)
        flipped = masks.rasterize_mask_group([d], [d.vertices], [texture], [0], (0, 0, 1, 1), 32,
                                             texture_origin='top-left')
        np.testing.assert_array_equal(result, flipped[::-1])

    def test_png_grayscale_orientation_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'alpha.png'
            values = np.array([[0, .25, .5], [.75, 1, 0]], np.float32)
            masks.write_grayscale_png(path, values)
            np.testing.assert_array_equal(decode_png(path), np.rint(values * 255).astype(np.uint8))
            with self.assertRaises(FileExistsError):
                masks.write_grayscale_png(path, values)
            with self.assertRaises(Live2DError):
                masks.write_grayscale_png(Path(directory) / 'bad.png', [[float('nan')]])


@unittest.skipUnless('--blender' in sys.argv, 'Run with Blender for native integration tests')
class BlenderMaskTests(unittest.TestCase):
    def setUp(self):
        import bpy
        self.bpy = bpy
        bpy.ops.wm.read_factory_settings(use_empty=True)
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        bpy.ops.wm.save_as_mainfile(filepath=str(self.directory / 'synthetic.blend'))
        self.root = bpy.data.objects.new('Output container', None)
        bpy.context.scene.collection.objects.link(self.root)
        self.objects = {}
        for name in ('source', 'target', 'inverse'):
            mesh = bpy.data.meshes.new(name)
            mesh.from_pydata([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)], [], [(0, 1, 2), (0, 2, 3)])
            uv = mesh.uv_layers.new()
            for loop in mesh.loops:
                uv.data[loop.index].uv = quad().uvs[loop.vertex_index]
            obj = bpy.data.objects.new(name, mesh); bpy.context.scene.collection.objects.link(obj)
            obj.parent = self.root
            self.objects[name] = obj
        self.pieces = [{'id': 'source', 'texture': 0, 'masks': []},
                       {'id': 'target', 'texture': 0, 'masks': [0]},
                       {'id': 'inverse', 'texture': 0, 'masks': [0], 'inverted_mask': True}]
        self.image = bpy.data.images.new('Original alpha input', 2, 2, alpha=True)
        self.image.pixels.foreach_set([.2, .8, .1, .5] * 4)
        self.image.update()
        bpy.context.scene.frame_set(8, subframe=.25)

    def tearDown(self):
        self.bpy.ops.wm.read_factory_settings(use_empty=True)
        self.temporary.cleanup()

    def bake(self, output_directory=None, **kwargs):
        return masks.bake_mask_sequences(self.bpy.context, self.objects, self.pieces, {0: self.image},
                                         -1, 1, self.root, str(output_directory or self.directory), resolution=32, **kwargs)

    def test_native_evaluation_sequence_parent_transform_and_shader(self):
        bpy = self.bpy
        source = self.objects['source']
        # An actual native armature deform makes the bounds/mask follow evaluated
        # geometry; source object visibility/opacity should not affect coverage.
        armature = bpy.data.armatures.new('Original test armature')
        rig = bpy.data.objects.new('Original test rig', armature)
        bpy.context.scene.collection.objects.link(rig); rig.parent = self.root
        bpy.context.view_layer.objects.active = rig; rig.select_set(True)
        bpy.ops.object.mode_set(mode='EDIT')
        bone = armature.edit_bones.new('Move'); bone.head = (0, 0, 0); bone.tail = (0, 1, 0)
        bpy.ops.object.mode_set(mode='OBJECT')
        group = source.vertex_groups.new(name='Move'); group.add([0, 1, 2, 3], 1, 'REPLACE')
        modifier = source.modifiers.new('Native armature', 'ARMATURE'); modifier.object = rig
        for frame, x in ((-1, 0), (1, 1)):
            rig.pose.bones['Move'].location.x = x
            rig.pose.bones['Move'].keyframe_insert('location', frame=frame)
        self.objects['target'].location.x = -3
        source.hide_render = True
        self.root.location = (7, 4, 2); self.root.rotation_euler = (.2, -.1, .4); self.root.scale = (2, 3, 1)
        result = self.bake()
        self.assertEqual(bpy.context.scene.frame_current, 8)
        self.assertAlmostEqual(bpy.context.scene.frame_subframe, .25)
        self.assertEqual(len(result['groups']), 1)
        self.assertEqual(len(result['images']), 1)
        image, bounds, inverted = result['bindings']['target']
        self.assertIs(image, result['bindings']['inverse'][0])
        self.assertFalse(inverted); self.assertTrue(result['bindings']['inverse'][2])
        self.assertLess(bounds[0], -3); self.assertGreater(bounds[2], 2)
        self.assertEqual(image.source, 'SEQUENCE'); self.assertTrue(image.colorspace_settings.is_data)
        self.assertTrue(image.filepath.startswith('//'))
        pngs = sorted(Path(result['directory']).rglob('*.png'))
        self.assertEqual(len(pngs), 3)
        first, last = decode_png(pngs[0]), decode_png(pngs[-1])
        self.assertEqual(first.max(), 128); self.assertEqual(last.max(), 128)
        self.assertFalse(np.array_equal(first, last))
        self.root.location = (-4, 2, 6); self.root.rotation_euler = (-.2, .2, -.6); self.root.scale = (1.2, .7, 2)
        again = self.bake()
        np.testing.assert_allclose(again['groups'][0]['bounds'], bounds, atol=2e-6)
        for left, right in zip(pngs, sorted(Path(again['directory']).rglob('*.png'))):
            np.testing.assert_array_equal(decode_png(left), decode_png(right))
        self.assertNotEqual(result['directory'], again['directory'])
        material = bpy.data.materials.new('Synthetic material'); material.use_nodes = True
        node = material.node_tree.nodes.new('ShaderNodeMath'); node.name = 'NE Masked Alpha'; node.operation = 'MULTIPLY'
        node.inputs[0].default_value = .3; node.inputs[1].default_value = 1
        texture = masks.attach_mask(material, image, bounds, True, self.root, -1, 1)
        self.assertEqual(node.inputs[0].default_value, np.float32(.3))
        self.assertEqual(texture.extension, 'CLIP')
        self.assertEqual(texture.image_user.frame_start, -1)
        self.assertEqual(texture.image_user.frame_duration, 3)
        self.assertTrue(texture.image_user.use_auto_refresh)
        coordinates = material.node_tree.nodes['NE Mask Object Coordinates']
        self.assertIs(coordinates.object, self.root)
        self.assertEqual(node.inputs[1].links[0].from_node.name, 'NE Inverted Mask Union')
        with self.assertRaises(Live2DError):
            masks.attach_mask(material, image, bounds, False, self.root, -1, 1)

    def test_empty_source_geometry_is_zero(self):
        self.objects['source'].data.clear_geometry()
        result = self.bake()
        for path in Path(result['directory']).rglob('*.png'):
            self.assertFalse(decode_png(path).any())

    def test_failure_removes_only_new_directory_and_images(self):
        before_images = set(self.bpy.data.images)
        sentinel = self.directory / 'user-data.txt'; sentinel.write_text('keep')
        original_write = masks.write_grayscale_png
        calls = []
        def fail_second(path, alpha):
            calls.append(path)
            if len(calls) == 2:
                raise RuntimeError('synthetic disk failure')
            original_write(path, alpha)
        with mock.patch.object(masks, 'write_grayscale_png', side_effect=fail_second):
            with self.assertRaisesRegex(RuntimeError, 'synthetic disk failure'):
                self.bake()
        self.assertEqual(set(self.bpy.data.images), before_images)
        self.assertFalse(list(self.directory.glob('native_alpha_masks_*')))
        self.assertEqual(sentinel.read_text(), 'keep')
        self.assertEqual(self.bpy.context.scene.frame_current, 8)
        self.assertAlmostEqual(self.bpy.context.scene.frame_subframe, .25)

    def test_missing_uv_fails_and_cleans_after_bounds_pass(self):
        mesh = self.objects['source'].data
        mesh.uv_layers.remove(mesh.uv_layers.active)
        mesh.update()
        self.bpy.context.view_layer.update()
        with self.assertRaisesRegex(Live2DError, 'UV coordinates'):
            self.bake()
        self.assertFalse(list(self.directory.glob('native_alpha_masks_*')))

    def test_missing_output_root_created_and_late_image_failure_cleaned(self):
        root = self.directory / 'new-parent' / 'configured-root'
        before = set(self.bpy.data.images)
        with mock.patch.object(self.bpy.path, 'relpath', side_effect=RuntimeError('synthetic path failure')):
            with self.assertRaisesRegex(RuntimeError, 'synthetic path failure'):
                self.bake(output_directory=root)
        self.assertEqual(set(self.bpy.data.images), before)
        self.assertFalse(root.parent.exists())
        result = self.bake(output_directory=root)
        self.assertTrue(root.is_dir())
        self.assertEqual(Path(result['directory']).parent, root)
        self.assertEqual(len(result['images']), 1)

    def test_restore_failure_rolls_back_files_and_images(self):
        bpy = self.bpy
        before = set(bpy.data.images)
        original = bpy.context.scene.frame_set
        failed = []
        def frame_set(frame, **kwargs):
            if frame == 8 and not failed:
                failed.append(True)
                raise RuntimeError('synthetic restore failure')
            original(frame, **kwargs)
        proxy = SimpleNamespace(scene=SimpleNamespace(frame_current=8, frame_subframe=.25, frame_set=frame_set),
                                evaluated_depsgraph_get=bpy.context.evaluated_depsgraph_get)
        with self.assertRaisesRegex(RuntimeError, 'synthetic restore failure'):
            masks.bake_mask_sequences(proxy, self.objects, self.pieces, {0: self.image},
                                      -1, 1, self.root, str(self.directory), resolution=32)
        self.assertFalse(list(self.directory.glob('native_alpha_masks_*')))
        self.assertEqual(set(bpy.data.images), before)
        self.assertEqual(bpy.context.scene.frame_current, 8)
        self.assertAlmostEqual(bpy.context.scene.frame_subframe, .25)

    def test_unsaved_blend_rejected_before_files(self):
        self.bpy.ops.wm.read_factory_settings(use_empty=True)
        with self.assertRaisesRegex(Live2DError, 'Save the .blend'):
            self.bake()
        self.assertFalse(list(self.directory.glob('native_alpha_masks_*')))


if __name__ == '__main__':
    result = unittest.main(argv=[sys.argv[0]], exit=False)
    if not result.result.wasSuccessful():
        raise SystemExit(1)
