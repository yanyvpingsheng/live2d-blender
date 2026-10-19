"""Independent, wholly original SDK-free v0.3 regression fixture.

Run in a disposable checkout with Blender --factory-startup --disable-autoexec:
  --python tests/blender_v03_appearance.py -- build
  --python tests/blender_v03_appearance.py -- verify
Optional second/third arguments select source root and output directory.
Generated files default to demo/v03_appearance, never the source manifest.

This tests flat interior pixels away from shared triangle seams. It does not
certify edge filtering or SDK parity. Opaque color PNGs avoid Blender's existing
8-bit semitransparent-color association quantization; the companion float-image
material test separately verifies texture alpha and fractional destination alpha.
"""
import hashlib
import json
import math
import shutil
import struct
import sys
sys.dont_write_bytecode = True
import tempfile
import zlib
from pathlib import Path

import bpy
import numpy as np

HERE = Path(__file__).resolve().parent
ARGS = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else ['verify']
MODE = ARGS[0]
ROOT = HERE.parent
OUTPUT = Path(ARGS[2]) if len(ARGS) > 2 else ROOT / 'demo' / 'v03_appearance'
BUNDLE = OUTPUT / 'original_bundle'
MOVED = OUTPUT / 'relocated_bundle'
CHECKS = []


def check(value, label):
    assert value, label
    CHECKS.append(label)
    print('PASS', label, flush=True)


def linear(rgb):
    rgb = np.asarray(rgb, dtype=float)
    return np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)


def png(path, rgba):
    def chunk(name, payload):
        return struct.pack('>I', len(payload)) + name + payload + struct.pack('>I', zlib.crc32(name + payload) & 0xffffffff)
    data = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', 2, 2, 8, 6, 0, 0, 0))
    data += chunk(b'IDAT', zlib.compress((b'\0' + bytes(rgba) * 2) * 2)) + chunk(b'IEND', b'')
    path.write_bytes(data)


def pixels(frame):
    scene = bpy.context.scene
    scene.frame_set(frame)
    with tempfile.TemporaryDirectory() as folder:
        scene.render.filepath = str(Path(folder) / 'rgba.exr')
        bpy.ops.render.render(write_still=True)
        im = bpy.data.images.load(scene.render.filepath, check_existing=False)
        result = np.empty(scene.render.resolution_x * scene.render.resolution_y * 4, np.float32)
        im.pixels.foreach_get(result)
        bpy.data.images.remove(im)
    return result.reshape(scene.render.resolution_y, scene.render.resolution_x, 4)


def expected_pixels(frame):
    # Source-1 and Source-2 are raw mask alpha 128/255 and 64/255. Animated
    # geometry removes Source-2 at frame 7 and Source-1 at frame 9.
    coverage = {5: round((1 - (1 - 128/255) * (1 - 64/255)) * 255) / 255,
                7: 128/255, 9: 64/255}[frame]
    source = linear(np.array([51, 128, 204]) / 255)
    base = linear(np.array([153, 102, 51]) / 255)
    alpha = .6 * .5
    normal = source * alpha * coverage + base * .6 * (1 - alpha * coverage)
    add = base * .6 + source * alpha * (1 - coverage)
    multiply = base * (1 - alpha * coverage + source * alpha * coverage)
    return [np.r_[normal, 1], np.r_[add, 1], np.r_[multiply, 1]]


def numeric_render(label):
    rows = []
    for frame in (5, 7, 9):
        rendered = pixels(frame)
        for index, expected in enumerate(expected_pixels(frame)):
            # Interior points away from shared triangle seams; this is an
            # algebra/sequence test, not a ray-intersection edge certification.
            observed = rendered[16, (15, 47, 79)[index]]
            error = float(np.max(np.abs(observed - expected)))
            print('REVIEW_PIXEL', label, frame, index, observed.tolist(), expected.tolist(), error, flush=True)
            check(error < 1e-4, f'{label}: frame {frame}, blend {index}, raw union/inversion and once-only alpha ({error:g})')
            rows.append({'frame': frame, 'mode': index, 'actual': observed.tolist(),
                         'expected': expected.tolist(), 'error': error})
    return rows


if MODE == 'build':
    SOURCE = Path(ARGS[1]) if len(ARGS) > 1 else HERE.parent
    sys.path.insert(0, str(SOURCE / 'addon'))
    import live2d_blender
    from live2d_blender import blender_bridge as bridge
    from live2d_blender.data import Drawable, Parameter, Snapshot, Live2DError
    from live2d_blender.independent_convert import convert_timeline

    BUNDLE.mkdir(parents=True, exist_ok=True)
    (BUNDLE / 'masks').mkdir(exist_ok=True)
    texture_dir = BUNDLE / 'source_textures'
    texture_dir.mkdir(exist_ok=True)
    # Opaque color PNGs avoid Blender's pre-existing 8-bit alpha association
    # quantization. The separate float-texture material test exercises texture
    # alpha; this end-to-end test supplies equivalent alpha via opacity.
    colors = [(0, 0, 0, 128), (0, 0, 0, 64), (153, 102, 51, 255),
              (153, 102, 51, 255), (51, 128, 204, 255), (0, 0, 0, 255)]
    for i, color in enumerate(colors):
        png(texture_dir / f'original_{i}.png', color)
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in texture_dir.glob('*.png')}

    def quad(name, x, y, half_x, half_y, texture, order, **kwargs):
        return Drawable(name, texture,
                        [(x-half_x, y-half_y), (x+half_x, y-half_y), (x+half_x, y+half_y), (x-half_x, y+half_y)],
                        [(0, 0), (1, 0), (1, 1), (0, 1)], [0, 1, 2, 0, 2, 3], render_order=order, **kwargs)

    class OriginalFixture:
        kind = 'SYNTHETIC'
        name = 'Original mask blend review'
        canvas = {'size': [6000, 2000], 'origin': [3000, 1000], 'pixels_per_unit': 1000}
        metadata = {'Synthetic': True, 'Notice': 'Original geometric test fixture only'}
        parameters = [Parameter('SourceOneShift', 0, 12, 0), Parameter('SourceTwoShift', 0, 12, 0)]
        parts = []
        warnings = []
        textures = [texture_dir / f'original_{i}.png' for i in range(len(colors))]

        def evaluate(self, values, parts=None):
            ds = [quad('Raw source one', values.get('SourceOneShift', 0), 0, 3.3, 1, 0, 0,
                       opacity=0, visible=False, masks=[1], inverted_mask=True,
                       multiply_color=(0, 0, 0, 1), screen_color=(1, 0, 0, 1)),
                  quad('Raw source two', values.get('SourceTwoShift', 0), 0, 3.3, 1, 1, 1,
                       opacity=0, visible=False)]
            for mode, x in enumerate((-2, 0, 2)):
                ds.append(quad(f'Base {mode}', x, 0, .9, .9, 3 if mode == 2 else 2, 2 + mode,
                               opacity=1 if mode == 2 else .6))
                ds.append(quad(f'Target {mode}', x, 0, .9, .9, 4, 5 + mode,
                               opacity=.3, masks=[0, 1], inverted_mask=(mode == 1), blend_mode=mode))
            # A zero-RGB additive layer with full alpha must never blacken skin.
            ds.append(quad('Black additive sentinel', 0, 0, .9, .9, 5, 8, blend_mode=1))
            return Snapshot(ds)

        def close(self):
            pass

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    live2d_blender.register()
    root = bridge.create_model(bpy.context, OriginalFixture(), allow_lossy=True)
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.render.film_transparent = False
    scene.world.use_nodes = True
    scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value = 0
    for frame, a, b in ((5, 0, 0), (7, 0, 12), (9, 12, 0)):
        for index, value in ((0, a), (1, b)):
            root[bridge.parameter_key(index)] = float(value)
            root.keyframe_insert(data_path=bridge.id_path(bridge.parameter_key(index)), frame=frame)
    bridge.linearize(root.animation_data.action)
    scene.frame_set(6, subframe=.5)
    bpy.ops.wm.save_as_mainfile(filepath=str(BUNDLE / 'original_fixture.blend'))
    counts = lambda: {name: len(getattr(bpy.data, name)) for name in
                      ('objects', 'meshes', 'armatures', 'materials', 'images', 'actions', 'collections', 'node_groups')}
    initial = counts()
    for options, phrase in (({}, 'default strict'),
                            ({'appearance_mode': 'NATIVE_APPROXIMATE'}, 'opaque acknowledgement'),
                            ({'appearance_mode': 'NATIVE_APPROXIMATE', 'allow_opaque_multiply': True, 'native_effects': False}, 'native effects required')):
        try:
            convert_timeline(bpy.context, root, 5, 9, 2, 8, 2, **options)
        except (Live2DError, ValueError):
            pass
        else:
            raise AssertionError(phrase + ' should reject')
        check(initial == counts() and bridge.runtime_for_root(root) is not None and not bridge._BUSY,
              phrase + ' rejects without orphan data or source disconnect')
    scene.render.film_transparent = True
    try:
        convert_timeline(bpy.context, root, 5, 9, 2, 8, 2,
                         appearance_mode='NATIVE_APPROXIMATE', allow_opaque_multiply=True)
    except (Live2DError, ValueError):
        pass
    else:
        raise AssertionError('Transparent multiply must reject')
    check(initial == counts() and bridge.runtime_for_root(root) is not None, 'transparent multiply rejects safely')
    scene.render.film_transparent = False
    from live2d_blender import mask_bake as mask_module
    initial_files = {str(p.relative_to(BUNDLE)) for p in (BUNDLE / 'masks').rglob('*')}
    for function in ('write_grayscale_png', 'attach_mask'):
        original = getattr(mask_module, function)
        def injected_failure(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError('Injected independent rollback failure')
        setattr(mask_module, function, injected_failure)
        try:
            try:
                convert_timeline(bpy.context, root, 5, 9, 2, 8, 2,
                                 appearance_mode='NATIVE_APPROXIMATE', allow_opaque_multiply=True,
                                 mask_resolution=64, mask_directory='//masks', grouping='DRAWABLE', subdivisions=3)
            except RuntimeError as error:
                check(str(error) == 'Injected independent rollback failure', function + ' injected failure reached')
            else:
                raise AssertionError(function + ' failure was swallowed')
        finally:
            setattr(mask_module, function, original)
        check(initial == counts() and bridge.runtime_for_root(root) is not None and not bridge._BUSY,
              function + ' rollback restores all data counts and connected runtime')
        check(initial_files == {str(p.relative_to(BUNDLE)) for p in (BUNDLE / 'masks').rglob('*')},
              function + ' rollback removes all generated files and subdirectories')
        check(scene.frame_current == 6 and scene.frame_subframe == .5 and
              not any(o.hide_render for o in bridge.runtime_for_root(root).objects),
              function + ' rollback restores scene time and source visibility')
    settings = root.live2d
    settings.bake_start = 5
    settings.bake_end = 9
    settings.bake_step = 2
    settings.skin_bones = 8
    settings.skin_iterations = 4
    settings.skin_appearance = 'NATIVE_APPROXIMATE'
    settings.skin_allow_opaque_multiply = True
    settings.skin_mask_resolution = 512
    settings.skin_mask_directory = '//masks'
    settings.skin_grouping = 'DRAWABLE'
    settings.skin_subdivisions = '3'
    bpy.context.view_layer.objects.active = root
    check(bpy.ops.live2d.convert_independent() == {'FINISHED'}, 'registered UI native-appearance conversion completes')
    rig = bpy.context.object
    result = {'armature': rig, 'root': rig.parent,
              'meshes': [o for o in rig.users_collection[0].objects if o.type == 'MESH'],
              'report': json.loads(rig['conversion_report'])}
    check(scene.frame_current == 6 and scene.frame_subframe == .5 and not bridge._BUSY, 'successful conversion restores frame and guard')
    check(not bridge.RUNTIMES and root['_l2d_state'] == 'DISCONNECTED', 'source runtime disconnected after conversion')
    check(len(result['meshes']) == 9 and all(o.data.shape_keys is None for o in result['meshes']), 'all original drawables retained without shape keys')
    check(bool(result['report']['appearance_loss']), 'approximate appearance warning remains explicit')
    for source in list(root.children_recursive) + [root]:
        bpy.data.objects.remove(source, do_unlink=True)
    for text in list(bpy.data.texts):
        bpy.data.texts.remove(text)
    live2d_blender.unregister()
    for _ in range(3):
        bpy.data.orphans_purge(do_recursive=True)
    for im in bpy.data.images:
        if im.source not in ('SEQUENCE', 'VIEWER'):
            check(bool(im.packed_file), 'source color texture packed for standalone replay: ' + im.name)
            original_name = im.name.split(' Independent packed')[0]
            check(hashlib.sha256(bytes(im.packed_file.data)).hexdigest() == source_hashes[original_name],
                  'original texture bytes preserved: ' + original_name)
        elif im.source == 'SEQUENCE':
            check(im.filepath.startswith('//'), 'alpha sequence is blend-relative')
            folder = Path(bpy.path.abspath(im.filepath)).parent
            files = sorted(folder.glob('*.png'))
            check([p.name for p in files] == [f'{i:06d}.png' for i in range(1, 6)],
                  'all five integer frames exist despite sample step 2')
            check(all(p.read_bytes()[:8] == b'\x89PNG\r\n\x1a\n' and p.read_bytes()[24:26] == b'\x08\x00' for p in files),
                  'every mask is eight-bit grayscale, no RGB replacement')
    for obj in result['meshes']:
        for mat in obj.data.materials:
            tex = mat.node_tree.nodes.get('NE Alpha Mask Sequence')
            if tex:
                check(tex.image_user.frame_start == 5 and tex.image_user.frame_duration == 5,
                      'sequence timing matches non-one scene start: ' + obj.name)
    scene.cycles.samples = 32
    scene.cycles.use_denoising = False
    scene.cycles.transparent_max_bounces = 64
    scene.cycles.pixel_filter_type = 'GAUSSIAN'
    scene.cycles.filter_width = .01
    scene.render.resolution_x = 96
    scene.render.resolution_y = 32
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'OPEN_EXR'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.image_settings.color_depth = '32'
    camera_data = bpy.data.cameras.new('Original camera')
    camera = bpy.data.objects.new('Original camera', camera_data)
    scene.collection.objects.link(camera)
    camera.location = (0, 0, 5)
    camera_data.type = 'ORTHO'
    camera_data.ortho_scale = 6
    scene.camera = camera
    rows = numeric_render('built')
    # Move the full output and its viewing camera together. Model-local mask
    # coordinates should remain valid without any additional bake.
    output_root = result['root']
    output_root.location = (4, -3, 1)
    output_root.rotation_mode = 'XYZ'
    output_root.rotation_euler.z = .45
    from mathutils import Matrix, Vector
    bpy.context.view_layer.update()
    camera.matrix_world = output_root.matrix_world @ Matrix.Translation((0, 0, 5))
    rows += numeric_render('relocated root')
    scene['independent_review_expected'] = json.dumps({'checks': CHECKS, 'rows': rows,
                                                       'report': result['report'], 'meshes': len(result['meshes'])})
    scene.frame_start = 5
    scene.frame_end = 9
    scene.frame_set(5)
    bpy.ops.wm.save_as_mainfile(filepath=str(BUNDLE / 'original_fixture.blend'))
    shutil.copytree(BUNDLE, MOVED, dirs_exist_ok=True, ignore=shutil.ignore_patterns('source_textures', '*.blend1'))
    (OUTPUT / 'build_review.json').write_text(json.dumps({'checks': CHECKS, 'rows': rows, 'report': result['report']}, indent=2))
    print('INDEPENDENT_BUILD_OK', len(CHECKS), flush=True)
else:
    bpy.ops.wm.open_mainfile(filepath=str(MOVED / 'original_fixture.blend'))
    check(not any('live2d' in name.lower() or 'cubism' in name.lower() for name in sys.modules), 'fresh replay imports no Live2D or Cubism modules')
    check(not bpy.context.preferences.filepaths.use_scripts_auto_execute, 'automatic scripts disabled')
    check(not bpy.data.texts and not bpy.data.libraries, 'no embedded scripts or blend library dependencies')
    check(not any(getattr(bpy.app.handlers, name) for name in ('frame_change_pre', 'frame_change_post', 'depsgraph_update_pre', 'depsgraph_update_post')), 'no runtime playback handlers')
    check(not any(o.animation_data and o.animation_data.drivers for o in bpy.data.objects), 'no object drivers')
    check(not any(m.node_tree and m.node_tree.animation_data and m.node_tree.animation_data.drivers for m in bpy.data.materials), 'no material drivers')
    check('Live2DCubismCore' not in Path('/proc/self/maps').read_text(), 'no native Core library loaded')
    for im in bpy.data.images:
        if im.source == 'SEQUENCE':
            path = Path(bpy.path.abspath(im.filepath)).resolve()
            check(path.is_relative_to(MOVED.resolve()) and path.is_file(),
                  'sequence resolves inside relocated bundle')
    rows = numeric_render('fresh moved bundle')
    check(len([o for o in bpy.data.objects if o.type == 'MESH']) == 9, 'all nine drawables survive reopening')
    report = {'checks': CHECKS, 'rows': rows, 'max_abs_error': max(r['error'] for r in rows),
              'blender': bpy.app.version_string, 'private_assets_used': False}
    (OUTPUT / 'reopen_review.json').write_text(json.dumps(report, indent=2))
    print('INDEPENDENT_REOPEN_OK', json.dumps(report), flush=True)
