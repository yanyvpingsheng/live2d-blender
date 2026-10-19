"""Original float-texture material tests: alpha/blend algebra, culling, visibility.
Run Blender --factory-startup --disable-autoexec --python this_file.py.
Generated result JSON is written to demo/v03_appearance. Optional argument after
-- selects an alternative source root. No external models or SDK are used.
"""
import json
import math
import sys
sys.dont_write_bytecode = True
import tempfile
from pathlib import Path
import bpy
import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = Path(sys.argv[sys.argv.index('--') + 1]) if '--' in sys.argv else HERE.parent
OUTPUT = SOURCE / 'demo' / 'v03_appearance'
OUTPUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(SOURCE / 'addon'))
from live2d_blender.native_effects import make_native_material

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
s = bpy.context.scene

def image(name, rgb, alpha):
    im = bpy.data.images.new(name, width=1, height=1, alpha=True, float_buffer=True)
    im.colorspace_settings.name = 'Linear Rec.709'
    im.alpha_mode = 'PREMUL'
    im.pixels[:] = [*(np.asarray(rgb) * alpha), alpha]
    im.pack()
    return im

def quad(name, x, z, material):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([(-.9, -.9, 0), (.9, -.9, 0), (.9, .9, 0), (-.9, .9, 0)], [], [(0, 1, 2), (0, 2, 3)])
    uv = mesh.uv_layers.new()
    for p in mesh.polygons:
        for loop in p.loop_indices:
            uv.data[loop].uv = ((0, 0), (1, 0), (1, 1), (0, 1))[mesh.loops[loop].vertex_index]
    obj = bpy.data.objects.new(name, mesh)
    s.collection.objects.link(obj)
    obj.location = (x, 0, z)
    mesh.materials.append(material)
    return obj

source = image('Original source', (.2, .5, .8), .6)
fronts = []
for mode, x in enumerate((-2, 0, 2)):
    base = image('Original base ' + str(mode), (.6, .4, .2), 1 if mode == 2 else .6)
    quad('Base ' + str(mode), x, 0, make_native_material(base, color_math='linear'))
    mat = make_native_material(source, 'Front ' + str(mode), double_sided=False,
                               color_math='linear', blend_mode=mode, allow_opaque_multiply=True)
    mat.node_tree.nodes['NE Opacity'].outputs[0].default_value = .5
    mat.node_tree.nodes['NE Masked Alpha'].inputs[1].default_value = .625
    fronts.append(quad('Front ' + str(mode), x, .1, mat))
camera_data = bpy.data.cameras.new('Camera')
camera = bpy.data.objects.new('Camera', camera_data)
s.collection.objects.link(camera)
camera.location = (0, 0, 5)
camera_data.type = 'ORTHO'
camera_data.ortho_scale = 6
s.camera = camera
s.render.engine = 'CYCLES'
s.cycles.samples = 32
s.cycles.use_denoising = False
s.cycles.transparent_max_bounces = 32
s.cycles.pixel_filter_type = 'GAUSSIAN'
s.cycles.filter_width = .01
s.render.resolution_x = 96
s.render.resolution_y = 32
s.render.resolution_percentage = 100
s.render.film_transparent = True
s.render.image_settings.file_format = 'OPEN_EXR'
s.render.image_settings.color_mode = 'RGBA'
s.render.image_settings.color_depth = '32'
rows = []
for phase in ('front', 'back', 'invisible'):
    if phase == 'back':
        for obj in fronts:
            obj.rotation_euler.y = math.pi
    elif phase == 'invisible':
        for obj in fronts:
            obj.rotation_euler.y = 0
            obj.data.materials[0].node_tree.nodes['NE Visibility'].outputs[0].default_value = 0
    with tempfile.TemporaryDirectory() as folder:
        s.render.filepath = str(Path(folder) / 'result.exr')
        bpy.ops.render.render(write_still=True)
        im = bpy.data.images.load(s.render.filepath)
        array = np.empty(96 * 32 * 4, np.float32)
        im.pixels.foreach_get(array)
        bpy.data.images.remove(im)
    array = array.reshape(32, 96, 4)
    for mode, x in enumerate((16, 48, 80)):
        alpha = .6 * .5 * .625 if phase == 'front' else 0
        source_pma = np.r_[np.asarray((.2, .5, .8)) * alpha, alpha]
        base_alpha = 1 if mode == 2 else .6
        destination = np.r_[np.asarray((.6, .4, .2)) * base_alpha, base_alpha]
        if mode == 0:
            expected = source_pma + destination * (1 - alpha)
        elif mode == 1:
            expected = np.r_[destination[:3] + source_pma[:3], base_alpha]
        else:
            expected = np.r_[destination[:3] * (1 - alpha + source_pma[:3]), base_alpha]
        observed = array[16, x]
        error = float(np.max(np.abs(observed - expected)))
        rows.append({'phase': phase, 'mode': mode, 'actual': observed.tolist(), 'expected': expected.tolist(), 'error': error})
        print('MATERIAL_PIXEL', phase, mode, error, observed.tolist(), expected.tolist(), flush=True)
(OUTPUT / 'material_review.json').write_text(json.dumps({'rows': rows, 'max_abs_error': max(r['error'] for r in rows)}, indent=2))
assert max(r['error'] for r in rows) < 3e-6, rows
print('NATIVE_MATERIAL_REVIEW_OK', len(rows), flush=True)
