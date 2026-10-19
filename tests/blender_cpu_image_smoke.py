"""SDK-free CPU-frame -> Blender image -> Cycles pixel numeric smoke test.

Run: blender -b --factory-startup --python tests/blender_cpu_image_smoke.py
Only original 2x2 colors are used. No project is saved and temporary EXR is
removed automatically. Results are printed and saved as a small JSON report.
"""
import json
from pathlib import Path
import sys
import tempfile

import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'addon'))
from live2d_blender.cpu_renderer import render
from live2d_blender.data import Drawable, Snapshot

checks=[]
def check(condition,label):
    assert condition,label
    checks.append(label)
    print('PASS',label,flush=True)

before=tuple(bpy.app.handlers.depsgraph_update_post)
tex=np.array([[[.8,.2,.1,.5],[.1,.8,.2,1]],[[.2,.1,.8,.25],[.7,.6,.2,.75]]],np.float32)
d=Drawable('original_quad',0,[(0,0),(1,0),(1,1),(0,1)],[(0,0),(1,0),(1,1),(0,1)],[0,1,2,0,2,3])
pixels=render(Snapshot([d]),[tex],2,2,(0,0,1,1))
expected=tex[::-1].copy();expected[:,:,:3]*=expected[:,:,3:4]
check(np.allclose(pixels,expected,atol=1e-7),'CPU bottom-up premultiplied four-color fixture')
check(tuple(bpy.app.handlers.depsgraph_update_post)==before,'CPU render installs no depsgraph callbacks')
image=bpy.data.images.new('Original CPU fixture',width=2,height=2,alpha=True,float_buffer=True)
image.alpha_mode='PREMUL'
image.colorspace_settings.name='Linear Rec.709'
image.pixels.foreach_set(pixels.ravel())
image.update()
roundtrip=np.empty(16,np.float32);image.pixels.foreach_get(roundtrip)
check(np.array_equal(roundtrip.reshape(2,2,4),pixels),'Blender float image preserves uploaded PMA rows and values')

bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
bpy.ops.mesh.primitive_plane_add(size=2)
plane=bpy.context.object
mat=bpy.data.materials.new('Original CPU image material');mat.use_nodes=True
nodes=mat.node_tree.nodes;nodes.clear();links=mat.node_tree.links
out=nodes.new('ShaderNodeOutputMaterial')
sampler=nodes.new('ShaderNodeTexImage');sampler.image=image;sampler.interpolation='Closest'
emission=nodes.new('ShaderNodeEmission');transparent=nodes.new('ShaderNodeBsdfTransparent');mix=nodes.new('ShaderNodeMixShader')
links.new(sampler.outputs['Color'],emission.inputs['Color'])
links.new(sampler.outputs['Alpha'],mix.inputs[0]);links.new(transparent.outputs[0],mix.inputs[1]);links.new(emission.outputs[0],mix.inputs[2]);links.new(mix.outputs[0],out.inputs['Surface'])
plane.data.materials.append(mat)
scene=bpy.context.scene
camera_data=bpy.data.cameras.new('Numeric camera');camera=bpy.data.objects.new('Numeric camera',camera_data)
scene.collection.objects.link(camera);camera.location=(0,0,5);camera_data.type='ORTHO';camera_data.ortho_scale=2.4;scene.camera=camera
scene.render.engine='CYCLES';scene.cycles.samples=64;scene.cycles.use_denoising=False
scene.render.resolution_x=32;scene.render.resolution_y=32;scene.render.resolution_percentage=100
scene.render.film_transparent=True
scene.view_settings.view_transform='Standard'
scene.render.image_settings.file_format='OPEN_EXR';scene.render.image_settings.color_mode='RGBA';scene.render.image_settings.color_depth='32'
with tempfile.TemporaryDirectory() as folder:
    scene.render.filepath=str(Path(folder)/'original_numeric.exr')
    bpy.ops.render.render(write_still=True)
    result=bpy.data.images.load(scene.render.filepath,check_existing=False)
    values=np.empty(32*32*4,np.float32);result.pixels.foreach_get(values)
    values=values.reshape(32,32,4)
    samples=values[np.array([8,8,24,24]),np.array([8,24,8,24])]
    expected_samples=pixels.reshape(4,4)
    error=float(np.max(np.abs(samples-expected_samples)))
    print('NUMERIC_COMPARISON',json.dumps({'actual':samples.tolist(),'expected':expected_samples.tolist(),'max_error':error}),flush=True)
    check(error<2e-5,'Cycles rendered interior pixels match CPU PMA RGBA')
    bpy.data.images.remove(result)
report={'blender':bpy.app.version_string,'passed':len(checks),'checks':checks,'max_render_abs_error':error,
        'sample_rgba':samples.tolist(),'expected_rgba':expected_samples.tolist(),
        'assets':'Original synthetic 2x2 colors only; no SDK or user model; temporary image removed'}
(ROOT/'docs'/'blender_cpu_image_smoke_results.json').write_text(json.dumps(report,indent=2))
print('BLENDER_CPU_IMAGE_SMOKE_OK',json.dumps(report),flush=True)
