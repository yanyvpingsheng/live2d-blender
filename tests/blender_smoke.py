"""Run: blender -b --factory-startup --python tests/blender_smoke.py"""
import json,sys,math
from pathlib import Path
import bpy
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'addon'))
import live2d_blender
from live2d_blender import blender_bridge as b
from live2d_blender.fixture import SyntheticBackend
from live2d_blender.data import Live2DError

checks=[]
def check(condition,label):
    assert condition,label
    checks.append(label)
    print('PASS',label,flush=True)

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
live2d_blender.register()
check(b.update_models in bpy.app.handlers.frame_change_post,'handler registered')
root=b.create_model(bpy.context,SyntheticBackend())
runtime=b.RUNTIMES[root['_l2d_uuid']]
check(len(runtime.objects)==8,'8 drawable meshes imported')
check(len(root.children_recursive)==10,'part hierarchy retained')
check(all(o.data.uv_layers for o in runtime.objects),'UV layers created')
check(all(o.data.materials[0].node_tree.nodes.get('L2D Opacity') for o in runtime.objects),'RGBA opacity material nodes created')
old=runtime.objects[3].data.vertices[1].co.copy()
root[b.parameter_key(0)]=30.0
runtime.evaluate(bpy.context.scene,force=True)
new=runtime.objects[3].data.vertices[1].co.copy()
check((old-new).length>.1,'parameter updates actual vertices')
root[b.parameter_key(0)]=-30.0
root.keyframe_insert(data_path=b.id_path(b.parameter_key(0)),frame=1)
root[b.parameter_key(0)]=30.0
root.keyframe_insert(data_path=b.id_path(b.parameter_key(0)),frame=11)
b.linearize(root.animation_data.action)
bpy.context.scene.frame_set(1)
left=runtime.objects[3].data.vertices[0].co.x
bpy.context.scene.frame_set(11)
right=runtime.objects[3].data.vertices[0].co.x
check(right-left>.29,'frame handler consumes animated parameter')
# Strict renderer must refuse unsupported clipping without mutating scene.
class Masked(SyntheticBackend):
    def evaluate(self,*args):
        s=super().evaluate(*args);s.drawables[1].masks=[0];return s
before=len(bpy.data.objects)
try:b.create_model(bpy.context,Masked())
except Live2DError:pass
else:raise AssertionError('strict masking should fail')
check(len(bpy.data.objects)==before,'unsupported masks rejected without partial objects')
# Explicit reference rasterization never happens inside a frame-change handler.
from live2d_blender.reference_preview import reference_frame
reference=reference_frame(bpy.context,root,resolution=64,texture_limit=64)
reference_meta=json.loads(reference['_l2d_reference'])
check(reference.size[:]==(64,64) and reference.alpha_mode=='PREMUL','offline CPU reference Image created')
check(reference_meta['raw_image'] in bpy.data.images,'raw PMA reference retained separately from display transform')
# Single-clip playback, then expression composition.
motion=ROOT/'fixtures'/'demo.motion3.json'
motion.write_text(json.dumps({'Version':3,'Meta':{'Duration':1,'Loop':False},'Curves':[{'Target':'Parameter','Id':'ParamMouthOpenY','Segments':[0,0,0,1,1]}]}))
root.live2d.motion_path=str(motion)
root.live2d.start_frame=1
bpy.context.scene.frame_set(1)
runtime.evaluate(bpy.context.scene,force=True)
closed=runtime.objects[5].data.vertices[9].co.y
bpy.context.scene.frame_set(25)
runtime.evaluate(bpy.context.scene,force=True)
opened=runtime.objects[5].data.vertices[9].co.y
check(opened-closed>.1,'motion3 drives animated geometry')
root.live2d.motion_path=''
# Bake 1/6/11 and compare evaluated geometry.
bpy.context.scene.frame_set(6)
runtime.evaluate(bpy.context.scene,bpy.context.evaluated_depsgraph_get(),force=True)
expected=runtime.objects[3].data.vertices[0].co.copy()
baked=b.bake_copy(bpy.context,root,1,11,5)
bpy.context.scene.frame_set(6)
target=next(o for o in baked.children if o.get('_l2d_drawable_id')=='EyeLeft')
actual=target.evaluated_get(bpy.context.evaluated_depsgraph_get()).data.vertices[0].co
check((expected-actual).length<1e-5,'bake reproduces sampled geometry')
check(len(target.data.shape_keys.key_blocks)==4,'bake stores basis plus 3 sampled keys')
check(all(o.hide_render for o in runtime.objects),'bake hides but preserves source meshes')
# Persisted demo intentionally contains no scripts or SDK.
scene=bpy.context.scene
scene.frame_start=1;scene.frame_end=11
for image in bpy.data.images:
    if image.source=='FILE':image.pack()
camdata=bpy.data.cameras.new('Demo Camera');cam=bpy.data.objects.new('Demo Camera',camdata);scene.collection.objects.link(cam)
cam.location=(0,0.08,5);camdata.type='ORTHO';camdata.ortho_scale=2.4;scene.camera=cam
scene.render.engine='CYCLES';scene.cycles.samples=16;scene.cycles.use_denoising=False
scene.render.resolution_x=640;scene.render.resolution_y=640;scene.render.resolution_percentage=100
scene.render.film_transparent=True
scene.view_settings.view_transform='Standard'
scene.render.filepath=str(ROOT/'demo'/'synthetic_preview.png')
bpy.ops.wm.save_as_mainfile(filepath=str(ROOT/'demo'/'synthetic_demo.blend'))
bpy.ops.render.render(write_still=True)
check((ROOT/'demo'/'synthetic_preview.png').is_file(),'headless render saved')
# Loading does not implicitly execute a native library. Manual reconnect is explicit.
bpy.ops.wm.open_mainfile(filepath=str(ROOT/'demo'/'synthetic_demo.blend'))
check(not b.RUNTIMES,'save/reload clears all native runtime pointers')
root=next(o for o in bpy.data.objects if o.get('_l2d_kind')=='SYNTHETIC')
check(root.get('_l2d_state')=='DISCONNECTED','reloaded root accurately reports disconnected')
b.reconnect(bpy.context,root)
check(root['_l2d_uuid'] in b.RUNTIMES,'explicit synthetic reconnect works')
live2d_blender.unregister()
check(b.update_models not in bpy.app.handlers.frame_change_post and not b.RUNTIMES,'unregister cleans handlers and backend memory')
report={'blender':bpy.app.version_string,'checks':checks,'passed':len(checks),'real_core_test':False}
(ROOT/'docs'/'blender_smoke_results.json').write_text(json.dumps(report,indent=2))
print('BLENDER_SMOKE_OK',json.dumps(report),flush=True)
