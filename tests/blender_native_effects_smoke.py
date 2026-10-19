"""Create original bone+effect fixture, numeric-render it, then save standalone.
Run with Blender --background --factory-startup --python this_file.py.
A separate verifier opens the saved file without importing the addon.
"""
import json,sys,tempfile
from pathlib import Path
import bpy
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'addon'))
from live2d_blender.native_effects import bake_native_effects
from live2d_blender.data import Live2DError

checks=[]
def check(value,label):
    assert value,label
    checks.append(label);print('PASS',label,flush=True)
def linear(rgb):
    rgb=np.asarray(rgb)
    return np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)
def xy(obj):
    depsgraph=bpy.context.evaluated_depsgraph_get();evaluated=obj.evaluated_get(depsgraph)
    return [[float(x) for x in (evaluated.matrix_world@v.co)[:2]] for v in evaluated.data.vertices]

bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
scene=bpy.context.scene
rig_data=bpy.data.armatures.new('Original fixture rig');rig=bpy.data.objects.new('Original fixture rig',rig_data);scene.collection.objects.link(rig)
bpy.context.view_layer.objects.active=rig;rig.select_set(True)
bpy.ops.object.mode_set(mode='EDIT');bone=rig_data.edit_bones.new('Fixture bone');bone.head=(0,0,0);bone.tail=(0,1,0);bpy.ops.object.mode_set(mode='OBJECT')
for frame,x in ((1,0),(2,.125),(3,.25)):
    rig.pose.bones['Fixture bone'].location.x=x
    rig.pose.bones['Fixture bone'].keyframe_insert('location',index=0,frame=frame)
for curve in rig.animation_data.action.fcurves:
    for point in curve.keyframe_points:point.interpolation='LINEAR'
objects={};pieces=[]
for i,ident in enumerate(('A','B')):
    mesh=bpy.data.meshes.new('Original quad '+ident)
    mesh.from_pydata([(-1,-1,i*.05),(1,-1,i*.05),(1,1,i*.05),(-1,1,i*.05)],[],[(0,1,2),(0,2,3)])
    mesh.uv_layers.new(name='UVMap')
    coordinates=[(0,0),(1,0),(1,1),(0,1)]
    for polygon in mesh.polygons:
        for loop in polygon.loop_indices:mesh.uv_layers[0].data[loop].uv=coordinates[mesh.loops[loop].vertex_index]
    obj=bpy.data.objects.new('Original '+ident,mesh);scene.collection.objects.link(obj)
    group=obj.vertex_groups.new(name='Fixture bone');group.add([0,1,2,3],1,'REPLACE')
    modifier=obj.modifiers.new('Native Armature','ARMATURE');modifier.object=rig
    objects[ident]=obj;pieces.append({'id':ident,'count':4,'texture':i,'render_order':i,'blend_mode':0,'masks':[],'double_sided':True})
# Existing object Y animation must survive, as must the independent rig action.
for frame,y in ((1,0),(2,.05),(3,.1)):
    objects['A'].location.y=y;objects['A'].keyframe_insert('location',index=1,frame=frame)
for curve in objects['A'].animation_data.action.fcurves:
    for point in curve.keyframe_points:point.interpolation='LINEAR'
images=[]
for i,color in enumerate(((.8,.4,.2,.6),(.1,.7,.3,1))):
    image=bpy.data.images.new('Original linear texture '+str(i),width=1,height=1,alpha=True,float_buffer=True)
    image.alpha_mode='PREMUL';image.colorspace_settings.name='Linear Rec.709'
    image.pixels[:]=[*(linear(color[:3])*color[3]),color[3]];image.pack();images.append(image)
channels={'opacity':np.array([[.25,1],[.5,1],[.75,1]]),'visible':np.array([[1,1],[1,0],[1,1]]),
          'multiply_color':np.ones((3,2,4)),'screen_color':np.zeros((3,2,4)),
          'render_order':np.array([[0,1],[2,0],[1,2]])}
channels['multiply_color'][:,0,:3]=[[1,1,1],[.5,.8,1],[.2,.4,.6]]
channels['screen_color'][:,0,:3]=[[0,0,0],[.1,.2,.3],[.2,.1,.05]]
baseline={}
for frame in (1,2,3):
    scene.frame_set(frame);baseline[str(frame)]={name:xy(obj) for name,obj in objects.items()}
scene.frame_set(1)
rig_action=rig.animation_data.action
handlers=tuple(bpy.app.handlers.depsgraph_update_post),tuple(bpy.app.handlers.frame_change_post)
report=bake_native_effects(objects,[1,2,3],channels,pieces,images,depth_step=.05)
check(report['animated']['opacity']==['A'],'only varying opacity gets an Action')
check(report['animated']['visible']==['B'],'visibility is a separate keyed channel')
check(report['animated']['render_order']==['A','B'],'per-Drawable order channels baked')
check(rig.animation_data.action==rig_action,'armature Action retained')
check(handlers==(tuple(bpy.app.handlers.depsgraph_update_post),tuple(bpy.app.handlers.frame_change_post)),'bake installs no playback handlers')
for frame in (1,2,3):
    scene.frame_set(frame)
    for name,obj in objects.items():
        check(np.max(np.abs(np.asarray(xy(obj))-baseline[str(frame)][name]))<1e-6,f'{name} bone/object XY unchanged at frame {frame}')
scene.frame_set(1,subframe=.5)
check(abs(objects['A'].data.materials[0].node_tree.nodes['NE Opacity'].outputs[0].default_value-.375)<1e-7,'opacity interpolates linearly between samples')
check(objects['B'].data.materials[0].node_tree.nodes['NE Visibility'].outputs[0].default_value==1,'visibility holds until next sampled event')
check(abs(objects['A'].location.z)<1e-7,'order holds instead of crossing at midpoint')
# Conflicting bone-depth ownership fails before modifying materials/actions.
old=objects['A'].data.materials[0]
try:bake_native_effects(objects,[1,2,3],channels,pieces,images,depth_step=.05,bone_depth_animated=True)
except Live2DError:pass
else:raise AssertionError('Double depth bake should fail')
check(objects['A'].data.materials[0]==old,'double depth ownership rejected without material mutation')
# Actual shading check: B is naturally invisible at frame 2, exposing tinted A.
scene.frame_set(2)
camdata=bpy.data.cameras.new('Fixture camera');camera=bpy.data.objects.new('Fixture camera',camdata);scene.collection.objects.link(camera)
camera.location=(0,0,5);camdata.type='ORTHO';camdata.ortho_scale=4;scene.camera=camera
scene.render.engine='CYCLES';scene.cycles.samples=64;scene.cycles.use_denoising=False
scene.render.resolution_x=32;scene.render.resolution_y=32;scene.render.resolution_percentage=100;scene.render.film_transparent=True
scene.render.image_settings.file_format='OPEN_EXR';scene.render.image_settings.color_mode='RGBA';scene.render.image_settings.color_depth='32'
with tempfile.TemporaryDirectory() as folder:
    scene.render.filepath=str(Path(folder)/'original_native_effects.exr');bpy.ops.render.render(write_still=True)
    rendered=bpy.data.images.load(scene.render.filepath,check_existing=False)
    pixels=np.empty(32*32*4,np.float32);rendered.pixels.foreach_get(pixels);pixel=pixels.reshape(32,32,4)[16,16]
    encoded=np.array([.8,.4,.2])*np.array([.5,.8,1]);encoded=encoded*(1-np.array([.1,.2,.3]))+np.array([.1,.2,.3])
    expected=np.r_[linear(encoded)*.3,.3]
    error=float(np.max(np.abs(pixel-expected)))
    print('RGBA_COMPARE',pixel.tolist(),expected.tolist(),error,flush=True)
    check(error<3e-5,'piecewise sRGB tint and opacity match expected rendered PMA')
    bpy.data.images.remove(rendered)
scene['_native_effects_fixture_expected']=json.dumps({'baseline_xy':baseline,'channels':{k:v.tolist() for k,v in channels.items()},'frames':[1,2,3]})
scene.frame_start=1;scene.frame_end=3;scene.frame_set(1)
scene.render.filepath='//native_effects_fixture'
path=ROOT/'demo'/'native_effects_smoke.blend';bpy.ops.wm.save_as_mainfile(filepath=str(path))
report.update({'blender':bpy.app.version_string,'checks':checks,'passed':len(checks),'max_render_abs_error':error,'standalone_file':str(path),'fixture_only':True})
(ROOT/'docs'/'native_effects_smoke_results.json').write_text(json.dumps(report,indent=2))
print('NATIVE_EFFECTS_SMOKE_OK',json.dumps(report),flush=True)
