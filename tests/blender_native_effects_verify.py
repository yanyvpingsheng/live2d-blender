"""Standalone reopen verifier: deliberately does NOT import the addon or Core.
blender -b demo/native_effects_smoke.blend --disable-autoexec --python this_file.py
"""
import json,sys,tempfile
from pathlib import Path
import bpy
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
scene=bpy.context.scene
expected=json.loads(scene['_native_effects_fixture_expected'])
channels=expected['channels'];checks=[]
def check(condition,label):
    assert condition,label
    checks.append(label);print('PASS',label,flush=True)
def xy(obj):
    evaluated=obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    return [[float(x) for x in (evaluated.matrix_world@v.co)[:2]] for v in evaluated.data.vertices]
def linear(rgb):
    rgb=np.asarray(rgb);return np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)
check(not any(name.startswith('live2d_blender') for name in sys.modules),'no addon module imported in verifier')
check('live2d_blender' not in bpy.context.preferences.addons,'addon not enabled')
check(not any('live2d' in getattr(f,'__module__','').lower() for f in list(bpy.app.handlers.frame_change_post)+list(bpy.app.handlers.depsgraph_update_post)),'no Live2D playback callbacks')
check(len(bpy.data.texts)==0,'no embedded scripts')
check(all(obj.data.shape_keys is None for obj in bpy.data.objects if obj.type=='MESH'),'no shape keys')
check(all(not obj.animation_data or not obj.animation_data.drivers for obj in bpy.data.objects),'no object drivers')
check(all(not mat.node_tree.animation_data or not mat.node_tree.animation_data.drivers for mat in bpy.data.materials if mat.use_nodes),'no material drivers')
for index,frame in enumerate(expected['frames']):
    scene.frame_set(frame)
    for column,name in enumerate(('A','B')):
        obj=bpy.data.objects['Original '+name];nodes=obj.data.materials[0].node_tree.nodes
        check(np.max(np.abs(np.asarray(xy(obj))-expected['baseline_xy'][str(frame)][name]))<1e-6,f'{name} bone/object XY survives reopen frame {frame}')
        for key,node in (('opacity','NE Opacity'),('visible','NE Visibility')):
            check(abs(nodes[node].outputs[0].default_value-channels[key][index][column])<1e-6,f'{name} {key} survives frame {frame}')
        for key,node in (('multiply_color','NE Multiply'),('screen_color','NE Screen')):
            check(np.max(np.abs(np.asarray(nodes[node].outputs[0].default_value[:3])-channels[key][index][column][:3]))<1e-6,f'{name} {key} survives frame {frame}')
        z=(channels['render_order'][index][column]-channels['render_order'][0][column])*.05
        check(abs(obj.location.z-z)<1e-7,f'{name} Z order survives frame {frame}')
        check(len(obj.modifiers)==1 and obj.modifiers[0].type=='ARMATURE','only native Armature modifier used')
scene.frame_set(1,subframe=.5)
a=bpy.data.objects['Original A'];b=bpy.data.objects['Original B']
check(abs(a.data.materials[0].node_tree.nodes['NE Opacity'].outputs[0].default_value-.375)<1e-7,'linear opacity midpoint survives reopen')
check(b.data.materials[0].node_tree.nodes['NE Visibility'].outputs[0].default_value==1,'visibility constant midpoint survives reopen')
check(abs(a.location.z)<1e-7,'order constant midpoint survives reopen')
scene.frame_set(2)
with tempfile.TemporaryDirectory() as folder:
    scene.render.filepath=str(Path(folder)/'reopened_original.exr');bpy.ops.render.render(write_still=True)
    image=bpy.data.images.load(scene.render.filepath,check_existing=False)
    data=np.empty(32*32*4,np.float32);image.pixels.foreach_get(data);actual=data.reshape(32,32,4)[16,16]
    encoded=np.array([.8,.4,.2])*np.array([.5,.8,1]);encoded=encoded*(1-np.array([.1,.2,.3]))+np.array([.1,.2,.3])
    target=np.r_[linear(encoded)*.3,.3];error=float(np.max(np.abs(actual-target)))
    check(error<3e-5,'standalone reopened Cycles pixels match reference tint/alpha')
    bpy.data.images.remove(image)
report={'blender':bpy.app.version_string,'checks':checks,'passed':len(checks),'addon_imported':False,
        'autoexec_disabled':not bpy.context.preferences.filepaths.use_scripts_auto_execute,
        'max_render_abs_error':error,'actual_rgba':actual.tolist(),'expected_rgba':target.tolist(),'fixture_only':True}
(ROOT/'docs'/'native_effects_reopen_results.json').write_text(json.dumps(report,indent=2))
print('NATIVE_EFFECTS_REOPEN_OK',json.dumps(report),flush=True)
