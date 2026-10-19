"""Integration: offline capture -> weighted bones -> native effects -> no runtime."""
import json, sys
from pathlib import Path
import bpy
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'addon'))
import live2d_blender
from live2d_blender import blender_bridge as b
from live2d_blender.fixture import SyntheticBackend
from live2d_blender.independent_convert import convert_timeline

bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
live2d_blender.register()
root=b.create_model(bpy.context,SyntheticBackend());root.location=(2,3,.5)
checks=[]
def check(value,label):
    assert value,label
    checks.append(label);print('PASS',label,flush=True)
for frame,angle,opacity in ((1,0,1),(7,30,.4),(13,-15,1)):
    root[b.parameter_key(0)]=float(angle)
    root[b.parameter_key(5)]=float(opacity)
    root.keyframe_insert(data_path=b.id_path(b.parameter_key(0)),frame=frame)
    root.keyframe_insert(data_path=b.id_path(b.parameter_key(5)),frame=frame)
b.linearize(root.animation_data.action)
bpy.context.scene.frame_set(3)
result=convert_timeline(bpy.context,root,1,13,2,8,6,native_effects=True)
check(not b.RUNTIMES and root['_l2d_state']=='DISCONNECTED','source Core runtime closed after conversion')
check(not b._BUSY and bpy.context.scene.frame_current==3,'frame and handler guard restored')
check(len(result['meshes'])==8 and len(result['armature'].data.bones)==16,'8 separate skinned pieces share 16 bones')
check(all(o.data.shape_keys is None for o in result['meshes']),'no shape keys')
check(all(len(o.modifiers)==1 and o.modifiers[0].type=='ARMATURE' for o in result['meshes']),'only ordinary Armature modifiers')
check(not result['armature'].animation_data.drivers,'no bone drivers')
check(len(result['report']['native_effects']['animated']['opacity'])==8,'opacity captured into native material actions')
check(tuple(result['root'].location)==(2,3,.5),'placement retained on independent transform parent')
for obj in result['meshes']:
    for vert in obj.data.vertices:
        check_weights=[g.weight for g in vert.groups]
        assert len(check_weights)<=4 and abs(sum(check_weights)-1)<1e-6
check(True,'all weights normalized with at most four influences')
obj=result['objects_by_id']['EyeLeft'];live2d_blender.unregister()
bpy.context.scene.frame_set(1);a=np.asarray([v.co[:] for v in obj.evaluated_get(bpy.context.evaluated_depsgraph_get()).data.vertices])
bpy.context.scene.frame_set(7);c=np.asarray([v.co[:] for v in obj.evaluated_get(bpy.context.evaluated_depsgraph_get()).data.vertices])
check(float(np.max(np.abs(a-c)))>.1,'geometry animates with add-on disabled')
for source in list(root.children_recursive)+[root]:bpy.data.objects.remove(source,do_unlink=True)
for text in list(bpy.data.texts):bpy.data.texts.remove(text)
for image in bpy.data.images:
    if image.source=='FILE':image.pack()
path='/tmp/live2d_independent_integration.blend'
bpy.ops.wm.save_as_mainfile(filepath=path)
report={'blender':bpy.app.version_string,'passed':len(checks),'checks':checks,'conversion':result['report'],'file':path}
(ROOT/'docs'/'independent_integration_results.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
print('INDEPENDENT_INTEGRATION_OK',flush=True)
