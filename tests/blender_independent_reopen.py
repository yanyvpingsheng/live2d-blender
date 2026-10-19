"""Run in a separate Blender process: no add-on imports or registration."""
import json,sys
from pathlib import Path
import bpy
ROOT=Path(__file__).resolve().parents[1]
path=sys.argv[sys.argv.index('--')+1] if '--' in sys.argv else '/tmp/live2d_independent_integration.blend'
bpy.ops.wm.open_mainfile(filepath=path)
assert not any('live2d_blender' in key for key in sys.modules)
objects=[o for o in bpy.data.objects if o.type=='MESH']
rigs=[o for o in bpy.data.objects if o.type=='ARMATURE']
assert len(objects)==8 and len(rigs)==1
assert not any(o.get('_l2d_uuid') for o in bpy.data.objects)
assert not any(o.data.shape_keys for o in objects)
assert not any(o.animation_data and o.animation_data.drivers for o in bpy.data.objects)
assert not any(o.data.materials[0].node_tree.animation_data and o.data.materials[0].node_tree.animation_data.drivers for o in objects)
assert all(len(o.modifiers)==1 and o.modifiers[0].type=='ARMATURE' for o in objects)
obj=next(o for o in objects if o['source_drawable_id']=='EyeLeft')
values=[];alpha=[]
for frame in (1,7):
    bpy.context.scene.frame_set(frame)
    evaluated=obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    values.append(tuple(evaluated.data.vertices[0].co))
    alpha.append(obj.data.materials[0].node_tree.nodes['NE Opacity'].outputs[0].default_value)
assert max(abs(x-y) for x,y in zip(*values))>.1
assert abs(alpha[0]-1)<1e-6 and abs(alpha[1]-.4)<1e-6
report={'blender':bpy.app.version_string,'addon_imported':False,'autoexec_enabled':bpy.context.preferences.filepaths.use_scripts_auto_execute,'meshes':len(objects),'bones':len(rigs[0].data.bones),'frames':[1,7],'animated_opacity':alpha,'passed':True}
(ROOT/'docs'/'independent_reopen_results.json').write_text(json.dumps(report,indent=2))
print('INDEPENDENT_REOPEN_OK',json.dumps(report))
