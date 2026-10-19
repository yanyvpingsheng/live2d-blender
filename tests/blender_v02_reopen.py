"""Fresh-process verifier, deliberately no add-on imports or SDK access."""
import json, sys
from pathlib import Path
import bpy
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
bpy.ops.wm.open_mainfile(filepath=str(ROOT/'demo/independent_sprout.blend'))
expected=json.loads((ROOT/'demo/v02_expected.json').read_text())
assert not any(n.startswith('live2d_blender') for n in sys.modules)
assert 'live2d_blender' not in bpy.context.preferences.addons
assert not bpy.context.preferences.filepaths.use_scripts_auto_execute
assert not bpy.data.texts
meshes=[o for o in bpy.data.objects if o.type=='MESH']
rigs=[o for o in bpy.data.objects if o.type=='ARMATURE']
assert len(meshes)==8 and len(rigs)==1
assert all(o.data.shape_keys is None and [m.type for m in o.modifiers]==['ARMATURE'] for o in meshes)
assert not any(o.animation_data and o.animation_data.drivers for o in bpy.data.objects)
assert not any('live2d' in getattr(f,'__module__','') for f in list(bpy.app.handlers.frame_change_post)+list(bpy.app.handlers.depsgraph_update_post))
error=0;world_error=0
for frame, poses, root_matrix in zip(expected['times'],expected['positions'],expected['root_world']):
    bpy.context.scene.frame_set(int(frame),subframe=frame%1)
    graph=bpy.context.evaluated_depsgraph_get()
    world_error=max(world_error,float(np.max(np.abs(np.asarray(bpy.data.objects[expected['root_name']].matrix_world)-root_matrix))))
    for obj in meshes:
        actual=np.array([v.co[:] for v in obj.evaluated_get(graph).data.vertices])
        error=max(error,float(np.max(np.abs(actual-poses[obj.name]))))
assert error<1e-7,error
assert world_error<1e-7,world_error
opacity=[]
for frame in (1,9,17):
    bpy.context.scene.frame_set(frame)
    opacity.append(meshes[0].data.materials[0].node_tree.nodes['NE Opacity'].outputs[0].default_value)
assert np.max(np.abs(np.array(opacity)-[1,.4,.8]))<1e-6
report={'blender':bpy.app.version_string,'times_checked':len(expected['times']),'max_reopen_vertex_difference':error,'max_reopen_root_world_difference':world_error,'ordinary_bones':len(rigs[0].data.bones),'opacity':opacity,'sdk_or_addon_imported':False,'autoexec_enabled':False,'drivers':0,'shape_keys':0,'embedded_scripts':0}
(ROOT/'docs/v02_reopen_results.json').write_text(json.dumps(report,indent=2))
print('V02_REOPEN_OK',json.dumps(report))
