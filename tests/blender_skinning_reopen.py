"""Run in a fresh Blender process opening demo/independent_sprout.blend.

Intentionally imports no live2d_blender module and loads no Core library.
"""
from pathlib import Path
import sys,json
import numpy as np
import bpy
ROOT=Path(__file__).resolve().parents[1]
expected=np.load(ROOT/'demo/pure_skinning_expected.npz')
meshes={o['source_drawable_id']:o for o in bpy.data.objects if o.type=='MESH'}
rig=next(o for o in bpy.data.objects if o.type=='ARMATURE')
assert not any(name.startswith('live2d_blender') for name in sys.modules)
assert not bpy.app.handlers.frame_change_post and not bpy.app.handlers.depsgraph_update_post
assert not bpy.data.texts
assert all(o.data.shape_keys is None and [m.type for m in o.modifiers]==['ARMATURE'] for o in meshes.values())
assert all(not o.animation_data for o in meshes.values())
assert rig.animation_data and rig.animation_data.action and not rig.animation_data.drivers
maximum=0.0
for i,frame in enumerate(expected['positions']):
    bpy.context.scene.frame_set(i+1);dg=bpy.context.evaluated_depsgraph_get();vertices=[]
    for name in expected['ids']:
        obj=meshes[str(name)].evaluated_get(dg);mesh=obj.to_mesh();vertices.extend(tuple(v.co) for v in mesh.vertices);obj.to_mesh_clear()
    maximum=max(maximum,float(np.abs(np.asarray(vertices)-frame).max()))
assert maximum<2e-6
report={'fresh_process_reopen':True,'live2d_module_imported':False,'core_loaded_by_test':False,'frames_tested':len(expected['positions']),'max_vertex_difference_from_export_run':maximum,'shape_keys':False,'frame_handlers':0,'mesh_count':len(meshes),'armature_bones':len(rig.data.bones),'deform_bones':sum(b.use_deform for b in rig.data.bones),'bone_action':True,'embedded_scripts':0}
(ROOT/'docs/pure_skinning_reopen_results.json').write_text(json.dumps(report,indent=2));print('PURE_SKELETON_REOPEN_OK',json.dumps(report))
