"""Create and verify a fully independent original skeletal demo, no SDK needed."""
import sys,json
from pathlib import Path
import numpy as np
import bpy
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'addon'))
from live2d_blender.fixture import SyntheticBackend
from live2d_blender.sampling import sample_sequence
from live2d_blender.skinning import fit_skinning,predict,error_metrics
from live2d_blender.skinning_blender import create_skinned
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
parameters=[]
for i in range(61):
    t=i/60*2*np.pi
    parameters.append({'ParamAngleX':25*np.sin(t),'ParamEyeOpen':1-.4*(1-np.cos(t)),'ParamMouthOpenY':.5*(1-np.cos(t)),'ParamBreath':.5*(1-np.cos(t)),'ParamLeafWave':np.sin(t)})
seq=sample_sequence(SyntheticBackend(),parameters,fps=60);train=np.arange(0,61,2);test=np.arange(1,61,2)
groups=np.concatenate([np.full(p['count'],i) for i,p in enumerate(seq.pieces)])
fit=fit_skinning(seq.rest_positions,seq.positions[train],8,groups=groups,regularization=1e-10)
materials=[]
for name,color in [('Mint',(.31,.73,.52,1)),('Ink',(.025,.055,.065,1)),('Peach',(.98,.43,.40,1)),('Leaf',(.12,.48,.27,1))]:
    material=bpy.data.materials.new(name);material.diffuse_color=color;material.use_nodes=True
    nodes=material.node_tree.nodes;nodes.clear();output=nodes.new('ShaderNodeOutputMaterial');emission=nodes.new('ShaderNodeEmission');emission.inputs['Color'].default_value=color;material.node_tree.links.new(emission.outputs[0],output.inputs['Surface']);materials.append(material)
ends=np.cumsum([p['count'] for p in seq.pieces]);owners=np.searchsorted(ends,seq.triangles[:,0],side='right');tri_materials=[seq.pieces[i]['texture'] for i in owners]
export=create_skinned(bpy.context,fit,seq.triangles,seq.uvs,frame_numbers=train+1,name='Independent Skeletal Sprout',materials=materials,triangle_materials=tri_materials,pieces=seq.pieces)
got=[]
for frame in range(1,62):
    bpy.context.scene.frame_set(frame);dg=bpy.context.evaluated_depsgraph_get();coords=[]
    for piece in seq.pieces:
        obj=export['objects_by_id'][piece['id']].evaluated_get(dg);mesh=obj.to_mesh();coords.extend(tuple(v.co) for v in mesh.vertices);obj.to_mesh_clear()
    got.append(coords)
got=np.asarray(got)
report={'vertices':len(seq.rest_positions),'logical_bones':8,'actual_bones':len(export['armature'].data.bones),'deform_bones':sum(b.use_deform for b in export['armature'].data.bones),'mesh_objects':len(export['meshes']),'max_influences':max(len(v.groups) for o in export['meshes'] for v in o.data.vertices),'blender_vs_numpy':error_metrics(seq.rest_positions,predict(fit),got[train]),'train_vs_source':error_metrics(seq.rest_positions,seq.positions[train],got[train]),'action_only_midpoints':error_metrics(seq.rest_positions,seq.positions[test],got[test]),'shape_keys':any(o.data.shape_keys for o in export['meshes']),'frame_handlers':len(bpy.app.handlers.frame_change_post),'bone_depth_animated':export['bone_depth_animated']}
assert report['blender_vs_numpy']['max']<2e-6
assert not report['shape_keys'] and report['max_influences']<=4 and len(export['meshes'])==8
assert all([m.type for m in o.modifiers]==['ARMATURE'] for o in export['meshes'])
(ROOT/'docs').mkdir(exist_ok=True);(ROOT/'demo').mkdir(exist_ok=True)
(ROOT/'docs/pure_skinning_smoke_results.json').write_text(json.dumps(report,indent=2))
# This small original fixture is safe to ship; it contains no real Live2D data.
np.savez_compressed(ROOT/'demo/pure_skinning_expected.npz',positions=got,source=seq.positions,rest=seq.rest_positions,ids=np.asarray([p['id'] for p in seq.pieces]))
scene=bpy.context.scene;scene.frame_start=1;scene.frame_end=61;scene.render.fps=60
camera_data=bpy.data.cameras.new('Demo Camera');camera=bpy.data.objects.new('Demo Camera',camera_data);scene.collection.objects.link(camera);camera.location=(0,.04,5);camera_data.type='ORTHO';camera_data.ortho_scale=2.1;scene.camera=camera
scene.render.engine='BLENDER_EEVEE_NEXT';scene.render.resolution_x=512;scene.render.resolution_y=512;scene.render.resolution_percentage=100;scene.render.film_transparent=True;scene.view_settings.view_transform='Standard'
scene.frame_set(16);scene.render.filepath=str(ROOT/'demo/independent_sprout.png');bpy.ops.render.render(write_still=True)
scene.frame_set(1);bpy.ops.wm.save_as_mainfile(filepath=str(ROOT/'demo/independent_sprout.blend'))
print('PURE_SKINNING_SMOKE_OK',json.dumps({k:v for k,v in report.items() if not isinstance(v,dict)}))
