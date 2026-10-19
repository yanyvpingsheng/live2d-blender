"""Regression: ordinary bone matrices match shear/reflection/singular maps."""
import json,sys
from pathlib import Path
import bpy
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'addon'))
from live2d_blender.skinning import SkinningFit,predict
from live2d_blender.skinning_blender import create_skinned
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
rest=np.asarray([(x,y,.15) for x in np.linspace(1.5,2.5,5) for y in [-3.2,-3.,-2.8]])
matrices=[]
# Smooth rotation through 90/180 degrees, signed scale through zero, changing shear.
for t in np.linspace(0,1,31):
    a=4*np.pi*t;rotation=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
    matrices.append(rotation@np.array([[1+.2*np.sin(2*np.pi*t),.3*np.sin(4*np.pi*t)],[0,np.cos(2*np.pi*t)]]))
# Circle/near-circle, reflection, exactly collapsed dimension and both dimensions.
matrices.extend([np.eye(2),np.diag([1+1e-11,1-1e-11]),np.array([[0,-1],[1,0]]),np.diag([-1.,1.]),np.diag([0.,1.]),np.zeros((2,2))])
affines=np.zeros((len(matrices),1,2,3));affines[:,0,:,:2]=matrices;affines[:,0,:,2]=np.column_stack((np.linspace(-.2,.2,len(matrices)),np.linspace(.1,-.1,len(matrices))))
fit=SkinningFit(rest,np.ones((len(rest),1)),affines,np.linspace(-.1,.1,len(matrices))[:,None],np.array([[2.,-3.,.15]]),{})
expected=predict(fit);output=create_skinned(bpy.context,fit,[[0,1,3],[1,4,3]],name='Affine Corner Regression');obj=output['mesh'];rig=output['armature'];errors=[];matrix_errors=[]
for i in range(len(matrices)):
    bpy.context.scene.frame_set(i+1);dg=bpy.context.evaluated_depsgraph_get();evaluated=obj.evaluated_get(dg);mesh=evaluated.to_mesh();got=np.array([tuple(v.co) for v in mesh.vertices]);evaluated.to_mesh_clear();errors.append(float(np.max(np.abs(got-expected[i]))))
    actual=np.asarray(rig.pose.bones['Skin 000'].matrix@rig.data.bones['Skin 000'].matrix_local.inverted());want=np.eye(4);want[:2,:2]=affines[i,0,:,:2];want[:2,3]=affines[i,0,:,2];want[2,3]=fit.depth_offsets[i,0];matrix_errors.append(float(np.max(np.abs(actual-want))))
report={'cases':len(matrices),'actual_bones':len(rig.data.bones),'max_vertex_component_error':max(errors),'max_skinning_matrix_component_error':max(matrix_errors),'shape_keys':obj.data.shape_keys is not None,'max_influences':max(len(v.groups) for v in obj.data.vertices)}
print('AFFINE_CASES',json.dumps(report),flush=True)
assert max(errors)<3e-6 and max(matrix_errors)<3e-6
assert not report['shape_keys'] and report['max_influences']==1
(ROOT/'docs/pure_skinning_affine_cases_results.json').write_text(json.dumps(report,indent=2))
