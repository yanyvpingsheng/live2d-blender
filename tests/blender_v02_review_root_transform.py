"""Original fixture: placement must survive conversion or be explicitly rejected."""
from pathlib import Path
import json,sys
import bpy
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'addon'))
import live2d_blender
from live2d_blender import blender_bridge as bridge
from live2d_blender.fixture import SyntheticBackend
from live2d_blender.independent_convert import convert_timeline
live2d_blender.register();results=[];failures=[]

def animate_x(obj, keys):
    for frame,value in keys:
        obj.location.x=value;obj.keyframe_insert('location',frame=frame)

def extra_object(name):
    obj=bpy.data.objects.new(name,None);bpy.context.scene.collection.objects.link(obj);return obj

for case in ('static_normal','parameter_only','direct_animation','unsampled_middle_animation','parent_animation','constraint_animation','static_parent_shear'):
    bridge.clear_runtimes();bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
    root=bridge.create_model(bpy.context,SyntheticBackend());step=1
    if case=='static_normal':root.location=(2,3,.5);root.rotation_euler.z=.3;root.scale=(1.2,.8,1)
    if case=='parameter_only':
        for frame,value in ((1,0.),(3,20.)):
            root[bridge.parameter_key(0)]=value;root.keyframe_insert(bridge.id_path(bridge.parameter_key(0)),frame=frame)
    if case=='direct_animation':animate_x(root,[(1,0),(3,1)])
    if case=='unsampled_middle_animation':animate_x(root,[(1,0),(2,1),(3,0)]);step=2
    if case=='parent_animation':
        parent=extra_object('Moving parent');root.parent=parent;animate_x(parent,[(1,0),(3,1)])
    if case=='constraint_animation':
        target=extra_object('Moving constraint target');constraint=root.constraints.new('COPY_LOCATION');constraint.target=target;animate_x(target,[(1,0),(3,1)])
    if case=='static_parent_shear':
        parent=extra_object('Nonuniform parent');parent.scale=(2,1,1);root.parent=parent;root.rotation_euler.z=.7
    source=[]
    for frame in (1,2,3):
        bpy.context.scene.frame_set(frame);bpy.context.view_layer.update();source.append(np.asarray(root.matrix_world).copy())
    bpy.context.scene.frame_set(1);runtime=bridge.runtime_for_root(root);before_objects=set(bpy.data.objects)
    try:result=convert_timeline(bpy.context,root,1,3,step=step,bone_count=8,iterations=1,native_effects=False)
    except ValueError as error:
        valid=case not in ('static_normal','parameter_only') and any(word in str(error).lower() for word in ('root','transform','placement','parent','shear','constraint'))
        valid=valid and bridge.runtime_for_root(root) is runtime and not bridge._BUSY and set(bpy.data.objects)==before_objects
        results.append({'case':case,'status':'rejected','message':str(error),'valid':valid})
        if not valid:failures.append(case+' unexpected/nontransactional rejection')
    else:
        output=[]
        for frame in (1,2,3):
            bpy.context.scene.frame_set(frame);bpy.context.view_layer.update();output.append(np.asarray(result['root'].matrix_world).copy())
        error=float(np.max(np.abs(np.asarray(source)-output)));results.append({'case':case,'status':'converted','max_world_matrix_error':error,'valid':error<2e-6})
        if error>=2e-6:failures.append(case+' placement silently changed')
print('ROOT_TRANSFORM_REVIEW',json.dumps({'cases':results,'failures':failures}),flush=True)
live2d_blender.unregister()
assert not failures,failures
