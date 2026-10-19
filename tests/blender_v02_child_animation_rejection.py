"""Reject unsupported per-piece animation, even at identity sampling endpoints."""
from pathlib import Path
import sys,json
import bpy
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'addon'))
import live2d_blender
from live2d_blender import blender_bridge as bridge
from live2d_blender.fixture import SyntheticBackend
from live2d_blender.independent_convert import convert_timeline
live2d_blender.register();checks=[]
for case in ('action','driver','nla'):
    bridge.clear_runtimes();bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
    root=bridge.create_model(bpy.context,SyntheticBackend());runtime=bridge.runtime_for_root(root);obj=runtime.objects[0]
    if case in ('action','nla'):
        for frame,x in ((1,0),(2,1),(3,0)):
            obj.location.x=x;obj.keyframe_insert('location',index=0,frame=frame)
        if case=='nla':
            action=obj.animation_data.action;track=obj.animation_data.nla_tracks.new();track.strips.new('Unsupported child motion',1,action);obj.animation_data.action=None
    else:
        driver=obj.driver_add('location',0).driver;driver.expression='0.0'
    bpy.context.scene.frame_set(1)
    before={kind:len(getattr(bpy.data,kind)) for kind in ('objects','meshes','armatures','materials','actions','collections')}
    try:convert_timeline(bpy.context,root,1,3,2,8,1)
    except ValueError as error:assert 'child' in str(error) and any(word in str(error) for word in ('Action','driver','NLA'))
    else:raise AssertionError(case+' must not silently lose child animation')
    assert not bridge._BUSY and bridge.runtime_for_root(root) is runtime
    assert before=={kind:len(getattr(bpy.data,kind)) for kind in before}
    checks.append(case+' rejected before sampling or output mutation')
live2d_blender.unregister()
report={'passed':len(checks),'checks':checks,'fixture_only':True}
(ROOT/'docs/v02_child_animation_results.json').write_text(json.dumps(report,indent=2))
print('CHILD_ANIMATION_REJECTION_OK',json.dumps(report))
