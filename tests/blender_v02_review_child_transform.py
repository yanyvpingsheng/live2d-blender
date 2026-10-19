"""Original fixture: unsupported child animation must not be silently dropped."""
from pathlib import Path
import json
import sys

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'addon'))
import live2d_blender
from live2d_blender import blender_bridge as bridge
from live2d_blender.fixture import SyntheticBackend
from live2d_blender.independent_convert import convert_timeline

live2d_blender.register()
results = []
for case in ('action', 'nla', 'driver'):
    bridge.clear_runtimes()
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    root = bridge.create_model(bpy.context, SyntheticBackend())
    runtime = bridge.runtime_for_root(root)
    child = runtime.objects[0]
    if case == 'driver':
        child.driver_add('location', 0).driver.expression = '1.0 if frame == 2 else 0.0'
    else:
        for frame, value in ((1, 0.0), (2, 1.0), (3, 0.0)):
            child.location.x = value
            child.keyframe_insert('location', index=0, frame=frame)
        if case == 'nla':
            action = child.animation_data.action
            child.animation_data.action = None
            track = child.animation_data.nla_tracks.new()
            track.strips.new('Unsupported placement', 1, action)
    bpy.context.scene.frame_set(1)
    bpy.context.view_layer.update()
    assert abs(child.location.x) < 1e-6, 'Guard must catch animation while current transform is identity'
    before_objects = set(bpy.data.objects)
    try:
        convert_timeline(bpy.context, root, 1, 3, step=2, bone_count=8,
                         iterations=1, native_effects=False)
    except ValueError as error:
        assert any(word in str(error).lower() for word in ('child', 'transform', 'animation')), str(error)
        assert bridge.runtime_for_root(root) is runtime
        assert set(bpy.data.objects) == before_objects
        assert not bridge._BUSY
        results.append({'case': case, 'status': 'transactionally_rejected', 'message': str(error)})
    else:
        raise AssertionError(f'{case}: unsupported child animation was silently discarded')
print('CHILD_TRANSFORM_REVIEW ' + json.dumps(results), flush=True)
live2d_blender.unregister()
