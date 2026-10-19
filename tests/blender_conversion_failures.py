"""Transactional failures cannot lock handlers or leave partial converted data."""
import json,sys
from pathlib import Path
import bpy
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'addon'))
import live2d_blender
from live2d_blender import blender_bridge as b, independent_convert as converter, native_effects
from live2d_blender.fixture import SyntheticBackend
live2d_blender.register();root=b.create_model(bpy.context,SyntheticBackend())
kinds=('objects','meshes','armatures','materials','actions','collections','node_groups')
def counts():return {key:len(getattr(bpy.data,key)) for key in kinds}
initial=counts();checks=[]
root.live2d.motion_path='/tmp/this-live2d-test-file-does-not-exist.motion3.json'
try:converter.convert_timeline(bpy.context,root,1,3)
except FileNotFoundError:pass
else:raise AssertionError('missing motion should fail')
assert not b._BUSY and counts()==initial
checks.append('failed sampling and failed restoration preserve Busy and data counts')
root.live2d.motion_path=''
original=native_effects.bake_native_effects

def fail_after_material(*args,**kwargs):
    bpy.data.materials.new('Intentional transaction failure')
    raise ValueError('Intentional failure after complete armature creation')
native_effects.bake_native_effects=fail_after_material
try:converter.convert_timeline(bpy.context,root,1,3,bone_count=8,iterations=1)
except ValueError as error:assert str(error).startswith('Intentional')
else:raise AssertionError('bake should fail')
finally:native_effects.bake_native_effects=original
assert not b._BUSY and counts()==initial,(counts(),initial)
assert b.runtime_for_root(root) and not any(o.hide_render for o in b.runtime_for_root(root).objects)
checks.append('failed effects removes new bones meshes materials actions and collections')
checks.append('source remains connected and visible after failure')
# Reconnect detects same-ID/same-count UV changes rather than silently retaining old UV.
class ChangedUV(SyntheticBackend):
    def evaluate(self,*args):
        snapshot=super().evaluate(*args)
        snapshot.drawables[0].uvs[0]=(.123,.456)
        return snapshot
original_factory=b.SyntheticBackend;b.SyntheticBackend=ChangedUV
try:b.reconnect(bpy.context,root)
except ValueError as error:assert 'topology' in str(error)
else:raise AssertionError('changed source UVs should fail')
finally:b.SyntheticBackend=original_factory
assert root['_l2d_state']=='DISCONNECTED' and not b.runtime_for_root(root)
checks.append('reconnect rejects changed source UVs and leaves accurate disconnected state')
live2d_blender.unregister()
(ROOT/'docs'/'conversion_failure_results.json').write_text(json.dumps({'passed':len(checks),'checks':checks},indent=2))
print('CONVERSION_FAILURES_OK',checks)
