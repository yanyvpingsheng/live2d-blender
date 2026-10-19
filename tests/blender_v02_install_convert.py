"""Install the ZIP, use registered UI operators, convert only original fixtures.
Produces a packed SDK-free demo and numeric expectations for separate replay.
"""
import json, sys, hashlib
from pathlib import Path
import bpy
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
ZIP = Path(sys.argv[sys.argv.index('--') + 1]) if '--' in sys.argv else ROOT/'dist/live2d_offline_converter-0.2.0.zip'
bpy.ops.preferences.addon_install(filepath=str(ZIP), overwrite=True)
bpy.ops.preferences.addon_enable(module='live2d_blender')
import live2d_blender
from live2d_blender import blender_bridge as bridge
from live2d_blender.fixture import SyntheticBackend
assert live2d_blender.bl_info['version'] == (0, 2, 0)
assert 'addons' in str(Path(live2d_blender.__file__))
bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete(use_global=False)
checks = []
def check(value, label):
    assert value, label
    checks.append(label); print('PASS', label, flush=True)

class OriginalRepairFixture(SyntheticBackend):
    def evaluate(self, parameters, parts=None):
        result = super().evaluate(parameters, parts)
        if parameters.get('ParamMouthOpenY', 0) <= 0:
            d = result.drawables[2]
            d.vertices = [d.vertices[0]] * len(d.vertices)
        return result

modes = []
for mode in ('GLOBAL', 'DRAWABLE', 'MANUAL'):
    root = bridge.create_model(bpy.context, OriginalRepairFixture())
    root.name = 'Original v02 ' + mode
    bpy.context.view_layer.objects.active = root
    for frame, angle, mouth, leaf, alpha in ((1, 0, 0, 0, 1), (9, 22, .8, .5, .4), (17, -15, .3, -.6, .8)):
        for index, value in ((0, angle), (2, mouth), (4, leaf), (5, alpha)):
            root[bridge.parameter_key(index)] = float(value)
            root.keyframe_insert(data_path=bridge.id_path(bridge.parameter_key(index)), frame=frame)
    for frame, x, rotation in ((1, 0, 0), (4, .18, .12), (9, -.1, -.08), (17, 0, 0)):
        root.location.x=x; root.rotation_euler.z=rotation
        root.keyframe_insert('location', frame=frame); root.keyframe_insert('rotation_euler', frame=frame)
    bridge.linearize(root.animation_data.action)
    source_world = []
    for frame in (1,4,9,17):
        bpy.context.scene.frame_set(frame); source_world.append(np.asarray(root.matrix_world).copy())
    settings = root.live2d
    settings.bake_start = 1; settings.bake_end = 17; settings.bake_step = 2
    settings.skin_bones = 8; settings.skin_iterations = 5
    settings.skin_grouping = mode; settings.skin_subdivisions = '9'
    if mode == 'MANUAL':
        assert bpy.ops.live2d.create_grouping_template() == {'FINISHED'}
        text = bpy.data.texts[settings.skin_mapping_text]
        initial = json.loads(text.as_string())
        ids = initial['groups'][0]['drawables']
        check(len(ids) == 8 and len(set(ids)) == 8, 'manual template uses actual unique original IDs')
        manual = {'version': 1, 'groups': [{'name': 'user_group_' + str(i), 'bones': 1, 'drawables': [pid]} for i, pid in enumerate(ids)]}
        text.clear(); text.write(json.dumps(manual))
        from live2d_blender.independent_convert import convert_timeline
        tracked = ('objects', 'meshes', 'armatures', 'materials', 'actions', 'collections', 'node_groups')
        before = {kind: len(getattr(bpy.data, kind)) for kind in tracked}
        bad = {'version': 1, 'groups': [{'name': 'bad', 'bones': 1, 'drawables': ['missing_original_id']}]}
        try:
            convert_timeline(bpy.context, root, 1, 17, 2, 8, 1, grouping='MANUAL', manual_mapping=bad)
        except ValueError as error:
            assert 'unknown Drawable' in str(error)
        else:
            raise AssertionError('Unknown mapping must fail')
        check(before == {kind: len(getattr(bpy.data, kind)) for kind in tracked} and not bridge._BUSY and bridge.runtime_for_root(root) is not None,
              'invalid manual mapping rolls back without disconnecting source')

    bpy.context.scene.frame_set(3, subframe=.5)
    assert bpy.ops.live2d.convert_independent() == {'FINISHED'}
    rig = bpy.context.object
    report = json.loads(rig['conversion_report'])
    check(report['version'] == '0.2.0' and report['fitting']['grouping'] == mode, mode + ' registered UI conversion')
    check(report['fitting']['bind_selection']['summary']['changed_rest_pieces'] == 1, mode + ' training-only collapsed bind repair')
    check(report['fitting']['static_vertices'] >= 33, mode + ' independent static vertices')
    check(len(report['bone_key_times']) == 73, mode + ' 9x matrix-key refinement')
    out_root=rig.parent
    output_world=[]
    for frame in (1,4,9,17):
        bpy.context.scene.frame_set(frame); output_world.append(np.asarray(out_root.matrix_world).copy())
    check(report['root_transform']['animated'] and np.max(np.abs(np.asarray(source_world)-output_world))<2e-6, mode + ' root movement/rotation including untrained frame retained')
    bpy.context.scene.frame_set(3, subframe=.5)
    check(report['logical_bones'] <= 8 and report['actual_bones'] == 2 * report['logical_bones'], mode + ' total bone budget respected')
    check(not report['holdout_tested'], mode + ' no invented holdout precision')
    check(not bridge.RUNTIMES and not bridge._BUSY and bpy.context.scene.frame_subframe == .5, mode + ' source detached and frame/guard restored')
    collection = rig.users_collection[0]
    meshes = [o for o in collection.objects if o.type == 'MESH']
    check(len(meshes) == 8 and all(o.data.shape_keys is None for o in meshes), mode + ' eight fixed meshes without shape keys')
    check(all([m.type for m in o.modifiers] == ['ARMATURE'] for o in meshes), mode + ' ordinary Armature playback')
    # Verify at integer/half times not only generated matrix keys.
    times = np.linspace(1, 17, 33)
    observations = []; root_world = []
    for frame in times:
        bpy.context.scene.frame_set(int(frame), subframe=float(frame % 1))
        graph = bpy.context.evaluated_depsgraph_get()
        root_world.append(np.asarray(out_root.matrix_world).tolist())
        observations.append({o.name: [[float(c) for c in v.co] for v in o.evaluated_get(graph).data.vertices] for o in meshes})
    for source in list(root.children_recursive) + [root]: bpy.data.objects.remove(source, do_unlink=True)
    if mode != 'MANUAL':
        for obj in list(collection.objects): bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.collections.remove(collection)
    else:
        expected = {'times': times.tolist(), 'positions': observations, 'mode': mode, 'root_name': out_root.name, 'root_world': root_world}
    modes.append({'mode': mode, 'conversion': report})
# Source-only orphan actions and meshes are not part of the demonstration.
for _ in range(3): bpy.data.orphans_purge(do_recursive=True)
for text in list(bpy.data.texts): bpy.data.texts.remove(text)
for image in bpy.data.images:
    if image.source == 'FILE': image.pack()
bpy.ops.preferences.addon_disable(module='live2d_blender')
check(not any('live2d' in getattr(f, '__module__', '') for f in list(bpy.app.handlers.frame_change_post)+list(bpy.app.handlers.depsgraph_update_post)), 'disable removes all plugin playback handlers')
scene = bpy.context.scene; scene.frame_start = 1; scene.frame_end = 17
camera_data=bpy.data.cameras.new('Original demo camera'); camera=bpy.data.objects.new('Original demo camera',camera_data);scene.collection.objects.link(camera)
camera.location=(0,.04,5);camera_data.type='ORTHO';camera_data.ortho_scale=2.1;scene.camera=camera
scene.render.engine='CYCLES';scene.cycles.samples=16;scene.cycles.use_denoising=False
scene.render.resolution_x=256;scene.render.resolution_y=256;scene.render.resolution_percentage=100
scene.render.film_transparent=True;scene.view_settings.view_transform='Standard'
scene.frame_set(5);scene.render.filepath=str(ROOT/'demo/independent_sprout.png');bpy.ops.render.render(write_still=True)
scene.frame_set(1);scene.render.filepath='//independent_sprout'
bpy.ops.wm.save_as_mainfile(filepath=str(ROOT/'demo/independent_sprout.blend'))
(ROOT/'demo/v02_expected.json').write_text(json.dumps(expected))
(ROOT/'docs/v02_install_conversion_results.json').write_text(json.dumps({'blender':bpy.app.version_string,'passed':len(checks),'checks':checks,'modes':modes,'sdk_used':False,'installer_sha256':hashlib.sha256(ZIP.read_bytes()).hexdigest()},indent=2))
print('V02_INSTALL_CONVERT_OK',len(checks),flush=True)
