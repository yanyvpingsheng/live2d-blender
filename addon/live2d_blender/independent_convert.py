"""Explicit, offline conversion to ordinary Blender bones and native effects.

Core/Framework are only used while sampling. The output receives no runtime
handlers, drivers, shape keys or mesh caches. Affine skinning is approximate.
"""
from dataclasses import replace
import json
import math
from types import SimpleNamespace
import numpy as np
import bpy
from mathutils import Matrix
from . import blender_bridge as bridge
from .data import Live2DError
from .sampling import sample_sequence
from .fit_pipeline import fit_animation
from .skinning_blender import create_skinned


def _capture_root_world(context, root, start, end, bone_frames):
    """Capture original placement at integer and generated bone-key times.

    Parents, constraints and drivers are evaluated, but no output dependency on
    them survives. Shear is rejected instead of silently lost by LRS channels.
    """
    if end - start + 1 > 20000:
        raise Live2DError('Root placement bake is limited to 20000 integer frames; shorten the clip')
    times = np.unique(np.concatenate((np.arange(start, end + 1, dtype=float), np.asarray(bone_frames, dtype=float))))
    # Collapse only duplicate representable Blender times (not distinct source
    # frames), since node F-curve time storage is single precision.
    stored, indices = np.unique(times.astype(np.float32), return_index=True)
    times = stored.astype(float)
    if len(times) > 20000:
        raise Live2DError('Root placement bake exceeds 20000 key times; reduce subdivisions or the range')
    captured = []; previous = None
    for frame in times:
        base = math.floor(float(frame))
        context.scene.frame_set(base, subframe=float(frame) - base)
        matrix = root.evaluated_get(context.evaluated_depsgraph_get()).matrix_world.copy()
        values = np.asarray(matrix, dtype=float)
        if not np.isfinite(values).all():
            raise Live2DError('Source root has a non-finite world transform')
        location, rotation, scale = matrix.decompose()
        rebuilt = Matrix.Translation(location) @ rotation.to_matrix().to_4x4() @ Matrix.Diagonal((*scale, 1.0))
        if not np.allclose(np.asarray(rebuilt), values, atol=2e-6, rtol=2e-6):
            raise Live2DError('Source root world transform contains shear that cannot be preserved by independent location/rotation/scale channels; remove or apply the shearing parent/constraint first')
        if previous is not None and previous.dot(rotation) < 0:
            rotation.negate()
        previous = rotation.copy()
        captured.append((float(frame), location.copy(), rotation.copy(), scale.copy(), matrix))
    return captured


def _bake_root_world(container, captured):
    first = captured[0][4]
    animated = any(not np.allclose(np.asarray(item[4]), np.asarray(first), atol=1e-8, rtol=1e-8) for item in captured[1:])
    container.rotation_mode = 'QUATERNION'
    if animated:
        for frame, location, rotation, scale, _ in captured:
            container.location = location; container.rotation_quaternion = rotation; container.scale = scale
            for path in ('location', 'rotation_quaternion', 'scale'):
                container.keyframe_insert(data_path=path, frame=frame, group='Independent root placement')
        bridge.linearize(container.animation_data.action)
    else:
        container.matrix_world = first
    return {'animated': animated, 'sampled_times': len(captured),
            'source_parent_constraints_drivers_retained': False,
            'method': 'Evaluated world placement sampled at every integer frame plus generated bone-key times; quaternion channels linearized',
            'shear_supported': False,
            'subframe_equivalence_certified': False}


def convert_timeline(context, root, start, end, step=1, bone_count=8, iterations=20,
                     native_effects=True, allow_lossy=False, *, grouping='GLOBAL',
                     manual_mapping=None, repair_bind=True, separate_static=True,
                     static_tolerance=1e-6, subdivisions=9):
    """Sample a direct Blender timeline and generate independent skeletal data.

    Healthy first-sample meshes remain at rest; optional training-only repair
    can choose a nondegenerate rest per Drawable. Bone count is logical; each
    affine influence exports two actual bones. Layer depth is frozen in mesh
    rest coordinates and optionally animated by native effect object tracks.
    """
    runtime = bridge.runtime_for_root(root)
    if runtime is None:
        raise Live2DError('Reconnect the conversion source first')
    if end < start or step < 1 or not 1 <= bone_count <= 128 or not 1 <= iterations <= 100:
        raise Live2DError('Invalid sampling range, bone count or iteration count')
    frames = list(range(int(start), int(end) + 1, int(step)))
    if frames[-1] != end:
        frames.append(int(end))
    if len(frames) < 2:
        raise Live2DError('Select at least two samples for an animated skeletal conversion')
    count = sum(len(obj.data.vertices) for obj in runtime.objects)
    if count * len(frames) > 10_000_000:
        raise Live2DError('Interactive fit is limited to 10 million vertex samples; reduce the range or sampling density')
    # Core controls own the source geometry. Extra child transforms cannot be
    # silently discarded by a 2D fit; only root placement is supported.
    for obj in root.children_recursive:
        animation = obj.animation_data
        # Only the root's parameter tracks are conversion inputs. Checking the
        # current matrix alone misses actions that return to identity here.
        if animation and (animation.action is not None or len(animation.drivers) or any(len(track.strips) for track in animation.nla_tracks)):
            raise Live2DError('Extra child-object Action, driver or NLA animation is not supported; animate Live2D root parameters or the overall root placement instead, or clear the child animation before conversion')
        basis = obj.matrix_basis.copy()
        if '_l2d_drawable_index' in obj:
            basis.translation.z = 0.0
        if obj.constraints or any(abs(basis[row][col] - (1.0 if row == col else 0.0)) > 1e-6
                                  for row in range(4) for col in range(4)):
            raise Live2DError('Child transforms/constraints are outside the Core sampling model; restore them or apply animation through source parameters')
    original_frame = context.scene.frame_current
    original_subframe = context.scene.frame_subframe
    previous_busy = bridge._BUSY
    existing = {kind: set(getattr(bpy.data, kind)) for kind in
                ('objects', 'meshes', 'armatures', 'materials', 'actions', 'collections', 'node_groups')}
    result = None
    succeeded = False
    bridge._BUSY = True
    try:
        class TimelineBackend:
            def evaluate(self, values, _parts):
                context.scene.frame_set(int(values['frame']))
                runtime.evaluate(context.scene, context.evaluated_depsgraph_get(), force=True)
                snapshot = runtime.last_snapshot
                # motion3 Model/Opacity is outside the Core drawable opacity.
                opacity = float(runtime.last_model_opacity)
                return replace(snapshot, drawables=[replace(d, opacity=max(0.0, min(1.0, d.opacity * opacity)))
                                                    for d in snapshot.drawables])
        sample = sample_sequence(TimelineBackend(), [{'frame': frame} for frame in frames],
                                 fps=context.scene.render.fps / context.scene.render.fps_base,
                                 depth_step=root.live2d.depth_step)
        unsupported = [p for p in sample.pieces if p['masks'] or p['blend_mode'] != 0]
        if unsupported and not allow_lossy:
            raise Live2DError(f'{len(unsupported)} pieces use clipping or destination-dependent blends; enable explicit lossy appearance conversion to proceed')
        training_fit, fit, export_frames, fit_report = fit_animation(
            sample, frames, bone_budget=bone_count, iterations=iterations,
            grouping=grouping, manual_mapping=manual_mapping, repair_bind=repair_bind,
            separate_static=separate_static, static_tolerance=static_tolerance,
            subdivisions=subdivisions)
        if len(export_frames) * fit.weights.shape[1] > 100_000:
            raise Live2DError('Bone/key budget exceeds 100000 logical bone key-times; reduce samples, bones or subdivisions')
        root_placements = _capture_root_world(context, root, int(start), int(end), export_frames)
        materials = [obj.data.materials[0] for obj in runtime.objects]
        triangle_materials = []
        for index, drawable in enumerate(runtime.last_snapshot.drawables):
            triangle_materials.extend([index] * (len(drawable.indices) // 3))
        result = create_skinned(context, fit, sample.triangles, sample.uvs, export_frames,
                                name=root.name + ' Independent', materials=materials,
                                triangle_materials=triangle_materials, pieces=sample.pieces)
        effects_report = None
        if native_effects:
            from .native_effects import bake_native_effects
            images = {p['texture']: runtime.objects[index].data.materials[0].node_tree.nodes['L2D Texture'].image
                      for index, p in enumerate(sample.pieces)}
            effects_report = bake_native_effects(result['objects_by_id'], frames, sample.channels, sample.pieces,
                                                  images=images, depth_step=root.live2d.depth_step,
                                                  bone_depth_animated=result['bone_depth_animated'],
                                                  allow_unsupported=allow_lossy)
        else:
            # Frozen appearance uses the first sample, not the last pose visited
            # during offline capture. Animation remains geometric only.
            for piece in sample.pieces:
                obj = result['objects_by_id'][piece['id']]
                for material in obj.data.materials:
                    bridge.set_material(material, SimpleNamespace(**piece))
        # Split export and effect replacement may leave temporary material copies.
        for material in set(bpy.data.materials) - existing['materials']:
            if material.users == 0:
                bpy.data.materials.remove(material)
        # A plain transform parent preserves placement without conflicting with
        # per-Drawable local Z curves, and has no runtime-linked properties.
        container = bpy.data.objects.new(root.name + ' Independent', None)
        result['collection'].objects.link(container)
        root_transform_report = _bake_root_world(container, root_placements)
        result['root'] = container
        for obj in [result['armature']] + result['meshes']:
            obj.parent = container
            obj.matrix_parent_inverse.identity()
        report = {'method': 'Sparse affine weighted bones; approximate sampled timeline',
                  'logical_bones': int(fit.weights.shape[1]), 'actual_bones': len(result['armature'].data.bones),
                  'max_influences': int((fit.weights > 1e-10).sum(axis=1).max()),
                  'version': '0.2.0', 'frames': frames, 'bone_key_times': export_frames.tolist(),
                  'training_metrics': training_fit.metrics, 'fitting': fit_report,
                  'non_geometric_changes': sample.non_geometric_changes, 'sampling_warnings': sample.warnings,
                  'native_effects': effects_report, 'root_transform': root_transform_report,
                  'requires_live2d_runtime': False,
                  'holdout_tested': False,
                  'appearance_loss': 'Clipping masks and destination-dependent blending are not preserved' if unsupported else ''}
        result['armature']['conversion_report'] = json.dumps(report, ensure_ascii=False)
        result['report'] = report
        for obj in runtime.objects:
            obj.hide_set(True)
            obj.hide_render = True
        # Output playback does not keep an invisible Core updating in the background.
        bridge.RUNTIMES.pop(root['_l2d_uuid'], None)
        runtime.close()
        root['_l2d_state'] = 'DISCONNECTED'
        root.live2d.physics_enabled = False
        succeeded = True
        return result
    except Exception:
        if context.object and context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        for kind in ('objects', 'collections', 'meshes', 'armatures', 'materials', 'actions', 'node_groups'):
            data = getattr(bpy.data, kind)
            for item in set(data) - existing[kind]:
                data.remove(item, do_unlink=True)
        raise
    finally:
        try:
            try:
                context.scene.frame_set(original_frame, subframe=original_subframe)
                if not succeeded:
                    runtime.evaluate(context.scene, context.evaluated_depsgraph_get(), force=True)
            except Exception as restore_error:
                root['_l2d_runtime_error'] = f'After conversion: {restore_error}'
        finally:
            bridge._BUSY = previous_busy
