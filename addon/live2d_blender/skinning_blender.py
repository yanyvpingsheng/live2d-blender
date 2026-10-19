"""Export offline affine fits as ordinary Blender Armature skinning and Actions.

No handlers, drivers, shape keys, mesh caches, or Core calls are installed.
Each affine influence uses a nondeforming scale/rotation helper followed by one
weighted deform bone. FULL inherited scale represents shear through hierarchy.
"""
import math
import numpy as np
import bpy


def continuous_affine_channels(matrices, tolerance=1e-9):
    """Return angle U, signed scales, angle V for A=R(U) S R(V).

    Track equivalent axis/sign choices across samples. At an isotropic matrix,
    reuse the preceding right rotation to avoid arbitrary SVD orientation flips.
    """
    out = []
    previous = (0.0, np.ones(2), 0.0)
    for matrix in np.asarray(matrices, dtype=np.float64):
        if matrix.shape != (2, 2) or not np.isfinite(matrix).all():
            raise ValueError('Expected finite 2D affine linear maps')
        u, scales, vt = np.linalg.svd(matrix)
        if np.linalg.det(u) < 0:
            u[:, 1] *= -1; scales[1] *= -1
        if np.linalg.det(vt) < 0:
            vt[1, :] *= -1; scales[1] *= -1
        angle_u = math.atan2(u[1, 0], u[0, 0])
        angle_v = math.atan2(vt[1, 0], vt[0, 0])
        candidates = []
        for swap in (False, True):
            a = angle_u + (math.pi / 2 if swap else 0)
            v = angle_v - (math.pi / 2 if swap else 0)
            s = scales[::-1].copy() if swap else scales.copy()
            for halfturn in (0, math.pi):
                au, av = a + halfturn, v + halfturn
                au += 2 * math.pi * round((previous[0] - au) / (2 * math.pi))
                av += 2 * math.pi * round((previous[2] - av) / (2 * math.pi))
                cost = (au - previous[0]) ** 2 + (av - previous[2]) ** 2 + 0.05 * float(np.sum((s - previous[1]) ** 2))
                candidates.append((cost, au, s, av))
        # Isotropic orientation is underdetermined. Preserve the previous split.
        if scales[0] * scales[1] > 0 and abs(abs(scales[0]) - abs(scales[1])) <= tolerance * max(1.0, abs(scales).max()):
            scale = math.sqrt(abs(float(np.linalg.det(matrix))))
            rotation = math.atan2(matrix[1, 0], matrix[0, 0])
            av = previous[2]
            au = rotation - av
            au += 2 * math.pi * round((previous[0] - au) / (2 * math.pi))
            candidates.append((-1.0, au, np.array([scale, scale]), av))
        _, au, s, av = min(candidates, key=lambda item: item[0])
        # Validate the chosen decomposition before it can enter Blender data.
        ru = np.array([[math.cos(au), -math.sin(au)], [math.sin(au), math.cos(au)]])
        rv = np.array([[math.cos(av), -math.sin(av)], [math.sin(av), math.cos(av)]])
        if not np.allclose(ru @ np.diag(s) @ rv, matrix, atol=5e-8, rtol=5e-8):
            raise ValueError('Could not decompose an affine map into hierarchical bone channels')
        out.append((au, float(s[0]), float(s[1]), av))
        previous = (au, s, av)
    return np.asarray(out)


def _linearize(action):
    if action and hasattr(action, 'fcurves'):
        for curve in action.fcurves:
            for point in curve.keyframe_points:
                point.interpolation = 'LINEAR'


def create_skinned(context, fit, triangles, uvs=None, frame_numbers=None,
                   name='Pure Skinning', materials=None, triangle_materials=None, pieces=None):
    """Create an independent mesh + armature. Returns a dictionary of Blender IDs.

    Input material copies are frozen at current values. Material effects and
    ordering are outside the fitted LBS geometry; inspect sampling warnings.
    """
    rest = np.asarray(fit.rest, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int32)
    weights = np.asarray(fit.weights, dtype=np.float64)
    poses, bones = fit.affines.shape[:2]
    frames = np.arange(1, poses + 1, dtype=float) if frame_numbers is None else np.asarray(frame_numbers, dtype=float)
    if frames.shape != (poses,) or not np.isfinite(frames).all() or np.any(np.diff(frames) <= 0):
        raise ValueError('Provide one strictly increasing finite frame per fitted pose')
    if triangles.ndim != 2 or triangles.shape[1] != 3 or np.any(triangles < 0) or np.any(triangles >= len(rest)):
        raise ValueError('Invalid triangle topology')
    if weights.shape != (len(rest), bones) or not np.isfinite(weights).all() or np.any(weights < -1e-10):
        raise ValueError('Invalid skinning weights')
    if not np.allclose(weights.sum(axis=1), 1.0, atol=1e-7, rtol=0) or np.any((weights > 1e-10).sum(axis=1) > 4):
        raise ValueError('Weights must be normalized and use at most 4 influences')
    # Omit numerically negligible influences without accidentally exporting a
    # fifth group. Renormalize exactly the weights that Blender will receive.
    weights = np.where(weights > 1e-10, weights, 0.0)
    weights /= weights.sum(axis=1, keepdims=True)
    if uvs is not None and np.asarray(uvs).shape != (len(rest), 2):
        raise ValueError('UVs must provide one pair per rest vertex')
    channels = [continuous_affine_channels(fit.affines[:, bone, :, :2]) for bone in range(bones)]
    existing = {kind: set(getattr(bpy.data, kind)) for kind in ('objects', 'meshes', 'armatures', 'materials', 'actions', 'collections')}
    original_frame = context.scene.frame_current
    try:
        if context.object and context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        collection = bpy.data.collections.new(name)
        context.scene.collection.children.link(collection)
        data = bpy.data.armatures.new(name + ' Armature')
        rig = bpy.data.objects.new(name + ' Armature', data)
        collection.objects.link(rig)
        rig.show_in_front = True
        rig['skinning_method'] = 'Offline sparse affine fit; helper + deform bone per influence'
        rig['logical_bones'] = bones
        rig['requires_live2d_runtime'] = False
        bpy.ops.object.select_all(action='DESELECT')
        rig.select_set(True);context.view_layer.objects.active = rig
        bpy.ops.object.mode_set(mode='EDIT')
        length = max(float(np.ptp(rest[:, :2], axis=0).max()) * 0.04, 0.01)
        for bone in range(bones):
            center = fit.centers[bone]
            helper = data.edit_bones.new(f'Affine {bone:03d}')
            helper.head = tuple(center);helper.tail = tuple(center + np.array([0.0, length, 0.0]))
            helper.use_deform = False
            deform = data.edit_bones.new(f'Skin {bone:03d}')
            deform.head = helper.head;deform.tail = helper.tail
            deform.parent = helper;deform.use_connect = False;deform.use_deform = True
        bpy.ops.object.mode_set(mode='OBJECT')
        for bone in data.bones:
            bone.inherit_scale = 'FULL'
        mesh = bpy.data.meshes.new(name + ' Mesh')
        mesh.from_pydata(rest.tolist(), [], triangles.tolist());mesh.update()
        obj = bpy.data.objects.new(name + ' Mesh', mesh);collection.objects.link(obj)
        if uvs is not None:
            layer = mesh.uv_layers.new(name='UVMap')
            for loop in mesh.loops:
                layer.data[loop.index].uv = tuple(uvs[loop.vertex_index])
        if materials:
            for material in materials:
                copy = material.copy();copy.animation_data_clear()
                if copy.node_tree:copy.node_tree.animation_data_clear()
                mesh.materials.append(copy)
        if triangle_materials is not None:
            if len(triangle_materials) != len(mesh.polygons):
                raise ValueError('Triangle material assignments must match topology')
            for face, index in zip(mesh.polygons, triangle_materials):
                face.material_index = int(index)
        for bone in range(bones):
            group = obj.vertex_groups.new(name=f'Skin {bone:03d}')
            for vertex in np.flatnonzero(weights[:, bone] > 1e-12):
                group.add([int(vertex)], float(weights[vertex, bone]), 'REPLACE')
        modifier = obj.modifiers.new(name='Pure skeletal deformation', type='ARMATURE')
        modifier.object = rig
        modifier.use_deform_preserve_volume = False
        modifier.use_vertex_groups = True
        for bone in range(bones):
            helper = rig.pose.bones[f'Affine {bone:03d}'];deform = rig.pose.bones[f'Skin {bone:03d}']
            helper.rotation_mode = deform.rotation_mode = 'XYZ'
            center = fit.centers[bone, :2]
            for sample, frame in enumerate(frames):
                affine = fit.affines[sample, bone]
                au, sx, sy, av = channels[bone][sample]
                translated = affine[:, 2] + affine[:, :2] @ center - center
                helper.location = (float(translated[0]), float(translated[1]), float(fit.depth_offsets[sample, bone]))
                helper.rotation_euler = (0.0, 0.0, float(au));helper.scale = (float(sx), float(sy), 1.0)
                deform.rotation_euler = (0.0, 0.0, float(av))
                helper.keyframe_insert('location', frame=float(frame), group=helper.name)
                helper.keyframe_insert('rotation_euler', frame=float(frame), group=helper.name)
                helper.keyframe_insert('scale', frame=float(frame), group=helper.name)
                deform.keyframe_insert('rotation_euler', frame=float(frame), group=deform.name)
        _linearize(rig.animation_data.action)
        rig.animation_data.action.name = name + ' Skeletal Animation'
        context.scene.frame_set(original_frame)
        context.view_layer.update()
        bpy.ops.object.select_all(action='DESELECT');obj.select_set(True);context.view_layer.objects.active = obj
        result = {'mesh': obj, 'meshes': [obj], 'objects_by_id': {}, 'armature': rig,
                  'collection': collection, 'action': rig.animation_data.action,
                  'bone_depth_animated': bool(np.max(np.abs(fit.depth_offsets)) > 1e-9)}
        if pieces is not None:
            _split_pieces(result, pieces, triangles, uvs)
            rig.select_set(True)
            context.view_layer.objects.active = rig
        return result
    except Exception:
        if context.object and context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        for kind in ('objects', 'collections', 'meshes', 'armatures', 'materials', 'actions'):
            group = getattr(bpy.data, kind)
            for item in set(group) - existing[kind]:group.remove(item, do_unlink=True)
        raise


def _split_pieces(result, pieces, triangles, uvs):
    """Keep independent materials/layer tracks while sharing one skeleton."""
    source = result['mesh'];mesh = source.data
    vertices = [tuple(v.co) for v in mesh.vertices]
    bounds = []
    expected = 0
    for piece in pieces:
        start, count = int(piece['offset']), int(piece['count'])
        if start != expected or count < 0:
            raise ValueError('Drawable pieces must partition consecutive vertices')
        bounds.append(start + count);expected += count
    if expected != len(vertices) or len({p['id'] for p in pieces}) != len(pieces):
        raise ValueError('Drawable pieces must cover vertices and have unique IDs')
    # Every triangle must stay within one original Drawable.
    owners = np.searchsorted(np.asarray(bounds), triangles[:, 0], side='right')
    if not np.all(np.searchsorted(np.asarray(bounds), triangles, side='right') == owners[:, None]):
        raise ValueError('A triangle crosses Drawable boundaries')
    out = {}
    for index, piece in enumerate(pieces):
        start, count = int(piece['offset']), int(piece['count'])
        selected = np.flatnonzero(owners == index)
        data = bpy.data.meshes.new(str(piece['id']) + ' Skinned Mesh')
        data.from_pydata(vertices[start:start + count], [], (triangles[selected] - start).tolist());data.update()
        obj = bpy.data.objects.new(str(piece['id']) + ' Skinned', data)
        result['collection'].objects.link(obj)
        obj.matrix_world = source.matrix_world.copy()
        obj['source_drawable_id'] = str(piece['id'])
        obj['rest_layer_depth_in_vertices'] = True
        if uvs is not None:
            layer = data.uv_layers.new(name='UVMap')
            for loop in data.loops:
                layer.data[loop.index].uv = tuple(uvs[start + loop.vertex_index])
        material_ids = sorted({mesh.polygons[int(face)].material_index for face in selected})
        material_map = {}
        for material_id in material_ids:
            if material_id < len(mesh.materials):
                material_map[material_id] = len(data.materials)
                # Per-Drawable copies allow opacity/color keys without cross-talk.
                data.materials.append(mesh.materials[material_id].copy())
        for face, old_index in zip(data.polygons, selected):
            face.material_index = material_map.get(mesh.polygons[int(old_index)].material_index, 0)
        groups = {}
        for local in range(count):
            for membership in mesh.vertices[start + local].groups:
                name = source.vertex_groups[membership.group].name
                if name not in groups:groups[name] = obj.vertex_groups.new(name=name)
                groups[name].add([local], membership.weight, 'REPLACE')
        modifier = obj.modifiers.new(name='Pure skeletal deformation', type='ARMATURE')
        modifier.object = result['armature'];modifier.use_deform_preserve_volume = False
        out[str(piece['id'])] = obj
    temporary_materials = list(mesh.materials)
    bpy.data.objects.remove(source, do_unlink=True)
    bpy.data.meshes.remove(mesh)
    for material in temporary_materials:
        if material.users == 0:
            bpy.data.materials.remove(material)
    result['mesh'] = None
    result['meshes'] = list(out.values())
    result['objects_by_id'] = out
