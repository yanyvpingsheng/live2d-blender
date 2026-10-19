"""Blender mesh/material adapter and deterministic timeline bridge."""
import json
import hashlib
import uuid
from dataclasses import asdict
from pathlib import Path
import bpy
from bpy.app.handlers import persistent
from .assets import load_model3, read_json
from .core import NativeBackend
from .data import Live2DError, validate_snapshot
from .fixture import SyntheticBackend
from .motion import Motion, apply_expression

RUNTIMES = {}
_BUSY = False


def parameter_key(index):
    # Stable short keys handle long/unusual Live2D IDs without collisions or RNA injection.
    return f'l2d_p_{index:04d}'


def part_key(index):
    return f'l2d_part_{index:04d}'


def id_path(key):
    return '["' + bpy.utils.escape_identifier(key) + '"]'


def runtime_for_root(root):
    runtime = RUNTIMES.get(root.get('_l2d_uuid')) if root else None
    return runtime if runtime and runtime.root == root else None


def model_root(obj):
    while obj:
        if obj.get('_l2d_uuid'):
            return obj
        obj = obj.parent
    return None


def topology_fingerprint(backend, snapshot):
    """Identity of saved static geometry and control indexing, not pose values."""
    payload = {'parameters': [asdict(p) for p in backend.parameters],
               'parts': [asdict(p) for p in backend.parts],
               'drawables': [{'id': d.id, 'vertices': len(d.vertices), 'uvs': d.uvs,
                              'indices': d.indices, 'texture': d.texture, 'masks': d.masks,
                              'blend_mode': d.blend_mode, 'inverted_mask': d.inverted_mask,
                              'parent_part': d.parent_part, 'double_sided': d.double_sided}
                             for d in snapshot.drawables]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def make_material(drawable, image):
    material = bpy.data.materials.new(f'L2D {drawable.id}')
    material.use_nodes = True
    material.use_backface_culling = not drawable.double_sided
    if hasattr(material, 'surface_render_method'):
        material.surface_render_method = 'DITHERED'
    elif hasattr(material, 'blend_method'):
        material.blend_method = 'BLEND'
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    out = nodes.new('ShaderNodeOutputMaterial')
    out.location = (700, 0)
    tex = nodes.new('ShaderNodeTexImage')
    tex.name = 'L2D Texture'
    tex.image = image
    tex.extension = 'CLIP'
    tex.location = (-700, 100)
    multiply = nodes.new('ShaderNodeMixRGB')
    multiply.name = 'L2D Multiply'
    multiply.blend_type = 'MULTIPLY'
    multiply.inputs[0].default_value = 1
    multiply.location = (-450, 140)
    screen = nodes.new('ShaderNodeMixRGB')
    screen.name = 'L2D Screen'
    screen.blend_type = 'SCREEN'
    screen.inputs[0].default_value = 1
    screen.location = (-200, 140)
    emission = nodes.new('ShaderNodeEmission')
    emission.location = (50, 150)
    transparent = nodes.new('ShaderNodeBsdfTransparent')
    transparent.location = (50, -100)
    opacity = nodes.new('ShaderNodeMath')
    opacity.name = 'L2D Opacity'
    opacity.operation = 'MULTIPLY'
    opacity.location = (-200, -160)
    mix = nodes.new('ShaderNodeMixShader')
    mix.location = (420, 0)
    links.new(tex.outputs['Color'], multiply.inputs[1])
    links.new(multiply.outputs[0], screen.inputs[1])
    links.new(screen.outputs[0], emission.inputs['Color'])
    links.new(tex.outputs['Alpha'], opacity.inputs[0])
    links.new(opacity.outputs[0], mix.inputs[0])
    links.new(transparent.outputs[0], mix.inputs[1])
    links.new(emission.outputs[0], mix.inputs[2])
    links.new(mix.outputs[0], out.inputs['Surface'])
    set_material(material, drawable)
    return material


def set_material(material, drawable, model_opacity=1.0):
    nodes = material.node_tree.nodes
    nodes['L2D Multiply'].inputs[2].default_value = drawable.multiply_color
    nodes['L2D Screen'].inputs[2].default_value = drawable.screen_color
    nodes['L2D Opacity'].inputs[1].default_value = max(0.0, min(1.0, drawable.opacity * model_opacity)) if drawable.visible else 0.0


def check_render_support(snapshot, allow_lossy):
    issues = []
    masks = sum(bool(d.masks) for d in snapshot.drawables)
    blends = sum(d.blend_mode != 0 for d in snapshot.drawables)
    if masks:
        issues.append(f'{masks} masked drawables: clipping masks are preserved as metadata but NOT rendered')
    if blends:
        issues.append(f'{blends} non-normal blend drawables: rendered as normal alpha ONLY')
    if issues and not allow_lossy:
        raise Live2DError('; '.join(issues) + '. Enable explicitly lossy preview only if this is acceptable.')
    return issues


class Runtime:
    def __init__(self, root, backend, objects):
        self.root = root
        self.backend = backend
        self.objects = objects
        self.last_signature = None
        self.cache = {}
        self.physics = None

    def cached_json(self, path):
        path = Path(bpy.path.abspath(path)).expanduser().resolve()
        stamp = (str(path), path.stat().st_mtime_ns)
        if stamp not in self.cache:
            self.cache = {k: v for k, v in self.cache.items() if k[0] != str(path)}
            self.cache[stamp] = read_json(path)
        return self.cache[stamp]

    def evaluate(self, scene, depsgraph=None, force=False):
        root = self.root
        # Animation is evaluated by Blender before frame_change_post.
        evaluated = root.evaluated_get(depsgraph) if depsgraph else root
        parameters = {p.id: float(evaluated.get(parameter_key(i), p.default)) for i, p in enumerate(self.backend.parameters)}
        parts = {p.id: float(evaluated.get(part_key(i), p.opacity)) for i, p in enumerate(self.backend.parts)}
        settings = root.live2d
        seconds = (scene.frame_current_final - settings.start_frame) / (scene.render.fps / scene.render.fps_base)
        signature = (tuple(parameters.items()), tuple(parts.items()), settings.motion_path, settings.expression_path,
                     settings.expression_weight, settings.play_motion, settings.start_frame, settings.depth_step,
                     scene.frame_current_final, scene.render.fps, scene.render.fps_base,
                     settings.physics_enabled, settings.simulation_fps, settings.simulation_max_steps, id(self.physics))
        if signature == self.last_signature and not force:
            return
        motion = Motion.from_json(self.cached_json(settings.motion_path)) if settings.play_motion and settings.motion_path else None
        expression = self.cached_json(settings.expression_path) if settings.expression_path else None
        fps = scene.render.fps / scene.render.fps_base
        action = root.animation_data.action if root.animation_data else None
        curves = {c.data_path: c for c in action.fcurves} if action and hasattr(action, 'fcurves') else {}
        if settings.physics_enabled and self.physics:
            if root.animation_data and (root.animation_data.use_nla and root.animation_data.nla_tracks):
                raise Live2DError('Physics replay currently supports direct Action curves, not NLA strips')
            if root.animation_data and root.animation_data.drivers:
                raise Live2DError('Physics replay currently supports direct Action curves, not drivers')

        def sample_inputs(t):
            sampled_params, sampled_parts = dict(parameters), dict(parts)
            # For physics replay, evaluate the control Action at each simulated time.
            if settings.physics_enabled and self.physics:
                frame = settings.start_frame + t * fps
                for i, parameter in enumerate(self.backend.parameters):
                    curve = curves.get(id_path(parameter_key(i)))
                    if curve:
                        sampled_params[parameter.id] = curve.evaluate(frame)
                sampled_parts = {p.id: parts[p.id] for p in self.backend.parts if abs(parts[p.id] - p.opacity) > 1e-7}
                for i, part in enumerate(self.backend.parts):
                    curve = curves.get(id_path(part_key(i)))
                    if curve:
                        sampled_parts[part.id] = curve.evaluate(frame)
            opacity = 1.0
            if motion and t >= 0:
                sample = motion.sample(t)
                sampled_params.update(sample['Parameter'])
                sampled_parts.update(sample['PartOpacity'])
                opacity = sample['Model'].get('Opacity', 1.0)
            if expression:
                sampled_params = apply_expression(sampled_params, expression, settings.expression_weight)
            return sampled_params, sampled_parts, opacity

        if settings.physics_enabled:
            if not self.physics:
                raise Live2DError('Connect the official Physics/Pose bridge explicitly before enabling it')
            from .timeline import replay_physics
            parameters, parts, model_opacity = replay_physics(self.physics, sample_inputs, seconds,
                                                             fps=settings.simulation_fps,
                                                             max_steps=settings.simulation_max_steps)
        else:
            parameters, parts, model_opacity = sample_inputs(seconds)
        self.last_model_opacity = model_opacity
        self.last_parameters = dict(parameters)
        snapshot = self.backend.evaluate(parameters, parts)
        if not getattr(self.backend, 'trusted_validated_snapshot', False):
            validate_snapshot(snapshot)
        self.last_snapshot = snapshot
        if len(snapshot.drawables) != len(self.objects):
            raise Live2DError('Backend topology changed; reimport the model')
        for drawable, obj in zip(snapshot.drawables, self.objects):
            if len(obj.data.vertices) != len(drawable.vertices):
                raise Live2DError(f'Topology changed on {drawable.id}; reimport required')
            obj.data.vertices.foreach_set('co', [c for xy in drawable.vertices for c in (xy[0], xy[1], 0.0)])
            obj.data.update()
            obj.location.z = drawable.render_order * settings.depth_step
            # Do not hide objects in depsgraph: alpha=0 safely handles invisibility and masks metadata.
            set_material(obj.data.materials[0], drawable, model_opacity)
        self.last_signature = signature
        if root.get('_l2d_runtime_error'):
            root['_l2d_runtime_error'] = ''
        return snapshot

    def connect_physics(self, library_path):
        package = getattr(self.backend, 'package', None)
        if not package:
            raise Live2DError('Official physics requires a real native model, not the synthetic fixture')
        physics_path = package.references.get('Physics')
        pose_path = package.references.get('Pose')
        if not physics_path and not pose_path:
            raise Live2DError('Model manifest has no physics3 or pose3 file')
        from .physics import OfficialPhysics
        candidate = OfficialPhysics(library_path, package.moc.read_bytes(), physics_path, pose_path)
        if candidate.core_version != self.backend.core.version:
            candidate.close()
            raise Live2DError('Framework bridge and mesh Core versions differ; build them from the same SDK')
        if self.physics:
            self.physics.close()
        self.physics = candidate
        self.root.live2d.physics_enabled = True
        self.root['_l2d_physics_state'] = 'OFFICIAL R5 CONNECTED'
        warnings = json.loads(self.root.get('_l2d_warnings', '[]'))
        warnings = [w for w in warnings if 'evaluation is not implemented' not in w]
        warnings.append('Official Physics/Pose bridge enabled; deterministic replay, no complete application parity certification.')
        self.root['_l2d_warnings'] = json.dumps(warnings)
        self.last_signature = None

    def close(self):
        if self.physics:
            self.physics.close()
            self.physics = None
        self.backend.close()


def create_model(context, backend, source='', allow_lossy=False):
    snapshot = validate_snapshot(backend.evaluate({}, {}))
    issues = check_render_support(snapshot, allow_lossy)
    # Validate textures before creating any Blender objects.
    for d in snapshot.drawables:
        if d.texture < 0 or d.texture >= len(backend.textures):
            raise Live2DError(f'{d.id}: texture index outside model texture list')
    for texture in backend.textures:
        if not Path(texture).is_file():
            raise Live2DError(f'Missing texture: {texture}')
    existing = {kind: set(getattr(bpy.data, kind)) for kind in ('objects', 'meshes', 'materials', 'images', 'collections', 'texts')}
    uid = str(uuid.uuid4())
    try:
        collection = bpy.data.collections.new(f'L2D {backend.name}')
        context.scene.collection.children.link(collection)
        root = bpy.data.objects.new(backend.name, None)
        collection.objects.link(root)
        root.empty_display_type = 'PLAIN_AXES'
        root.empty_display_size = 0.2
        root['_l2d_uuid'] = uid
        root['_l2d_kind'] = backend.kind
        root['_l2d_source'] = source
        root['_l2d_allow_lossy'] = bool(allow_lossy)
        root['_l2d_warnings'] = json.dumps(snapshot.warnings + issues, ensure_ascii=False)
        root['_l2d_parameters'] = json.dumps([asdict(p) for p in backend.parameters], ensure_ascii=False)
        root['_l2d_parts'] = json.dumps([asdict(p) for p in backend.parts], ensure_ascii=False)
        root['_l2d_canvas'] = json.dumps(backend.canvas)
        root['_l2d_display_info'] = json.dumps(getattr(backend, 'display_info', {}), ensure_ascii=False)
        root['_l2d_state'] = 'LIVE'
        root['_l2d_topology_fingerprint'] = topology_fingerprint(backend, snapshot)
        for i, p in enumerate(backend.parameters):
            key = parameter_key(i)
            root[key] = float(p.default)
            root.id_properties_ui(key).update(min=p.minimum, max=p.maximum, soft_min=p.minimum, soft_max=p.maximum, default=p.default, description=p.id)
        for i, p in enumerate(backend.parts):
            key = part_key(i)
            root[key] = float(p.opacity)
            root.id_properties_ui(key).update(min=0.0, max=1.0, default=p.opacity, description=p.id)
        report = bpy.data.texts.new(f'{backend.name} manifest.json')
        report.write(json.dumps({'source': source, 'model3': backend.metadata, 'warnings': snapshot.warnings + issues,
                                 'backend': backend.kind, 'canvas': backend.canvas}, indent=2, ensure_ascii=False))
        root['_l2d_manifest_text'] = report.name
        images = [bpy.data.images.load(str(p), check_existing=True) for p in backend.textures]
        for image in images:
            image.alpha_mode = 'STRAIGHT'
        part_objects = []
        for i, part in enumerate(backend.parts):
            obj = bpy.data.objects.new(part.id, None)
            collection.objects.link(obj)
            obj['_l2d_part_index'] = i
            obj['_l2d_part_id'] = part.id
            obj.empty_display_size = 0.03
            part_objects.append(obj)
        for i, part in enumerate(backend.parts):
            # Guard cyclic or invalid part ancestry from corrupt adapters.
            seen, parent = {i}, part.parent
            while parent >= 0:
                if parent in seen or parent >= len(backend.parts):
                    raise Live2DError('Invalid or cyclic part hierarchy')
                seen.add(parent)
                parent = backend.parts[parent].parent
            part_objects[i].parent = part_objects[part.parent] if part.parent >= 0 else root
        objects = []
        for i, d in enumerate(snapshot.drawables):
            mesh = bpy.data.meshes.new(d.id)
            vertices = [(x, y, 0) for x, y in d.vertices]
            triangles = [d.indices[j:j + 3] for j in range(0, len(d.indices), 3)]
            mesh.from_pydata(vertices, [], triangles)
            uv = mesh.uv_layers.new(name='Live2D UV')
            for loop in mesh.loops:
                uv.data[loop.index].uv = d.uvs[loop.vertex_index]
            mesh.update()
            obj = bpy.data.objects.new(d.id, mesh)
            collection.objects.link(obj)
            if not -1 <= d.parent_part < len(part_objects):
                raise Live2DError(f'{d.id}: invalid parent part index')
            obj.parent = part_objects[d.parent_part] if d.parent_part >= 0 else root
            obj['_l2d_drawable_index'] = i
            obj['_l2d_drawable_id'] = d.id
            obj['_l2d_mask_indices'] = json.dumps(d.masks)
            obj['_l2d_inverted_mask'] = d.inverted_mask
            obj['_l2d_blend_mode'] = d.blend_mode
            mesh.materials.append(make_material(d, images[d.texture]))
            objects.append(obj)
        runtime = Runtime(root, backend, objects)
        RUNTIMES[uid] = runtime
        runtime.evaluate(context.scene, force=True)
        bpy.ops.object.select_all(action='DESELECT')
        root.select_set(True)
        context.view_layer.objects.active = root
        context.scene.render.use_lock_interface = True
        return root
    except Exception:
        RUNTIMES.pop(uid, None)
        backend.close()
        # Transactional rollback: do not leave half an imported model in the scene.
        for kind in ('objects', 'collections', 'meshes', 'materials', 'images', 'texts'):
            group = getattr(bpy.data, kind)
            for item in set(group) - existing[kind]:
                group.remove(item, do_unlink=True)
        raise


def reconnect(context, root, library_path=''):
    uid = root['_l2d_uuid']
    if uid in RUNTIMES and RUNTIMES[uid].root != root:
        uid = str(uuid.uuid4())
        root['_l2d_uuid'] = uid
    if uid in RUNTIMES:
        RUNTIMES.pop(uid).close()
    root['_l2d_state'] = 'DISCONNECTED'
    root.live2d.physics_enabled = False
    if root['_l2d_kind'] == 'SYNTHETIC':
        backend = SyntheticBackend()
    elif root['_l2d_kind'] == 'NATIVE':
        backend = NativeBackend(load_model3(bpy.path.abspath(root['_l2d_source'])), library_path)
    else:
        raise Live2DError('This is a baked model; no runtime is needed')
    objects = sorted([o for o in root.children_recursive if '_l2d_drawable_index' in o], key=lambda o: o['_l2d_drawable_index'])
    try:
        snapshot = backend.evaluate({}, {})
        saved = root.get('_l2d_topology_fingerprint')
        if not saved or saved != topology_fingerprint(backend, snapshot):
            raise Live2DError('Source topology, UVs or control definitions changed (or old scene lacks fingerprint); reimport instead')
        if [d.id for d in snapshot.drawables] != [o['_l2d_drawable_id'] for o in objects]:
            raise Live2DError('Model identity/topology differs from saved scene; reimport instead')
        check_render_support(snapshot, bool(root.get('_l2d_allow_lossy')))
        runtime = Runtime(root, backend, objects)
        root.live2d.physics_enabled = False
        runtime.evaluate(context.scene, force=True)
        RUNTIMES[uid] = runtime
        root['_l2d_state'] = 'LIVE'
    except Exception:
        backend.close()
        raise


@persistent
def update_models(scene, depsgraph=None):
    global _BUSY
    if _BUSY:
        return
    _BUSY = True
    try:
        for uid, runtime in list(RUNTIMES.items()):
            try:
                root = runtime.root
                if root.name not in scene.objects:
                    continue
                if root.get('_l2d_state') == 'LIVE':
                    runtime.evaluate(scene, depsgraph)
            except ReferenceError:
                RUNTIMES.pop(uid, None)
                runtime.close()
            except Exception as error:
                message = str(error)
                if runtime.root.get('_l2d_runtime_error') != message:
                    print('[Live2D] Runtime stopped:', message)
                    runtime.root['_l2d_runtime_error'] = message
                runtime.root['_l2d_state'] = 'ERROR'
    finally:
        _BUSY = False


@persistent
def clear_runtimes(_dummy=None):
    for runtime in RUNTIMES.values():
        runtime.close()
    RUNTIMES.clear()
    # Never load native libraries automatically from a .blend or execute embedded code.
    for obj in bpy.data.objects:
        if obj.get('_l2d_uuid') and obj.get('_l2d_state') == 'LIVE':
            obj['_l2d_state'] = 'DISCONNECTED'



@persistent
def mark_loaded_disconnected(_dummy=None):
    # load_pre sees the old scene; load_post must mark the newly opened file.
    for obj in bpy.data.objects:
        if obj.get('_l2d_uuid') and obj.get('_l2d_kind') in ('NATIVE', 'SYNTHETIC'):
            obj['_l2d_state'] = 'DISCONNECTED'
            obj.live2d.physics_enabled = False
            obj['_l2d_physics_state'] = 'DISCONNECTED'


def linearize(action):
    if action and hasattr(action, 'fcurves'):
        for curve in action.fcurves:
            for key in curve.keyframe_points:
                key.interpolation = 'LINEAR'


def bake_copy(context, root, start, end, step=1):
    """Sample full evaluated vertex states into per-frame shape keys on a NEW copy.

    This bakes the timeline, not the full multi-dimensional Live2D parameter space.
    Only currently supported preview shading is baked. Source is kept intact.
    """
    global _BUSY
    if end < start or step < 1:
        raise Live2DError('Invalid bake range')
    runtime = runtime_for_root(root)
    if not runtime:
        raise Live2DError('Reconnect runtime before baking')
    frames = list(range(start, end + 1, step))
    if frames[-1] != end:
        frames.append(end)
    count = sum(len(o.data.vertices) for o in runtime.objects) * len(frames)
    if count > 5_000_000:
        raise Live2DError('Bake exceeds 5 million vertex samples; reduce range/step or use a future cache exporter')
    existing_bake = {kind: set(getattr(bpy.data, kind)) for kind in ('objects', 'meshes', 'materials', 'collections', 'actions')}
    collection = bpy.data.collections.new(root.name + ' Baked')
    context.scene.collection.children.link(collection)
    baked_root = bpy.data.objects.new(root.name + ' Baked', None)
    collection.objects.link(baked_root)
    baked_root.matrix_world = root.matrix_world.copy()
    baked_root['_l2d_uuid'] = str(uuid.uuid4())
    baked_root['_l2d_kind'] = 'BAKED'
    baked_root['_l2d_state'] = 'BAKED'
    baked_root['_l2d_warnings'] = root.get('_l2d_warnings', '[]')
    baked_root['_l2d_bake'] = json.dumps({'start': start, 'end': end, 'step': step, 'source': root.name,
                                         'note': 'Sampled timeline only; not an editable Cubism rig.'})
    copies = []
    original_frame = context.scene.frame_current
    previous_busy = _BUSY
    _BUSY = True
    try:
        for source in runtime.objects:
            target = source.copy()
            target.data = source.data.copy()
            target.animation_data_clear()
            target.name = source.name + ' Baked'
            collection.objects.link(target)
            target.parent = baked_root
            target.matrix_parent_inverse.identity()
            target.matrix_basis = root.matrix_world.inverted() @ source.matrix_world
            for index, material in enumerate(target.data.materials):
                target.data.materials[index] = material.copy()
            target.shape_key_add(name='Basis')
            copies.append(target)
        for index, frame in enumerate(frames):
            context.scene.frame_set(frame)
            runtime.evaluate(context.scene, context.evaluated_depsgraph_get(), force=True)
            for source, target in zip(runtime.objects, copies):
                key = target.shape_key_add(name=f'Frame {frame}')
                coords = [0.0] * (len(source.data.vertices) * 3)
                source.data.vertices.foreach_get('co', coords)
                key.data.foreach_set('co', coords)
                if index > 0:
                    key.value = 0.0
                    key.keyframe_insert('value', frame=frames[index - 1])
                key.value = 1.0
                key.keyframe_insert('value', frame=frame)
                if index + 1 < len(frames):
                    key.value = 0.0
                    key.keyframe_insert('value', frame=frames[index + 1])
                target.location.z = source.location.z
                target.keyframe_insert('location', index=2, frame=frame)
                srcnodes = source.data.materials[0].node_tree.nodes
                tree = target.data.materials[0].node_tree
                for name, socket in (('L2D Opacity', 1), ('L2D Multiply', 2), ('L2D Screen', 2)):
                    dst = tree.nodes[name].inputs[socket]
                    value = srcnodes[name].inputs[socket].default_value
                    dst.default_value = value
                    dst.keyframe_insert('default_value', frame=frame)
        for source, target in zip(runtime.objects, copies):
            linearize(target.data.shape_keys.animation_data.action)
            linearize(target.animation_data.action)
            tree = target.data.materials[0].node_tree
            linearize(tree.animation_data.action)
            source.hide_set(True)
            source.hide_render = True
        baked_root['_l2d_vertex_samples'] = count
        return baked_root
    except Exception:
        # Keep the source intact on a failed bake. Newly-created disposable output is removed.
        for kind in ('objects', 'collections', 'meshes', 'materials', 'actions'):
            group = getattr(bpy.data, kind)
            for item in set(group) - existing_bake[kind]:
                group.remove(item, do_unlink=True)
        raise
    finally:
        try:
            context.scene.frame_set(original_frame)
            runtime.evaluate(context.scene, context.evaluated_depsgraph_get(), force=True)
        except Exception as restore_error:
            # A restoration failure must not suppress the primary bake error or lock all models.
            root['_l2d_runtime_error'] = str(restore_error)
            root['_l2d_state'] = 'ERROR'
        finally:
            _BUSY = previous_busy


def register_handlers():
    for handlers, fn in ((bpy.app.handlers.frame_change_post, update_models),
                         (bpy.app.handlers.depsgraph_update_post, update_models),
                         (bpy.app.handlers.load_pre, clear_runtimes),
                         (bpy.app.handlers.load_post, mark_loaded_disconnected),
                         (bpy.app.handlers.undo_pre, clear_runtimes),
                         (bpy.app.handlers.undo_post, mark_loaded_disconnected),
                         (bpy.app.handlers.redo_pre, clear_runtimes),
                         (bpy.app.handlers.redo_post, mark_loaded_disconnected)):
        if fn not in handlers:
            handlers.append(fn)


def unregister_handlers():
    for handlers, fn in ((bpy.app.handlers.frame_change_post, update_models),
                         (bpy.app.handlers.depsgraph_update_post, update_models),
                         (bpy.app.handlers.load_pre, clear_runtimes),
                         (bpy.app.handlers.load_post, mark_loaded_disconnected),
                         (bpy.app.handlers.undo_pre, clear_runtimes),
                         (bpy.app.handlers.undo_post, mark_loaded_disconnected),
                         (bpy.app.handlers.redo_pre, clear_runtimes),
                         (bpy.app.handlers.redo_post, mark_loaded_disconnected)):
        if fn in handlers:
            handlers.remove(fn)
    clear_runtimes()
