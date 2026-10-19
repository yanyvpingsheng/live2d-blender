"""Bake sampled non-geometric channels into ordinary Blender data.

Conversion-time helper only. Saved materials, node groups and F-curves play in
Blender without this module, Core, drivers, frame/depsgraph handlers, or geometry caches.
Pass exported per-Drawable meshes, NEVER live source objects. XY location curves,
mesh vertices, vertex weights and armature actions are not modified.

Color math matches legacy per-Drawable multiply/screen operations in encoded
sRGB: Blender's scene-linear texture color is encoded, tinted, then decoded to
scene-linear emission with exact piecewise transfer functions. Texture alpha,
Drawable opacity and visibility remain separate. This is not a complete match
for encoded-space multi-layer SDK compositing. Optional legacy add/multiply use
scene-linear Cycles shader arithmetic; multiply requires an opaque backdrop.
Alpha-only masks are attached by the separate conversion-time mask baker.
"""
import math
import numpy as np

from .data import Live2DError

__all__ = ['validate_channels', 'bake_native_effects', 'make_native_material']


def validate_channels(pieces, frames, channels):
    """Validate/copy lightweight arrays from SampledAnimation.channels (no bpy)."""
    frames = np.asarray(frames, dtype=np.float64)
    if frames.ndim != 1 or not len(frames) or not np.isfinite(frames).all() or np.any(np.diff(frames) <= 0):
        raise Live2DError('Effect frames must be finite and strictly increasing')
    ids = [piece['id'] for piece in pieces]
    if len(set(ids)) != len(ids):
        raise Live2DError('Effect pieces contain duplicate Drawable IDs')
    shapes = {'opacity': (len(frames), len(ids)), 'visible': (len(frames), len(ids)),
              'multiply_color': (len(frames), len(ids), 4), 'screen_color': (len(frames), len(ids), 4),
              'render_order': (len(frames), len(ids))}
    normalized = {}
    for key, shape in shapes.items():
        if key not in channels:
            raise Live2DError(f'Missing sampled effect channel: {key}')
        array = np.asarray(channels[key], dtype=np.float64)
        if array.shape != shape or not np.isfinite(array).all():
            raise Live2DError(f'{key} must have finite shape {shape}')
        if key == 'render_order':
            if np.any(array != np.round(array)):
                raise Live2DError('render_order values must be integers')
        elif np.any(array < -1e-6) or np.any(array > 1 + 1e-6):
            raise Live2DError(f'{key} values must be in [0, 1]')
        elif key != 'visible':
            # Core interpolation can overshoot 1 by a few float32 ULPs.
            array = np.clip(array, 0, 1)
        if key == 'visible' and np.any((array != 0) & (array != 1)):
            raise Live2DError('visible must contain boolean values')
        normalized[key] = array.copy()
    return frames.copy(), normalized


def _collections(owner):
    """Obtain only this datablock's action slot, supporting legacy and 4.4+."""
    animation = owner.animation_data
    if not animation or not animation.action:
        return []
    action = animation.action
    slot = getattr(animation, 'action_slot', None)
    if slot is not None and hasattr(action, 'layers'):
        collections = []
        for layer in action.layers:
            for strip in layer.strips:
                if hasattr(strip, 'channelbag'):
                    bag = strip.channelbag(slot)
                    if bag is not None:
                        collections.append(bag.fcurves)
        if collections:
            return collections
    return [action.fcurves] if hasattr(action, 'fcurves') else []


def _curves(owner):
    return [curve for collection in _collections(owner) for curve in collection]


def _math(nodes, links, operation, a, b=None):
    node = nodes.new('ShaderNodeMath')
    node.operation = operation
    for socket, value in zip(node.inputs, (a, b)):
        if value is None:
            continue
        if isinstance(value, (int, float)):
            socket.default_value = value
        else:
            links.new(value, socket)
    return node.outputs[0]


def _transfer_group(kind):
    import bpy
    signature = 'native_effects_piecewise_srgb_v1_' + kind
    for existing in bpy.data.node_groups:
        if existing.get('_native_effects_transfer') == signature:
            return existing
    group = bpy.data.node_groups.new('Native Effects ' + kind, 'ShaderNodeTree')
    group['_native_effects_transfer'] = signature
    if hasattr(group, 'interface'):
        group.interface.new_socket(name='Color', in_out='INPUT', socket_type='NodeSocketColor')
        group.interface.new_socket(name='Color', in_out='OUTPUT', socket_type='NodeSocketColor')
    else:
        group.inputs.new('NodeSocketColor', 'Color')
        group.outputs.new('NodeSocketColor', 'Color')
    nodes, links = group.nodes, group.links
    incoming = nodes.new('NodeGroupInput'); outgoing = nodes.new('NodeGroupOutput')
    split = nodes.new('ShaderNodeSeparateColor'); split.mode = 'RGB'
    combine = nodes.new('ShaderNodeCombineColor'); combine.mode = 'RGB'
    links.new(incoming.outputs['Color'], split.inputs['Color'])
    links.new(combine.outputs['Color'], outgoing.inputs['Color'])
    for channel in range(3):
        x = split.outputs[channel]
        if kind == 'Linear to sRGB':
            condition = _math(nodes, links, 'LESS_THAN', x, 0.0031308)
            low = _math(nodes, links, 'MULTIPLY', x, 12.92)
            positive = _math(nodes, links, 'MAXIMUM', x, 0.0)
            power = _math(nodes, links, 'POWER', positive, 1.0 / 2.4)
            high = _math(nodes, links, 'SUBTRACT', _math(nodes, links, 'MULTIPLY', power, 1.055), 0.055)
        else:
            condition = _math(nodes, links, 'LESS_THAN', x, 0.04045)
            low = _math(nodes, links, 'DIVIDE', x, 12.92)
            offset = _math(nodes, links, 'ADD', x, 0.055)
            scaled = _math(nodes, links, 'DIVIDE', offset, 1.055)
            high = _math(nodes, links, 'POWER', _math(nodes, links, 'MAXIMUM', scaled, 0.0), 2.4)
        low_term = _math(nodes, links, 'MULTIPLY', low, condition)
        high_term = _math(nodes, links, 'MULTIPLY', high, _math(nodes, links, 'SUBTRACT', 1.0, condition))
        links.new(_math(nodes, links, 'ADD', low_term, high_term), combine.inputs[channel])
    return group


def make_native_material(image, name='Native Effects', *, double_sided=True, color_math='srgb',
                         blend_mode=0, allow_opaque_multiply=False):
    """New standalone emission/transmission material; never edits input image.

    blend_mode 0 is normal alpha, 1 adds radiance without attenuating the
    destination, and 2 tints transmission. Mode 2 requires explicit acceptance
    of opaque-backdrop-only correctness; it changes fractional backdrop alpha.
    Add/multiply arithmetic is scene-linear, with no per-Drawable UNORM clamp.
    The named NE Masked Alpha node accepts baked coverage on input 1.

    The texture Color socket must supply scene-linear straight RGB. Use properly
    tagged sRGB FILE images, or generated float images explicitly decoded to
    linear, premultiplied by alpha, and labeled Linear Rec.709 / PREMUL.
    Do not pass Non-Color premultiplied input.
    color_math='linear' intentionally applies tint math in scene-linear space;
    'srgb' is the default legacy encoded-space per-Drawable tinting behavior.
    """
    import bpy
    if color_math not in ('srgb', 'linear'):
        raise Live2DError('color_math must be srgb or linear')
    if blend_mode not in (0, 1, 2):
        raise Live2DError('Only legacy normal/add/multiply blend modes are supported')
    if blend_mode == 2 and not allow_opaque_multiply:
        raise Live2DError('Multiply needs explicit opaque-backdrop acceptance; transparent RGBA export is not supported')
    if image is None:
        raise Live2DError('A texture image is required for each exported Drawable')
    if image.colorspace_settings.is_data:
        raise Live2DError('Non-Color/data images are ambiguous for tinting; supply a scene-linear Color texture')
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    material.use_backface_culling = not double_sided
    if hasattr(material, 'surface_render_method'):
        material.surface_render_method = 'DITHERED'
    elif hasattr(material, 'blend_method'):
        material.blend_method = 'BLEND'
    material['_native_effects_version'] = 2
    material['_native_blend_mode'] = blend_mode
    material['_native_multiply_requires_opaque_backdrop'] = blend_mode == 2
    material['_native_blend_space'] = 'scene-linear'
    material['_native_effects_color_math'] = color_math
    nodes, links = material.node_tree.nodes, material.node_tree.links
    nodes.clear()
    output = nodes.new('ShaderNodeOutputMaterial')
    texture = nodes.new('ShaderNodeTexImage'); texture.name = 'NE Texture'
    texture.image = image; texture.extension = 'EXTEND'
    multiply = nodes.new('ShaderNodeRGB'); multiply.name = 'NE Multiply'
    multiply.outputs[0].default_value = (1, 1, 1, 1)
    screen = nodes.new('ShaderNodeRGB'); screen.name = 'NE Screen'
    screen.outputs[0].default_value = (0, 0, 0, 1)
    opacity = nodes.new('ShaderNodeValue'); opacity.name = 'NE Opacity'; opacity.outputs[0].default_value = 1
    visible = nodes.new('ShaderNodeValue'); visible.name = 'NE Visibility'; visible.outputs[0].default_value = 1
    color = texture.outputs['Color']
    if color_math == 'srgb':
        encode = nodes.new('ShaderNodeGroup'); encode.node_tree = _transfer_group('Linear to sRGB')
        links.new(color, encode.inputs['Color']); color = encode.outputs['Color']
    product = nodes.new('ShaderNodeMixRGB'); product.name = 'NE Color Multiply'; product.blend_type = 'MULTIPLY'
    product.inputs[0].default_value = 1
    links.new(color, product.inputs[1]); links.new(multiply.outputs[0], product.inputs[2])
    screened = nodes.new('ShaderNodeMixRGB'); screened.name = 'NE Color Screen'; screened.blend_type = 'SCREEN'
    screened.inputs[0].default_value = 1
    links.new(product.outputs[0], screened.inputs[1]); links.new(screen.outputs[0], screened.inputs[2])
    color = screened.outputs[0]
    if color_math == 'srgb':
        decode = nodes.new('ShaderNodeGroup'); decode.node_tree = _transfer_group('sRGB to Linear')
        links.new(color, decode.inputs['Color']); color = decode.outputs['Color']
    emission = nodes.new('ShaderNodeEmission'); links.new(color, emission.inputs['Color'])
    transparent = nodes.new('ShaderNodeBsdfTransparent')
    alpha = _math(nodes, links, 'MULTIPLY', texture.outputs['Alpha'], opacity.outputs[0])
    alpha = _math(nodes, links, 'MULTIPLY', alpha, visible.outputs[0])
    masked = nodes.new('ShaderNodeMath'); masked.name = 'NE Masked Alpha'
    masked.operation = 'MULTIPLY'; masked.inputs[1].default_value = 1.0
    links.new(alpha, masked.inputs[0]); alpha = masked.outputs[0]
    if blend_mode == 0:
        mix = nodes.new('ShaderNodeMixShader')
        links.new(alpha, mix.inputs[0]); links.new(transparent.outputs[0], mix.inputs[1]); links.new(emission.outputs[0], mix.inputs[2])
        surface = mix.outputs[0]
    elif blend_mode == 1:
        # A black additive texel must leave the backdrop unchanged, not cover it.
        links.new(alpha, emission.inputs['Strength'])
        add = nodes.new('ShaderNodeAddShader'); add.name = 'NE Legacy Add'
        links.new(transparent.outputs[0], add.inputs[0]); links.new(emission.outputs[0], add.inputs[1])
        surface = add.outputs[0]
    else:
        factor = nodes.new('ShaderNodeMixRGB'); factor.name = 'NE Opaque Multiply'
        factor.blend_type = 'MIX'; factor.inputs[1].default_value = (1, 1, 1, 1)
        links.new(alpha, factor.inputs[0]); links.new(color, factor.inputs[2])
        links.new(factor.outputs[0], transparent.inputs['Color'])
        surface = transparent.outputs[0]
    if not double_sided:
        geometry = nodes.new('ShaderNodeNewGeometry'); cull = nodes.new('ShaderNodeMixShader')
        passthrough = nodes.new('ShaderNodeBsdfTransparent')
        links.new(geometry.outputs['Backfacing'], cull.inputs[0])
        links.new(surface, cull.inputs[1]); links.new(passthrough.outputs[0], cull.inputs[2]); surface = cull.outputs[0]
    links.new(surface, output.inputs['Surface'])
    return material


def _image_for(obj, piece, images):
    if images is not None:
        try:
            return images[piece['texture']]
        except (KeyError, IndexError, TypeError) as error:
            raise Live2DError(f'{piece["id"]}: missing texture image') from error
    for slot in obj.material_slots:
        if slot.material and slot.material.use_nodes:
            for node in slot.material.node_tree.nodes:
                if node.type == 'TEX_IMAGE' and node.image:
                    return node.image
    raise Live2DError(f'{piece["id"]}: no texture; pass images indexed by texture number')


def bake_native_effects(objects_by_id, frames, channels, pieces, images=None, *,
                        depth_step=0.0001, color_math='srgb', bone_depth_animated=False,
                        allow_unsupported=False, replace_z_animation=False, native_blending=False,
                        masks_baked=False, allow_opaque_multiply=False):
    """Bake channels aligned [sample, piece] onto separate exported meshes.

    Samples use sampling.SampledAnimation.channels. Rest mesh geometry already
    contains the first sample's render_order * depth_step. Z keys add only order
    DELTAS to each object's existing local Z. Do not also bake dynamic layer depth
    into the armature: bone_depth_animated=True explicitly prevents double use.

    Existing exported material slots are replaced with new standalone materials.
    Only object.location[2] is keyed; location X/Y, rotations, weights, modifiers
    and armature actions are preserved. Opacity and RGB tint curves are LINEAR;
    visibility and render order are CONSTANT. Static channels get no F-curves.
    native_blending=True emits legacy blend nodes; multiply additionally requires
    allow_opaque_multiply=True. masks_baked=True is reserved for a caller that
    attaches coverage with mask_bake before exposing the result. Remaining
    unknown behavior requires allow_unsupported=True and is reported as a
    normal-alpha approximation. Unknown blend IDs are never guessed in native mode.
    """
    frames, channels = validate_channels(pieces, frames, channels)
    if not math.isfinite(depth_step) or depth_step <= 0:
        raise Live2DError('depth_step must be finite and positive')
    if color_math not in ('srgb', 'linear'):
        raise Live2DError('color_math must be srgb or linear')
    ordering_changes = bool(np.any(channels['render_order'] != channels['render_order'][0]))
    if bone_depth_animated and ordering_changes:
        raise Live2DError('Render order is already animated in bone depth; object Z would apply it twice')
    if native_blending and any(p.get('blend_mode', 0) not in (0, 1, 2) for p in pieces):
        raise Live2DError('Native materials support only legacy normal/add/multiply blend modes')
    if native_blending and not allow_opaque_multiply and any(p.get('blend_mode', 0) == 2 for p in pieces):
        raise Live2DError('Multiply needs explicit opaque-backdrop acceptance')
    unsupported = {'masks': [p['id'] for p in pieces if p.get('masks') and not masks_baked],
                   'blend_modes': [p['id'] for p in pieces if p.get('blend_mode', 0) != 0 and not native_blending]}
    if any(unsupported.values()) and not allow_unsupported:
        raise Live2DError('Complex masks/non-normal Drawable blend modes need a separate native solution; explicitly allow approximation to bake these channels alone')
    active = []
    for index, piece in enumerate(pieces):
        if piece.get('count', 1) == 0:
            continue
        obj = objects_by_id.get(piece['id'])
        if obj is None or obj.type != 'MESH':
            raise Live2DError(f'{piece["id"]}: expected a separate exported mesh object')
        if any(obj is other[1] for other in active):
            raise Live2DError('Each Drawable needs a distinct mesh object for independent layer order')
        if obj.data.users > 1:
            raise Live2DError(f'{piece["id"]}: shared mesh data; make the exported mesh single-user first')
        if obj.animation_data and (obj.animation_data.nla_tracks or obj.animation_data.drivers):
            raise Live2DError(f'{piece["id"]}: object NLA/drivers are not supported by this Z bake')
        changes_order = np.any(channels['render_order'][:, index] != channels['render_order'][0, index])
        old_z = [c for c in _curves(obj) if c.data_path == 'location' and c.array_index == 2]
        if changes_order and old_z and not replace_z_animation:
            raise Live2DError(f'{piece["id"]}: existing Z animation; explicitly request replacement')
        image = _image_for(obj, piece, images)
        if image is None or image.colorspace_settings.is_data:
            raise Live2DError(f'{piece["id"]}: provide a correctly tagged color texture')
        active.append((index, obj, piece, image, float(obj.location.z), changes_order))
    result = {'objects': len(active), 'animated': {key: [] for key in channels},
              'material_actions': 0, 'object_z_actions': 0, 'unsupported': unsupported,
              'color_math': color_math, 'frames': frames.tolist(), 'runtime_dependencies': [],
              'warnings': ['Per-Drawable native tinting and opacity are baked; this does not certify complete multi-layer SDK compositing parity.']}
    result['native_blending'] = native_blending
    result['blend_counts'] = {str(mode): sum(p.get('blend_mode', 0) == mode for p in pieces) for mode in (0, 1, 2)}
    result['multiply_requires_opaque_backdrop'] = bool(native_blending and result['blend_counts']['2'])
    if native_blending:
        result['warnings'].append('Cycles-only legacy blend approximation: scene-linear blending, no per-Drawable saturation; multiply requires an opaque backdrop. Transparent exports are not generally SDK-equivalent.')
    if any(unsupported.values()):
        result['warnings'].append('Masks and non-normal Drawable blend modes remain explicit normal-alpha approximations.')
    for index, obj, piece, image, original_z, changes_order in active:
        mat = make_native_material(image, 'Native ' + piece['id'], double_sided=piece.get('double_sided', True), color_math=color_math,
                                   blend_mode=piece.get('blend_mode', 0) if native_blending else 0,
                                   allow_opaque_multiply=allow_opaque_multiply)
        obj.data.materials.clear(); obj.data.materials.append(mat)
        for polygon in obj.data.polygons:
            polygon.material_index = 0
        nodes = mat.node_tree.nodes
        mapping = {'opacity': 'NE Opacity', 'visible': 'NE Visibility',
                   'multiply_color': 'NE Multiply', 'screen_color': 'NE Screen'}
        for channel, node_name in mapping.items():
            socket = nodes[node_name].outputs[0]
            values = channels[channel][:, index]
            colors = values.ndim == 2
            if colors:
                values = values[:, :3]  # Color alpha is deliberately ignored.
                socket.default_value = (*values[0], 1.0)
            else:
                socket.default_value = float(values[0])
            changing = bool(np.any(np.abs(values - values[0]) > 1e-6))
            if changing:
                result['animated'][channel].append(piece['id'])
                for frame, value in zip(frames, values):
                    if colors:
                        socket.default_value = (*value, 1.0)
                        for component in range(3):
                            socket.keyframe_insert('default_value', index=component, frame=float(frame), group='Native Effects')
                    else:
                        socket.default_value = float(value)
                        socket.keyframe_insert('default_value', frame=float(frame), group='Native Effects')
        for curve in _curves(mat.node_tree):
            interpolation = 'CONSTANT' if 'NE Visibility' in curve.data_path else 'LINEAR'
            for point in curve.keyframe_points:
                point.interpolation = interpolation
            curve.extrapolation = 'CONSTANT'
        if mat.node_tree.animation_data and mat.node_tree.animation_data.action:
            mat.node_tree.animation_data.action.name = 'Native material ' + piece['id']
            result['material_actions'] += 1
        if changes_order:
            if obj.animation_data and obj.animation_data.action and obj.animation_data.action.users > 1:
                obj.animation_data.action = obj.animation_data.action.copy()
            if replace_z_animation:
                for collection in _collections(obj):
                    for curve in list(collection):
                        if curve.data_path == 'location' and curve.array_index == 2:
                            collection.remove(curve)
            for frame, order in zip(frames, channels['render_order'][:, index]):
                obj.location.z = original_z + float(order - channels['render_order'][0, index]) * depth_step
                obj.keyframe_insert('location', index=2, frame=float(frame), group='Native layer order')
            for curve in _curves(obj):
                if curve.data_path == 'location' and curve.array_index == 2:
                    curve.extrapolation = 'CONSTANT'
                    for point in curve.keyframe_points:
                        point.interpolation = 'CONSTANT'
            result['animated']['render_order'].append(piece['id'])
            result['object_z_actions'] += 1
        # Returning the editable data to the first sampled values is predictable.
        # The next scene.frame_set() evaluates the ordinary Actions as usual.
        obj.location.z = original_z
        for channel, node_name in mapping.items():
            value = channels[channel][0, index]
            nodes[node_name].outputs[0].default_value = (*value[:3], 1.0) if np.ndim(value) else float(value)
        obj['_native_effects_version'] = 2
    return result
