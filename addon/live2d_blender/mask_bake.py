"""Offline alpha-only clipping masks from evaluated, exported Blender meshes.

No reference RGB, runtime callbacks, drivers, SDK, or Pillow are used. Each
unique sorted source set shares a grayscale PNG sequence. Raw texture alpha
ignores source visibility, opacity, colors, blending, and its own clipping.
Sources combine as 1-product(1-alpha); target inversion happens in the shader.

Projection is fixed in the output container's local XY space over every integer
frame, including source AND target extents. Moving the whole output afterwards
therefore remains safe. Editing the relative geometry or animation needs a new
bake. Subframes, filtering boundaries and arbitrary 3D viewpoints are not
certified. Keep the external sequence directory with the saved .blend file.
"""
from pathlib import Path
import shutil
import struct
import tempfile
import zlib

import numpy as np

from .cpu_renderer import _MaskCache, prepare_textures
from .data import Drawable, Live2DError

__all__ = ['mask_groups', 'validate_bake_request', 'fit_projection_bounds',
           'rasterize_mask_group', 'write_grayscale_png',
           'bake_mask_sequences', 'attach_mask']

MAX_FRAMES = 2000
MAX_OUTPUT_PIXELS = 1_073_741_824
MAX_VERTEX_SAMPLES = 20_000_000
MAX_TEXTURE_PIXELS = 32_000_000


def _integer(value, label, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise Live2DError(f'{label} must be an integer in [{low}, {high}]')
    return int(value)


def _bounds(bounds):
    box = np.asarray(bounds, dtype=np.float64)
    if box.shape != (4,) or not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
        raise Live2DError('Mask bounds must be finite (min_x, min_y, max_x, max_y) with positive area')
    return box


def mask_groups(pieces):
    """Return sorted source indices and target indices into the original pieces.

    Duplicate source indices collapse into one source. An empty masks list is
    an unmasked Drawable and creates no image. Referenced empty source geometry
    still creates a group whose coverage is zero, including for inversion.
    """
    ids = [p['id'] for p in pieces]
    if len(set(ids)) != len(ids):
        raise Live2DError('Mask pieces contain duplicate Drawable IDs')
    groups = {}
    for target, piece in enumerate(pieces):
        sources = tuple(sorted({_integer(i, 'Mask source index', 0, len(pieces) - 1)
                                for i in piece.get('masks', [])}))
        if sources:
            groups.setdefault(sources, []).append(target)
    return [{'sources': list(sources), 'targets': groups[sources]}
            for sources in sorted(groups)]


def validate_bake_request(start, end, resolution, group_count):
    """Reject unreasonable work before creating directories or image data."""
    start = _integer(start, 'Mask start frame', -1_000_000, 1_000_000)
    end = _integer(end, 'Mask end frame', start, 1_000_000)
    resolution = _integer(resolution, 'Mask resolution', 32, 2048)
    group_count = _integer(group_count, 'Mask group count', 0, 100_000)
    frames = end - start + 1
    pixels = frames * group_count * resolution * resolution
    if frames > MAX_FRAMES:
        raise Live2DError(f'Mask bake is limited to {MAX_FRAMES} integer frames; shorten the clip')
    if pixels > MAX_OUTPUT_PIXELS:
        raise Live2DError(f'Mask bake would write {pixels:,} pixels across {frames:,} frames and {group_count:,} groups; limit is {MAX_OUTPUT_PIXELS:,} pixels (1 GiB raw grayscale). Reduce the frame range, mask groups, or resolution')
    return frames


def fit_projection_bounds(point_sets, resolution=512, padding=0.04, pixel_padding=3):
    """Fit a finite fixed XY box, with fractional plus final-image-pixel padding.

    Empty groups have a deterministic fallback; collapsed axes get a small
    positive span. Call with all source and target positions across the clip.
    """
    resolution = _integer(resolution, 'Mask resolution', 32, 2048)
    if not np.isfinite(padding) or padding < 0 or not np.isfinite(pixel_padding) or not 0 <= pixel_padding < resolution / 2:
        raise Live2DError('Mask padding must be finite, nonnegative, and leave interior pixels')
    low, high = np.full(2, np.inf), np.full(2, -np.inf)
    for points in point_sets:
        points = np.asarray(points, dtype=np.float64)
        if points.size == 0:
            continue
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise Live2DError('Mask projection needs finite N x 2 positions')
        low = np.minimum(low, points.min(axis=0))
        high = np.maximum(high, points.max(axis=0))
    if not np.isfinite(low).all():
        low, high = np.array((-0.5, -0.5)), np.array((0.5, 0.5))
    span = high - low
    span = np.maximum(span, max(float(span.max()), 1.0) * 1e-6)
    middle = low * 0.5 + high * 0.5
    half_span = span * (0.5 + padding) / (1.0 - 2.0 * pixel_padding / resolution)
    return tuple(float(x) for x in _bounds((*list(middle - half_span), *list(middle + half_span))))


def _raw_union(drawables, positions, textures, sources, bounds, resolution, texture_origin):
    box = _bounds(bounds)
    scale = resolution / (box[2:] - box[:2])
    points = [(np.asarray(p, dtype=np.float64).reshape(-1, 2) - box[:2]) * scale for p in positions]
    if any(not np.isfinite(p).all() for p in points):
        raise Live2DError('Non-finite mask geometry')
    cache = _MaskCache(drawables, points, textures, resolution, resolution,
                       texture_origin, 'bilinear', 16 * 1024 * 1024)
    remaining = np.ones((resolution, resolution), np.float32)
    for index in sources:
        x, y, alpha = cache.get(index)
        remaining[y:y + alpha.shape[0], x:x + alpha.shape[1]] *= 1.0 - alpha
    return 1.0 - remaining


def rasterize_mask_group(drawables, positions, textures, source_indices, bounds,
                         resolution=512, *, texture_origin='bottom-left'):
    """Pure NumPy raw alpha union, bottom row first; an empty union is zero.

    Positions use model-local XY, separately from the Drawable's UV/topology.
    Sources may have arbitrary opacity, visibility, own masks and RGB colors.
    """
    resolution = _integer(resolution, 'Mask resolution', 32, 2048)
    if texture_origin not in ('bottom-left', 'top-left'):
        raise Live2DError('Mask texture origin must be bottom-left or top-left')
    if len(drawables) != len(positions):
        raise Live2DError('Mask drawable and position counts differ')
    sources = sorted({_integer(i, 'Mask source index', 0, len(drawables) - 1) for i in source_indices})
    textures = prepare_textures(textures)
    for i in sources:
        d = drawables[i]
        p = np.asarray(positions[i], dtype=np.float64).reshape(-1, 2)
        Drawable(d.id, d.texture, p.tolist(), d.uvs, d.indices).validate()
        _integer(d.texture, 'Mask texture index', 0, len(textures) - 1)
    return _raw_union(drawables, positions, textures, sources, bounds, resolution, texture_origin)


def write_grayscale_png(path, alpha):
    """Write an 8-bit grayscale PNG without Pillow, never replacing a file.

    Input is finite alpha in [0,1], bottom row first, as used by Blender pixels.
    PNG scanlines are flipped to top-first. No RGB or model texture is written.
    """
    alpha = np.asarray(alpha)
    if alpha.ndim != 2 or not all(alpha.shape) or not np.isfinite(alpha).all() or np.any(alpha < 0) or np.any(alpha > 1):
        raise Live2DError('PNG mask must be a nonempty finite H x W alpha array in [0, 1]')
    pixels = np.rint(alpha[::-1] * 255.0).astype(np.uint8)
    rows = np.zeros((pixels.shape[0], pixels.shape[1] + 1), np.uint8)
    rows[:, 1:] = pixels
    def chunk(kind, payload):
        return struct.pack('>I', len(payload)) + kind + payload + struct.pack('>I', zlib.crc32(kind + payload) & 0xffffffff)
    data = (b'\x89PNG\r\n\x1a\n' +
            chunk(b'IHDR', struct.pack('>IIBBBBB', pixels.shape[1], pixels.shape[0], 8, 0, 0, 0, 0)) +
            chunk(b'IDAT', zlib.compress(rows.tobytes(), 6)) + chunk(b'IEND', b''))
    with Path(path).open('xb') as output:
        output.write(data)


def _local_mesh(obj, depsgraph, inverse_projection, with_topology=False):
    """Copy evaluated mesh data, always releasing Blender's temporary mesh."""
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh(preserve_all_data_layers=with_topology, depsgraph=depsgraph)
    try:
        if mesh is None:
            raise Live2DError(f'{obj.name}: an evaluated mesh is required for mask projection')
        xyz = np.empty(len(mesh.vertices) * 3, np.float64)
        mesh.vertices.foreach_get('co', xyz)
        matrix = inverse_projection @ np.asarray(evaluated.matrix_world, dtype=np.float64)
        xy = (xyz.reshape(-1, 3) @ matrix[:3, :3].T + matrix[:3, 3])[:, :2]
        if not np.isfinite(xy).all():
            raise Live2DError(f'{obj.name}: non-finite evaluated mask coordinates')
        if not with_topology:
            return xy
        mesh.calc_loop_triangles()
        if not len(mesh.loop_triangles):
            return xy[:0], [], []
        if mesh.uv_layers.active is None:
            raise Live2DError(f'{obj.name}: mask sources require UV coordinates')
        # Expand corners: per-loop UV seams must not be averaged or dropped.
        vertices = np.empty(len(mesh.loops), np.int32)
        mesh.loops.foreach_get('vertex_index', vertices)
        uv = np.empty(len(mesh.loops) * 2, np.float64)
        mesh.uv_layers.active.data.foreach_get('uv', uv)
        corners = np.empty(len(mesh.loop_triangles) * 3, np.int32)
        mesh.loop_triangles.foreach_get('loops', corners)
        uv = uv.reshape(-1, 2)[corners]
        if not np.isfinite(uv).all():
            raise Live2DError(f'{obj.name}: non-finite mask UV coordinates')
        return xy[vertices[corners]], uv.tolist(), list(range(len(corners)))
    finally:
        evaluated.to_mesh_clear()


def _projection_inverse(projection_object, depsgraph):
    matrix = np.asarray(projection_object.evaluated_get(depsgraph).matrix_world, dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise Live2DError('Mask projection object has a non-finite world transform')
    try:
        inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError as error:
        raise Live2DError('Mask projection object has a singular world transform') from error
    if not np.isfinite(inverse).all():
        raise Live2DError('Mask projection transform cannot be inverted')
    return inverse


def _read_alpha_textures(images, texture_indices):
    """Blender pixels are bottom-first; RGB and color space are immaterial."""
    total = 0
    for index in texture_indices:
        image = images.get(index) if hasattr(images, 'get') else images[index]
        if image is None or len(image.size) != 2 or min(image.size) < 1:
            raise Live2DError(f'Mask texture {index} is missing or empty')
        if image.source not in ('FILE', 'GENERATED', 'VIEWER'):
            raise Live2DError('Mask baking requires static source textures, not movies or sequences')
        total += int(image.size[0]) * int(image.size[1])
    if total > MAX_TEXTURE_PIXELS:
        raise Live2DError(f'Mask textures exceed {MAX_TEXTURE_PIXELS:,} pixels; use smaller conversion textures')
    textures = []
    for index in texture_indices:
        image = images[index]
        width, height = map(int, image.size)
        pixels = np.empty(len(image.pixels), np.float32)
        image.pixels.foreach_get(pixels)
        channels = len(pixels) // (width * height)
        if channels not in (1, 3, 4) or len(pixels) != width * height * channels:
            raise Live2DError(f'Mask texture {index} has an unsupported pixel layout')
        alpha = (pixels.reshape(height, width, channels)[:, :, 3].copy()
                 if channels == 4 else np.ones((height, width), np.float32))
        if not np.isfinite(alpha).all() or np.any(alpha < 0) or np.any(alpha > 1):
            raise Live2DError(f'Mask texture {index} alpha must be finite and in [0, 1]')
        # The rasterizer reads only channel 3. Broadcasting avoids retaining RGB.
        textures.append(np.broadcast_to(alpha[:, :, None], (height, width, 4)))
    return textures


def bake_mask_sequences(context, objects_by_id, pieces, images, start, end,
                        projection_object, output_directory, resolution=512):
    """Bake exported native mesh alpha and return images, report and bindings.

    Requires a saved .blend. Creates the configured output root if missing and a
    unique subdirectory, never overwrites, and stores Blender-relative paths.
    Only new images/files are made; selection, materials, meshes and animation
    are not changed. Scene time is restored, including its subframe. A failure
    removes this call's new images and directory. The caller owns rollback after
    a successful return if a later conversion step fails.

    bindings maps piece ID to (image, bounds, inverted). groups is JSON-ready;
    images and bindings contain Blender datablocks and are not report JSON.
    created_roots lists newly created ancestors for caller-owned later rollback.
    """
    import bpy
    groups = mask_groups(pieces)
    frame_count = validate_bake_request(start, end, resolution, len(groups))
    report = {'groups': [], 'directory': None, 'frames': [int(start), int(end)],
              'frame_count': frame_count, 'resolution': int(resolution),
              'bindings': {}, 'images': [], 'projection': 'output-container local XY',
              'geometry': 'evaluated exported native meshes at integer frames',
              'source_alpha': 'raw texture alpha; ignores opacity, visibility and own masks',
              'union': '1-product(1-alpha)', 'inversion': 'after union in target shader',
              'subframe_equivalence_certified': False}
    if not groups:
        return report
    if not bpy.data.filepath:
        raise Live2DError('Save the .blend before baking external mask sequences')
    if projection_object is None:
        raise Live2DError('An output container is required for local mask projection')
    if not output_directory or not str(output_directory).strip():
        raise Live2DError('Choose an output directory for mask sequences')
    root = Path(bpy.path.abspath(str(output_directory))).expanduser().resolve()
    if root.exists() and not root.is_dir():
        raise Live2DError('Mask output root must be a directory, not a file')
    indices = sorted({i for g in groups for i in g['sources'] + g['targets']})
    objects = {}
    for index in indices:
        obj = objects_by_id.get(pieces[index]['id'])
        if obj is None or obj.type != 'MESH':
            raise Live2DError(f"Missing exported mask mesh: {pieces[index]['id']}")
        objects[index] = obj
    if sum(len(o.data.vertices) for o in objects.values()) * frame_count > MAX_VERTEX_SAMPLES:
        raise Live2DError(f'Mask bake exceeds {MAX_VERTEX_SAMPLES:,} vertex samples; shorten the clip')
    sources = sorted({i for g in groups for i in g['sources']})
    texture_indices = sorted({_integer(pieces[i]['texture'], 'Mask texture index', 0, 1_000_000) for i in sources})
    texture_lookup = {index: n for n, index in enumerate(texture_indices)}
    textures = _read_alpha_textures(images, texture_indices)
    old_frame, old_subframe = context.scene.frame_current, context.scene.frame_subframe
    directory, created_images, created_roots = None, [], []
    restored = False
    try:
        # A first pass accumulates only min/max, not a full animation in memory.
        lows = [np.full(2, np.inf) for _ in groups]
        highs = [np.full(2, -np.inf) for _ in groups]
        for frame in range(start, end + 1):
            context.scene.frame_set(frame)
            depsgraph = context.evaluated_depsgraph_get()
            inverse = _projection_inverse(projection_object, depsgraph)
            extents = {}
            for index, obj in objects.items():
                xy = _local_mesh(obj, depsgraph, inverse)
                if len(xy):
                    extents[index] = (xy.min(axis=0), xy.max(axis=0))
            for n, group in enumerate(groups):
                for index in set(group['sources'] + group['targets']):
                    if index in extents:
                        lows[n] = np.minimum(lows[n], extents[index][0])
                        highs[n] = np.maximum(highs[n], extents[index][1])
        boxes = [fit_projection_bounds([np.stack((low, high))] if np.isfinite(low).all() else [], resolution)
                 for low, high in zip(lows, highs)]
        missing = []
        ancestor = root
        while not ancestor.exists():
            missing.append(ancestor)
            ancestor = ancestor.parent
        for folder in reversed(missing):
            try:
                folder.mkdir()
                created_roots.append(folder)
            except FileExistsError:
                if not folder.is_dir():
                    raise
        directory = Path(tempfile.mkdtemp(prefix='native_alpha_masks_', dir=str(root)))
        # Files are numbered from 1, independently of scene frame_start. This
        # supports zero/negative starts without Blender filename-offset tricks.
        for n in range(len(groups)):
            (directory / f'mask_{n:04d}').mkdir()
        source_lookup = {index: n for n, index in enumerate(sources)}
        for frame in range(start, end + 1):
            context.scene.frame_set(frame)
            depsgraph = context.evaluated_depsgraph_get()
            inverse = _projection_inverse(projection_object, depsgraph)
            drawables, positions = [], []
            for index in sources:
                xy, uvs, triangles = _local_mesh(objects[index], depsgraph, inverse, True)
                piece = pieces[index]
                positions.append(xy)
                drawables.append(Drawable(str(piece['id']), texture_lookup[piece['texture']], [], uvs, triangles,
                                          double_sided=bool(piece.get('double_sided', True))))
            for n, group in enumerate(groups):
                alpha = _raw_union(drawables, positions, textures, [source_lookup[i] for i in group['sources']],
                                   boxes[n], resolution, 'bottom-left')
                write_grayscale_png(directory / f'mask_{n:04d}' / f'{frame - start + 1:06d}.png', alpha)
        for n, group in enumerate(groups):
            first = directory / f'mask_{n:04d}' / '000001.png'
            image = bpy.data.images.load(str(first), check_existing=False)
            created_images.append(image)
            image.name = f'Native alpha mask {n:04d}'
            image.source = 'SEQUENCE'
            image.colorspace_settings.name = 'Non-Color'
            image.filepath = bpy.path.relpath(str(first))
            if not image.filepath.startswith('//'):
                raise Live2DError('Mask output must support a path relative to the saved .blend')
            item = dict(group, bounds=list(boxes[n]), path=image.filepath,
                        source_ids=[pieces[i]['id'] for i in group['sources']],
                        target_ids=[pieces[i]['id'] for i in group['targets']])
            report['groups'].append(item)
            for i in group['targets']:
                report['bindings'][pieces[i]['id']] = (image, boxes[n], bool(pieces[i].get('inverted_mask', False)))
        report['directory'] = str(directory)
        report['images'] = created_images
        report['created_roots'] = [str(folder) for folder in created_roots]
        # Restore inside the transaction, so a restore failure also rolls back
        # our output instead of leaving files behind after a failed return.
        context.scene.frame_set(old_frame, subframe=old_subframe)
        restored = True
        return report
    except Exception:
        for image in reversed(created_images):
            bpy.data.images.remove(image, do_unlink=True)
        if directory is not None:
            shutil.rmtree(directory)
        for folder in reversed(created_roots):
            try:
                folder.rmdir()  # Only our new, still-empty ancestors.
            except OSError:
                pass
        raise
    finally:
        if not restored:
            context.scene.frame_set(old_frame, subframe=old_subframe)


def attach_mask(material, image, bounds, inverted, projection_object, start, end):
    """Attach raw union coverage to the material's NE Masked Alpha input 1.

    The Object coordinate refers to the output container, never target UVs or
    absolute world position. Inversion is after the union and after CLIP (so
    outside a normal mask is zero, outside an inverted mask is one).
    """
    box = _bounds(bounds)
    duration = validate_bake_request(start, end, 32, 1)
    if material is None or material.node_tree is None or image is None or projection_object is None:
        raise Live2DError('A native material, mask image and projection object are required')
    nodes, links = material.node_tree.nodes, material.node_tree.links
    alpha = nodes.get('NE Masked Alpha')
    if alpha is None or alpha.bl_idname != 'ShaderNodeMath' or alpha.operation != 'MULTIPLY':
        raise Live2DError('Native material is missing its NE Masked Alpha multiplier')
    if alpha.inputs[1].is_linked:
        raise Live2DError('Native material already has mask coverage; create a fresh material before attaching')
    if image.source != 'SEQUENCE' or not image.colorspace_settings.is_data:
        raise Live2DError('Mask image must be a Non-Color image sequence')
    created = []
    def node(kind, name):
        result = nodes.new(kind); created.append(result); result.name = name
        return result
    try:
        coordinates = node('ShaderNodeTexCoord', 'NE Mask Object Coordinates')
        coordinates.object = projection_object
        subtract = node('ShaderNodeVectorMath', 'NE Mask Bounds Origin')
        subtract.operation = 'SUBTRACT'
        subtract.inputs[1].default_value = (float(box[0]), float(box[1]), 0.0)
        links.new(coordinates.outputs['Object'], subtract.inputs[0])
        divide = node('ShaderNodeVectorMath', 'NE Mask Bounds Scale')
        divide.operation = 'DIVIDE'
        divide.inputs[1].default_value = (float(box[2] - box[0]), float(box[3] - box[1]), 1.0)
        links.new(subtract.outputs['Vector'], divide.inputs[0])
        texture = node('ShaderNodeTexImage', 'NE Alpha Mask Sequence')
        texture.image = image
        texture.extension = 'CLIP'
        texture.interpolation = 'Linear'
        texture.projection = 'FLAT'
        texture.image_user.frame_start = int(start)
        texture.image_user.frame_duration = duration
        texture.image_user.frame_offset = 0
        texture.image_user.use_auto_refresh = True
        texture.image_user.use_cyclic = False
        links.new(divide.outputs['Vector'], texture.inputs['Vector'])
        coverage = texture.outputs['Color']
        if inverted:
            invert = node('ShaderNodeMath', 'NE Inverted Mask Union')
            invert.operation = 'SUBTRACT'; invert.inputs[0].default_value = 1.0
            links.new(coverage, invert.inputs[1]); coverage = invert.outputs[0]
        links.new(coverage, alpha.inputs[1])
        return texture
    except Exception:
        for item in reversed(created):
            nodes.remove(item)
        raise
