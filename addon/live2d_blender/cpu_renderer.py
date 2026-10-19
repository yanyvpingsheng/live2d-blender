"""Independent NumPy reference rasterizer for Cubism 3/4/5.2 Drawables.

This is original code implementing published rendering behavior, not SDK code.
Behavior references (read 2026-10-19):
https://github.com/Live2D/CubismNativeFramework/tree/develop/src/Rendering/OpenGL
https://docs.live2d.com/en/cubism-editor-manual/clipping-mask/
https://docs.live2d.com/en/cubism-editor-manual/multiply-color-screen-color/

Contract
--------
* Geometry and UVs are unmodified Core coordinates: +Y up; V=0 is bottom.
* NumPy textures are H x W x RGB(A), uint8 or floats in [0, 1]. By default
  their first row is the top, as in a PNG. Set texture_origin='bottom-left'
  for Blender image.pixels. Paths require optional Pillow; no bpy dependency.
* Output is float32 H x W x RGBA, bottom row first by default, with PREMULTIPLIED
  alpha. Set Blender image.alpha_mode='PREMUL'. RGB can exceed alpha for old
  additive blending, including nonzero RGB at zero alpha. Straight-alpha output
  rejects that last case because it cannot represent it without losing light.
* Numbers are blended in the supplied texture color space, without an implicit
  sRGB/linear transfer. Caller owns color management (including Blender images).
* Core already evaluated per-drawable opacity and colors, including inheritance.
  Do not apply part opacity again. Multiply/screen color alpha is unused.
* Mask geometry uses raw texture alpha, ignoring its visible flag, opacity,
  blend, color adjustments and its own mask. Multiple masks combine as an alpha
  union, 1-product(1-alpha). Inversion is applied AFTER that union.
* CCW triangles face forward. A top-left edge rule prevents shared-edge seams.

This is a single-sample CPU reference, NOT a GPU parity certification. No mask
atlas downsampling, mipmaps, MSAA, anisotropy, physics, pose, model tint, Cubism
5.3 offscreen groups or new color/alpha blend types are implemented. It uses
output-resolution masks; SDK mask-atlas filtering can differ at boundaries.
"""
from collections import OrderedDict
from pathlib import Path

import numpy as np

from .data import Live2DError, validate_snapshot

__all__ = ['render', 'fit_bounds', 'prepare_textures']


def _choice(value, choices, label):
    if value not in choices:
        raise Live2DError(f'{label} must be one of {", ".join(choices)}')


def prepare_textures(textures):
    """Load/normalize textures once for reuse. Does not flip row orientation.

    File paths always load top-first. Do not mix paths and bottom-first arrays
    when rendering with texture_origin='bottom-left'. Float32 inputs may be
    returned without copying; treat them as read-only while a frame is rendered.
    """
    prepared = []
    for index, value in enumerate(textures):
        if isinstance(value, (str, Path)):
            try:
                from PIL import Image
            except ImportError as error:
                raise Live2DError('Texture paths require Pillow; pass NumPy RGBA arrays instead') from error
            with Image.open(value) as image:
                value = np.asarray(image.convert('RGBA'))
        array = np.asarray(value)
        if array.ndim != 3 or array.shape[2] not in (3, 4) or not all(array.shape[:2]):
            raise Live2DError(f'Texture {index}: expected nonempty H x W x RGB(A)')
        if array.dtype == np.uint8:
            array = array.astype(np.float32) / 255.0
        elif np.issubdtype(array.dtype, np.floating):
            array = array.astype(np.float32, copy=False)
        else:
            raise Live2DError(f'Texture {index}: use uint8 or floating-point pixels')
        if not np.isfinite(array).all() or array.min() < 0 or array.max() > 1:
            raise Live2DError(f'Texture {index}: pixels must be finite and in [0, 1]')
        if array.shape[2] == 3:
            array = np.concatenate((array, np.ones((*array.shape[:2], 1), np.float32)), axis=2)
        prepared.append(array)
    return prepared


def fit_bounds(snapshot, padding=0.02):
    """Geometry bounds with fractional padding; choose fixed bounds for animation.

    Includes hidden geometry to avoid jumps when visibility changes. This still
    cannot anticipate future deformation; prefer backend canvas bounds for a
    timeline. Returns (min_x, min_y, max_x, max_y).
    """
    if not np.isfinite(padding) or padding < 0:
        raise Live2DError('Padding must be finite and nonnegative')
    points = [np.asarray(d.vertices, dtype=np.float64) for d in snapshot.drawables if d.vertices]
    if not points:
        return (-1.0, -1.0, 1.0, 1.0)
    xy = np.concatenate(points)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise Live2DError('Cannot fit bounds to invalid geometry')
    low, high = xy.min(axis=0), xy.max(axis=0)
    span = high - low
    span = np.maximum(span, max(float(span.max()), 1.0) * 1e-6)
    middle = (low + high) * 0.5
    low, high = middle - span * (0.5 + padding), middle + span * (0.5 + padding)
    return (float(low[0]), float(low[1]), float(high[0]), float(high[1]))


def _edge(a, b, x, y):
    return (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0])


def _inclusive(a, b):
    # Top/left edges with +Y up, for a counterclockwise triangle.
    dy, dx = b[1] - a[1], b[0] - a[0]
    return dy < 0 or (dy == 0 and dx < 0)


def _fragments(drawable, vertices, rect):
    """Yield absolute pixel rows/columns and interpolated UVs per triangle."""
    left, bottom, right, top = rect
    uv = np.asarray(drawable.uvs, dtype=np.float64)
    triangles = np.asarray(drawable.indices, dtype=np.int64).reshape(-1, 3)
    for indices in triangles:
        p = vertices[indices]
        area = _edge(p[0], p[1], p[2, 0], p[2, 1])
        if area == 0 or (area < 0 and not drawable.double_sided):
            continue
        if area < 0:
            indices = indices[[0, 2, 1]]
            p = vertices[indices]
            area = -area
        x0 = max(left, int(np.ceil(p[:, 0].min() - 0.5)))
        x1 = min(right, int(np.floor(p[:, 0].max() - 0.5)) + 1)
        y0 = max(bottom, int(np.ceil(p[:, 1].min() - 0.5)))
        y1 = min(top, int(np.floor(p[:, 1].max() - 0.5)) + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        x, y = np.meshgrid(np.arange(x0, x1, dtype=np.float64) + 0.5,
                           np.arange(y0, y1, dtype=np.float64) + 0.5)
        edges = [_edge(p[1], p[2], x, y), _edge(p[2], p[0], x, y), _edge(p[0], p[1], x, y)]
        inside = np.ones(x.shape, dtype=bool)
        for e, a, b in zip(edges, (p[1], p[2], p[0]), (p[2], p[0], p[1])):
            inside &= (e > 0) | ((e == 0) & _inclusive(a, b))
        rows, cols = np.nonzero(inside)
        if rows.size:
            weights = np.stack([e[rows, cols] for e in edges], axis=1) / area
            yield rows + y0, cols + x0, weights @ uv[indices]


def _sample(texture, uv, origin, filtering):
    h, w = texture.shape[:2]
    u = np.clip(uv[:, 0], 0, 1)
    v = np.clip(uv[:, 1], 0, 1)
    if origin == 'top-left':
        v = 1.0 - v
    if filtering == 'nearest':
        ix = np.minimum((u * w).astype(np.int64), w - 1)
        iy = np.minimum((v * h).astype(np.int64), h - 1)
        return texture[iy, ix]
    # GL normalized texture coordinates locate texel centers at (i+0.5)/size.
    x, y = u * w - 0.5, v * h - 0.5
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    fx, fy = (x - x0).astype(np.float32)[:, None], (y - y0).astype(np.float32)[:, None]
    x1, y1 = np.clip(x0 + 1, 0, w - 1), np.clip(y0 + 1, 0, h - 1)
    x0, y0 = np.clip(x0, 0, w - 1), np.clip(y0, 0, h - 1)
    low = texture[y0, x0] * (1 - fx) + texture[y0, x1] * fx
    high = texture[y1, x0] * (1 - fx) + texture[y1, x1] * fx
    return low * (1 - fy) + high * fy


class _MaskCache:
    """Budgeted LRU of compact raw mask alpha tiles; no frame-to-frame state."""
    def __init__(self, drawables, vertices, textures, width, height, origin, filtering, budget):
        self.drawables, self.vertices, self.textures = drawables, vertices, textures
        self.width, self.height, self.origin, self.filtering = width, height, origin, filtering
        self.budget, self.used, self.cache = budget, 0, OrderedDict()

    def get(self, index):
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        d, points = self.drawables[index], self.vertices[index]
        if not len(points) or not d.indices:
            tile = (0, 0, np.zeros((0, 0), np.float32))
        else:
            x0 = max(0, min(self.width, int(np.ceil(points[:, 0].min() - 0.5))))
            x1 = max(x0, min(self.width, int(np.floor(points[:, 0].max() - 0.5)) + 1))
            y0 = max(0, min(self.height, int(np.ceil(points[:, 1].min() - 0.5))))
            y1 = max(y0, min(self.height, int(np.floor(points[:, 1].max() - 0.5)) + 1))
            alpha = np.zeros((y1 - y0, x1 - x0), np.float32)
            for rows, cols, uv in _fragments(d, points, (x0, y0, x1, y1)):
                sample_alpha = _sample(self.textures[d.texture], uv, self.origin, self.filtering)[:, 3]
                r, c = rows - y0, cols - x0
                alpha[r, c] += sample_alpha * (1.0 - alpha[r, c])
            tile = (x0, y0, alpha)
        size = tile[2].nbytes
        if size <= self.budget:
            while self.cache and self.used + size > self.budget:
                _, old = self.cache.popitem(last=False)
                self.used -= old[2].nbytes
            self.cache[index] = tile
            self.used += size
        return tile


def render(snapshot, textures, width, height, bounds=None, *,
           output_origin='bottom-left', alpha_mode='premultiplied',
           texture_origin='top-left', texture_alpha_mode='straight',
           filtering='bilinear', background=(0.0, 0.0, 0.0, 0.0),
           clamp=True, mask_cache_bytes=64 * 1024 * 1024):
    """Render a Snapshot into an independent float32 RGBA frame.

    Bounds are (min_x, min_y, max_x, max_y). Pixel centers sample evenly across
    bounds; the image fills them without aspect correction. Background is a
    straight RGBA color in [0, 1]. With clamp=True each blend is clamped to the
    [0, 1] range like an SDK UNORM render target (but without 8-bit quantization).
    With clamp=False the working buffer is HDR. Only legacy blend IDs 0/1/2 are
    accepted; unsupported modes fail rather than silently rendering as normal.
    Mask cache size bounds retained mask tiles; the framebuffer, textures and
    currently used mask group require additional memory. No input is mutated.
    """
    _choice(output_origin, ('bottom-left', 'top-left'), 'output_origin')
    _choice(texture_origin, ('bottom-left', 'top-left'), 'texture_origin')
    _choice(alpha_mode, ('premultiplied', 'straight'), 'alpha_mode')
    _choice(texture_alpha_mode, ('premultiplied', 'straight'), 'texture_alpha_mode')
    _choice(filtering, ('bilinear', 'nearest'), 'filtering')
    if any(isinstance(n, bool) or not isinstance(n, (int, np.integer)) or not 1 <= n <= 8192 for n in (width, height)):
        raise Live2DError('Frame dimensions must be integers in [1, 8192]')
    if isinstance(mask_cache_bytes, bool) or not isinstance(mask_cache_bytes, int) or mask_cache_bytes < 0:
        raise Live2DError('mask_cache_bytes must be a nonnegative integer')
    validate_snapshot(snapshot)
    tex = prepare_textures(textures)
    ds = snapshot.drawables
    for d in ds:
        if not isinstance(d.texture, (int, np.integer)) or not 0 <= d.texture < len(tex):
            raise Live2DError(f'{d.id}: texture index outside supplied textures')
        if d.blend_mode not in (0, 1, 2):
            raise Live2DError(f'{d.id}: unsupported blend mode {d.blend_mode}; only legacy 0/1/2 are implemented')
        for value, label in ((d.multiply_color, 'multiply_color'), (d.screen_color, 'screen_color')):
            color = np.asarray(value, dtype=np.float64)
            if color.shape != (4,) or not np.isfinite(color).all():
                raise Live2DError(f'{d.id}: invalid {label}')
        if not np.isfinite(d.render_order):
            raise Live2DError(f'{d.id}: invalid render order')
    box = np.asarray(fit_bounds(snapshot) if bounds is None else bounds, dtype=np.float64)
    if box.shape != (4,) or not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
        raise Live2DError('Bounds must be finite (min_x, min_y, max_x, max_y) with positive area')
    bg = np.asarray(background, dtype=np.float32)
    if bg.shape != (4,) or not np.isfinite(bg).all() or np.any(bg < 0) or np.any(bg > 1):
        raise Live2DError('Background must be a finite straight RGBA color in [0, 1]')
    frame = np.empty((height, width, 4), np.float32)
    frame[:] = bg
    frame[:, :, :3] *= bg[3]
    scale = np.array((width / (box[2] - box[0]), height / (box[3] - box[1])))
    points = [(np.asarray(d.vertices, dtype=np.float64).reshape(-1, 2) - box[:2]) * scale for d in ds]
    if any(not np.isfinite(p).all() for p in points):
        raise Live2DError('Projected coordinates are not finite')
    cache = _MaskCache(ds, points, tex, width, height, texture_origin, filtering, mask_cache_bytes)
    for index in sorted(range(len(ds)), key=lambda i: (ds[i].render_order, i)):
        d = ds[index]
        opacity = float(np.clip(d.opacity, 0, 1))
        if not d.visible or opacity == 0 or not d.indices:
            continue
        masks = [cache.get(i) for i in d.masks]
        multiply = np.asarray(d.multiply_color[:3], dtype=np.float32)
        screen = np.asarray(d.screen_color[:3], dtype=np.float32)
        for rows, cols, uv in _fragments(d, points[index], (0, 0, width, height)):
            sampled = _sample(tex[d.texture], uv, texture_origin, filtering)
            alpha = sampled[:, 3:4]
            rgb = sampled[:, :3] * multiply
            if texture_alpha_mode == 'premultiplied':
                rgb = rgb * (1 - screen) + screen * alpha
            else:
                rgb = (rgb * (1 - screen) + screen) * alpha
            source = np.concatenate((rgb, alpha), axis=1) * opacity
            if masks:
                remaining = np.ones(len(rows), np.float32)
                for x0, y0, tile in masks:
                    r, c = rows - y0, cols - x0
                    inside = (r >= 0) & (c >= 0) & (r < tile.shape[0]) & (c < tile.shape[1])
                    remaining[inside] *= 1.0 - tile[r[inside], c[inside]]
                coverage = remaining if d.inverted_mask else 1.0 - remaining
                source *= coverage[:, None]
            destination = frame[rows, cols]
            if d.blend_mode == 0:
                result = source + destination * (1.0 - source[:, 3:4])
            elif d.blend_mode == 1:
                result = destination.copy()
                result[:, :3] += source[:, :3]
            else:
                result = destination.copy()
                result[:, :3] *= 1.0 - source[:, 3:4] + source[:, :3]
            if clamp:
                np.clip(result, 0, 1, out=result)
            frame[rows, cols] = result
    if alpha_mode == 'straight':
        alpha = frame[:, :, 3:4]
        if np.any((alpha == 0) & (frame[:, :, :3] != 0)):
            raise Live2DError('Additive RGB at zero alpha cannot be represented as straight RGBA; request premultiplied output')
        np.divide(frame[:, :, :3], alpha, out=frame[:, :, :3], where=alpha > 0)
    if output_origin == 'top-left':
        frame = frame[::-1].copy()
    return frame
