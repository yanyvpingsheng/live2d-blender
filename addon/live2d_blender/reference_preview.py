"""Explicit offline reference frame, never run automatically by a frame handler."""
import json
from pathlib import Path
import bpy
import numpy as np
from .data import Live2DError
from .cpu_renderer import render, fit_bounds
from . import blender_bridge as bridge


def _textures(backend, limit):
    """Downsample separate in-memory copies; source atlases are never modified."""
    result = []
    try:
        from PIL import Image
    except ImportError:
        Image = None
    for path in backend.textures:
        if Image is not None:
            with Image.open(path) as image:
                image = image.convert('RGBA')
                if limit:
                    image.thumbnail((limit, limit), Image.Resampling.LANCZOS)
                result.append(np.asarray(image).astype(np.float32) / 255.0)
        else:
            image = bpy.data.images.load(str(path), check_existing=True)
            width, height = image.size
            values = np.empty(width * height * 4, dtype=np.float32)
            image.pixels.foreach_get(values)
            # Blender buffers are bottom-up; normalize this fallback to top-first.
            values = values.reshape(height, width, 4)[::-1]
            stride = max(1, int(np.ceil(max(width, height) / limit))) if limit else 1
            result.append(values[::stride, ::stride].copy())
    return result


def reference_frame(context, root, resolution=512, texture_limit=2048):
    runtime = bridge.runtime_for_root(root)
    if not runtime:
        raise Live2DError('Reconnect the model before reference rendering')
    runtime.evaluate(context.scene, context.evaluated_depsgraph_get(), force=True)
    snapshot = runtime.last_snapshot
    opacity = getattr(runtime, 'last_model_opacity', 1.0)
    if opacity != 1.0:
        from dataclasses import replace
        from .data import Snapshot
        snapshot = Snapshot([replace(d, opacity=d.opacity * opacity) for d in snapshot.drawables], snapshot.warnings)
    canvas = runtime.backend.canvas
    width, height = canvas.get('size', (0, 0))
    origin = canvas.get('origin', (width / 2, height / 2))
    ppu = canvas.get('pixels_per_unit', 0)
    if width > 0 and height > 0 and ppu > 0:
        bounds = (-origin[0] / ppu, -(height - origin[1]) / ppu, (width - origin[0]) / ppu, origin[1] / ppu)
        aspect = width / height
    else:
        bounds = fit_bounds(snapshot)
        aspect = (bounds[2] - bounds[0]) / (bounds[3] - bounds[1])
    output_width = resolution if aspect >= 1 else max(1, round(resolution * aspect))
    output_height = resolution if aspect <= 1 else max(1, round(resolution / aspect))
    pixels = render(snapshot, _textures(runtime.backend, texture_limit), output_width, output_height, bounds,
                    output_origin='bottom-left', alpha_mode='premultiplied', texture_origin='top-left')
    name = f'{root.name} • Reference frame {context.scene.frame_current}'
    # Keep exact CPU numbers separately; generated float images are interpreted
    # as linear by Blender even when merely labelled sRGB (verified in 4.3.2).
    raw_image = bpy.data.images.new(name + ' RAW', width=output_width, height=output_height, alpha=True, float_buffer=True)
    raw_image.alpha_mode = 'PREMUL'
    raw_image.colorspace_settings.name = 'Non-Color'
    raw_image.pixels.foreach_set(pixels.ravel())
    raw_image.update()
    alpha = pixels[:, :, 3:4]
    straight = np.divide(pixels[:, :, :3], alpha, out=pixels[:, :, :3].copy(), where=alpha > 0)
    decoded = np.where(straight <= 0.04045, straight / 12.92, ((straight + 0.055) / 1.055) ** 2.4)
    display = pixels.copy()
    # Additive RGB at alpha zero has no unassociated color. Preserve an explicit
    # decoded-glow display approximation; raw_image remains numerically exact.
    display[:, :, :3] = np.where(alpha > 0, decoded * alpha, decoded)
    image = bpy.data.images.new(name, width=output_width, height=output_height, alpha=True, float_buffer=True)
    image.alpha_mode = 'PREMUL'
    image.colorspace_settings.name = 'Linear Rec.709'
    image.pixels.foreach_set(display.ravel())
    image.update()
    image['_l2d_reference'] = json.dumps({'frame': context.scene.frame_current_final, 'bounds': bounds,
        'texture_limit': texture_limit, 'renderer': 'CPU single-sample reference, premultiplied RGBA',
        'raw_image': raw_image.name, 'display': 'explicit sRGB-to-linear preview; zero-alpha additive glow is a display approximation',
        'warning': 'Not a GPU parity certificate. No automatic refresh; use Render reference again.'})
    root['_l2d_reference_image'] = image.name
    # If an Image Editor already exists, show this actual image without changing the user's layout.
    if context.screen:
        for area in context.screen.areas:
            if area.type == 'IMAGE_EDITOR':
                area.spaces.active.image = image
    return image
