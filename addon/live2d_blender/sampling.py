"""Conversion-time mesh sampling for independent bone/weight fitting.

The output is numerical training data, NOT a runtime dependency and NOT a claim
that skinning can reproduce masks, opacity changes or arbitrary Live2D poses.
Do not distribute samples from a model whose license forbids converted assets.
"""
from dataclasses import dataclass, field
import numpy as np
from .data import Live2DError, Snapshot, validate_snapshot


@dataclass
class SampledAnimation:
    positions: np.ndarray       # [frames, vertices, 3], first frame is rest
    triangles: np.ndarray       # [triangles, 3], global vertex indices
    uvs: np.ndarray             # [vertices, 2]
    pieces: list                # Drawable boundaries and baseline material facts
    fps: float
    warnings: list
    non_geometric_changes: dict
    channels: dict = field(default_factory=dict)  # Native material/layer animation samples

    @property
    def rest_positions(self):
        return self.positions[0].copy()


def sample_sequence(backend, parameter_frames, part_frames=None, physics=None,
                    fps=60.0, depth_step=0.0001, sample_render_order=False):
    """Evaluate ordered conversion samples and capture fixed-topology geometry.

    The solver is reset/stabilized at the first input then advanced by 1/fps.
    part_frames are sparse explicit overrides when an official Pose is active.
    sample_render_order=False freezes layer depths at the first sample and emits
    a warning if later orders change. Setting True samples depth but still does
    not make general 2D transparent compositing equivalent to bone skinning.
    """
    if not parameter_frames:
        raise Live2DError('At least one parameter sample is required')
    if not np.isfinite(fps) or fps <= 0:
        raise Live2DError('Sampling FPS must be finite and positive')
    if not np.isfinite(depth_step) or depth_step < 0:
        raise Live2DError('Depth step must be finite and nonnegative')
    if part_frames is not None and len(part_frames) != len(parameter_frames):
        raise Live2DError('Parameter and part frame counts differ')
    part_frames = part_frames if part_frames is not None else [{} for _ in parameter_frames]
    positions = None
    baseline = None
    pieces, triangles, uvs = [], [], []
    changes = {name: set() for name in ('opacity', 'visibility', 'multiply_color', 'screen_color', 'render_order')}
    warnings = []
    channels = {name: [] for name in ('opacity', 'visible', 'multiply_color', 'screen_color', 'render_order')}
    for frame, (inputs, overrides) in enumerate(zip(parameter_frames, part_frames)):
        if not isinstance(inputs, dict) or not isinstance(overrides, dict):
            raise Live2DError('Every parameter/part frame must be a dictionary')
        if physics is not None:
            if frame == 0:
                parameters, parts = physics.reset(inputs, overrides, stabilize=True)
            else:
                parameters, parts = physics.evaluate(inputs, overrides, 1.0 / fps)
        else:
            parameters, parts = inputs, overrides
        snapshot = backend.evaluate(parameters, parts)
        if baseline is None:
            validate_snapshot(snapshot)
            baseline = snapshot
            total = sum(len(d.vertices) for d in snapshot.drawables)
            if not total or total * len(parameter_frames) > 100_000_000:
                raise Live2DError('Sampling needs a nonempty model and at most 100 million vertex samples')
            positions = np.empty((len(parameter_frames), total, 3), np.float32)
            offset = 0
            for d in snapshot.drawables:
                pieces.append({'id': d.id, 'offset': offset, 'count': len(d.vertices), 'texture': d.texture,
                               'opacity': d.opacity, 'visible': d.visible, 'blend_mode': d.blend_mode,
                               'masks': list(d.masks), 'inverted_mask': d.inverted_mask,
                               'double_sided': d.double_sided, 'render_order': d.render_order,
                               'multiply_color': d.multiply_color, 'screen_color': d.screen_color})
                triangles.extend([tuple(offset + i for i in d.indices[j:j+3]) for j in range(0, len(d.indices), 3)])
                uvs.extend(d.uvs)
                offset += len(d.vertices)
            masked = sum(bool(d.masks) for d in snapshot.drawables)
            blended = sum(d.blend_mode != 0 for d in snapshot.drawables)
            if masked:
                warnings.append(f'{masked} drawables need clipping masks; bones and weights alone do not encode masks.')
            if blended:
                warnings.append(f'{blended} drawables use destination-dependent blend modes; skinning does not encode this compositing.')
        for name in channels:
            channels[name].append([getattr(d, name) for d in snapshot.drawables])
        if len(snapshot.drawables) != len(baseline.drawables):
            raise Live2DError('Drawable count changed during sampling')
        offset = 0
        for d, initial in zip(snapshot.drawables, baseline.drawables):
            if d.id != initial.id or len(d.vertices) != len(initial.vertices) or d.indices != initial.indices or d.uvs != initial.uvs:
                raise Live2DError('Topology/UVs changed; fixed-weight skinning requires fixed topology')
            count = len(d.vertices)
            positions[frame, offset:offset+count, :2] = np.asarray(d.vertices, np.float32).reshape(-1, 2)
            positions[frame, offset:offset+count, 2] = (d.render_order if sample_render_order else initial.render_order) * depth_step
            if abs(d.opacity - initial.opacity) > 1e-6: changes['opacity'].add(d.id)
            if d.visible != initial.visible: changes['visibility'].add(d.id)
            if np.max(np.abs(np.asarray(d.multiply_color) - initial.multiply_color)) > 1e-6: changes['multiply_color'].add(d.id)
            if np.max(np.abs(np.asarray(d.screen_color) - initial.screen_color)) > 1e-6: changes['screen_color'].add(d.id)
            if d.render_order != initial.render_order: changes['render_order'].add(d.id)
            offset += count
    if not np.isfinite(positions).all():
        raise Live2DError('Non-finite sampled vertex coordinates')
    changed = {key: sorted(ids) for key, ids in changes.items() if ids}
    for key, ids in changed.items():
        warnings.append(f'{key} changes on {len(ids)} drawables are not encoded by geometric bone weights.')
    warnings.append('Fit accuracy on these sampled frames does not prove accuracy on unseen parameter combinations.')
    return SampledAnimation(positions, np.asarray(triangles, np.int32).reshape(-1, 3),
                            np.asarray(uvs, np.float32).reshape(-1, 2), pieces, float(fps), warnings, changed,
                            {key: np.asarray(value, dtype=np.bool_ if key == 'visible' else np.int32 if key == 'render_order' else np.float32)
                             for key, value in channels.items()})
