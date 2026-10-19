"""Conversion-time deterministic samples for downstream bone/weight fitting.

This utility produces plain sample data. A final Blender armature/Action may
consume a fit of those samples without loading this module or any Cubism SDK.
It does not itself fit bones or promise lossless conversion.
"""
from dataclasses import dataclass
import math
from .data import Live2DError


@dataclass
class PhysicsSample:
    frame: int
    seconds: float
    parameters: dict
    parts: dict
    snapshot: object


def sample_sequence(backend, physics, input_at_frame, frame_count, *, fps=60.0,
                    parts_at_frame=None, stabilize=True, include_mesh=True):
    """Yield frames 0..frame_count-1 from a new deterministic initial state.

    input_at_frame(frame, seconds) returns PRE-physics parameter inputs, already
    combining the desired motion/expression layers. parts_at_frame, if supplied,
    returns only explicitly animated part-opacity overrides, not a default map.
    NativeBackend remains owned by the caller; no original asset is modified.
    Every sequence uses a single regular sample clock. Seeking is done by
    replaying this generator from frame zero, never by taking one large delta.
    """
    if type(frame_count) is not int or not 1 <= frame_count <= 1000000:
        raise Live2DError('Sample count must be an integer in 1..1000000')
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or not 1 <= fps <= 1000:
        raise Live2DError('Sample FPS must be finite and inside 1..1000')
    if include_mesh and backend is None: raise Live2DError('Mesh sampling needs a Core backend')
    base = input_at_frame(0, 0.0)
    part_base = parts_at_frame(0, 0.0) if parts_at_frame else {}
    values, parts = physics.reset(base, part_base, stabilize=stabilize)
    for frame in range(frame_count):
        seconds = frame / fps
        if frame:
            inputs = input_at_frame(frame, seconds)
            overrides = parts_at_frame(frame, seconds) if parts_at_frame else {}
            values, parts = physics.evaluate(inputs, overrides, 1.0 / fps)
        snapshot = backend.evaluate(values, parts) if include_mesh else None
        yield PhysicsSample(frame, seconds, values, parts, snapshot)
