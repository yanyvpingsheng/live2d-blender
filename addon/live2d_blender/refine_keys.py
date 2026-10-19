"""Training-only affine key refinement; no source/held-out target input."""
import numpy as np
from .skinning import SkinningFit

MAX_KEY_TIMES = 20000
MAX_BONE_KEY_TIMES = 100000


def refined_frame_times(frames, subdivisions=9, logical_bones=None):
    """Validate limits before expensive fitting and return safe Blender key times."""
    if type(subdivisions) is not int or not 1 <= subdivisions <= 31 or subdivisions % 2 != 1:
        raise ValueError('Use an odd subdivision count from 1 to 31')
    frames = np.asarray(frames, dtype=float)
    if frames.ndim != 1 or len(frames) < 1 or not np.isfinite(frames).all() or np.any(np.diff(frames) <= 0):
        raise ValueError('Frames must be finite and increase strictly')
    count = (len(frames) - 1) * subdivisions + 1
    if count > MAX_KEY_TIMES:
        raise ValueError('Refined animation exceeds 20000 key times; reduce samples or subdivisions')
    if logical_bones is not None:
        if not isinstance(logical_bones, (int, np.integer)) or isinstance(logical_bones, (bool, np.bool_)) or logical_bones < 1:
            raise ValueError('Logical bone count must be a positive integer')
        if count * logical_bones > MAX_BONE_KEY_TIMES:
            raise ValueError('Refinement exceeds 100000 logical-bone key times; reduce bones, samples or subdivisions')
    out = np.empty(count, dtype=float)
    u = np.arange(subdivisions, dtype=float) / subdivisions
    for i in range(len(frames) - 1):
        out[i * subdivisions:(i + 1) * subdivisions] = (1 - u) * frames[i] + u * frames[i + 1]
    out[-1] = frames[-1]
    # Blender F-curve coordinates are single precision. Distinct float64 times
    # may otherwise overwrite each other, especially near a very large frame.
    with np.errstate(over='ignore', invalid='ignore'):
        stored = out.astype(np.float32)
    if not np.isfinite(stored).all() or np.any(np.diff(stored) <= 0):
        raise ValueError('Refined frame times collide in Blender precision; reduce subdivisions or move the animation closer to frame zero')
    return out


def refine_affine_keys(fit, frames, subdivisions=9):
    """Interpolate fitted matrices, not vertices or Live2D parameters.

    Odd subdivision counts keep interval midpoints between generated bone keys.
    Transforms must be finite and dimensionally consistent. Limits bound both
    time samples and total logical-bone poses before Blender allocates F-curves.
    """
    affines = np.asarray(fit.affines, dtype=float)
    depths = np.asarray(fit.depth_offsets, dtype=float)
    rest = np.asarray(fit.rest, dtype=float)
    weights = np.asarray(fit.weights, dtype=float)
    centers = np.asarray(fit.centers, dtype=float)
    if affines.ndim != 4 or affines.shape[2:] != (2, 3) or min(affines.shape[:2]) < 1:
        raise ValueError('Affine transforms must have nonempty shape [T, B, 2, 3]')
    poses, bones = affines.shape[:2]
    if depths.shape != (poses, bones) or rest.ndim != 2 or rest.shape[1] != 3 or len(rest) < 1:
        raise ValueError('Depth offsets or rest vertices do not match the fitted animation')
    if weights.shape != (len(rest), bones) or centers.shape != (bones, 3):
        raise ValueError('Weights and bone centers must match rest vertices and transforms')
    if not all(np.isfinite(array).all() for array in (affines, depths, rest, weights, centers)):
        raise ValueError('Fitted geometry, weights, and transforms must be finite')
    if np.any(weights < -1e-10) or not np.allclose(weights.sum(axis=1), 1, atol=1e-7, rtol=0) or np.any((weights > 1e-10).sum(axis=1) > 4):
        raise ValueError('Weights must be nonnegative, normalized, and use at most four influences')
    frames = np.asarray(frames, dtype=float)
    if frames.shape != (poses,):
        raise ValueError('Frames must match transforms')
    out_frames = refined_frame_times(frames, subdivisions, bones)
    out_affines = np.empty((len(out_frames), bones, 2, 3), dtype=float)
    out_depths = np.empty((len(out_frames), bones), dtype=float)
    for i in range(poses - 1):
        for k in range(subdivisions):
            u = k / subdivisions
            out_affines[i * subdivisions + k] = (1 - u) * affines[i] + u * affines[i + 1]
            out_depths[i * subdivisions + k] = (1 - u) * depths[i] + u * depths[i + 1]
    out_affines[-1] = affines[-1]
    out_depths[-1] = depths[-1]
    result = SkinningFit(rest, weights, out_affines, out_depths, centers, fit.metrics.copy(),
                         settings={**fit.settings, 'refinement_subdivisions': subdivisions,
                                   'source_target_meshes_used': False})
    return result, out_frames
