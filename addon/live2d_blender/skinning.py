"""Offline sparse affine skinning fit. No Core, bpy, SciPy or runtime dependency.

This is an original alternating least-squares affine prototype, not an
implementation of the rigid-bone SSDR paper. Two hierarchical Blender bones per
logical influence can express each affine map using rotations and scales.
Only the exported sampled animation is independent of Live2D. Refitting held-out
poses still needs their target vertices and does not predict new Live2D controls.
"""
from dataclasses import dataclass, field
from itertools import combinations
import numpy as np


@dataclass
class SkinningFit:
    rest: np.ndarray
    weights: np.ndarray
    affines: np.ndarray
    depth_offsets: np.ndarray
    centers: np.ndarray
    metrics: dict
    history: list = field(default_factory=list)
    settings: dict = field(default_factory=dict)


def _inputs(rest, poses):
    rest = np.asarray(rest, dtype=np.float64)
    poses = np.asarray(poses, dtype=np.float64)
    if rest.ndim != 2 or rest.shape[1] not in (2, 3) or len(rest) < 3:
        raise ValueError('Rest vertices must have shape [N, 2 or 3], with at least 3 vertices')
    if poses.ndim != 3 or poses.shape[1:] != rest.shape or len(poses) < 1:
        raise ValueError('Poses must have shape [T, N, dimensions], matching rest')
    if not np.isfinite(rest).all() or not np.isfinite(poses).all():
        raise ValueError('Skinning inputs must be finite')
    if rest.shape[1] == 2:
        rest = np.pad(rest, ((0, 0), (0, 1)))
        poses = np.pad(poses, ((0, 0), (0, 0), (0, 1)))
    return rest, poses


def project_simplex(values):
    """Project each row onto nonnegative weights summing to one."""
    values = np.asarray(values, dtype=np.float64)
    order = np.sort(values, axis=1)[:, ::-1]
    total = np.cumsum(order, axis=1) - 1.0
    ranks = np.arange(1, values.shape[1] + 1)
    active = order - total / ranks > 0
    rho = np.maximum(active.sum(axis=1) - 1, 0)
    theta = total[np.arange(len(values)), rho] / (rho + 1)
    result = np.maximum(values - theta[:, None], 0.0)
    return result / np.maximum(result.sum(axis=1, keepdims=True), 1e-30)


def _initial_weights(rest, poses, bones, groups=None):
    if groups is not None:
        groups = np.asarray(groups)
        if groups.shape != (len(rest),):
            raise ValueError('Initial groups must contain one label per vertex')
        labels, assign = np.unique(groups, return_inverse=True)
        if len(labels) == bones:
            return np.eye(bones)[assign]
    # Deterministic motion-space clustering with a small spatial tie-breaker.
    displacement = (poses[:, :, :2] - rest[None, :, :2]).transpose(1, 0, 2).reshape(len(rest), -1)
    features = np.concatenate((displacement, rest[:, :2] * 0.05), axis=1)
    centers = [int(np.argmax(np.sum((features - features.mean(axis=0)) ** 2, axis=1)))]
    distance = np.full(len(rest), np.inf)
    for _ in range(1, bones):
        distance = np.minimum(distance, np.sum((features - features[centers[-1]]) ** 2, axis=1))
        distance[centers] = -1
        centers.append(int(np.argmax(distance)))
    centroids = features[centers].copy()
    assignments = np.zeros(len(rest), dtype=int)
    for _ in range(20):
        # Avoid an [vertices, bones, motion_dimensions] temporary on real models.
        distances = (np.sum(features * features, axis=1)[:, None]
                     + np.sum(centroids * centroids, axis=1)[None, :]
                     - 2 * features @ centroids.T)
        assignments = np.argmin(distances, axis=1)
        for bone in range(bones):
            selected = assignments == bone
            if selected.any():
                centroids[bone] = features[selected].mean(axis=0)
            else:
                # Retain at least one seed for otherwise unused influences.
                vertex = int(np.argmax(distances.min(axis=1)))
                assignments[vertex] = bone
                centroids[bone] = features[vertex]
    return np.eye(bones)[assignments]


def _solve_xy(rest_xy, poses_xy, weights, regularization):
    """Solve all logical bone affine displacements together for fixed weights."""
    x = np.column_stack((rest_xy, np.ones(len(rest_xy))))
    design = (weights[:, :, None] * x[:, None, :]).reshape(len(rest_xy), -1)
    displacement = (poses_xy - rest_xy[None]).transpose(1, 0, 2).reshape(len(rest_xy), -1)
    gram = design.T @ design
    ridge = regularization * max(float(np.trace(gram)) / max(len(gram), 1), 1e-12)
    rhs = design.T @ displacement
    delta = np.linalg.solve(gram + np.eye(len(gram)) * ridge, rhs)
    # [N, B*3] @ [B*3, T*2] -> per-pose [B, 2, 3].
    affines = delta.reshape(weights.shape[1], 3, len(poses_xy), 2).transpose(2, 0, 3, 1)
    affines[:, :, 0, 0] += 1.0
    affines[:, :, 1, 1] += 1.0
    return affines


def _predict_xy(rest_xy, weights, affines):
    x = np.column_stack((rest_xy, np.ones(len(rest_xy))))
    design = (weights[:, :, None] * x[:, None, :]).reshape(len(rest_xy), -1)
    return (design @ affines.transpose(1, 3, 0, 2).reshape(weights.shape[1] * 3, -1)).reshape(len(rest_xy), len(affines), 2).transpose(1, 0, 2)


def _solve_weights(rest_xy, poses_xy, affines, previous, max_influences, iterations, chunk=256):
    x = np.column_stack((rest_xy, np.ones(len(rest_xy))))
    out = np.empty_like(previous)
    bones = previous.shape[1]
    for start in range(0, len(rest_xy), chunk):
        end = min(start + chunk, len(rest_xy))
        predicted = np.einsum('tbqc,nc->ntbq', affines, x[start:end], optimize=True)
        error = predicted - poses_xy[:, start:end, :].transpose(1, 0, 2)[:, :, None, :]
        gram = np.einsum('ntbi,ntci->nbc', error, error, optimize=True)
        if bones <= 4:
            support = np.tile(np.arange(bones), (end - start, 1))
            out[start:end] = _exact_sparse_simplex(gram, support, max_influences)
            continue
        lipschitz = np.maximum(np.abs(gram).sum(axis=2).max(axis=1), 1e-14)
        weights = previous[start:end].copy()
        accelerated, momentum = weights.copy(), 1.0
        for _ in range(iterations):
            gradient = np.einsum('nbc,nc->nb', gram, accelerated, optimize=True)
            updated = project_simplex(accelerated - gradient / lipschitz[:, None])
            next_momentum = (1.0 + np.sqrt(1.0 + 4 * momentum * momentum)) / 2
            accelerated = updated + ((momentum - 1.0) / next_momentum) * (updated - weights)
            weights, momentum = updated, next_momentum
        if max_influences < bones:
            keep = np.argsort(weights, axis=1)[:, -max_influences:]
            weights = _exact_sparse_simplex(gram, keep)
        out[start:end] = weights
    return out


def _exact_sparse_simplex(gram, support, max_active=None):
    """Enumerate the <=15 active subsets within at most four candidate bones."""
    rows, bones = gram.shape[:2]
    best = np.zeros((rows, bones))
    error = np.full(rows, np.inf)
    row = np.arange(rows)
    limit = min(support.shape[1], max_active) if max_active is not None else support.shape[1]
    for size in range(1, limit + 1):
        for subset in combinations(range(support.shape[1]), size):
            selected = support[:, subset]
            if size == 1:
                values = np.ones((rows, 1))
            else:
                local = gram[row[:, None, None], selected[:, :, None], selected[:, None, :]]
                system = np.zeros((rows, size + 1, size + 1))
                system[:, :size, :size] = local
                # A tiny diagonal term selects a stable solution in flat regions.
                ridge = np.maximum(np.max(np.abs(local), axis=(1, 2)), 1e-20) * 1e-12
                system[:, np.arange(size), np.arange(size)] += ridge[:, None]
                system[:, :size, size] = 1; system[:, size, :size] = 1
                rhs = np.zeros((rows, size + 1, 1));rhs[:, size, 0] = 1
                try:
                    values = np.linalg.solve(system, rhs)[:, :size, 0]
                except np.linalg.LinAlgError:
                    values = (np.linalg.pinv(system) @ rhs)[:, :size, 0]
            feasible = np.all(values >= -1e-9, axis=1)
            values = np.maximum(values, 0);values /= np.maximum(values.sum(axis=1, keepdims=True), 1e-30)
            candidate = np.zeros((rows, bones));np.put_along_axis(candidate, selected, values, axis=1)
            residual = np.einsum('nb,nbc,nc->n', candidate, gram, candidate, optimize=True)
            improved = feasible & (residual < error)
            best[improved] = candidate[improved];error[improved] = residual[improved]
    return best


def _solve_depth(rest, poses, weights, regularization):
    difference = (poses[:, :, 2] - rest[None, :, 2]).T
    gram = weights.T @ weights
    ridge = regularization * max(float(np.trace(gram)) / len(gram), 1e-12)
    return np.linalg.solve(gram + np.eye(len(gram)) * ridge, weights.T @ difference).T


def _world_affines(affines, center, scale):
    result = affines.copy()
    result[:, :, :, 2] = scale * affines[:, :, :, 2] + center - np.einsum('tbij,j->tbi', affines[:, :, :, :2], center)
    return result


def predict(fit, affines=None, depth_offsets=None):
    affines = fit.affines if affines is None else np.asarray(affines)
    depth_offsets = fit.depth_offsets if depth_offsets is None else np.asarray(depth_offsets)
    xy = _predict_xy(fit.rest[:, :2], fit.weights, affines)
    z = fit.rest[None, :, 2] + depth_offsets @ fit.weights.T
    return np.concatenate((xy, z[:, :, None]), axis=2)


def error_metrics(rest, target, predicted, image_size=1024):
    diagonal = max(float(np.linalg.norm(np.ptp(rest, axis=0))), 1e-12)
    frame_error = np.linalg.norm(np.asarray(predicted) - np.asarray(target), axis=2)
    xy_extent = max(float(np.ptp(rest[:, :2], axis=0).max()), 1e-12)
    pixels_per_unit = image_size / xy_extent
    rows = []
    for i, values in enumerate(frame_error):
        rms = float(np.sqrt(np.mean(values * values)))
        rows.append({'pose': i, 'rms': rms, 'p95': float(np.percentile(values, 95)), 'max': float(values.max()),
                     'rms_bbox_diagonal': rms / diagonal, 'rms_at_1024px': rms * pixels_per_unit,
                     'max_at_1024px': float(values.max()) * pixels_per_unit})
    return {'per_pose': rows, 'rms': float(np.sqrt(np.mean(frame_error ** 2))), 'max': float(frame_error.max()),
            'bbox_diagonal': diagonal, 'pixels_per_unit_1024': pixels_per_unit}


def fit_skinning(rest, poses, bone_count=8, max_influences=4, groups=None,
                 iterations=30, weight_iterations=80, regularization=1e-8, tolerance=1e-7):
    rest, poses = _inputs(rest, poses)
    if not 1 <= bone_count <= min(128, len(rest) // 3):
        raise ValueError('Logical bone budget must be 1..128 and no more than one per 3 vertices')
    if not 1 <= max_influences <= 4:
        raise ValueError('Only 1..4 nonzero influences per vertex are supported')
    if iterations < 0 or weight_iterations < 1 or regularization <= 0:
        raise ValueError('Invalid fitting iteration count or regularization')
    center = (rest[:, :2].min(axis=0) + rest[:, :2].max(axis=0)) / 2
    scale = max(float(np.linalg.norm(np.ptp(rest[:, :2], axis=0))), 1e-8)
    normalized = (rest[:, :2] - center) / scale
    normalized_poses = (poses[:, :, :2] - center) / scale
    weights = _initial_weights(normalized, normalized_poses, bone_count, groups)
    affines = _solve_xy(normalized, normalized_poses, weights, regularization)
    error = float(np.mean((_predict_xy(normalized, weights, affines) - normalized_poses) ** 2))
    history = [error]
    for _ in range(iterations):
        if np.sqrt(error) <= tolerance:
            break
        candidate = _solve_weights(normalized, normalized_poses, affines, weights,
                                   min(max_influences, bone_count), weight_iterations)
        transformed = _solve_xy(normalized, normalized_poses, candidate, regularization)
        next_error = float(np.mean((_predict_xy(normalized, candidate, transformed) - normalized_poses) ** 2))
        if not np.isfinite(next_error):
            raise ValueError('Skinning fitting produced a non-finite error')
        if next_error > error * (1 + 1e-7):
            break  # Do not replace a better sparse solution with a worse step.
        weights, affines = candidate, transformed
        history.append(next_error)
        improvement, error = error - next_error, next_error
        if improvement < max(1e-16, error * 1e-6):
            break
    depths = _solve_depth(rest, poses, weights, regularization)
    world = _world_affines(affines, center, scale)
    centers = weights.T @ rest / np.maximum(weights.sum(axis=0)[:, None], 1e-12)
    result = SkinningFit(rest.copy(), weights, world, depths, centers, {}, history,
                         {'logical_bones': bone_count, 'blender_bones': bone_count * 2,
                          'max_influences': min(max_influences, bone_count), 'regularization': regularization,
                          'method': 'sparse affine alternating least squares; not rigid SSDR'})
    result.metrics = error_metrics(rest, poses, predict(result))
    return result


def fit_pose_transforms(fit, target_poses):
    """Fixed-weight fit requiring target meshes; NOT inference from Live2D controls."""
    rest, target = _inputs(fit.rest, target_poses)
    center = (rest[:, :2].min(axis=0) + rest[:, :2].max(axis=0)) / 2
    scale = max(float(np.linalg.norm(np.ptp(rest[:, :2], axis=0))), 1e-8)
    regularization = fit.settings.get('regularization', 1e-8)
    affines = _solve_xy((rest[:, :2] - center) / scale, (target[:, :, :2] - center) / scale, fit.weights, regularization)
    affines = _world_affines(affines, center, scale)
    depths = _solve_depth(rest, target, fit.weights, regularization)
    return affines, depths
