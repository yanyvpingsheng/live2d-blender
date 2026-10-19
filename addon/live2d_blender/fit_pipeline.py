"""Generic conversion-time fitting policy; no model-specific IDs or anatomy.

Manual groups are supplied by the user. Automatic Drawable boundaries are
structural, not anatomical, and may fit worse than global motion clustering.
"""
from collections.abc import Mapping
import numpy as np
from .bind_selection import select_training_rest
from .skinning import SkinningFit, fit_skinning, predict, error_metrics
from .refine_keys import refine_affine_keys, refined_frame_times


def grouping_template(pieces, bone_budget=8):
    """An editable explicit mapping, not an inferred semantic assignment."""
    return {'version': 1, 'groups': [{'name': 'user_group',
            'bones': max(1, int(bone_budget) - 1),
            'drawables': [str(p['id']) for p in pieces]}]}


def _piece_labels(pieces, count):
    labels = np.full(count, -1, dtype=np.int32)
    ids = []
    for index, piece in enumerate(pieces):
        pid = piece['id']; offset = piece['offset']; size = piece['count']
        if not isinstance(pid, str) or not pid or not isinstance(offset, (int, np.integer)) or isinstance(offset, (bool, np.bool_)) or not isinstance(size, (int, np.integer)) or isinstance(size, (bool, np.bool_)):
            raise ValueError('Drawable IDs must be nonempty strings and vertex spans must be integers')
        offset, size = int(offset), int(size)
        if pid in ids or offset < 0 or size < 1 or offset + size > count:
            raise ValueError('Drawable IDs and vertex spans must be unique and valid')
        if np.any(labels[offset:offset + size] != -1):
            raise ValueError('Drawable vertex spans overlap')
        labels[offset:offset + size] = index; ids.append(pid)
    if np.any(labels < 0):
        raise ValueError('Drawables must partition all vertices')
    return labels, ids


def _manual_groups(mapping, ids, piece_labels, moving):
    if not isinstance(mapping, Mapping) or type(mapping.get('version')) is not int or mapping.get('version') != 1:
        raise ValueError('Manual mapping needs version 1 and a groups array')
    groups = mapping.get('groups')
    if not isinstance(groups, list) or not groups:
        raise ValueError('Manual mapping groups must be a nonempty array')
    owners = {}; names = set(); labels = np.full(len(piece_labels), -1, dtype=np.int32)
    budgets = {}; descriptions = {}
    for index, group in enumerate(groups):
        if not isinstance(group, Mapping): raise ValueError('Every group must be an object')
        name = group.get('name'); budget = group.get('bones'); drawables = group.get('drawables')
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError('Manual group names must be unique nonempty strings')
        if type(budget) is not int or not 1 <= budget <= 128:
            raise ValueError('Each manual group needs 1–128 logical bones')
        if not isinstance(drawables, list) or not drawables:
            raise ValueError('Each manual group needs at least one Drawable ID')
        for pid in drawables:
            if not isinstance(pid, str) or pid not in ids:
                raise ValueError('Manual mapping contains an unknown Drawable ID')
            if pid in owners: raise ValueError('A Drawable may belong to only one manual group')
            owners[pid] = index
            labels[piece_labels == ids.index(pid)] = index
        names.add(name); budgets[index] = budget; descriptions[index] = name
    if np.any(labels[moving] < 0):
        raise ValueError('Every moving Drawable needs an explicit manual group')
    return labels, budgets, descriptions


def _allocate(labels, moving, budget, rest, poses):
    groups = sorted(int(x) for x in np.unique(labels[moving]))
    if len(groups) > budget:
        raise ValueError(f'{len(groups)} moving groups exceed the {budget} available logical bones. Increase the total budget or merge Drawables with a manual mapping.')
    counts = {g: int(np.count_nonzero((labels == g) & moving)) for g in groups}
    capacities = {g: max(1, counts[g] // 3) for g in groups}
    budgets = {g: 1 for g in groups}
    scores = {g: float(np.linalg.norm(poses[:, (labels == g) & moving] - rest[(labels == g) & moving], axis=2).max()) * counts[g] for g in groups}
    while sum(budgets.values()) < budget:
        eligible = [g for g in groups if budgets[g] < capacities[g]]
        if not eligible: break
        chosen = max(eligible, key=lambda g: (scores[g] / budgets[g], counts[g], -g))
        budgets[chosen] += 1
    return budgets


def fit_animation(sample, frames, *, bone_budget=8, iterations=20, grouping='GLOBAL',
                  manual_mapping=None, repair_bind=True, separate_static=True,
                  static_tolerance=1e-6, subdivisions=9):
    """Return (training_fit, export_fit, refined_frames, JSON report).

    Only supplied training poses inform bind selection, weights and generated
    matrix keys. This function does not claim held-out or pixel validation.
    """
    poses = np.asarray(sample.positions, dtype=np.float64)
    frames = np.asarray(frames, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[2] != 3 or min(poses.shape[:2]) < 1 or not np.isfinite(poses).all():
        raise ValueError('Expected finite nonempty poses[T,N,3]')
    if frames.shape != (len(poses),) or not np.isfinite(frames).all() or np.any(np.diff(frames) <= 0):
        raise ValueError('Training frames must match poses and increase strictly')
    if type(bone_budget) is not int or not 1 <= bone_budget <= 128:
        raise ValueError('Total logical bone budget must be 1–128')
    if type(iterations) is not int or not 1 <= iterations <= 100:
        raise ValueError('Iterations must be 1–100')
    if grouping not in ('GLOBAL', 'DRAWABLE', 'MANUAL'):
        raise ValueError('Unsupported grouping mode')
    if not np.isfinite(static_tolerance) or static_tolerance < 0:
        raise ValueError('Static tolerance must be finite and nonnegative')
    # Reject invalid/oversized/colliding time grids before any bind or fit work.
    refined_frame_times(frames, subdivisions)
    piece_labels, ids = _piece_labels(sample.pieces, poses.shape[1])
    rest = poses[0].copy(); bind_report = {'enabled': False}
    if repair_bind:
        selected = select_training_rest(poses, sample.pieces, sample.triangles)
        rest = selected.rest
        bind_report = {'enabled': True, 'summary': selected.summary,
                       'selected_training_frames': [float(frames[i]) for i in selected.selected_sample_indices]}
    moving = np.max(np.abs(poses - rest[None]), axis=(0, 2)) > static_tolerance if separate_static else np.ones(len(rest), dtype=bool)
    static_count = int(np.count_nonzero(~moving)); reserved = int(static_count > 0)
    available = bone_budget - reserved
    if moving.any() and available < 1:
        raise ValueError('The total budget needs a static influence plus at least one moving influence')
    if grouping == 'MANUAL':
        labels, budgets, descriptions = _manual_groups(manual_mapping, ids, piece_labels, moving)
        budgets = {g: n for g, n in budgets.items() if np.any((labels == g) & moving)}
        if sum(budgets.values()) > available:
            raise ValueError('Manual group budgets plus the static influence exceed the total budget')
    else:
        labels = piece_labels if grouping == 'DRAWABLE' else np.zeros(len(rest), dtype=np.int32)
        budgets = _allocate(labels, moving, available, rest, poses)
        descriptions = {g: ids[g] if grouping == 'DRAWABLE' else 'global_motion_cluster' for g in budgets}
    total = reserved + sum(budgets.values())
    refined_frame_times(frames, subdivisions, total)
    weights = np.zeros((len(rest), total)); affines = np.zeros((len(poses), total, 2, 3))
    depths = np.zeros((len(poses), total)); centers = np.zeros((total, 3))
    details = []; offset = 0
    if reserved:
        weights[~moving, 0] = 1; affines[:, 0, 0, 0] = 1; affines[:, 0, 1, 1] = 1
        centers[0] = rest[~moving].mean(axis=0); offset = 1
    for label, count in budgets.items():
        selected = np.flatnonzero((labels == label) & moving)
        if count > max(1, len(selected) // 3):
            raise ValueError(f'Group {descriptions[label]} has too few moving vertices for its requested bone budget')
        # One/two moving points still have valid affine solutions. Repetition
        # meets the underlying solver's minimum sample count without adding
        # influences or moving static vertices in the exported model.
        solver_indices = selected if len(selected) >= 3 else np.resize(selected, 3)
        local = fit_skinning(rest[solver_indices], poses[:, solver_indices], bone_count=count,
                             max_influences=4, iterations=iterations,
                             weight_iterations=60, regularization=1e-6)
        weights[selected, offset:offset+count] = local.weights[:len(selected)]
        affines[:, offset:offset+count] = local.affines
        depths[:, offset:offset+count] = local.depth_offsets
        centers[offset:offset+count] = local.centers
        details.append({'name': descriptions[label], 'moving_vertices': len(selected),
                        'logical_bones': count,
                        'training_metrics': error_metrics(rest, poses[:, selected], predict(local)[:, :len(selected)])})
        offset += count
    training = SkinningFit(rest, weights, affines, depths, centers, {})
    training.metrics = error_metrics(rest, poses, predict(training))
    report = {'grouping': grouping, 'manual_semantics': grouping == 'MANUAL',
              'anatomy_inferred': False, 'requested_logical_budget': bone_budget,
              'logical_bones': total, 'actual_bones': 2 * total,
              'unused_budget': bone_budget - total, 'static_vertices': static_count,
              'static_influence_reserved': bool(reserved), 'static_tolerance': float(static_tolerance),
              'bind_selection': bind_report, 'groups': details,
              'training_samples': len(frames), 'training_metrics': training.metrics,
              'holdout_tested': False, 'subdivisions': subdivisions,
              'warning': 'Drawable boundaries and manual mappings can improve or worsen a fit. No model-specific precision is guaranteed.'}
    training.settings = report.copy()
    refined, export_frames = refine_affine_keys(training, frames, subdivisions)
    report['export_key_times'] = len(export_frames)
    return training, refined, export_frames, report
