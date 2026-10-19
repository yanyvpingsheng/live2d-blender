"""Training-only per-Drawable rest selection; no runtime/Core/fitter dependency.

Pass *only* the allowed training positions, not a full sequence. Sample labels are
provenance, never an array index into a full or held-out sequence. Selection uses
XY geometry. Default Z/depth is preserved so layer animation is not baked twice.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass
class BindSelection:
    rest: np.ndarray
    selected_sample_indices: np.ndarray
    pieces: list[dict]
    summary: dict


def _validate_pieces(pieces, vertex_count):
    spans = []
    covered = np.zeros(vertex_count, dtype=np.int32)
    for index, piece in enumerate(pieces):
        if any(not isinstance(piece[key], (int, np.integer)) or isinstance(piece[key], (bool, np.bool_))
               for key in ('offset', 'count')):
            raise ValueError('Piece offsets and counts must be integers')
        offset, count = int(piece['offset']), int(piece['count'])
        if offset < 0 or count < 1 or offset + count > vertex_count:
            raise ValueError('Each piece needs a nonempty valid vertex span.')
        covered[offset:offset + count] += 1
        spans.append((offset, count, str(piece.get('id', index))))
    if not spans or np.any(covered != 1):
        raise ValueError('Pieces must partition the complete vertex array once.')
    return spans


def _triangles_by_piece(triangles, spans, vertex_count):
    if triangles is None:
        return [None] * len(spans)
    tri = np.asarray(triangles)
    if tri.ndim != 2 or tri.shape[1] != 3 or not np.issubdtype(tri.dtype, np.integer):
        raise ValueError('Triangles must be an integer [N, 3] array.')
    if tri.size and (tri.min() < 0 or tri.max() >= vertex_count):
        raise ValueError('Triangle indices are outside the vertex array.')
    owner = np.empty(vertex_count, dtype=np.int64)
    for i, (offset, count, _) in enumerate(spans):
        owner[offset:offset + count] = i
    owners = owner[tri]
    if np.any(owners != owners[:, :1]):
        raise ValueError('A triangle may not cross Drawable boundaries.')
    return [tri[owners[:, 0] == i] - span[0] for i, span in enumerate(spans)]


def select_training_rest(
    training_positions: np.ndarray,
    pieces: Sequence[Mapping],
    triangles: np.ndarray | None = None,
    *,
    sample_indices: Sequence[int] | None = None,
    default_sample_index: int | None = None,
    method: str = 'covariance',
    mode: str = 'repair_degenerate',
    relative_rank_tolerance: float = 1e-8,
    absolute_std_fraction: float = 1e-8,
    preserve_depth: bool = True,
) -> BindSelection:
    """Choose one observed training rest per Drawable.

    ``covariance`` maximizes the smaller centered XY covariance eigenvalue
    (E-optimal support); ``area`` maximizes total absolute triangle area. Eligible
    candidates first maximize numerical planar rank, then the requested score.
    Ties favor the earliest input sample, so input order must be deterministic.

    ``repair_degenerate`` changes only deficient default pieces for which a
    higher-rank training pose exists. ``best`` chooses for every piece. No
    fabricated coordinates, epsilon expansion, parameter writes, opacity edits,
    or licensing/watermark modifications are performed. A collection of rest
    pieces need not represent a single globally realized animation pose.

    Numerical planar rank uses eigenvalues greater than both
    max_eigenvalue * relative_rank_tolerance and
    (training XY scene span * absolute_std_fraction) ** 2. These tolerances are
    reported, because tiny-but-full-rank geometry differs from exact collapse.
    """
    source = np.asarray(training_positions)
    if source.ndim != 3 or source.shape[0] < 1 or source.shape[1] < 1 or source.shape[2] not in (2, 3):
        raise ValueError('Training positions must have shape [T, V, 2 or 3].')
    if not np.isfinite(source).all():
        raise ValueError('Training positions must be finite.')
    if not np.isfinite([relative_rank_tolerance, absolute_std_fraction]).all() or relative_rank_tolerance < 0 or absolute_std_fraction < 0:
        raise ValueError('Rank tolerances must be finite and nonnegative.')
    if method not in ('covariance', 'area') or mode not in ('repair_degenerate', 'best'):
        raise ValueError('Unsupported selection method or mode.')
    if method == 'area' and triangles is None:
        raise ValueError('Area selection needs triangles.')
    if sample_indices is not None:
        raw_labels = np.asarray(sample_indices)
        if not np.issubdtype(raw_labels.dtype, np.integer):
            raise ValueError('Sample indices must be integers, not rounded labels')
    if default_sample_index is not None and (not isinstance(default_sample_index, (int, np.integer)) or isinstance(default_sample_index, (bool, np.bool_))):
        raise ValueError('Default sample index must be an integer')
    labels = np.arange(len(source), dtype=np.int64) if sample_indices is None else np.asarray(sample_indices, dtype=np.int64)
    if labels.shape != (len(source),) or len(np.unique(labels)) != len(labels):
        raise ValueError('Unique sample indices must match the training array.')
    default_label = labels[0] if default_sample_index is None else default_sample_index
    matches = np.flatnonzero(labels == default_label)
    if len(matches) != 1:
        raise ValueError('Default sample must be one of the supplied training samples.')
    default = int(matches[0])
    spans = _validate_pieces(pieces, source.shape[1])
    local_triangles = _triangles_by_piece(triangles, spans, source.shape[1])
    xy = source[:, :, :2].astype(np.float64)
    scene_span = float(np.ptp(xy, axis=(0, 1)).max())
    absolute_variance = (scene_span * absolute_std_fraction) ** 2
    rest = source[default].copy()
    selected = np.full(len(spans), int(labels[default]), dtype=np.int64)
    records = []
    for i, (offset, count, piece_id) in enumerate(spans):
        pts = xy[:, offset:offset + count]
        centered = pts - pts.mean(axis=1, keepdims=True)
        covariance = np.einsum('tvi,tvj->tij', centered, centered) / count
        eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
        threshold = np.maximum(absolute_variance, eigenvalues[:, 1] * relative_rank_tolerance)
        ranks = (eigenvalues > threshold[:, None]).sum(axis=1)
        extents = np.linalg.norm(np.ptp(pts, axis=1), axis=1)
        areas = np.zeros(len(pts))
        local_tri = local_triangles[i]
        if local_tri is not None and len(local_tri):
            a = pts[:, local_tri[:, 1]] - pts[:, local_tri[:, 0]]
            b = pts[:, local_tri[:, 2]] - pts[:, local_tri[:, 0]]
            areas = np.abs(a[:, :, 0] * b[:, :, 1] - a[:, :, 1] * b[:, :, 0]).sum(axis=1) / 2
        score = eigenvalues[:, 0] if method == 'covariance' else areas
        candidates = np.flatnonzero(ranks == ranks.max())
        best = int(candidates[np.argmax(score[candidates])])
        chosen = best if mode == 'best' or (ranks[default] < 2 and ranks[best] > ranks[default]) else default
        selected[i] = labels[chosen]
        if preserve_depth:
            rest[offset:offset + count, :2] = source[chosen, offset:offset + count, :2]
        else:
            rest[offset:offset + count] = source[chosen, offset:offset + count]
        records.append({
            'piece_index': i, 'id': piece_id, 'offset': offset, 'count': count,
            'default_rank': int(ranks[default]), 'best_rank': int(ranks.max()),
            'selected_rank': int(ranks[chosen]),
            'default_min_covariance': float(eigenvalues[default, 0]),
            'selected_min_covariance': float(eigenvalues[chosen, 0]),
            'best_min_covariance': float(eigenvalues[:, 0].max()),
            'default_area': float(areas[default]), 'selected_area': float(areas[chosen]),
            'best_area': float(areas.max()),
            'default_extent': float(extents[default]),
            'selected_extent': float(extents[chosen]),
            'max_training_extent': float(extents.max()),
            'max_vertex_motion_from_default': float(np.linalg.norm(pts - pts[default], axis=2).max()),
            'selected_sample_index': int(labels[chosen]),
            'best_candidate_sample_index': int(labels[best]),
            'exact_default_point': bool(extents[default] == 0),
            'exact_always_point': bool(extents.max() == 0),
            'exact_static': bool(np.array_equal(pts, np.broadcast_to(pts[default], pts.shape))),
        })
    default_bad = [r for r in records if r['default_rank'] < 2]
    summary = {
        'method': method, 'mode': mode, 'training_samples': len(source),
        'training_sample_indices': labels.tolist(), 'default_sample_index': int(labels[default]),
        'piece_count': len(records), 'vertex_count': source.shape[1],
        'scene_xy_span': scene_span, 'relative_rank_tolerance': relative_rank_tolerance,
        'absolute_std_fraction': absolute_std_fraction, 'absolute_variance_tolerance': absolute_variance,
        'default_rank_counts': [sum(r['default_rank'] == k for r in records) for k in range(3)],
        'selected_rank_counts': [sum(r['selected_rank'] == k for r in records) for k in range(3)],
        'default_degenerate_pieces': len(default_bad),
        'corrected_to_rank2_pieces': sum(r['default_rank'] < 2 and r['selected_rank'] == 2 for r in records),
        'still_degenerate_pieces': sum(r['selected_rank'] < 2 for r in records),
        'changed_rest_pieces': int(np.count_nonzero(selected != labels[default])),
        'static_default_degenerate_pieces': sum(r['exact_static'] for r in default_bad),
        'moving_default_degenerate_pieces': sum(not r['exact_static'] for r in default_bad),
        'exact_default_point_pieces': sum(r['exact_default_point'] for r in records),
        'exact_always_point_pieces': sum(r['exact_always_point'] for r in records),
        'max_rest_piece_xy_extent': max(r['selected_extent'] for r in records),
        'max_rest_scene_xy_extent': float(np.linalg.norm(np.ptp(rest[:, :2].astype(float), axis=0))),
        'max_rest_displacement_from_default': float(np.linalg.norm(rest[:, :2] - source[default, :, :2], axis=1).max()),
        'preserve_default_depth': preserve_depth,
    }
    return BindSelection(rest, selected, records, summary)


def select_training_bind(training_positions, pieces, *, sample_indices=None):
    """Minimal fitter interface: (rest, JSON-serializable report).

    Preserve each healthy default Drawable; replace only a rank-deficient rest
    with strictly higher-rank observed training geometry. ``sample_indices`` is
    optional provenance; without it, report indices address the training array.
    """
    selected = select_training_rest(training_positions, pieces,
                                    sample_indices=sample_indices)
    return selected.rest, {'summary': selected.summary, 'pieces': selected.pieces}
