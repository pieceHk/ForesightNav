"""Eq.(11)--(13) candidate ordering, in geodesic metres.

The immutable snapshot is also the complete input for calibration replay. No
benchmark labels, perception models or simulator imports enter the recursion.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
import time

import numpy as np

from pbsp.types import GeodesicStatus
from .candidates import reconcile_spatial_candidates
from .planning import active_candidate, bind_projected_candidates


@dataclass(frozen=True)
class BeliefParameters:
    a: float = 10.0
    b0: float = -2.5
    pi: float = 0.3
    c_exp: float = 10.0

    def __post_init__(self):
        if not all(math.isfinite(v) for v in (self.a, self.b0, self.pi, self.c_exp)):
            raise ValueError('Calibration parameters must be finite')
        if self.a <= 0 or not 0 <= self.pi <= 1 or self.c_exp < 0:
            raise ValueError('Invalid calibration parameters')

    def evidence(self, semantic, n_views):
        z = self.a * float(semantic) + self.b0
        q = 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))
        s = -math.expm1(-int(n_views) / 3.0)
        return s, q, s * q + (1 - s) * self.pi


def solve_snapshot(snapshot, parameters, variant='full', active_raw_rank=None,
                   switch_delta_m=1.0):
    """Solve each root Q exactly with (task, location, removed-mask) memoization.

    Later tasks always enter with their complete candidate set. Exhausting one
    task never mutates the candidate sets belonging to later ordered tasks.
    """
    if variant not in {'nearest', 'suffix_only', 'myopic', 'full'}:
        raise ValueError('Unknown BLP variant')
    started = time.perf_counter()
    k = snapshot['k']
    layers = snapshot['layers']
    distances = snapshot['distances']
    frontier = snapshot['nearest_frontier']
    n_tasks = len(layers)
    fallback_counts = {'empty_candidates': 0, 'no_reachable_frontier': 0,
                       'frontier_coordinates_unavailable': 0}
    beliefs = {}
    for layer in layers:
        for item in layer:
            s, q, b = parameters.evidence(item['E'], item['N_views'])
            beliefs[item['location']] = (s, q, b)

    def distance(x, y):
        value = distances[x][y]
        return float('inf') if value is None else value

    def branch(m, x, removed, index):
        item = layers[m][index]
        y = item['location']
        d = distance(x, y)
        if not math.isfinite(d):
            return float('inf'), 0.0, 0.0
        b = 1.0 if variant == 'suffix_only' else beliefs[y][2]
        succ = value(m + 1, y, 0) if b > 0 else 0.0
        fail = value(m, y, removed | (1 << index)) if b < 1 else 0.0
        return d + b * succ + (1 - b) * fail, succ, fail

    @lru_cache(None)
    def value(m, x, removed):
        if m >= n_tasks or (variant == 'myopic' and m > k):
            return 0.0
        remaining = [i for i in range(len(layers[m])) if not removed & (1 << i)]
        if not remaining:
            fallback_counts['empty_candidates'] += 1
            if not snapshot['frontier_available']:
                fallback_counts['frontier_coordinates_unavailable'] += 1
                return parameters.c_exp + value(m + 1, x, 0)
            f = frontier[x]
            if f is None:
                fallback_counts['no_reachable_frontier'] += 1
                return 1000.0
            return distance(x, f) + parameters.c_exp + value(m + 1, f, 0)
        return min(branch(m, x, removed, i)[0] for i in remaining)

    rows = []
    for i, item in enumerate(layers[k]):
        location = item['location']
        s, q, b = beliefs[location]
        if variant == 'nearest':
            total, succ, fail = distance(0, location), 0.0, 0.0
        else:
            total, succ, fail = branch(k, 0, 0, i)
        rows.append(dict(item, S=s, q=q, B=1.0 if variant == 'suffix_only' else b,
                         d=distance(0, location), V_succ=succ, V_fail=fail, Q=total))
    rows.sort(key=lambda item: (item['Q'], item['raw_rank']))
    nearest = min(rows, key=lambda item: (item['d'], item['raw_rank'])) if rows else None
    active = next((r for r in rows if r['raw_rank'] == active_raw_rank), None)
    held = False
    proposal = rows[0]['raw_rank'] if rows else None
    if active and rows and rows[0] is not active:
        if not rows[0]['Q'] < active['Q'] - switch_delta_m:
            rows.remove(active)
            rows.insert(0, active)
            held = True
    order = [row['raw_rank'] for row in rows]
    order.extend(sorted(set(snapshot['all_raw_ranks']) - set(order)))
    chosen = rows[0]['raw_rank'] if rows else None
    return {'variant': variant, 'k': k, 'candidates': rows,
            'chosen_raw_rank': chosen,
            'unconstrained_raw_rank': proposal,
            'nearest_raw_rank': None if nearest is None else nearest['raw_rank'],
            'ordered_raw_ranks': order, 'hysteresis_held': held,
            'switched': active is not None and chosen != active_raw_rank,
            'fallback': 'empty_candidates' if not rows else None,
            'fallback_counts': fallback_counts, 'memo_states': value.cache_info().currsize,
            'plan_ms': (time.perf_counter() - started) * 1000}


def build_snapshot(*, candidate_batch, projected_goals, projected_indices,
                   current_task, origin, geodesic, agent, distance_cache,
                   query_future=True, project_frontier=None):
    """Construct raw-rank top-five sets after the native blacklist filter."""
    started = time.perf_counter()
    batch = bind_projected_candidates(candidate_batch, projected_goals, projected_indices)
    filtered, _, _ = reconcile_spatial_candidates(
        getattr(agent, 'gs_poses', {}), projected_goals,
        getattr(agent, 'frontier_flag', {}),
        getattr(agent, 'visited_candidate_positions', {}),
        getattr(agent, 'candidate_match_radius', 1.0),
        getattr(agent, 'candidate_blacklist_radius', 1.5), preserve_active=False)
    token = batch.version_token
    for key in list(distance_cache):
        if key[0] != token:
            del distance_cache[key]
    hits = misses = 0

    def measured(start, end):
        nonlocal hits, misses
        a, b = tuple(map(float, start)), tuple(map(float, end))
        if a == b:
            return 0.0
        key = (token, 'blp', a, b)
        if key in distance_cache:
            hits += 1
        else:
            distance_cache[key] = geodesic(a, b)
            misses += 1
        result = distance_cache[key]
        return result.distance_m if result.status is GeodesicStatus.AVAILABLE else float('inf')

    points = [list(map(float, origin))]
    layers = [[] for _ in batch.ordered_categories]
    bank = getattr(agent, 'candidate_evidence', None)
    for task in range(current_task, len(layers) if query_future else current_task + 1):
        for record in sorted(batch.records_for_task(task), key=lambda r: r.raw_rank):
            if not any(np.allclose(record.position, p, atol=1e-6, rtol=0)
                       for p in filtered.get(task, [])):
                continue
            if not math.isfinite(measured(origin, record.position)):
                continue
            location = len(points)
            points.append(list(record.position))
            n_views = bank.coverage_views(task, record.position) if bank else 0
            layers[task].append({'raw_rank': record.raw_rank, 'position': list(record.position),
                                 'E': record.semantic_score, 'N_views': n_views,
                                 'location': location})
            if len(layers[task]) == 5:
                break
    frontier_points = getattr(agent, 'blp_frontier_world', None)
    if frontier_points is not None and project_frontier is not None:
        snapped = []
        seen = set()
        for point in frontier_points:
            target = np.asarray(project_frontier(point), dtype=float)
            if target.shape != (3,) or not np.isfinite(target).all():
                continue
            if np.linalg.norm(target[[0, 2]] - np.asarray(point)[[0, 2]]) > 1.5:
                continue
            key = tuple(target.tolist())
            if key not in seen:
                seen.add(key)
                snapped.append(list(key))
        frontier_points = snapped
    nearest = []
    # The Euclidean lower bound prunes only points that cannot beat the current
    # geodesic best; no frontier subsampling or Euclidean fallback is used.
    for point in tuple(points):
        best, best_d = None, float('inf')
        sorted_frontiers = sorted(frontier_points or [],
                                  key=lambda f: float(np.linalg.norm(np.array(f) - point)))
        for f in sorted_frontiers:
            if float(np.linalg.norm(np.array(f) - point)) > best_d:
                break
            d = measured(point, f)
            if d < best_d:
                best, best_d = f, d
        if best is None:
            nearest.append(None)
        else:
            if best not in points:
                points.append(best)
            nearest.append(points.index(best))
    for i in range(len(nearest), len(points)):
        nearest.append(i)  # The selected frontier itself is its nearest frontier.
    matrix = [[measured(a, b) for b in points] for a in points]
    snapshot = {'k': int(current_task), 'layers': layers, 'points': points,
                'distances': [[v if math.isfinite(v) else None for v in row] for row in matrix],
                'nearest_frontier': nearest, 'frontier_available': frontier_points is not None,
                'all_raw_ranks': [r.raw_rank for r in batch.records_for_task(current_task)],
                'distance_cache': {'hits': hits, 'misses': misses},
                'snapshot_ms': (time.perf_counter() - started) * 1000}
    return batch, snapshot
