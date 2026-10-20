"""Habitat navmesh projection, strict distances and recovery sampling."""

import numpy as np
from pbsp.types import GeodesicDistance


def project_goals_to_navigable(
    goals, pathfinder, *, return_indices=False, max_distance=1.5, merge_radius_m=0.25
):
    projected = {}
    projected_indices = {}
    for task_idx, candidates in (goals or {}).items():
        accepted = []
        accepted_indices = []
        for candidate_index, candidate in enumerate(candidates):
            source = np.asarray(candidate, dtype=np.float32)
            try:
                snapped = np.asarray(pathfinder.snap_point(source), dtype=np.float32)
            except Exception as exc:
                print(f"[CANDIDATE_PROJECTION] task={task_idx} failed={exc}", flush=True)
                continue
            if snapped.shape != (3,) or not np.isfinite(snapped).all():
                continue
            distance = float(np.linalg.norm(snapped[[0, 2]] - source[[0, 2]]))
            if distance > max_distance:
                continue
            if any(
                (
                    np.linalg.norm(snapped[[0, 2]] - np.asarray(old)[[0, 2]]) < merge_radius_m
                    for old in accepted
                )
            ):
                continue
            accepted.append(snapped.tolist())
            accepted_indices.append(candidate_index)
        projected[int(task_idx)] = accepted
        projected_indices[int(task_idx)] = accepted_indices
        print(
            f"[CANDIDATE_PROJECTION] task={task_idx} input={len(candidates)} accepted={len(accepted)}",
            flush=True,
        )
    if return_indices:
        return (projected, projected_indices)
    return projected


def strict_geodesic_distance(pathfinder, start, end):
    """Return an explicit Habitat geodesic status; never use Euclidean distance."""
    try:
        import habitat_sim
    except Exception:
        return GeodesicDistance.unavailable()
    try:
        path = habitat_sim.ShortestPath()
        path.requested_start = np.asarray(start, dtype=np.float32)
        path.requested_end = np.asarray(end, dtype=np.float32)
        found = bool(pathfinder.find_path(path))
    except Exception:
        return GeodesicDistance.unavailable()
    if not found:
        return GeodesicDistance.unreachable()
    try:
        distance = float(path.geodesic_distance)
    except (TypeError, ValueError, OverflowError):
        return GeodesicDistance.unavailable()
    if not np.isfinite(distance) or distance < 0.0:
        return GeodesicDistance.unavailable()
    return GeodesicDistance.available(distance)


def sample_navmesh_goal(
    pathfinder,
    origin,
    mode,
    *,
    max_vertical_delta=0.35,
    sample_count=32,
    escape_radius_m=3.0,
    escape_min_distance_m=1.25,
    random_min_distance_m=2.0,
):
    origin = np.asarray(origin, dtype=np.float32)
    if mode == "escape":
        radius = escape_radius_m
        min_distance = escape_min_distance_m
    elif mode == "random":
        radius = None
        min_distance = random_min_distance_m
    else:
        raise ValueError(f"unsupported navmesh sampling mode: {mode}")
    try:
        snapped_origin = np.asarray(pathfinder.snap_point(origin), dtype=np.float32)
        island = int(pathfinder.get_island(snapped_origin))
    except Exception as exc:
        return (None, {"error": f"origin_snap_failed:{exc}"})
    if snapped_origin.shape != (3,) or not np.isfinite(snapped_origin).all():
        return (None, {"error": "origin_not_navigable"})
    candidates = []
    vertical_rejections = 0
    island_rejections = 0
    for _ in range(sample_count):
        try:
            if mode == "escape":
                point = pathfinder.get_random_navigable_point_near(
                    snapped_origin, radius, 100, island
                )
            else:
                point = pathfinder.get_random_navigable_point(100, island)
            point = np.asarray(point, dtype=np.float32)
            if point.shape != (3,) or not np.isfinite(point).all():
                continue
            if int(pathfinder.get_island(point)) != island:
                island_rejections += 1
                continue
        except Exception:
            continue
        if abs(float(point[1] - snapped_origin[1])) > max_vertical_delta:
            vertical_rejections += 1
            continue
        distance = float(np.linalg.norm(point[[0, 2]] - snapped_origin[[0, 2]]))
        if distance < min_distance:
            continue
        candidates.append((distance, point))
    details = {
        "mode": mode,
        "sample_count": sample_count,
        "accepted": len(candidates),
        "vertical_rejections": vertical_rejections,
        "island_rejections": island_rejections,
        "max_vertical_delta": max_vertical_delta,
        "stairs_suspected": vertical_rejections >= max(3, sample_count // 4),
    }
    if not candidates:
        details["error"] = "no_same_floor_navigable_sample"
        return (None, details)
    (_, goal) = max(candidates, key=lambda item: item[0])
    details["goal"] = goal.tolist()
    return (goal, details)
