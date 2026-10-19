"""Resolve retained geometry controls once when assembling an experiment."""

from functools import partial
import os

from adapters.geometry import (
    project_goals_to_navigable,
    sample_navmesh_goal,
    strict_geodesic_distance,
)
from experiments.planning import PlanningBridge


def planning_bridge_from_environment(environ=None):
    environment = os.environ if environ is None else environ
    return PlanningBridge(
        project_goals=partial(
            project_goals_to_navigable,
            max_distance=float(environment.get("ES_NAV_PROJECTION_MAX_DISTANCE", "1.5")),
        ),
        geodesic=strict_geodesic_distance,
    )


def navmesh_sampler_from_environment(environ=None):
    environment = os.environ if environ is None else environ
    return partial(
        sample_navmesh_goal,
        max_vertical_delta=float(environment.get("ES_NAVMESH_MAX_VERTICAL_DELTA", "0.35")),
        sample_count=int(environment.get("ES_NAVMESH_SAMPLE_COUNT", "32")),
        escape_radius_m=float(environment.get("ES_NAVMESH_ESCAPE_RADIUS", "3.0")),
        escape_min_distance_m=float(environment.get("ES_NAVMESH_ESCAPE_MIN_DISTANCE", "1.25")),
        random_min_distance_m=float(environment.get("ES_NAVMESH_RANDOM_MIN_DISTANCE", "2.0")),
    )
