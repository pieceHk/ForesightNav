"""Replace local navigation while PBSP owns candidate order and verification."""

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Protocol


@dataclass(frozen=True)
class NavigationFrame:
    """RGB, metric depth, world [qw,qx,qy,qz,x,y,z], and episode-local GPS/compass.

    Habitat world uses metres and y up. GPS and compass retain the simulator's
    episode frame. Evaluation labels and the ground-truth top-down map are omitted.
    """

    rgb: Any
    depth_m: Any
    world_pose: Any
    gps: Any
    compass: Any


@dataclass(frozen=True)
class NavigationRequest:
    frame: NavigationFrame
    waypoint_world: Any
    task_idx: int
    task_step: int
    scene_id: str
    episode_id: str
    local_maps: Mapping[str, Any]
    use_dilation_for_stg: bool
    version_token: Optional[Mapping[str, Any]] = None


@dataclass(frozen=True)
class NavigationPlan:
    """Actions use the simulator adapter's enum; a planner STOP needs verification."""

    action: Any
    closest_goal_map: Any = None
    short_term_goal: Any = None
    dilated_obstacle_map: Any = None
    no_path: bool = False
    planner_stop: bool = False
    error: Optional[str] = None

    def values(self):
        return (
            self.action,
            self.closest_goal_map,
            self.short_term_goal,
            self.dilated_obstacle_map,
            self.no_path,
            self.planner_stop,
        )


class NavigationBackend(Protocol):
    def reset_episode(self): ...

    def reset_task(self): ...

    def plan(self, request: NavigationRequest) -> NavigationPlan: ...

    def close(self): ...


class LocalPlannerBackend:
    """Adapt the installed map planner without changing its arguments or results."""

    def __init__(self, planner):
        self.planner = planner

    def reset_episode(self):
        self.planner.reset()

    def reset_task(self):
        if hasattr(self.planner, "reset_subtask_state"):
            self.planner.reset_subtask_state()
        else:
            self.planner.dd = None
            self.planner.reset_collision_state()

    def plan(self, request):
        values = self.planner.plan(
            **request.local_maps,
            use_dilation_for_stg=request.use_dilation_for_stg,
            timestep=request.task_step,
            debug=False,
        )
        return NavigationPlan(
            *values, error=getattr(self.planner, "last_plan_error", None)
        )

    def close(self):
        close = getattr(self.planner, "close", None)
        if close is not None:
            close()


class WaypointNavigationBackend:
    """Use an external waypoint controller with a separate exploration policy.

    A controller's step(request) returns NavigationPlan and translates its native
    action to the simulator enum. Full VLFM/ApexNav policies need their own adapters.
    """

    def __init__(self, controller, *, exploration: NavigationBackend):
        self.controller = controller
        self.exploration = exploration

    def reset_episode(self):
        self.controller.reset_episode()
        self.exploration.reset_episode()

    def reset_task(self):
        self.controller.reset_task()
        self.exploration.reset_task()

    def plan(self, request):
        if request.waypoint_world is None:
            return self.exploration.plan(request)
        result = self.controller.step(request)
        if not isinstance(result, NavigationPlan):
            raise TypeError("Waypoint controller must return NavigationPlan")
        return result

    def close(self):
        try:
            self.controller.close()
        finally:
            self.exploration.close()


def navigation_backend(agent) -> NavigationBackend:
    backend = getattr(agent, "navigation_backend", None)
    return LocalPlannerBackend(agent.planner) if backend is None else backend
