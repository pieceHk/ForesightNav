"""Exercise controller replacement through the real Navigator action path."""

import numpy as np
import pytest

from pbsp.navigation.backends import (
    LocalPlannerBackend,
    NavigationPlan,
    WaypointNavigationBackend,
)
from test_navigator_replay import (
    DiscreteNavigationAction as Action,
    NativeAgent,
    make_agent,
    observation,
)


class Controller:
    def __init__(self, action=Action.TURN_LEFT):
        self.action = action
        self.requests = []
        self.resets = []
        self.closed = False

    def step(self, request):
        self.requests.append(request)
        return NavigationPlan(self.action, planner_stop=self.action == Action.STOP)

    def reset_episode(self):
        self.resets.append("episode")

    def reset_task(self):
        self.resets.append("task")

    def close(self):
        self.closed = True


def actor_with_controller(goals=([0.0, 0.0, -5.0],), action=Action.TURN_LEFT):
    actor = make_agent(NativeAgent, goals=goals)
    controller = Controller(action)
    actor.navigation_backend = WaypointNavigationBackend(
        controller,
        exploration=LocalPlannerBackend(actor.planner),
    )
    return actor, controller


def test_external_waypoint_controller_receives_sensor_frame_and_selected_world_goal():
    actor, controller = actor_with_controller()
    token = {"scene_id": "scene-a", "episode_id": "0", "map_revision": 7}
    actor.last_external_ranking_metadata = {"version_token": token}
    frame = observation()
    action, _ = actor.act(frame)
    assert action is Action.TURN_LEFT
    assert actor.planner.calls == []
    (request,) = controller.requests
    np.testing.assert_array_equal(request.waypoint_world, [0.0, 0.0, -5.0])
    assert request.task_idx == 0 and request.task_step == 1
    assert (request.scene_id, request.episode_id) == ("scene-a", "0")
    assert request.version_token == token
    assert request.frame.depth_m is frame.depth
    assert request.frame.world_pose is frame.globalpose
    assert not hasattr(request.frame, "task_observations")
    assert not hasattr(request.frame, "top_down_map")
    assert not hasattr(request.frame, "success")


def test_exploration_falls_back_until_a_waypoint_exists():
    actor, controller = actor_with_controller(goals=())
    action, _ = actor.act(observation())
    assert action is Action.MOVE_FORWARD
    assert controller.requests == []
    assert len(actor.planner.calls) == 1


@pytest.mark.parametrize(
    "step,expected,reason",
    ((1, Action.TURN_RIGHT, "planner_error"), (500, Action.STOP, "max_steps")),
)
def test_external_stop_keeps_the_verified_target_and_budget_rules(
    step, expected, reason
):
    actor, _ = actor_with_controller(action=Action.STOP)
    actor.sub_task_timesteps[0][0] = actor.total_timesteps[0] = step
    action, _ = actor.act(observation())
    assert action is expected
    assert actor.last_action_metadata["stop_reason"] == reason


def test_task_reset_and_close_reach_both_controller_and_exploration():
    actor, controller = actor_with_controller()
    actor.reset_sub_episode()
    assert controller.resets == ["task"]
    assert actor.planner.resets == 1
    actor.navigation_backend.close()
    assert controller.closed is True
