"""Keep failed action prefixes and arbitrary episode identities intact."""

import json
from urllib.parse import unquote

import pytest

from experiments.trajectory import TrajectoryEnvironment, component
from tests.experiments.test_runner_replay import Action, Environment


def test_failed_action_is_not_recorded_as_an_executed_action(tmp_path, monkeypatch):
    base = Environment(("chair",), "normal", [])
    observed = TrajectoryEnvironment(base, directory=tmp_path, identities={}, dataset_manifest_sha256=None,
                                     actions=Action, action_metadata=lambda: {})
    observed.reset()
    observed.get_observation()
    observed.apply_action(Action.MOVE_FORWARD)
    observed.get_observation()

    def fail(*args, **kwargs):
        raise RuntimeError("environment step failed")

    monkeypatch.setattr(base, "apply_action", fail)
    with pytest.raises(RuntimeError, match="step failed"):
        observed.apply_action(Action.MOVE_FORWARD)
    rows = [json.loads(line) for line in observed.output.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["action_name"] == "MOVE_FORWARD"
    assert rows[0]["world_pose"][4] == 0
    assert base.frame == 1
    with pytest.raises(FileExistsError):
        TrajectoryEnvironment(base, directory=tmp_path, identities={}, dataset_manifest_sha256=None,
                              actions=Action, action_metadata=lambda: {})


@pytest.mark.parametrize("value", ["../episode", "..", ".", "episode/with/slash", "a%2Fb", "有空 格"])
def test_encoded_identity_remains_one_reversible_path_component(tmp_path, value):
    encoded = component(value)
    path = tmp_path / encoded
    path.mkdir()
    assert path.resolve().parent == tmp_path.resolve()
    assert unquote(encoded) == value
