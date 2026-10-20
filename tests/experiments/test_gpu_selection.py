"""Keep CLI GPU overrides explicit and valid before input checks."""

import argparse
from dataclasses import replace

import pytest

import run
from tests.runtime.test_native_launch import plan


@pytest.mark.parametrize("value", ["0", "0,1,2", "-1,0", "0,-1", "zero,1", "0,", ""])
def test_invalid_gpu_pair_is_rejected(value):
    with pytest.raises(argparse.ArgumentTypeError):
        run.gpu_pair(value)


def test_gpu_override_changes_effective_plan_without_editing_machine_settings(plan, monkeypatch):
    captured = []
    original = plan.experiment
    original_gpus = (original.runtime.mapping_gpu, original.runtime.navigation_gpu)
    monkeypatch.setattr(run, "load_experiment", lambda *args: original)
    monkeypatch.setattr(run, "source_info", lambda *args: {"commit": plan.source_commit})

    def prepare(config, **kwargs):
        captured.append(config)
        return replace(plan, experiment=config)

    monkeypatch.setattr(run, "prepare_plan", prepare)
    monkeypatch.setattr(run, "validate_resources", lambda *args: {"habitat_revision": "checked", "models": {}})
    monkeypatch.setattr(run, "launch", lambda *args, **kwargs: pytest.fail("--check started a worker"))
    run.main(["--gpus", "7,6", "--check"])
    assert (captured[0].runtime.mapping_gpu, captured[0].runtime.navigation_gpu) == (7, 6)
    assert captured[0] is not original
    assert (original.runtime.mapping_gpu, original.runtime.navigation_gpu) == original_gpus
