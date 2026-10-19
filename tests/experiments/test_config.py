"""Check protocol parity and reject ambiguous experiment inputs."""

from pathlib import Path

import pytest
import yaml

try:
    import tomllib
except ImportError:
    import tomli as tomllib

from runtime.config import ConfigError, load_experiment, read_yaml


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("count", (2, 3, 4))
def test_new_configuration_preserves_frozen_full_method(count):
    config = load_experiment(ROOT / f"configs/experiments/hm3d_{count}mon.yaml")
    suffix = "mon" if count == 3 else f"{count}mon"
    frozen = tomllib.loads((ROOT / f"Exp/profiles/hm3dv1_{suffix}_scsp_suffix.toml").read_text())
    for key, value in frozen["mapping"].items():
        assert getattr(config.method.mapping, key) == value
    for key, value in frozen["navigation"].items():
        assert getattr(config.method.navigation, key) == value
    for key, value in frozen["scsp"].items():
        if key not in {"arm", "ordered_subtasks", "portable_dataset", "dataset_manifest_sha256"}:
            actual = getattr(config.method.suffix, key)
            assert list(actual) == value if isinstance(value, list) else actual == value
    assert config.method.ground_truth_semantics == frozen["ground_truth_semantics"]
    assert config.method.navigation.semantic_source == frozen["semantic_source"]
    assert config.method.query_switch_cooldown_by_category == frozen["query_switch_cooldown_by_category"]
    assert config.evaluation.max_steps_per_task == frozen["max_steps_per_subtask"]
    assert config.seed == frozen["seed"]
    assert config.dataset.expected_tasks_per_episode == count
    assert config.dataset.expected_episodes * count == 1200


def test_target_counts_share_exactly_one_method_and_evaluation():
    configurations = [load_experiment(ROOT / f"configs/experiments/hm3d_{count}mon.yaml") for count in (2, 3, 4)]
    assert all(config.method == configurations[0].method for config in configurations)
    assert all(config.evaluation == configurations[0].evaluation for config in configurations)
    assert len({config.dataset.manifest_sha256 for config in configurations}) == 3


def test_yaml_rejects_duplicate_keys(tmp_path):
    path = tmp_path / "duplicate.yaml"
    path.write_text("seed: 1\nseed: 2\n")
    with pytest.raises(ConfigError, match="unique"):
        read_yaml(path)


def _experiment_files(tmp_path):
    source = ROOT / "configs/experiments/hm3d_4mon.yaml"
    experiment = read_yaml(source)
    for name in ("method", "dataset", "evaluation"):
        value = read_yaml(source.parent / experiment[name])
        (tmp_path / f"{name}.yaml").write_text(yaml.safe_dump(value))
        experiment[name] = f"{name}.yaml"
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(experiment))
    return path


@pytest.mark.parametrize("field,value", (("suffix_weight", float("nan")), ("candidate_cap", True), ("strict_geodesic", False)))
def test_method_rejects_invalid_numeric_and_distance_controls(tmp_path, field, value):
    path = _experiment_files(tmp_path)
    method_path = tmp_path / "method.yaml"
    method = read_yaml(method_path)
    method["suffix"][field] = value
    method_path.write_text(yaml.safe_dump(method))
    with pytest.raises(ConfigError):
        load_experiment(path)


def test_unknown_parameter_is_reported(tmp_path):
    path = _experiment_files(tmp_path)
    method_path = tmp_path / "method.yaml"
    method = read_yaml(method_path)
    method["navigation"]["query_w_semantics"] = 99.0
    method_path.write_text(yaml.safe_dump(method))
    with pytest.raises(ConfigError, match="unknown.*query_w_semantics"):
        load_experiment(path)


def test_file_references_and_resource_paths_ignore_process_cwd(tmp_path, monkeypatch):
    path = _experiment_files(tmp_path)
    runtime_path = tmp_path / "runtime.yaml"
    runtime_path.write_text(yaml.safe_dump({
        "paths": {"data": "data", "weights": "weights", "outputs": "outputs"},
        "python": {"mapping": "env-map/bin/python", "navigation": "env-nav/bin/python"},
        "gpus": {"mapping": 2, "navigation": 3},
    }))
    monkeypatch.chdir(tmp_path.parent)
    config = load_experiment(path, runtime_path)
    assert config.runtime.data == tmp_path / "data"
    assert config.runtime.mapping_python == tmp_path / "env-map/bin/python"
    assert len(config.source_sha256) == 5
    assert config.snapshot()["runtime"]["data"] == str(tmp_path / "data")


def test_resolved_method_is_independent_of_inherited_algorithm_environment(monkeypatch):
    path = ROOT / "configs/experiments/hm3d_4mon.yaml"
    expected = load_experiment(path).snapshot()
    monkeypatch.setenv("ES_SCSP_SUFFIX_WEIGHT", "999")
    monkeypatch.setenv("ES_QUERY_W_SEMANTIC", "999")
    assert load_experiment(path).snapshot() == expected
