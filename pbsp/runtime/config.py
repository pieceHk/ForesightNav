"""Resolve experiment references without importing either model pbsp.runtime."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Mapping, Optional

import yaml

from pbsp.settings import (
    MappingSettings,
    MethodSettings,
    NavigationSettings,
    QuerySettings,
    SuffixSettings,
)


class ConfigError(ValueError):
    """A configuration is ambiguous or violates its declared schema."""


class _UniqueLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        result = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str) or key in result:
                raise ConfigError(f"Configuration keys must be unique strings: {key!r}")
            result[key] = self.construct_object(value_node, deep=deep)
        return result


def read_yaml(path: Path) -> dict:
    path = Path(path).resolve()
    try:
        value = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueLoader)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigError(f"Expected a configuration mapping: {path}")
    return value


def _keys(value, expected, context):
    expected = set(expected)
    if not isinstance(value, dict) or set(value) != expected:
        actual = set(value) if isinstance(value, dict) else set()
        raise ConfigError(
            f"{context}: missing {sorted(expected - actual)}, "
            f"unknown {sorted(actual - expected)}"
        )


def _typed(cls, value):
    _keys(value, (field.name for field in fields(cls)), cls.__name__)
    try:
        return cls(**value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{cls.__name__}: {exc}") from exc


def _positive(value, name):
    if type(value) is not int or value < 1:
        raise ConfigError(f"{name} must be a positive integer")


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{name} must be a non-empty string")


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    split: str
    root: str
    manifest: str
    identities: str
    manifest_sha256: str
    source_trace_sha256: str
    expected_tasks_per_episode: int
    expected_episodes: int
    expected_scenes: int

    def __post_init__(self):
        for name in (
            "name",
            "split",
            "root",
            "manifest",
            "identities",
            "manifest_sha256",
            "source_trace_sha256",
        ):
            _text(getattr(self, name), name)
        for name in ("root", "manifest", "identities"):
            path = Path(getattr(self, name))
            if path.is_absolute() or ".." in path.parts:
                raise ConfigError(
                    f"Dataset {name} must remain relative to its resource root"
                )
        for name in ("manifest_sha256", "source_trace_sha256"):
            digest = getattr(self, name)
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ConfigError(f"Dataset {name} must be a lowercase SHA-256")
        for name in (
            "expected_tasks_per_episode",
            "expected_episodes",
            "expected_scenes",
        ):
            _positive(getattr(self, name), name)


@dataclass(frozen=True)
class ObjectNavDatasetConfig:
    format: str
    name: str
    split: str
    root: str
    manifest: str
    manifest_sha256: str
    expected_episodes: int
    expected_scenes: int
    scene_directory: str
    scene_dataset: str

    def __post_init__(self):
        if self.format != 'objectnav' or self.split != 'val':
            raise ConfigError('Native ObjectNav requires format=objectnav and split=val')
        for name in ('name', 'root', 'manifest', 'scene_directory', 'scene_dataset'):
            value = getattr(self, name)
            _text(value, name)
            if name != 'name' and (Path(value).is_absolute() or '..' in Path(value).parts):
                raise ConfigError('ObjectNav resource paths must be relative: ' + name)
        digest = self.manifest_sha256
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ConfigError('ObjectNav manifest_sha256 must be a lowercase SHA-256')
        for name in ('expected_episodes', 'expected_scenes'):
            _positive(getattr(self, name), name)

    @property
    def expected_tasks_per_episode(self):
        return 1


def dataset_config(value):
    cls = ObjectNavDatasetConfig if isinstance(value, dict) and 'format' in value else DatasetConfig
    return _typed(cls, value)


@dataclass(frozen=True)
class EvaluationConfig:
    max_steps_per_task: int
    metrics: tuple[str, ...]
    record_decisions: bool
    visualize: bool

    def __post_init__(self):
        _positive(self.max_steps_per_task, "max_steps_per_task")
        metrics = tuple(self.metrics)
        if (
            not metrics
            or len(set(metrics)) != len(metrics)
            or not set(metrics) <= {"SR", "SPL", "PSR", "PeSPL"}
        ):
            raise ConfigError(
                "Evaluation metrics must be unique supported MON metric names"
            )
        if type(self.record_decisions) is not bool or type(self.visualize) is not bool:
            raise ConfigError("Evaluation recording options must be booleans")
        object.__setattr__(self, "metrics", metrics)


@dataclass(frozen=True)
class RuntimeConfig:
    data: Path
    weights: Path
    outputs: Path
    mapping_python: Path
    navigation_python: Path
    mapping_gpu: int
    navigation_gpu: int

    def __post_init__(self):
        for name in ("mapping_gpu", "navigation_gpu"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ConfigError(f"{name} must be a non-negative integer")


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    seed: int
    method: MethodSettings
    dataset: DatasetConfig
    evaluation: EvaluationConfig
    runtime: Optional[RuntimeConfig]
    source_sha256: Mapping[str, str]

    def snapshot(self) -> dict:
        """Return the effective configuration as JSON/YAML-safe values."""

        def plain(value):
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, dict):
                return {key: plain(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [plain(item) for item in value]
            return value

        return plain(asdict(self))


def load_runtime(path: Path) -> RuntimeConfig:
    path = Path(path).resolve()
    config = read_yaml(path)
    _keys(config, ("paths", "python", "gpus"), str(path))
    _keys(config["paths"], ("data", "weights", "outputs"), "pbsp.runtime.paths")
    _keys(config["python"], ("mapping", "navigation"), "pbsp.runtime.python")
    _keys(config["gpus"], ("mapping", "navigation"), "pbsp.runtime.gpus")

    def resource(value):
        _text(value, "runtime path")
        expanded = Path(value).expanduser()
        return (path.parent / expanded).resolve()

    return RuntimeConfig(
        **{key: resource(value) for key, value in config["paths"].items()},
        mapping_python=resource(config["python"]["mapping"]),
        navigation_python=resource(config["python"]["navigation"]),
        mapping_gpu=config["gpus"]["mapping"],
        navigation_gpu=config["gpus"]["navigation"],
    )


def load_experiment(
    path: Path, runtime_path: Optional[Path] = None
) -> ExperimentConfig:
    path = Path(path).resolve()
    experiment = read_yaml(path)
    _keys(experiment, ("name", "method", "dataset", "evaluation", "seed"), str(path))
    _text(experiment["name"], "experiment.name")
    if type(experiment["seed"]) is not int or experiment["seed"] < 0:
        raise ConfigError("Experiment seed must be a non-negative integer")
    sources = [path]
    sections = {}
    for name in ("method", "dataset", "evaluation"):
        _text(experiment[name], f"experiment.{name}")
        reference = (path.parent / experiment[name]).resolve()
        sources.append(reference)
        sections[name] = read_yaml(reference)

    settings = method_settings(sections["method"])
    runtime = load_runtime(runtime_path) if runtime_path is not None else None
    if runtime_path is not None:
        sources.append(Path(runtime_path).resolve())
    return ExperimentConfig(
        name=experiment["name"],
        seed=experiment["seed"],
        method=settings,
        dataset=dataset_config(sections["dataset"]),
        evaluation=_typed(EvaluationConfig, sections["evaluation"]),
        runtime=runtime,
        source_sha256={
            str(source): hashlib.sha256(source.read_bytes()).hexdigest()
            for source in sources
        },
    )


def method_settings(method: dict) -> MethodSettings:
    """Validate the method block from source YAML or a resolved run snapshot."""
    _keys(method, (field.name for field in fields(MethodSettings)), "method")
    resolved_method = dict(method)
    for name, cls in (
        ("mapping", MappingSettings),
        ("query", QuerySettings),
        ("navigation", NavigationSettings),
        ("suffix", SuffixSettings),
    ):
        resolved_method[name] = _typed(cls, method[name])
    if not isinstance(method["query_switch_cooldown_by_category"], dict):
        raise ConfigError("Category cooldowns must be a mapping")
    return _typed(MethodSettings, resolved_method)


def experiment_from_snapshot(value: dict) -> ExperimentConfig:
    """Restore the saved configuration without reading mutable source files."""
    _keys(
        value, (field.name for field in fields(ExperimentConfig)), "resolved experiment"
    )
    runtime = value["runtime"]
    if runtime is not None:
        _keys(
            runtime, (field.name for field in fields(RuntimeConfig)), "resolved runtime"
        )
        runtime = RuntimeConfig(
            **{
                name: (
                    Path(item)
                    if name not in {"mapping_gpu", "navigation_gpu"}
                    else item
                )
                for name, item in runtime.items()
            }
        )
    _text(value["name"], "experiment.name")
    if type(value["seed"]) is not int or value["seed"] < 0:
        raise ConfigError("Experiment seed must be a non-negative integer")
    return ExperimentConfig(
        name=value["name"],
        seed=value["seed"],
        method=method_settings(value["method"]),
        dataset=dataset_config(value["dataset"]),
        evaluation=_typed(EvaluationConfig, value["evaluation"]),
        runtime=runtime,
        source_sha256=value["source_sha256"],
    )
