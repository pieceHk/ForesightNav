"""Validate complete dataset identities and freeze selected episodes for workers."""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Optional

from experiments.dataset import build_identity_index, load_identity_index
from pbsp.types import EpisodeIdentity
from pbsp.mapping.spec import ModelInputSpec
from pbsp.runtime.config import ExperimentConfig, experiment_from_snapshot, read_yaml
from pbsp.runtime.resources import ResourcePaths

SCHEMA = "pbsp_launch_v1"


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, allow_nan=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


@dataclass(frozen=True)
class SceneSelection:
    scene_id: str
    scene_index: int
    episode_ids: tuple[str, ...]


@dataclass(frozen=True)
class RunPlan:
    experiment: ExperimentConfig
    resources: ResourcePaths
    scenes: tuple[SceneSelection, ...]
    identities: tuple[EpisodeIdentity, ...]
    source_commit: str
    model_specification: dict
    run_directory: Optional[Path] = None

    def __post_init__(self):
        if set(self.model_specification) != {"input", "camera"}:
            raise ValueError("Mapping model requires input and camera specifications")
        ModelInputSpec(**self.model_specification["input"])
        camera = self.model_specification["camera"]
        if (
            set(camera) != {"height", "width", "hfov_degrees"}
            or any(
                type(camera[key]) is not int or camera[key] < 1
                for key in ("height", "width")
            )
            or type(camera["hfov_degrees"]) not in (int, float)
            or not math.isfinite(camera["hfov_degrees"])
            or not 0 < camera["hfov_degrees"] < 180
        ):
            raise ValueError(
                "Camera requires positive dimensions and finite 0 < hfov < 180"
            )

    @property
    def configuration_sha256(self):
        return digest(
            {
                "experiment": self.experiment.snapshot(),
                "mapping_model": self.model_specification,
            }
        )

    @property
    def identity_sha256(self):
        return digest([asdict(item) for item in self.identities])

    @property
    def socket_path(self):
        if self.run_directory is None:
            raise ValueError("Run directory has not been allocated")
        suffix = hashlib.sha256(str(self.run_directory).encode()).hexdigest()[:20]
        return Path("/tmp") / f"pbsp_{suffix}.sock"

    def snapshot(self):
        return {
            "schema_version": SCHEMA,
            "configuration": self.experiment.snapshot(),
            "model_specification": self.model_specification,
            "configuration_sha256": self.configuration_sha256,
            "identities": [asdict(item) for item in self.identities],
            "identity_sha256": self.identity_sha256,
            "scenes": [asdict(item) for item in self.scenes],
            "resources": {
                name: str(value) for name, value in vars(self.resources).items()
            },
            "source_commit": self.source_commit,
            "run_directory": str(self.run_directory) if self.run_directory else None,
        }


def prepare_plan(
    experiment, *, root, source_commit, scenes=None, episodes=None, limit=None
):
    if experiment.runtime is None:
        raise ValueError("A runtime configuration is required")
    if (
        experiment.method.ground_truth_semantics
        or experiment.method.navigation.semantic_source != "detic"
    ):
        raise ValueError(
            "Full PBSP requires Detic semantics without ground-truth labels"
        )
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("--limit-episodes must be positive")
    resources = ResourcePaths(
        Path(root).resolve(), experiment.runtime.data, experiment.runtime.weights
    )
    dataset = resources.data / experiment.dataset.root
    config = experiment.dataset
    payload = build_identity_index(
        dataset,
        config.expected_tasks_per_episode,
        config.manifest_sha256,
        config.source_trace_sha256,
    )
    index = load_identity_index(dataset / config.identities, config.manifest_sha256)
    checked = {
        (row["scene_id"], row["runtime_episode_id"]): EpisodeIdentity(
            dataset_manifest_sha256=config.manifest_sha256, **row
        )
        for row in payload["episodes"]
    }
    if checked != index:
        raise ValueError("Prepared identity index differs from the validated dataset")
    if (
        len(index) != config.expected_episodes
        or len({key[0] for key in index}) != config.expected_scenes
    ):
        raise ValueError("Dataset cardinality differs from the selected configuration")
    available = list(dict.fromkeys(row["scene_id"] for row in payload["episodes"]))
    requested = set(scenes) if scenes is not None else set(available)
    if (
        not requested
        or len(requested) != len(scenes or requested)
        or not requested <= set(available)
    ):
        raise ValueError(f"Unknown, repeated or empty scene selection: {scenes}")
    if episodes is not None and (
        len(requested) != 1 or not episodes or len(set(episodes)) != len(episodes)
    ):
        raise ValueError("Episode IDs require one scene and distinct non-empty IDs")
    records = [identity for key, identity in checked.items() if key[0] in requested]
    if episodes is not None:
        selected_scene = next(iter(requested))
        if any((selected_scene, key) not in checked for key in episodes):
            raise ValueError("Episode ID is absent from the selected scene")
        records = [item for item in records if item.runtime_episode_id in set(episodes)]
    if limit is not None:
        records = records[:limit]
    selection = []
    for scene in available:
        ids = tuple(
            item.runtime_episode_id for item in records if item.scene_id == scene
        )
        if ids:
            selection.append(SceneSelection(scene, available.index(scene), ids))
    if not selection:
        raise ValueError("No episodes selected")
    specification = read_yaml(resources.root / "configs/models/embodied_splat.yaml")
    return RunPlan(
        experiment,
        resources,
        tuple(selection),
        tuple(records),
        source_commit,
        specification,
    )


def load_plan(path):
    def reject(value):
        raise ValueError(f"Non-finite value in launch plan: {value}")

    value = json.loads(Path(path).read_text(), parse_constant=reject)
    if (
        not isinstance(value, dict)
        or set(value)
        != {
            "schema_version",
            "configuration",
            "model_specification",
            "configuration_sha256",
            "identities",
            "identity_sha256",
            "scenes",
            "resources",
            "source_commit",
            "run_directory",
        }
        or value["schema_version"] != SCHEMA
    ):
        raise ValueError("Launch plan schema mismatch")
    experiment = experiment_from_snapshot(value["configuration"])
    if experiment.runtime is None or not value["run_directory"]:
        raise ValueError(
            "Launch plan requires runtime paths and an allocated run directory"
        )
    plan = RunPlan(
        experiment,
        ResourcePaths(
            **{name: Path(item) for name, item in value["resources"].items()}
        ),
        tuple(
            SceneSelection(
                item["scene_id"], item["scene_index"], tuple(item["episode_ids"])
            )
            for item in value["scenes"]
        ),
        tuple(EpisodeIdentity(**item) for item in value["identities"]),
        value["source_commit"],
        value["model_specification"],
        Path(value["run_directory"]),
    )
    if (
        plan.configuration_sha256 != value["configuration_sha256"]
        or plan.identity_sha256 != value["identity_sha256"]
    ):
        raise ValueError("Launch plan configuration or identity digest differs")
    expected = {(item.scene_id, item.runtime_episode_id) for item in plan.identities}
    selected = [
        (item.scene_id, key) for item in plan.scenes for key in item.episode_ids
    ]
    if (
        len(set(selected)) != len(selected)
        or set(selected) != expected
        or len(expected) != len(plan.identities)
    ):
        raise ValueError("Launch selection does not match its identity records")
    if (
        plan.resources.data != experiment.runtime.data
        or plan.resources.weights != experiment.runtime.weights
    ):
        raise ValueError("Launch resources differ from the resolved runtime")
    return plan
