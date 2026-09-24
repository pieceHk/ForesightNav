#!/usr/bin/env python3
"""Build deterministic three-object MON episodes from HM3D ObjectNav."""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CATEGORY_ORDER = ("bed", "chair", "plant", "sofa", "toilet", "tv_monitor")
RUNTIME_CATEGORY_ALIASES = {"sofa": "couch", "tv_monitor": "tv"}


def runtime_category(category: str) -> str:
    return RUNTIME_CATEGORY_ALIASES.get(category, category)


def read_gzip_json(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def write_gzip_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        # Fixed mtime makes repeated builds byte-for-byte reproducible.
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as handle:
            handle.write(canonical_json_bytes(value))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scene_name(path: Path) -> str:
    suffix = ".json.gz"
    if not path.name.endswith(suffix):
        raise ValueError(f"unexpected content filename: {path.name}")
    return path.name[: -len(suffix)]


def choose_category_triplet(categories: list[str], group_index: int) -> list[str]:
    if len(categories) < 3:
        raise ValueError(f"MON needs at least three categories, got {categories}")
    start = group_index % len(categories)
    return [categories[(start + offset) % len(categories)] for offset in range(3)]


def episode_identity(scene: str, source_index: int, episode: dict[str, Any]) -> dict[str, Any]:
    return {
        "scene": scene,
        "source_index": source_index,
        "source_episode_id": str(episode["episode_id"]),
        "object_category": episode["object_category"],
    }


def build_scene(
    source_path: Path,
    output_path: Path,
    episodes_per_scene: int,
    allowed_categories: tuple[str, ...],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    source = read_gzip_json(source_path)
    episodes = source["episodes"]
    scene = scene_name(source_path)
    by_category: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for source_index, episode in enumerate(episodes):
        by_category[episode["object_category"]].append((source_index, episode))

    categories = [category for category in allowed_categories if category in by_category]
    output_episodes: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    category_offsets: Counter[str] = Counter()

    for group_index in range(episodes_per_scene):
        task_categories = choose_category_triplet(categories, group_index)
        selected: list[tuple[int, dict[str, Any]]] = []
        for category in task_categories:
            pool = by_category[category]
            pool_index = category_offsets[category] % len(pool)
            category_offsets[category] += 1
            selected.append(pool[pool_index])

        first = selected[0][1]
        runtime_categories = [runtime_category(category) for category in task_categories]
        output_episode = {
            "episode_id": group_index,
            "start_position": first["start_position"],
            "start_rotation": first["start_rotation"],
            "scene_id": first["scene_id"],
            "scene_dataset_config": first["scene_dataset_config"],
            "tasks": [[category, "object", None] for category in runtime_categories],
        }
        output_episodes.append(output_episode)
        trace.append(
            {
                "mon_episode_id": group_index,
                "scene": scene,
                "subtasks": [
                    {
                        "subtask_id": subtask_id,
                        "runtime_category": runtime_categories[subtask_id],
                        **episode_identity(scene, *item),
                    }
                    for subtask_id, item in enumerate(selected)
                ],
            }
        )

    # GOAT's loader resolves every object task by category from this dictionary.
    # HM3D ObjectNav already provides the exact Habitat ObjectGoal/viewpoint schema.
    goals = copy.deepcopy(source["goals_by_category"])
    goals = {
        key: objects
        for key, objects in goals.items()
        if objects and objects[0].get("object_category") in allowed_categories
    }
    for objects in goals.values():
        for goal in objects:
            goal["object_category"] = runtime_category(goal["object_category"])
    payload = {"episodes": output_episodes, "goals": goals}
    write_gzip_json(output_path, payload)
    summary = {
        "scene": scene,
        "source_episode_count": len(episodes),
        "source_task_count": len(episodes),
        "source_categories": categories,
        "source_category_counts": dict(sorted(Counter(e["object_category"] for e in episodes).items())),
        "mon_episode_count": len(output_episodes),
        "mon_subtask_count": len(output_episodes) * 3,
        "mon_category_counts": dict(
            sorted(Counter(task[0] for episode in output_episodes for task in episode["tasks"]).items())
        ),
        "source_sha256": sha256(source_path),
        "output_sha256": sha256(output_path),
    }
    return summary, trace, source


def validate_build(
    output_root: Path,
    summaries: list[dict[str, Any]],
    traces: list[dict[str, Any]],
    expected_episodes_by_scene: dict[str, int],
    expected_categories: set[str],
) -> dict[str, Any]:
    errors: list[str] = []
    observed_categories: Counter[str] = Counter()
    trace_keys = {(item["scene"], item["mon_episode_id"]) for item in traces}

    for summary in summaries:
        scene = summary["scene"]
        payload = read_gzip_json(output_root / "content" / f"{scene}.json.gz")
        episodes = payload.get("episodes", [])
        goals = payload.get("goals", {})
        expected_scene_episodes = expected_episodes_by_scene[scene]
        if len(episodes) != expected_scene_episodes:
            errors.append(
                f"{scene}: expected {expected_scene_episodes} episodes, got {len(episodes)}"
            )
        for episode in episodes:
            tasks = episode.get("tasks", [])
            categories = [task[0] for task in tasks]
            if len(tasks) != 3:
                errors.append(f"{scene}/{episode.get('episode_id')}: task count {len(tasks)}")
            if len(set(categories)) != 3:
                errors.append(f"{scene}/{episode.get('episode_id')}: categories are not distinct")
            if any(task[1:] != ["object", None] for task in tasks):
                errors.append(f"{scene}/{episode.get('episode_id')}: invalid task modality")
            for category in categories:
                matches = [
                    value
                    for value in goals.values()
                    if value and value[0].get("object_category") == category
                ]
                if len(matches) != 1:
                    errors.append(
                        f"{scene}/{episode.get('episode_id')}: category {category} "
                        f"has {len(matches)} goal dictionaries"
                    )
            if (scene, episode.get("episode_id")) not in trace_keys:
                errors.append(f"{scene}/{episode.get('episode_id')}: missing source trace")
            observed_categories.update(categories)

    expected_episodes = sum(expected_episodes_by_scene.values())
    expected_subtasks = expected_episodes * 3
    if len(traces) != expected_episodes:
        errors.append(f"trace count {len(traces)} != {expected_episodes}")
    if not expected_categories.issubset(observed_categories):
        errors.append(f"global category coverage is incomplete: {sorted(observed_categories)}")

    result = {
        "valid": not errors,
        "errors": errors,
        "scene_count": len(summaries),
        "mon_episode_count": expected_episodes,
        "mon_subtask_count": expected_subtasks,
        "unique_category_count": len(observed_categories),
        "category_counts": dict(sorted(observed_categories.items())),
        "constraints": {
            "three_subtasks_per_episode": True,
            "distinct_categories_within_episode": True,
            "all_source_scenes_covered": True,
            "all_source_categories_covered": expected_categories.issubset(observed_categories),
            "source_trace_complete": len(traces) == expected_episodes,
        },
    }
    if errors:
        raise ValueError("MON validation failed:\n" + "\n".join(errors))
    return result


def write_scene_summary(path: Path, summaries: list[dict[str, Any]]) -> None:
    columns = (
        "selection_order",
        "scene",
        "source_episode_count",
        "source_task_count",
        "source_category_count",
        "source_categories",
        "mon_episode_count",
        "mon_subtask_count",
        "mon_category_counts",
        "source_sha256",
        "output_sha256",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for index, summary in enumerate(summaries, 1):
            writer.writerow(
                {
                    "selection_order": index,
                    "scene": summary["scene"],
                    "source_episode_count": summary["source_episode_count"],
                    "source_task_count": summary["source_task_count"],
                    "source_category_count": len(summary["source_categories"]),
                    "source_categories": "|".join(summary["source_categories"]),
                    "mon_episode_count": summary["mon_episode_count"],
                    "mon_subtask_count": summary["mon_subtask_count"],
                    "mon_category_counts": json.dumps(summary["mon_category_counts"], sort_keys=True),
                    "source_sha256": summary["source_sha256"],
                    "output_sha256": summary["output_sha256"],
                }
            )


def write_readme(path: Path, manifest: dict[str, Any]) -> None:
    validation = manifest["validation"]
    text = f"""# {manifest['dataset_name']}

This dataset deterministically converts official HM3D ObjectNav `val`
episodes into three-object MON episodes.

## Counts

- Source: {manifest['source']['scene_count']} scenes, {manifest['source']['episode_count']} episodes/tasks.
- MON: {validation['scene_count']} scenes, {validation['mon_episode_count']} episodes,
  {validation['mon_subtask_count']} ordered object subtasks.
- Each MON episode has exactly three distinct categories and executes `0 -> 1 -> 2`.

## Runtime contract

`tasks` uses the GOAT object-subtask schema: `[category, "object", null]`. The
Habitat scene is loaded once per MON episode. Subtask transitions call only
`reset_subtask`; the agent, online map, and memory are reset only at the next
MON episode boundary.

`goals` is copied from the source ObjectNav `goals_by_category`, preserving the
official object instances and view points used by Habitat success metrics.
Only two runtime vocabulary aliases are applied: `sofa -> couch` and
`tv_monitor -> tv`; `source_trace.json` retains the official labels.

## JSON organization controls

- `content/<scene>.json.gz` partitions episodes by scene.
- `episode_id` controls identity within a scene file.
- `scene_id` and `scene_dataset_config` select scene assets/configuration.
- `start_position` and `start_rotation` control the initial agent state.
- `tasks` controls ordered target categories and subtask modality.
- `val_seen.json.gz` provides dataset-level category id mappings.

JSON cannot create missing geometry, semantic annotations, or navmeshes; those
remain properties of the HM3D scene assets.

`source_trace.json` records the exact original episode used for each subtask.
`validation.json` records machine-checked invariants.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-version", choices=("v1", "v2"), default="v1")
    parser.add_argument("--episodes-per-scene", type=int, default=20)
    parser.add_argument("--total-episodes", type=int)
    parser.add_argument("--scene-limit", type=int)
    parser.add_argument(
        "--exclude-category",
        action="append",
        default=[],
        choices=CATEGORY_ORDER,
        help="source category to exclude; may be provided more than once",
    )
    args = parser.parse_args()

    excluded_categories = set(args.exclude_category)
    allowed_categories = tuple(
        category for category in CATEGORY_ORDER if category not in excluded_categories
    )
    if len(allowed_categories) < 3:
        raise ValueError("MON needs at least three allowed categories")

    if args.dataset_version == "v1":
        source_content = args.source_root / "data/datasets/objectnav/hm3d/v1/val/content"
    else:
        source_content = args.source_root / "val/content"
    source_files = sorted(source_content.glob("*.json.gz"))
    if args.scene_limit is not None:
        source_files = source_files[: args.scene_limit]
    if not source_files:
        raise FileNotFoundError(f"no source content files below {source_content}")
    if args.episodes_per_scene < 1:
        raise ValueError("episodes-per-scene must be positive")
    if args.total_episodes is not None and args.total_episodes < len(source_files):
        raise ValueError("total-episodes must be at least the selected scene count")

    if args.total_episodes is None:
        episode_counts = [args.episodes_per_scene] * len(source_files)
    else:
        quotient, remainder = divmod(args.total_episodes, len(source_files))
        episode_counts = [
            quotient + (1 if index < remainder else 0)
            for index in range(len(source_files))
        ]
    expected_episodes_by_scene = {
        scene_name(path): count for path, count in zip(source_files, episode_counts)
    }

    output_root = args.output_root.resolve()
    content_root = output_root / "content"
    output_root.mkdir(parents=True, exist_ok=False)
    content_root.mkdir()

    summaries: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    source_episode_count = 0
    source_categories: Counter[str] = Counter()
    for source_path, episode_count in zip(source_files, episode_counts):
        summary, scene_trace, source = build_scene(
            source_path,
            content_root / source_path.name,
            episode_count,
            allowed_categories,
        )
        summaries.append(summary)
        traces.extend(scene_trace)
        source_episode_count += len(source["episodes"])
        source_categories.update(e["object_category"] for e in source["episodes"])

    runtime_categories = [runtime_category(category) for category in allowed_categories]
    category_mapping = {category: index for index, category in enumerate(runtime_categories)}
    index_payload = {
        "episodes": [],
        "category_to_task_category_id": category_mapping,
        "category_to_scene_annotation_category_id": category_mapping,
    }
    write_gzip_json(output_root / "val_seen.json.gz", index_payload)
    write_json(output_root / "source_trace.json", traces)
    write_scene_summary(output_root / "scene_summary.csv", summaries)

    validation = validate_build(
        output_root,
        summaries,
        traces,
        expected_episodes_by_scene,
        {
            runtime_category(category)
            for category in source_categories
            if category in allowed_categories
        },
    )
    write_json(output_root / "validation.json", validation)
    manifest = {
        "schema_version": 1,
        "dataset_name": f"HM3D-{args.dataset_version} MON val",
        "source": {
            "root": str(args.source_root.resolve()),
            "split": "val",
            "scene_count": len(source_files),
            "episode_count": source_episode_count,
            "task_count": source_episode_count,
            "category_counts": dict(sorted(source_categories.items())),
        },
        "selection_policy": {
            "deterministic": True,
            "requested_total_episodes": args.total_episodes,
            "episodes_per_scene": (
                args.episodes_per_scene if args.total_episodes is None else None
            ),
            "episodes_by_scene": expected_episodes_by_scene,
            "subtasks_per_episode": 3,
            "category_order": list(CATEGORY_ORDER),
            "allowed_categories": list(allowed_categories),
            "excluded_categories": sorted(excluded_categories),
            "runtime_category_aliases": RUNTIME_CATEGORY_ALIASES,
            "category_triplet": "cyclic rotation over categories available in each scene",
            "source_episode_selection": "round-robin within each selected category",
            "episode_start": "start pose of source episode selected for subtask 0",
        },
        "runtime_contract": {
            "subtask_order": [0, 1, 2],
            "reload_scene_between_subtasks": False,
            "reset_map_between_subtasks": False,
            "reset_memory_between_subtasks": False,
            "reset_at_next_mon_episode": True,
        },
        "scene_summary": summaries,
        "validation": validation,
    }
    write_json(output_root / "dataset_manifest.json", manifest)
    write_readme(output_root / "README.md", manifest)
    print(json.dumps(validation, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
