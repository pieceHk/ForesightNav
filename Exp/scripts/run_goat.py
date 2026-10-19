#!/usr/bin/env python3
"""Run the GOAT-Bench object-only profile through 3DONav ownership boundaries."""

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

from _runtime import (
    ROOT,
    create_run,
    expand,
    git_head,
    load_toml,
    mapping_server,
    navigation_environment,
    resource_monitor,
    send_socket,
    socket_for_run,
    write_json,
    write_status,
)

NAVIGATION_SRC = ROOT / "Navigation" / "src"
if str(NAVIGATION_SRC) not in sys.path:
    sys.path.insert(0, str(NAVIGATION_SRC))

from experiments.metrics import (  # noqa: E402
    aggregate_run_artifacts,
    write_json_strict,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, default=ROOT / "Exp/runtime.local.toml")
    parser.add_argument("--profile", type=Path, default=ROOT / "Exp/profiles/goat_val_seen.toml")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--scene-start", type=int, default=0)
    parser.add_argument("--scene-end", type=int)
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--episode-end", type=int)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    es_gpu, nav_gpu = (int(value) for value in args.gpus.split(",", 1))
    profile = load_toml(args.profile)
    runtime = load_toml(args.runtime)
    max_steps_per_subtask = int(profile["max_steps_per_subtask"])
    if max_steps_per_subtask < 1:
        raise ValueError("max_steps_per_subtask must be positive")
    visualization = profile.get("visualization", {})
    dataset = Path(expand(runtime["datasets"][profile["dataset"]]))
    scenes = [row["scene"] for row in csv.DictReader((dataset / "scene_summary.csv").open())]
    end = len(scenes) if args.scene_end is None else min(args.scene_end, len(scenes))
    if not 0 <= args.scene_start < end:
        raise ValueError(f"invalid scene range [{args.scene_start}, {end})")
    run = create_run(profile, args.tag)
    socket_path = socket_for_run(run)
    shutil.copy2(args.profile, run / "profile.toml")
    shutil.copy2(dataset / "dataset_manifest.json", run / "dataset_manifest.json")
    write_json(
        run / "provenance.json",
        {
            "git_head": git_head(),
            "dataset": str(dataset),
            "scene_start": args.scene_start,
            "scene_end": end,
            "episode_start": args.episode_start,
            "episode_end": args.episode_end,
            "mapping_gpu": es_gpu,
            "navigation_gpu": nav_gpu,
            "max_steps_per_subtask": max_steps_per_subtask,
            "per_keyframe_snapshots": bool(
                visualization.get("per_keyframe_snapshots", False)
            ),
            "mon_metrics_schema": "mon_metrics_v3",
            "unreachable_shortest_path_convention": (
                "only failed subtasks may encode Habitat +inf as strict-JSON null; "
                "SR/PSR fail, SPL/PeSPL contribute zero, M*n is unchanged, and "
                "navigation efficiency reports finite coverage separately"
            ),
            "mon_metric_definitions": {
                "sr": "mean episode-all success over complete MON episodes",
                "spl": "mean episode-all success weighted by summed path efficiency",
                "pespl": "sum of successful subtask path-efficiency contributions divided by M*n",
                "psr": "successful ordered subtasks divided by M*n",
            },
            "frame_time_definition": (
                "one action-loop wall time from observation fetch through the "
                "Habitat action, including mapping IPC, scheduled rank refresh, "
                "policy inference and simulator step; excluding reset and metric "
                "serialization"
            ),
            "path_efficiency_convention": (
                "shortest_path_length / max(actual_path_length, "
                "shortest_path_length)"
            ),
            "in_place_resume_supported": False,
        },
    )
    write_status(run, "starting")
    try:
        with resource_monitor(run, (es_gpu, nav_gpu)), mapping_server(
            runtime, profile, run, socket_path, es_gpu
        ):
            write_status(run, "running", scene_start=args.scene_start, scene_end=end)
            env = navigation_environment(runtime, profile, nav_gpu)
            python = expand(runtime["python"]["navigation"])
            for scene_index in range(args.scene_start, end):
                scene = scenes[scene_index]
                write_status(run, "running", scene_index=scene_index, scene=scene)
                log_path = run / "logs" / f"navigation_{scene_index:02d}_{scene}.log"
                command = [
                    python,
                    "-m",
                    "tdonav_navigation.adapters.lagmemo_evaluator",
                    "--entrypoint",
                    "goat",
                    "--scene_idx",
                    str(scene_index),
                    "--scenes",
                    scene,
                    "--episode_start",
                    str(args.episode_start),
                    "--dataset_root",
                    str(dataset),
                    "--output_path",
                    str(run),
                    "--socket_path",
                    str(socket_path),
                    "--top_k",
                    str(profile["mapping"]["top_k"]),
                    "--max_steps_per_subtask",
                    str(max_steps_per_subtask),
                    "--live_mode",
                    "--enable_timing",
                ]
                if args.episode_end is not None:
                    command.extend(["--episode_end", str(args.episode_end)])
                command.append(
                    "--print-images"
                    if visualization.get("per_keyframe_snapshots", False)
                    else "--no-print-images"
                )
                with log_path.open("w", encoding="utf-8") as log:
                    result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                if result.returncode:
                    raise RuntimeError(f"Navigation failed for {scene}: {result.returncode}")
                write_json_strict(
                    run / "mon_metrics.json", aggregate_run_artifacts(run)
                )
                send_socket(socket_path, {"cmd": "flush"})
        write_status(run, "completed", scene_start=args.scene_start, scene_end=end)
    except BaseException as error:
        write_status(run, "failed", error=repr(error))
        raise
    finally:
        print(run)


if __name__ == "__main__":
    main()
