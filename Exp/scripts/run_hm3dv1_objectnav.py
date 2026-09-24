#!/usr/bin/env python3
"""Run HM3Dv1 single-target ObjectNav through 3DONav boundaries."""

import argparse
import shutil
import subprocess
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
    socket_for_run,
    write_json,
    write_status,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, default=ROOT / "Exp/runtime.local.toml")
    parser.add_argument("--profile", type=Path, default=ROOT / "Exp/profiles/hm3dv1_objectnav_val.toml")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--episode-end", type=int, default=2000)
    parser.add_argument("--episode-indices", default="")
    parser.add_argument("--resume-metrics", type=Path, default=None)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    es_gpu, nav_gpu = (int(value) for value in args.gpus.split(",", 1))
    profile = load_toml(args.profile)
    runtime = load_toml(args.runtime)
    visualization = profile.get("visualization", {})
    dataset = Path(expand(runtime["datasets"][profile["dataset"]]))
    dataset_file = dataset / "data/datasets/objectnav/hm3d/v1/val/val.json.gz"
    if not dataset_file.is_file():
        raise FileNotFoundError(
            "HM3Dv1 ObjectNav dataset is incomplete; expected "
            f"{dataset_file}"
        )
    run = create_run(profile, args.tag)
    socket_path = socket_for_run(run)
    shutil.copy2(args.profile, run / "profile.toml")
    write_json(
        run / "provenance.json",
        {
            "git_head": git_head(),
            "dataset": str(dataset),
            "episode_start": args.episode_start,
            "episode_end": args.episode_end,
            "episode_indices": args.episode_indices or None,
            "resume_metrics": (
                str(args.resume_metrics.resolve()) if args.resume_metrics else None
            ),
            "mapping_gpu": es_gpu,
            "navigation_gpu": nav_gpu,
            "per_keyframe_snapshots": bool(
                visualization.get("per_keyframe_snapshots", False)
            ),
        },
    )
    write_status(run, "starting")
    try:
        with resource_monitor(run, (es_gpu, nav_gpu)), mapping_server(
            runtime, profile, run, socket_path, es_gpu
        ):
            env = navigation_environment(runtime, profile, nav_gpu)
            command = [
                expand(runtime["python"]["navigation"]),
                "-m",
                "tdonav_navigation.adapters.lagmemo_evaluator",
                "--entrypoint",
                "hm3dv1",
                "--dataset-root",
                str(dataset),
                "--output-path",
                str(run),
                "--socket-path",
                str(socket_path),
                "--top-k",
                str(profile["mapping"]["top_k"]),
                "--episode-start",
                str(args.episode_start),
                "--episode-end",
                str(args.episode_end),
                "--enable-timing",
            ]
            if visualization.get("per_keyframe_snapshots", False):
                command.append("--print-images")
            if args.resume_metrics is not None:
                command.extend(
                    ["--resume-metrics", str(args.resume_metrics.resolve())]
                )
            if args.episode_indices:
                command.extend(["--episode-indices", args.episode_indices])
            write_status(run, "running", episode_start=args.episode_start, episode_end=args.episode_end)
            with (run / "logs/navigation.log").open("w", encoding="utf-8") as log:
                result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f"Navigation failed: {result.returncode}")
        write_status(run, "completed", episode_start=args.episode_start, episode_end=args.episode_end)
    except BaseException as error:
        write_status(run, "failed", error=repr(error))
        raise
    finally:
        print(run)


if __name__ == "__main__":
    main()
