#!/usr/bin/env python3
"""Run PBSP from shared experiment and machine-local YAML configuration."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

from pbsp.runtime.config import load_experiment
from pbsp.runtime.launch import launch, source_info
from pbsp.runtime.plan import prepare_plan
from pbsp.runtime.resources import validate_resources

ROOT = Path(__file__).resolve().parent


def gpu_pair(value):
    try:
        parts = tuple(int(part) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use mapping,navigation GPU IDs, for example 0,1") from error
    if len(parts) != 2 or any(part < 0 for part in parts):
        raise argparse.ArgumentTypeError("Use two non-negative GPU IDs, for example 0,1")
    return parts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    # 切换任务，替换hm3d_2mon.yaml/hm3d_3mon.yaml/hm3d_4mon.yaml的配置文件
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/experiments/hm3d_4mon.yaml"
    )
    # 声明数据集来源，权重路径，解释器。
    parser.add_argument("--runtime", type=Path, default=ROOT / "runtime.local.yaml")
    # 选择GPU，顺序是"MAPPING,NAVIGATION"
    parser.add_argument("--gpus", type=gpu_pair, metavar="MAPPING,NAVIGATION",
                        help="Override the two GPU IDs from the machine configuration")
    parser.add_argument(
        "--scene",
        action="append",
        help="Exact scene ID; repeat to select several scenes",
    )
    parser.add_argument(
        "--episode",
        action="append",
        help="Exact runtime episode ID within one selected scene",
    )
    parser.add_argument(
        "--limit-episodes", type=int, help="Maximum number of complete ordered episodes"
    )
    parser.add_argument("--tag", default="")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate inputs without starting workers or creating outputs",
    )
    args = parser.parse_args(argv)
    config = load_experiment(args.config, args.runtime)
    if args.gpus is not None:
        config = replace(config, runtime=replace(config.runtime, mapping_gpu=args.gpus[0],
                                                  navigation_gpu=args.gpus[1]))
    plan = prepare_plan(
        config,
        root=ROOT,
        source_commit=source_info(ROOT)["commit"],
        scenes=args.scene,
        episodes=args.episode,
        limit=args.limit_episodes,
    )
    resources = validate_resources(plan.resources, config.runtime)
    print(
        json.dumps(
            {
                "experiment": config.name,
                "dataset": config.dataset.name,
                "scenes": [item.scene_id for item in plan.scenes],
                "complete_episodes": len(plan.identities),
                "targets_per_episode": config.dataset.expected_tasks_per_episode,
                "gpus": {
                    "mapping": config.runtime.mapping_gpu,
                    "navigation": config.runtime.navigation_gpu,
                },
                "configuration_sha256": plan.configuration_sha256,
                "identity_sha256": plan.identity_sha256,
                "habitat_revision": resources["habitat_revision"],
                "models": resources["models"],
            },
            indent=2,
        ),
        flush=True,
    )
    if not args.check:
        launch(plan, tag=args.tag, resource_checks=resources)


if __name__ == "__main__":
    main()
