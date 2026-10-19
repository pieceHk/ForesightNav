#!/usr/bin/env python3
"""Run PBSP from shared experiment and machine-local YAML configuration."""

import argparse
import json
from pathlib import Path

from runtime.config import load_experiment
from runtime.launch import launch, source_info
from runtime.plan import prepare_plan
from runtime.resources import validate_resources

ROOT = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/experiments/hm3d_4mon.yaml"
    )
    parser.add_argument("--runtime", type=Path, default=ROOT / "runtime.local.yaml")
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
