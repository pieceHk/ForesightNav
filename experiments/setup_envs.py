#!/usr/bin/env python3
"""Prepare the mapping and navigation environments from their Linux locks."""

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from experiments import setup_mapping, setup_navigation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, default=ROOT / ".cache/envs")
    parser.add_argument("--environment", choices=("both", "mapping", "navigation"), default="both")
    manager = parser.add_mutually_exclusive_group()
    manager.add_argument("--conda", type=Path)
    manager.add_argument("--micromamba", type=Path)
    parser.add_argument("--cache", type=Path, default=ROOT / ".cache/pbsp-offline")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--cuda-home", type=Path, default=Path("/usr/local/cuda-11.8"))
    parser.add_argument("--cuda-arch")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    roles = ("mapping", "navigation") if args.environment == "both" else (args.environment,)
    installers = {"mapping": setup_mapping.install, "navigation": setup_navigation.install}
    locks = {role: json.loads((ROOT / f"configs/environments/{role}/linux-64.lock.json").read_text()) for role in roles}
    # Check both caches before either environment is created.
    for role in roles:
        options = SimpleNamespace(**{**vars(args), "prefix": args.prefix / role, "download_only": True})
        installers[role](options, locks[role])
    if not args.download_only:
        for role in roles:
            options = SimpleNamespace(**{**vars(args), "prefix": args.prefix / role})
            installers[role](options, locks[role])


if __name__ == "__main__":
    main()
