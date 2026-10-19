#!/usr/bin/env python3
"""Run a frozen Full PBSP profile with its complete ordered-episode identities."""

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path

from _runtime import ROOT, expand, load_toml, navigation_environment
from build_identity_sidecar import FROZEN, build_sidecar, expected_manifest_hash
from prepare_reproduction import HABITAT_REVISION, habitat_pythonpaths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, default=ROOT / "Exp/runtime.local.toml")
    parser.add_argument("--mon", type=int, choices=(2, 3, 4), default=4)
    parser.add_argument("--scope", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--gpus", default="0,1", help="mapping,navigation GPU IDs")
    parser.add_argument("--scene-start", type=int, default=0)
    parser.add_argument("--scene-end", type=int)
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--episode-end", type=int)
    parser.add_argument("--tag", default="reproduction")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    os.environ["TDONAV_ROOT"] = str(ROOT)
    dependencies = habitat_pythonpaths()
    inherited = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = os.pathsep.join(map(str, dependencies)) + (os.pathsep + inherited if inherited else "")
    profile_name = "hm3dv1_mon_scsp_suffix.toml" if args.mon == 3 else f"hm3dv1_{args.mon}mon_scsp_suffix.toml"
    profile_path = ROOT / "Exp/profiles" / profile_name
    profile = load_toml(profile_path)
    runtime = load_toml(args.runtime)
    dataset = Path(expand(runtime["datasets"][profile["dataset"]]))
    gpus = [int(value) for value in args.gpus.split(",")]
    if len(gpus) != 2 or min(gpus) < 0:
        raise ValueError("--gpus requires two non-negative mapping,navigation GPU IDs")
    environment = navigation_environment(runtime, profile, gpus[1])
    if environment["ES_SCSP_RUNTIME_MODE"] != "ordered_suffix_mvp":
        raise ValueError("Full PBSP requires ordered_suffix_mvp; remove conflicting runtime overrides")
    payload = build_sidecar(dataset, args.mon, expected_manifest_hash(args.mon),
                            json.loads(FROZEN.read_text()))
    sidecar = ROOT / "Exp/assets/identities" / f"full_pbsp_{args.mon}mon.json"
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if sidecar.exists() and sidecar.read_text() != text:
        raise ValueError(f"Existing identity sidecar differs from the dataset: {sidecar}")
    for name in ("mapping", "navigation"):
        if not Path(expand(runtime["python"][name])).is_file():
            raise FileNotFoundError(runtime["python"][name])

    scene_end = args.scene_end
    episode_end = args.episode_end
    if args.scope == "smoke":
        scene_end = args.scene_start + 1 if scene_end is None else scene_end
        episode_end = args.episode_start + 1 if episode_end is None else episode_end
    command = [
        expand(runtime["python"]["navigation"]), str(ROOT / "Exp/scripts/run_goat.py"),
        "--runtime", str(args.runtime.absolute()), "--profile", str(profile_path),
        "--gpus", args.gpus, "--scene-start", str(args.scene_start),
        "--episode-start", str(args.episode_start), "--tag", args.tag,
    ]
    if scene_end is not None:
        command += ["--scene-end", str(scene_end)]
    if episode_end is not None:
        command += ["--episode-end", str(episode_end)]
    print(json.dumps({
        "scope": args.scope, "ordered_subtasks": args.mon,
        "dataset_manifest_sha256": payload["dataset_manifest_sha256"],
        "additional_identity_records": len(payload["episodes"]),
        "profile": str(profile_path), "identity_sidecar": str(sidecar),
        "habitat_revision": HABITAT_REVISION,
        "dependency_pythonpaths": list(map(str, dependencies)),
        "required_environment": {
            "TDONAV_ROOT": str(ROOT),
            "PYTHONPATH": os.environ["PYTHONPATH"],
            "ES_SCSP_IDENTITY_SIDECAR": str(sidecar),
        },
        "command": shlex.join(command),
    }, indent=2), flush=True)
    if args.check_only:
        return
    subprocess.run([expand(runtime["python"]["navigation"]),
                    str(ROOT / "Exp/scripts/validate.py")], check=True)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    if not sidecar.exists():
        sidecar.write_text(text)
    # Let the original launcher choose independent environments for its children.
    environment = os.environ.copy()
    environment["ES_SCSP_IDENTITY_SIDECAR"] = str(sidecar)
    raise SystemExit(subprocess.call(command, env=environment, cwd=ROOT))


if __name__ == "__main__":
    main()
