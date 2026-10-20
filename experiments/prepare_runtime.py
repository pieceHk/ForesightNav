#!/usr/bin/env python3
"""Write machine-local YAML for provisioned PBSP data, weights and interpreters."""

import argparse
import hashlib
from pathlib import Path
import shutil
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TORCH_FILES = ("depth-save.pth", "disk_lightglue_v0-1_arxiv-pth")


def file_sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def prepare_runtime(
    *,
    mapping_python,
    navigation_python,
    data,
    weights,
    outputs,
    mapping_gpu,
    navigation_gpu,
    destination,
    torch_home_source=None,
):
    paths = {
        name: Path(value).expanduser().resolve()
        for name, value in (
            ("mapping", mapping_python),
            ("navigation", navigation_python),
            ("data", data),
            ("weights", weights),
            ("outputs", outputs),
        )
    }
    for name in ("mapping", "navigation"):
        if not paths[name].is_file():
            raise FileNotFoundError(paths[name])
    for name in ("data", "weights"):
        if not paths[name].is_dir():
            raise FileNotFoundError(paths[name])
    if (
        type(mapping_gpu) is not int
        or type(navigation_gpu) is not int
        or min(mapping_gpu, navigation_gpu) < 0
    ):
        raise ValueError("GPU IDs must be non-negative integers")
    payload = {
        "paths": {name: str(paths[name]) for name in ("data", "weights", "outputs")},
        "python": {name: str(paths[name]) for name in ("mapping", "navigation")},
        "gpus": {"mapping": mapping_gpu, "navigation": navigation_gpu},
    }
    destination = Path(destination).expanduser().resolve()
    text = yaml.safe_dump(payload, sort_keys=False)
    if destination.exists() and destination.read_text() != text:
        raise ValueError(f"Existing machine configuration differs: {destination}")
    copies = []
    for name in TORCH_FILES:
        target = paths["weights"] / "torch/hub/checkpoints" / name
        source = (
            Path(torch_home_source).expanduser().resolve() / "hub/checkpoints" / name
            if torch_home_source
            else target
        )
        if not source.is_file():
            raise FileNotFoundError(source)
        if target.exists() and file_sha256(target) != file_sha256(source):
            raise ValueError(f"Existing Torch cache differs: {target}")
        if not target.exists():
            copies.append((source, target))
    for source, target in copies:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        destination.write_text(text)
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping-python", type=Path, required=True)
    parser.add_argument("--navigation-python", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--weights", type=Path, default=ROOT / "third_party/weights")
    parser.add_argument("--outputs", type=Path, default=ROOT / "experiments/runs")
    parser.add_argument("--mapping-gpu", type=int, default=0)
    parser.add_argument("--navigation-gpu", type=int, default=1)
    parser.add_argument(
        "--torch-home-source",
        type=Path,
        help="Copy the two required Kornia checkpoints from a provisioned Torch cache",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "runtime.local.yaml")
    parser.add_argument("--cache", type=Path, default=Path.home() / ".cache/pbsp")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args(argv)
    from experiments.prepare_dependencies import prepare_habitat, prepare_superglue

    prepare_habitat(root=ROOT, cache=args.cache, offline=args.offline)
    prepare_superglue(root=ROOT, cache=args.cache, offline=args.offline)
    payload = prepare_runtime(
        mapping_python=args.mapping_python,
        navigation_python=args.navigation_python,
        data=args.data,
        weights=args.weights,
        outputs=args.outputs,
        mapping_gpu=args.mapping_gpu,
        navigation_gpu=args.navigation_gpu,
        destination=args.output,
        torch_home_source=args.torch_home_source,
    )
    print(yaml.safe_dump(payload, sort_keys=False), end="")


if __name__ == "__main__":
    main()
