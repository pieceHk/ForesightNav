#!/usr/bin/env python3
"""Check both installed environments and optionally run one complete MON episode."""

import argparse
from datetime import datetime, timezone
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def require_prefix(path, prefix):
    path, prefix = Path(path).resolve(), Path(prefix).resolve()
    try:
        path.relative_to(prefix)
    except ValueError as error:
        raise ValueError(f"Installed module came from outside its environment: {path}") from error
    return str(path)


def locked_versions(packages):
    versions = {}
    for item in packages:
        name = item.get("installed_distribution", item["name"])
        actual = importlib.metadata.version(name)
        if actual != item["version"]:
            raise ValueError(f"{name}: installed {actual}, expected {item['version']}")
        versions[name] = actual
    return versions


def probe(payload):
    """Run inside the selected interpreter, with explicit runtime search paths."""
    for path in reversed(payload["search_paths"]):
        sys.path.insert(0, path)
    role = payload["role"]
    lock = json.loads((ROOT / "envs" / role / "linux-64.lock.json").read_text())
    if list(sys.version_info[:2]) != lock["python"]:
        raise ValueError(f"{role} Python differs from the environment lock")
    versions = locked_versions(lock["packages"])
    names = ["torch", "torchvision"]
    if role == "mapping":
        names += ["diff_gaussian_rasterization._C", "diff_gaussian_rasterization_lang._C",
                  "simple_knn._C", "deformable_aggregation_ext"]
    else:
        names += ["habitat_sim", "detectron2", "detectron2._C", "_magnum"]
    origins = {name: require_prefix(importlib.import_module(name).__file__, sys.prefix)
               for name in names}
    for name in ("pbsp", "adapters", "runtime", "experiments"):
        path = Path(importlib.import_module(name).__file__).resolve()
        if path != ROOT / name / "__init__.py":
            raise ValueError(f"PBSP module came from a different checkout: {path}")
        origins[name] = str(path)
    if role == "navigation":
        habitat = Path(importlib.import_module("habitat").__file__).resolve()
        expected = ROOT / "Exp/assets/dependencies/habitat-lab/habitat-lab/habitat/__init__.py"
        if habitat != expected:
            raise ValueError(f"Habitat-Lab came from a different source tree: {habitat}")
        origins["habitat"] = str(habitat)
    import torch

    if torch.version.cuda != lock["pytorch_cuda"]:
        raise ValueError("PyTorch CUDA build differs from the environment lock")
    if torch.ones(4).sum().item() != 4:
        raise RuntimeError("CPU tensor check failed")
    devices = ["cpu"] if payload["cpu_only"] else ["cpu", "cuda"]
    if "cuda" in devices:
        if not torch.cuda.is_available():
            raise RuntimeError("The selected CUDA device is unavailable")
        if torch.ones(4, device="cuda").sum().item() != 4:
            raise RuntimeError("CUDA tensor check failed")
    if role == "navigation":
        from detectron2.layers import nms_rotated

        for device in devices:
            boxes = torch.tensor([[0., 0., 2., 2., 0.], [0., 0., 2., 2., 0.],
                                  [10., 10., 1., 1., 0.]], device=device)
            scores = torch.tensor([.9, .8, .7], device=device)
            if nms_rotated(boxes, scores, .5).cpu().tolist() != [0, 2]:
                raise RuntimeError(f"Detectron2 {device} rotated NMS failed")
    result = {"status": "passed", "role": role, "python": sys.executable,
              "prefix": sys.prefix, "versions": versions, "module_origins": origins,
              "devices_checked": devices, "cuda_initialized": torch.cuda.is_initialized(),
              "torch_cuda_build": torch.version.cuda}
    if payload["cpu_only"] and result["cuda_initialized"]:
        raise RuntimeError("CPU-only probe unexpectedly initialized CUDA")
    Path(payload["result"]).write_text(json.dumps(result, indent=2) + "\n")


def check(args):
    from runtime.bootstrap import worker_environment
    from runtime.config import load_experiment
    from runtime.launch import launch, source_info, write_json
    from runtime.plan import prepare_plan
    from runtime.resources import validate_resources

    if sys.platform != "linux" or os.uname().machine != "x86_64":
        raise ValueError("Installation checks require Linux x86_64")
    config = load_experiment(args.config, args.runtime)
    source = source_info(ROOT)
    plan = prepare_plan(config, root=ROOT, source_commit=source["commit"],
                        scenes=[args.scene] if args.scene else None,
                        episodes=[args.episode] if args.episode else None, limit=1)
    resources = validate_resources(plan.resources, config.runtime)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output = (args.output or ROOT / "Exp/runs/check_install" / stamp).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "source": source,
              "scope": "input hashes, locked package versions, module origins and native operations",
              "cpu_only": args.cpu_only, "resource_checks": resources,
              "configuration_sha256": plan.configuration_sha256,
              "identity_sha256": plan.identity_sha256,
              "model_inference": "pending" if args.smoke else "not_requested",
              "habitat_reset_and_actions": "pending" if args.smoke else "not_requested",
              "probes": {}}
    write_json(output / "status.json", report)
    try:
        for role in ("mapping", "navigation"):
            environment = worker_environment(plan, role)
            for name in ("PYTHONHOME", "PYTHONSTARTUP"):
                environment.pop(name, None)
            if args.cpu_only:
                environment["CUDA_VISIBLE_DEVICES"] = ""
            payload = {"role": role, "cpu_only": args.cpu_only,
                       "search_paths": environment["PYTHONPATH"].split(os.pathsep),
                       "result": str(output / f"{role}.json")}
            payload_file = output / f"{role}_request.json"
            write_json(payload_file, payload)
            python = getattr(config.runtime, role + "_python")
            with (output / f"{role}.log").open("w") as log:
                # -I excludes user-site packages and inherited PYTHONPATH. The probe
                # inserts only the explicit paths already used by native workers.
                command = [str(python), "-I", str(Path(__file__).resolve()),
                           "--probe", str(payload_file)]
                process = subprocess.run(command, cwd=ROOT, env=environment,
                                         stdout=log, stderr=subprocess.STDOUT, timeout=300)
            report["probes"][role] = {"returncode": process.returncode,
                                      "log": str(output / f"{role}.log")}
            write_json(output / "status.json", report)
            if process.returncode:
                raise RuntimeError(f"{role} installation probe failed; see {output / (role + '.log')}")
            result = json.loads(Path(payload["result"]).read_text())
            if result["status"] != "passed" or result["role"] != role:
                raise ValueError("Installation probe result is inconsistent")
        if args.smoke:
            directory = launch(plan, tag="check_install_" + stamp, resource_checks=resources)
            from scripts.summarize import validate_run

            validate_run(directory)
            report.update(model_inference="passed", habitat_reset_and_actions="passed",
                          smoke_run=str(directory))
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(output / "status.json", report)
        print(f"Installation evidence: {output}", flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiments/hm3d_4mon.yaml")
    parser.add_argument("--runtime", type=Path, default=ROOT / "runtime.local.yaml")
    parser.add_argument("--output", type=Path, help="New directory for diagnostic records")
    parser.add_argument("--scene", help="Exact scene for the optional smoke episode")
    parser.add_argument("--episode", help="Exact episode within --scene")
    parser.add_argument("--cpu-only", action="store_true", help="Import native modules and run CPU checks without initializing CUDA")
    parser.add_argument("--smoke", action="store_true", help="Also load models and run one complete ordered MON episode")
    parser.add_argument("--probe", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.probe:
        probe(json.loads(args.probe.read_text()))
        return
    if args.cpu_only and args.smoke:
        parser.error("--smoke requires CUDA; remove --cpu-only")
    if args.episode and not args.scene:
        parser.error("--episode requires --scene")
    check(args)


if __name__ == "__main__":
    main()
