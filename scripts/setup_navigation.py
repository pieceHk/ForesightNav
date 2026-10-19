#!/usr/bin/env python3
"""Install the locked Linux navigation environment from online or offline inputs."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from scripts._environment import (
    check_prefix, fetch_artifact, file_hash, materialize_source, source_files,
)
from scripts._sources import materialize_source_group


def checked_artifacts(lock, root):
    artifacts = lock["conda"] + lock["packages"] + lock["sources"]
    from runtime.external_sources import load_source_manifest

    for entry in lock.get("runtime_source_manifests", []):
        path = PurePosixPath(entry["path"])
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Unsafe runtime source manifest path")
        artifacts += load_source_manifest(root / str(path), entry["sha256"])["files"]
    paths = [item["cache_path"] for item in artifacts]
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate cache paths in the navigation lock")
    for value in paths:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsafe cache path: {value}")
    sources = {item["id"] for item in lock["sources"]}
    if len(sources) != len(lock["sources"]):
        raise ValueError("Duplicate source IDs")
    if set(lock["source_groups"]) != {"habitat_sim", "habitat_lab", "imath", "detectron2"}:
        raise ValueError("Expected the four pinned navigation source groups")
    for modules in lock["source_groups"].values():
        if not modules:
            raise ValueError("A source group cannot be empty")
        for module in modules:
            path = PurePosixPath(module["path"])
            if module["source"] not in sources or path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Invalid source module: {module}")
    for source in lock["tracked_sources"]:
        source_files(root, source)
    return artifacts


def install(args, lock, root=ROOT):
    cache = args.cache.resolve()
    artifacts = checked_artifacts(lock, root)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda item: fetch_artifact(cache, item, args.offline), artifacts))
    if args.download_only:
        print(f"Verified {len(artifacts)} Linux navigation artifacts under {cache}")
        return
    if sys.platform != "linux" or os.uname().machine != "x86_64":
        raise ValueError("Navigation installation requires Linux x86_64")
    if args.prefix is None:
        raise ValueError("Installation requires --prefix")
    if not (args.cuda_home / "bin/nvcc").is_file():
        raise ValueError(f"CUDA compiler not found under {args.cuda_home}")
    prefix = args.prefix.resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    control = root / "Exp/runs/setup_navigation" / stamp
    control.mkdir(parents=True)
    environment = {name: value for name, value in os.environ.items()
                   if name not in {"PYTHONPATH", "PYTHONHOME", "CONDA_PREFIX", "CONDA_DEFAULT_ENV", "LD_LIBRARY_PATH"}}
    environment.update({
        "PYTHONNOUSERSITE": "1", "PIP_NO_INDEX": "1",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1", "PIP_NO_BUILD_ISOLATION": "0",
        "CUDA_HOME": str(args.cuda_home.resolve()),
        "PATH": os.pathsep.join(map(str, (prefix / "bin", args.cuda_home / "bin", Path("/usr/bin"), Path("/bin")))),
        "CC": "/usr/bin/gcc", "CXX": "/usr/bin/g++", "MAX_JOBS": str(args.jobs),
        "CMAKE_BUILD_PARALLEL_LEVEL": str(args.jobs), "FORCE_CUDA": "1",
        "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
        "MAMBA_ROOT_PREFIX": str(cache / "mamba-root"),
        "HEADLESS": "ON", "WITH_BULLET": "OFF", "WITH_CUDA": "OFF",
    })
    if args.cuda_arch:
        environment["TORCH_CUDA_ARCH_LIST"] = args.cuda_arch
    journal = []
    status = {
        "prefix": str(prefix), "pid": os.getpid(), "status": "running",
        "started_empty_prefix": not prefix.exists(),
        "lock_sha256": hashlib.sha256(json.dumps(lock, sort_keys=True).encode()).hexdigest(),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "installer_files_sha256": {str(path.relative_to(root)): file_hash(path) for path in (
            Path(__file__), root / "scripts/_environment.py", root / "scripts/_sources.py",
        )},
        "build_environment": {name: environment.get(name) for name in (
            "CUDA_HOME", "TORCH_CUDA_ARCH_LIST", "CC", "CXX", "MAX_JOBS", "HEADLESS", "WITH_BULLET", "WITH_CUDA",
        )},
    }

    def save():
        (control / "status.json").write_text(json.dumps(status, indent=2) + "\n")
        (control / "commands.json").write_text(json.dumps(journal, indent=2) + "\n")

    def run(command, label, cwd=root):
        print(label, flush=True)
        journal.append({"label": label, "command": list(map(str, command)), "cwd": str(cwd)})
        save()
        with (control / f"{label}.log").open("w") as output:
            result = subprocess.run(list(map(str, command)), cwd=cwd, env=environment,
                                    stdout=output, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"{label} exited {result.returncode}; see {control / (label + '.log')}")

    save()
    try:
        if not prefix.exists():
            manager = args.micromamba or args.conda
            if not manager:
                raise ValueError("Creating a prefix requires --conda or --micromamba")
            explicit = control / "conda-explicit.txt"
            explicit.write_text("@EXPLICIT\n" + "\n".join(
                (cache / row["cache_path"]).as_uri() + "#" + row["md5"] for row in lock["conda"]
            ) + "\n")
            run([manager.resolve(), "--version"], "environment_manager")
            options = ["--no-rc", "create"] if args.micromamba else ["create", "--no-default-packages"]
            run([manager.resolve(), *options, "--offline", "-y", "--prefix", prefix,
                 "--file", explicit, "--json"], "conda")
        python = check_prefix(prefix, lock)
        run([python, "-I", "-c", "import sys,site,importlib.util; print(sys.version,sys.path); "
             "print('user_site_enabled',site.ENABLE_USER_SITE); "
             "print('torch_before_install',importlib.util.find_spec('torch')); "
             "print('habitat_before_install',importlib.util.find_spec('habitat_sim'))"], "python")
        run([args.cuda_home / "bin/nvcc", "--version"], "nvcc")
        run([environment["CXX"], "--version"], "compiler")
        pip = [python, "-m", "pip"]
        wheels = [cache / item["cache_path"] for item in lock["packages"] if item["filename"].endswith(".whl")]
        run(pip + ["install", "--no-deps", "--no-index", *wheels], "binary_packages")
        built = control / "wheels"
        built.mkdir()

        def wheel(source, label):
            run(pip + ["wheel", "--no-deps", "--no-build-isolation", "--no-index", "--wheel-dir", built, source], label)

        for index, item in enumerate(lock["packages"]):
            if not item["filename"].endswith(".whl"):
                wheel(cache / item["cache_path"], f"package_{index}")
        run(pip + ["install", "--no-deps", "--no-index", *sorted(built.glob("*.whl"))], "sdist_packages")
        sources = {}
        for name, modules in lock["source_groups"].items():
            directory = control / "sources" / name
            count = materialize_source_group(cache, lock["sources"], modules, directory)
            sources[name] = directory
            status.setdefault("source_groups", {})[name] = {"files": count, "path": str(directory)}
        environment["CMAKE_ARGS"] = shlex.join([
            f"-DFETCHCONTENT_SOURCE_DIR_IMATH={sources['imath']}", "-DFETCHCONTENT_FULLY_DISCONNECTED=ON",
        ])
        status["build_environment"]["CMAKE_ARGS"] = environment["CMAKE_ARGS"]
        # Habitat-Sim also installs the Magnum bindings it just compiled.
        run([python, "setup.py", "install", "--headless", "--no-bullet", "--no-update-submodules",
             "--build-type", "RelWithDebInfo", "build_ext", "--parallel", str(args.jobs)],
            "habitat_sim", cwd=sources["habitat_sim"])
        for name in ("habitat-lab", "habitat-baselines"):
            wheel(sources["habitat_lab"] / name, name.replace("-", "_"))
        wheel(sources["detectron2"], "detectron2")
        for source in lock["tracked_sources"]:
            directory = control / "sources" / source["name"]
            materialize_source(root, cache, source, directory)
            wheel(directory, source["name"])
        run(pip + ["install", "--no-deps", "--no-index", *sorted(built.glob("*.whl"))], "source_packages")
        run(pip + ["install", "--no-deps", "--no-build-isolation", "--no-index", "-e", root], "pbsp")
        run(pip + ["check"], "pip_check")
        run([python, "-I", "-c", "import json,sys,torch,habitat_sim,detectron2,detectron2._C,_magnum; "
             "from detectron2.data import MetadataCatalog; "
             "paths={'habitat_sim':habitat_sim.__file__,'detectron2':detectron2.__file__,"
             "'detectron2._C':detectron2._C.__file__,'magnum':_magnum.__file__}; "
             "assert all(p.startswith(sys.prefix+'/') for p in paths.values()); print(json.dumps(paths)); "
             "assert torch.cuda.is_available(); assert torch.ones(4,device='cuda').sum().item()==4"], "extensions")
        run([python, "-I", "-c", "import torch; from detectron2.layers import nms_rotated; "
             "b=torch.tensor([[0.,0.,2.,2.,0.],[0.,0.,2.,2.,0.],[10.,10.,1.,1.,0.]]); s=torch.tensor([.9,.8,.7]); "
             "assert nms_rotated(b,s,.5).tolist()==[0,2]; "
             "assert nms_rotated(b.cuda(),s.cuda(),.5).cpu().tolist()==[0,2]; print('CPU/CUDA NMS passed')"], "native_nms")
        status["status"] = "completed"
    except BaseException as error:
        status.update(status="failed", error=str(error))
        raise
    finally:
        save()
        print(f"Installation evidence: {control}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path)
    manager = parser.add_mutually_exclusive_group()
    manager.add_argument("--conda", type=Path)
    manager.add_argument("--micromamba", type=Path)
    parser.add_argument("--cache", type=Path, default=ROOT / ".cache/pbsp-offline")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--cuda-home", type=Path, default=Path("/usr/local/cuda-11.8"))
    parser.add_argument("--cuda-arch", help="PyTorch CUDA architectures; inferred from visible GPUs by default")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    lock = json.loads((ROOT / "envs/navigation/linux-64.lock.json").read_text())
    install(args, lock)


if __name__ == "__main__":
    main()
