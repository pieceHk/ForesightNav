#!/usr/bin/env python3
"""Prepare the locked Linux mapping environment, including offline source builds."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request


ROOT = Path(__file__).resolve().parents[1]


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def content_hash(files):
    return hashlib.sha256(json.dumps(sorted(files), separators=(",", ":")).encode()).hexdigest()


def archive_files(path):
    """Read regular files under one archive root before writing anything to disk."""
    entries = []
    with tarfile.open(path) as archive:
        roots = set()
        names = set()
        for member in archive:
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts:
                raise ValueError(f"Unsafe archive path: {member.name}")
            if not name.parts:
                continue
            roots.add(name.parts[0])
            if member.isdir():
                continue
            if not member.isfile() or len(name.parts) < 2:
                raise ValueError(f"Unsupported archive entry: {member.name}")
            relative = str(PurePosixPath(*name.parts[1:]))
            if relative in names:
                raise ValueError(f"Duplicate archive entry: {relative}")
            names.add(relative)
            mode = 0o755 if member.mode & 0o111 else 0o644
            entries.append((relative, mode, archive.extractfile(member).read()))
        if len(roots) != 1:
            raise ValueError("Source archive must contain exactly one root directory")
    return entries


def archive_hash(path):
    return content_hash([(name, mode, hashlib.sha256(data).hexdigest())
                         for name, mode, data in archive_files(path)])


def verify_artifact(path, item):
    if not path.is_file():
        raise FileNotFoundError(f"Missing cached artifact: {path}")
    actual = archive_hash(path) if "content_sha256" in item else file_hash(path)
    expected = item.get("content_sha256", item.get("sha256"))
    if actual != expected:
        raise ValueError(f"Checksum mismatch: {path}")


def fetch_artifact(cache, item, offline):
    path = cache / item["cache_path"]
    if path.exists():
        verify_artifact(path, item)
        return path
    if offline:
        raise FileNotFoundError(f"Missing cached artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
        try:
            with urllib.request.urlopen(item["url"], timeout=120) as response:
                shutil.copyfileobj(response, temporary, length=1024 * 1024)
            temporary.flush()
            verify_artifact(temporary_path, item)
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)
    return path


def source_files(root, source):
    relative = source["repository_path"]
    names = subprocess.check_output(
        ["git", "-C", str(root), "ls-files", "-z", "--", relative]
    ).decode().split("\0")
    entries = []
    for name in filter(None, names):
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Source must be a regular tracked file: {path}")
        path.resolve().relative_to(root.resolve())
        mode = 0o755 if path.stat().st_mode & 0o111 else 0o644
        entries.append((str(path.relative_to(root / relative)), mode, file_hash(path)))
    if not entries or content_hash(entries) != source["content_sha256"]:
        raise ValueError(f"Tracked source differs from the mapping lock: {relative}")
    return entries


def materialize_source(root, cache, source, destination):
    if "repository_path" in source:
        entries = source_files(root, source)
        for name, mode, _ in entries:
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / source["repository_path"] / name, target)
            target.chmod(mode)
    else:
        path = cache / source["cache_path"]
        verify_artifact(path, source)
        for name, mode, data in archive_files(path):
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            target.chmod(mode)


def check_prefix(prefix, lock):
    python = prefix / "bin/python"
    if not python.is_file():
        raise ValueError(f"Existing prefix is not a Python environment: {prefix}")
    observed = json.loads(subprocess.check_output([
        str(python), "-I", "-c",
        "import importlib.util,json,sys;print(json.dumps({'prefix':sys.prefix,"
        "'python':list(sys.version_info[:2]),'pip':importlib.util.find_spec('pip') is not None}))",
    ], text=True))
    if Path(observed["prefix"]).resolve() != prefix.resolve() or observed["python"] != lock["python"]:
        raise ValueError(f"Prefix does not match the locked Python version: {prefix}")
    for package in lock["conda"]:
        records = list((prefix / "conda-meta").glob(
            f"{package['name']}-{package['version']}-{package['build']}.json"
        ))
        if len(records) != 1:
            raise ValueError(f"Prefix is missing the locked Conda base package: {package['name']}")
    if not observed["pip"]:
        raise ValueError(
            "Conda did not link pip into the target Python environment. "
            "Recreate this prefix with a current Conda or micromamba."
        )
    return python


def install(args, lock, root=ROOT):
    cache = args.cache.resolve()
    artifacts = lock["conda"] + lock["packages"] + [s for s in lock["sources"] if "cache_path" in s]
    paths = [item["cache_path"] for item in artifacts]
    if len(paths) != len(set(paths)):
        raise ValueError("The lock contains duplicate cache paths")
    for path in paths:
        if PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts:
            raise ValueError(f"Unsafe cache path: {path}")
    for source in lock["sources"]:
        if "repository_path" in source:
            source_files(root, source)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda item: fetch_artifact(cache, item, args.offline), artifacts))
    if args.download_only:
        print(f"Verified {len(artifacts)} Linux mapping artifacts under {cache}")
        return
    if sys.platform != "linux" or os.uname().machine != "x86_64":
        raise ValueError("Mapping installation requires Linux x86_64")
    if args.prefix is None:
        raise ValueError("Installation requires --prefix")
    prefix = args.prefix.resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    control = root / "Exp/runs/setup_mapping" / stamp
    control.mkdir(parents=True)
    environment = {key: value for key, value in os.environ.items()
                   if key not in {"PYTHONPATH", "PYTHONHOME", "CONDA_PREFIX", "CONDA_DEFAULT_ENV", "LD_LIBRARY_PATH"}}
    environment.update({
        "PYTHONNOUSERSITE": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "CUDA_HOME": str(args.cuda_home.resolve()),
        "PATH": os.pathsep.join(map(str, (prefix / "bin", args.cuda_home / "bin", Path("/usr/bin"), Path("/bin")))),
        "CC": "/usr/bin/gcc", "CXX": "/usr/bin/g++", "MAX_JOBS": str(args.jobs),
        "FORCE_CUDA": "1",
        "MMCV_WITH_OPS": "0", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
        "MAMBA_ROOT_PREFIX": str(cache / "mamba-root"),
    })
    if args.offline:
        environment["PIP_NO_INDEX"] = "1"
    if args.cuda_arch:
        environment["TORCH_CUDA_ARCH_LIST"] = args.cuda_arch
    journal = []

    def run(command, label, cwd=root):
        print(label, flush=True)
        journal.append({"label": label, "command": list(map(str, command)), "cwd": str(cwd)})
        with (control / f"{label}.log").open("w") as output:
            result = subprocess.run(list(map(str, command)), cwd=cwd, env=environment,
                                    stdout=output, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"{label} exited {result.returncode}; see {control / (label + '.log')}")

    status = {
        "prefix": str(prefix), "pid": os.getpid(), "status": "running",
        "lock_sha256": hashlib.sha256(json.dumps(lock, sort_keys=True).encode()).hexdigest(),
        "installer_sha256": file_hash(Path(__file__)),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "build_environment": {name: environment.get(name) for name in (
            "CUDA_HOME", "TORCH_CUDA_ARCH_LIST", "CC", "CXX", "MAX_JOBS", "MMCV_WITH_OPS",
        )},
    }
    (control / "status.json").write_text(json.dumps(status, indent=2) + "\n")
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
            run([manager.resolve(), *options, "--offline", "-y",
                 "--prefix", prefix, "--file", explicit, "--json"], "conda")
        python = check_prefix(prefix, lock)
        run([python, "-I", "-c", "import sys,site,importlib.util; print(sys.version); "
             "print(sys.path); print('user_site_enabled',site.ENABLE_USER_SITE); "
             "print('torch_before_install',importlib.util.find_spec('torch'))"], "python")
        if not (args.cuda_home / "bin/nvcc").is_file():
            raise ValueError(f"CUDA compiler not found under {args.cuda_home}")
        run([args.cuda_home / "bin/nvcc", "--version"], "nvcc")
        run([environment["CXX"], "--version"], "compiler")
        pip = [python, "-m", "pip"]
        wheels = [cache / row["cache_path"] for row in lock["packages"] if row["filename"].endswith(".whl")]
        run(pip + ["install", "--no-deps", "--no-index", *wheels], "binary_packages")
        built = control / "wheels"
        built.mkdir()

        def wheel(source, label):
            run(pip + ["wheel", "--no-deps", "--no-build-isolation", "--no-index",
                       "--wheel-dir", built, source], label)

        for index, row in enumerate(lock["packages"]):
            if not row["filename"].endswith(".whl"):
                wheel(cache / row["cache_path"], f"package_{index}")
        for source in lock["sources"]:
            directory = control / "sources" / source["name"]
            materialize_source(root, cache, source, directory)
            wheel(directory, source["name"])
        run(pip + ["install", "--no-deps", "--no-index", *sorted(built.glob("*.whl"))], "source_packages")
        run(pip + ["install", "--no-deps", "--no-build-isolation", "--no-index", "-e", root], "pbsp")
        run(pip + ["check"], "pip_check")
        run([python, "-I", "-c", "import importlib,json; import torch; "
             "names=['diff_gaussian_rasterization._C','diff_gaussian_rasterization_lang._C',"
             "'simple_knn._C','deformable_aggregation_ext']; "
             "print(json.dumps({name:importlib.import_module(name).__file__ for name in names}))"], "extensions")
        run([python, "-I", "-c", "import torch; print(torch.__version__, torch.version.cuda); "
             "assert torch.cuda.is_available(); x=torch.ones(4,device='cuda'); "
             "assert x.sum().item()==4; print(torch.cuda.get_device_name())"], "cuda")
        status["status"] = "completed"
    except BaseException as error:
        status.update(status="failed", error=str(error))
        raise
    finally:
        (control / "commands.json").write_text(json.dumps(journal, indent=2) + "\n")
        (control / "status.json").write_text(json.dumps(status, indent=2) + "\n")
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
    parser.add_argument("--cuda-arch", help="PyTorch CUDA architecture list; inferred from visible GPUs by default")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    lock = json.loads((ROOT / "envs/mapping/linux-64.lock.json").read_text())
    install(args, lock)


if __name__ == "__main__":
    main()
