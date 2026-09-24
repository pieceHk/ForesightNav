#!/usr/bin/env python3
"""Reject ownership, standalone-runtime, or release-state violations."""

import argparse
import ast
import os
import subprocess
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("TDONAV_ROOT", str(ROOT))
OWNERS = {"Mapping", "Navigation", "Exp"}
ROOT_FILES = {".gitattributes", ".gitignore", "README.md", "pyproject.toml"}
LEGACY_TOKENS = tuple("/nvme/khy/" + name for name in ("goat" + "_rssfix", "hm3dv1" + "_exp"))
RUNTIME_ROOTS = (ROOT / "Mapping" / "runtime", ROOT / "Navigation" / "runtime")
DATA_ROOT = Path("/data/3DONav_datasets").resolve()
ALLOWED_EXTERNAL_DATA_LINKS = {
    ROOT / "Exp" / "assets" / "datasets",
    ROOT / "Navigation" / "runtime" / "lagmemo" / "data" / "scene_datasets",
}


def _inside_root(path: Path) -> bool:
    try:
        path.resolve().relative_to(ROOT.resolve())
        return True
    except ValueError:
        return False


def _runtime_values(value):
    if isinstance(value, dict):
        for nested in value.values():
            yield from _runtime_values(nested)
    elif isinstance(value, str):
        yield value


def _is_allowed_external_data_path(path: Path) -> bool:
    """Permit data only through the checked-in Exp/assets/datasets entrypoint."""
    try:
        path.relative_to(ROOT / "Exp" / "assets" / "datasets")
        path.resolve().relative_to(DATA_ROOT)
        return True
    except ValueError:
        return False


def _is_allowed_external_data_link(link: Path) -> bool:
    if link not in ALLOWED_EXTERNAL_DATA_LINKS:
        return False
    try:
        link.resolve().relative_to(DATA_ROOT)
        return True
    except ValueError:
        return False


def assert_tree_is_owned(path: Path) -> None:
    for link in path.rglob("*"):
        if link.is_symlink():
            assert _inside_root(link) or _is_allowed_external_data_link(link), (
                f"symlink escapes 3DONav: {link} -> {link.resolve()}"
            )


def validate_runtime() -> None:
    runtime = ROOT / "Exp" / "runtime.local.toml"
    if not runtime.exists():
        return
    config = tomllib.loads(runtime.read_text(encoding="utf-8"))
    for section in ("datasets", "shared"):
        for value in _runtime_values(config.get(section, {})):
            expanded = Path(os.path.expandvars(os.path.expanduser(value)))
            assert _inside_root(expanded) or _is_allowed_external_data_path(expanded), (
                f"runtime path escapes 3DONav: {expanded}"
            )
            assert expanded.exists(), f"runtime path missing: {expanded}"
            assert_tree_is_owned(expanded)
    assert set(config) <= {"python", "datasets", "shared"}, "runtime config may not name external source/workdir"
    _validate_navigation_import_origin(config)


def _validate_navigation_import_origin(runtime: dict) -> None:
    """Prove the launcher resolves LagMemo from the owned vendor tree."""
    from _runtime import navigation_environment

    profile_path = ROOT / "Exp" / "profiles" / "hm3dv1_objectnav_val.toml"
    profile = tomllib.loads(profile_path.read_text(encoding="utf-8"))
    env = navigation_environment(runtime, profile, nav_gpu=0)
    command = [
        os.path.expandvars(os.path.expanduser(runtime["python"]["navigation"])),
        "-c",
        "import importlib.util; print(importlib.util.find_spec('lagmemo').origin)",
    ]
    source = Path(subprocess.check_output(command, env=env, text=True).strip())
    assert _inside_root(source), f"LagMemo import escapes 3DONav: {source}"
    expected = ROOT / "Navigation" / "runtime" / "lagmemo" / "src" / "lagmemo"
    assert source.is_relative_to(expected), f"LagMemo import is not the owned package: {source}"


def validate_source_tree() -> None:
    for owner in ("Mapping", "Navigation"):
        source_root = ROOT / owner / "src"
        assert source_root.is_dir()
        for source in source_root.rglob("*.py"):
            ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    expected = {
        ROOT / "Mapping" / "runtime" / "server" / "es_mapper_server.py",
        ROOT / "Mapping" / "runtime" / "embodied_splat",
        ROOT / "Navigation" / "runtime" / "lagmemo",
        ROOT / "Navigation" / "runtime" / "evaluators" / "goat" / "entrypoint.py",
        ROOT / "Navigation" / "runtime" / "evaluators" / "hm3dv1" / "entrypoint.py",
    }
    for path in expected:
        assert path.exists(), f"owned runtime missing: {path.relative_to(ROOT)}"
        assert _inside_root(path), f"runtime escapes 3DONav: {path}"
        assert_tree_is_owned(path)
    extensions = {".py", ".toml", ".yaml", ".yml"}
    for owner in OWNERS:
        for path in (ROOT / owner).rglob("*"):
            if path == Path(__file__) or "runs" in path.parts or "assets" in path.parts or path.suffix not in extensions:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            assert not any(token in text for token in LEGACY_TOKENS), f"legacy runtime reference: {path.relative_to(ROOT)}"


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True)


def validate_tracked_runtime(require_clean: bool) -> None:
    tracked = set(filter(None, git_output("ls-files", "-z").split("\0")))
    missing = []
    for runtime_root in RUNTIME_ROOTS:
        for path in runtime_root.rglob("*"):
            if path.is_dir() or "__pycache__" in path.parts:
                continue
            relative = path.relative_to(ROOT).as_posix()
            if relative not in tracked:
                missing.append(relative)
    assert not missing, "untracked runtime files: " + ", ".join(sorted(missing)[:12])
    if require_clean:
        status = git_output("status", "--porcelain", "--untracked-files=all").strip()
        assert not status, "release tree is not clean:\n" + status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-clean", action="store_true", help="also require no staged or unstaged source changes")
    args = parser.parse_args()
    top_dirs = {path.name for path in ROOT.iterdir() if path.is_dir() and path.name != ".git"}
    assert top_dirs == OWNERS, f"top-level owners must be exactly {sorted(OWNERS)}: {sorted(top_dirs)}"
    extra_files = {path.name for path in ROOT.iterdir() if path.is_file() and path.name != ".git"} - ROOT_FILES
    assert not extra_files, f"unowned root files: {sorted(extra_files)}"
    validate_source_tree()
    validate_runtime()
    validate_tracked_runtime(args.require_clean)
    print("architecture=passed runtime=independent tracked=true owners=Mapping,Navigation,Exp")


if __name__ == "__main__":
    main()
