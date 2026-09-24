"""Run the 3DONav-owned EmbodiedSplat mapper runtime."""

import argparse
import os
import runpy
import sys
from pathlib import Path


MAPPING_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = MAPPING_ROOT.parent
RUNTIME_ROOT = MAPPING_ROOT / "runtime"
SERVER_SOURCE = RUNTIME_ROOT / "server" / "es_mapper_server.py"
ENGINE_ROOT = RUNTIME_ROOT / "embodied_splat"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    args, passthrough = parser.parse_known_args()
    del args
    if not SERVER_SOURCE.is_file():
        raise FileNotFoundError(SERVER_SOURCE)
    if not ENGINE_ROOT.is_dir():
        raise FileNotFoundError(ENGINE_ROOT)
    os.environ.setdefault("TDONAV_ROOT", str(PROJECT_ROOT))
    os.environ.setdefault("ES_ROOT", str(ENGINE_ROOT))
    os.chdir(ENGINE_ROOT)
    for path in (SERVER_SOURCE.parent, ENGINE_ROOT, ENGINE_ROOT / "src", ENGINE_ROOT / "src" / "third_party"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    sys.argv = [str(SERVER_SOURCE), *passthrough]
    runpy.run_path(str(SERVER_SOURCE), run_name="__main__")


if __name__ == "__main__":
    main()
