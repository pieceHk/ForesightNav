#!/usr/bin/env python3
"""Record bounded host and GPU resource snapshots as JSON Lines."""

import argparse
import json
import os
import signal
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path


STOP = False
_WAKE = threading.Event()


def command(args):
    return subprocess.run(
        args, check=False, capture_output=True, text=True
    ).stdout.strip()


def key_values(path: Path) -> dict:
    values = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            key, value = line.split(":", 1)
            try:
                values[key] = int(value.strip().split()[0])
            except (IndexError, ValueError):
                continue
    except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
        pass
    return values


def process_snapshot(root_pid: int) -> list:
    rows = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command_line = (
                (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode().strip()
            )
        except (FileNotFoundError, ProcessLookupError, PermissionError, UnicodeDecodeError):
            continue
        status = key_values(entry / "status")
        pid = int(entry.name)
        rows[pid] = {
            "pid": pid,
            "ppid": status.get("PPid"),
            "rss_kib": status.get("VmRSS"),
            "rss_anon_kib": status.get("RssAnon"),
            "command": command_line,
        }
    selected = {root_pid}
    while True:
        descendants = {pid for pid, row in rows.items() if row["ppid"] in selected}
        expanded = selected | descendants
        if expanded == selected:
            break
        selected = expanded
    result = []
    for pid in sorted(selected):
        if pid not in rows:
            continue
        row = rows[pid]
        smaps = key_values(Path("/proc") / str(pid) / "smaps_rollup")
        row["pss_anon_kib"] = smaps.get("Pss_Anon")
        row["private_dirty_kib"] = smaps.get("Private_Dirty")
        result.append(row)
    return result


def snapshot(root_pid: int, gpus: str) -> dict:
    memory = key_values(Path("/proc/meminfo"))
    processes = process_snapshot(root_pid)
    process_ids = {str(row["pid"]) for row in processes}
    gpu_processes = command(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_gpu_memory,gpu_uuid",
            "--format=csv,noheader,nounits",
        ]
    ).splitlines()
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "host_kib": {
            key: memory.get(key)
            for key in ("MemTotal", "MemAvailable", "Cached", "Buffers", "SReclaimable")
        },
        "gpu": command(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
                "-i",
                gpus,
            ]
        ).splitlines(),
        "gpu_processes": [
            line
            for line in gpu_processes
            if line.split(",", 1)[0].strip() in process_ids
        ],
        "processes": processes,
    }


def stop(_signum, _frame) -> None:
    global STOP
    STOP = True
    _WAKE.set()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--root-pid", type=int, default=os.getppid())
    parser.add_argument("--gpus", default="0,1,2,3")
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    handle = None
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        handle = args.output.open("a", encoding="utf-8")
    try:
        while True:
            line = json.dumps(snapshot(args.root_pid, args.gpus), ensure_ascii=True)
            if handle:
                handle.write(line + "\n")
                handle.flush()
            else:
                print(line)
            if not handle or STOP or not (Path("/proc") / str(args.root_pid)).exists():
                break
            if _WAKE.wait(args.interval):
                break
    finally:
        if handle:
            handle.close()


if __name__ == "__main__":
    main()
