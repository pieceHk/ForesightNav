"""Process termination may race with reads of Linux procfs files."""

from pathlib import Path
import os
import subprocess
import sys

import pytest

from runtime import monitor


@pytest.mark.parametrize("error", (FileNotFoundError, ProcessLookupError, PermissionError))
def test_key_values_tolerates_disappearing_or_unreadable_processes(monkeypatch, error):
    def read(*args, **kwargs):
        raise error("process is gone or unreadable")
    monkeypatch.setattr(Path, "read_text", read)
    assert monitor.key_values(Path("/proc/123/smaps_rollup")) == {}


def test_snapshot_survives_cmdline_and_rollup_exit_races(tmp_path, monkeypatch):
    for pid, ppid in ((10, 1), (11, 10), (12, 10)):
        process = tmp_path / str(pid)
        process.mkdir()
        (process / "cmdline").write_bytes(b"python\x00worker")
        (process / "status").write_text(f"PPid:\t{ppid}\nVmRSS:\t20 kB\nRssAnon:\t12 kB\n")
        (process / "smaps_rollup").write_text("Pss_Anon: 8 kB\nPrivate_Dirty: 4 kB\n")
    original_bytes, original_text = Path.read_bytes, Path.read_text

    def read_bytes(path):
        if path == tmp_path / "12/cmdline":
            raise ProcessLookupError("exited before cmdline read")
        return original_bytes(path)

    def read_text(path, *args, **kwargs):
        if path == tmp_path / "11/smaps_rollup":
            raise ProcessLookupError("exited after status read")
        return original_text(path, *args, **kwargs)

    def proc_path(*parts):
        return tmp_path.joinpath(*parts[1:]) if parts[0] == "/proc" else Path(*parts)

    monkeypatch.setattr(monitor, "Path", proc_path)
    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    monkeypatch.setattr(Path, "read_text", read_text)
    rows = monitor.process_snapshot(10)
    assert [row["pid"] for row in rows] == [10, 11]
    assert rows[0]["pss_anon_kib"] == 8
    assert rows[1]["rss_kib"] == 20
    assert rows[1]["pss_anon_kib"] is None
    assert rows[1]["private_dirty_kib"] is None


@pytest.mark.skipif(not Path("/proc/self/status").is_file(), reason="Linux procfs integration")
def test_procfs_sampling_with_short_lived_children():
    processes = []
    root_pid = os.getpid()
    try:
        for _ in range(25):
            processes.append(subprocess.Popen([sys.executable, "-c", "pass"]))
            rows = monitor.process_snapshot(root_pid)
            assert any(row["pid"] == root_pid for row in rows)
            for process in processes:
                process.poll()
        for process in processes:
            assert process.wait(timeout=10) == 0
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=10)
