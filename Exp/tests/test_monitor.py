import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "monitor.py"
if not MODULE_PATH.exists():
    MODULE_PATH = Path(__file__).with_name("monitor_36e6c403.py")


def load_monitor():
    spec = importlib.util.spec_from_file_location("resource_monitor", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_process(proc_root: Path, pid: int, ppid: int, children=()):
    root = proc_root / str(pid)
    (root / "task" / str(pid)).mkdir(parents=True)
    (root / "task" / str(pid) / "children").write_text(
        " ".join(str(child) for child in children), encoding="utf-8"
    )
    (root / "comm").write_text(f"process-{pid}\n", encoding="utf-8")
    (root / "status").write_text(
        f"PPid:\t{ppid}\nVmRSS:\t{pid} kB\nRssAnon:\t{pid - 1} kB\n",
        encoding="utf-8",
    )
    (root / "smaps_rollup").write_text(
        f"Pss_Anon:\t{pid - 2} kB\nPrivate_Dirty:\t{pid - 3} kB\n",
        encoding="utf-8",
    )


def test_process_snapshot_reads_only_root_descendants(tmp_path):
    monitor = load_monitor()
    write_process(tmp_path, 100, 1, (101, 102))
    write_process(tmp_path, 101, 100, (103,))
    write_process(tmp_path, 102, 100)
    write_process(tmp_path, 103, 101)
    (tmp_path / "100" / "task" / "1001").mkdir()
    (tmp_path / "100" / "task" / "1001" / "children").write_text(
        "104", encoding="utf-8"
    )
    write_process(tmp_path, 104, 100)
    write_process(tmp_path, 999, 1)

    rows = monitor.process_snapshot(100, proc_root=tmp_path)

    assert [row["pid"] for row in rows] == [100, 101, 102, 103, 104]
    assert rows[-1]["command"] == "process-104"
    assert rows[-1]["pss_anon_kib"] == 102
    assert rows[-1]["private_dirty_kib"] == 101


def test_process_snapshot_tolerates_exited_child(tmp_path):
    monitor = load_monitor()
    write_process(tmp_path, 100, 1, (101,))

    assert [row["pid"] for row in monitor.process_snapshot(100, tmp_path)] == [100]
