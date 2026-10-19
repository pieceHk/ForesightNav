"""Exercise the actual socket service and compare retained numerical handlers."""

import ast
import base64
from copy import deepcopy
from functools import lru_cache
import gzip
import json
import os
from pathlib import Path
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np
import pytest

from experiments.dataset import load_runtime_tasks
from runtime import mapping_client
from runtime.mapping_client import MappingClient
from runtime.mapping_worker import MappingServer, run_worker
from runtime.protocol import (
    decode_binary_payload,
    decode_message,
    encode_message,
    receive_line,
    request,
)


ROOT = Path(__file__).resolve().parents[2]
BASELINE = "01c65ba"
CLIENT = "Navigation/runtime/evaluators/goat/online_es_provider.py"
SERVER = "Mapping/runtime/server/es_mapper_server.py"


@pytest.fixture
def socket_path():
    with tempfile.TemporaryDirectory(prefix="pbsp-ipc-", dir="/tmp") as directory:
        yield Path(directory) / "map.sock"


@lru_cache()
def frozen_classes():
    client_tree = ast.parse(
        subprocess.check_output(["git", "show", BASELINE + ":" + CLIENT], cwd=ROOT)
    )
    server_tree = ast.parse(
        subprocess.check_output(["git", "show", BASELINE + ":" + SERVER], cwd=ROOT)
    )
    old_client = next(node for node in client_tree.body if isinstance(node, ast.ClassDef))
    old_server = next(node for node in server_tree.body if isinstance(node, ast.ClassDef))
    namespace = {
        **vars(mapping_client),
        "CandidateProvider": object,
        "SOCKET_PATH": "/tmp/unused",
        "CONTENT_ROOT": "/unused",
        "gzip": gzip,
        "np": np,
        "io": __import__("io"),
        "_decode_binary_payload": decode_binary_payload,
    }
    exec(
        compile(
            ast.Module(body=[old_client, old_server], type_ignores=[]),
            "<frozen-mapping-ipc>",
            "exec",
        ),
        namespace,
    )
    return namespace["OnlineESProvider"], namespace["ESMapperServer"]


class Clock:
    def __init__(self):
        self.value = 0

    def perf_counter(self):
        self.value += 1
        return self.value * 0.001


class Mapper:
    def __init__(self):
        self.calls, self.frames = [], []
        self.closed = self.flushes = self.queries = 0
        self.scene = self.episode = ""
        self.revision = 7

    def reset_episode(self, scene, episode):
        self.scene, self.episode = scene, episode
        self.calls.append(("reset", scene, episode))

    def add_live_frame(self, rgb, depth, extrinsic, intrinsic):
        self.frames.append((rgb, depth, extrinsic, intrinsic))
        self.calls.append(("frame", rgb.shape, depth.shape))

    def push_frame(self, frame):
        self.calls.append(("snapshot_frame", frame))

    def query_with_records(self, category, top_k):
        self.queries += 1
        self.calls.append(("query", category, top_k))
        point = [1.0, 0.0, 2.0]
        return [point], [
            {
                "position": point,
                "raw_rank": 0,
                "semantic_score": 0.5,
                "support_count": 4,
                "support_score": 1.0,
                "map_revision": self.revision,
            }
        ]

    def query_batch_with_records(self, categories, top_k):
        results, records = {}, {}
        for category in categories:
            results[category], records[category] = self.query_with_records(category, top_k)
        return results, records

    def query(self, task, category, top_k):
        return self.query_with_records(category, top_k)[0]

    def get_status(self):
        return {
            "scene_id": self.scene,
            "episode_id": self.episode,
            "last_fused_keyframes": self.revision,
            "rebuilding": False,
            "snap": "live",
            "nav_steps": len(self.frames),
        }

    def flush_episode(self):
        self.flushes += 1
        return self.get_status()

    def close(self):
        self.closed += 1


def serve_in_thread(path, mapper):
    ready, stopped = threading.Event(), threading.Event()
    errors = []

    def run():
        try:
            run_worker(MappingServer(mapper), str(path), stop_event=stopped, on_ready=ready.set)
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    assert ready.wait(3), errors
    return thread, stopped, errors


def packed_frame(monkeypatch, native, pose):
    if native:
        monkeypatch.setattr(mapping_client, "time", Clock())
        client = MappingClient("/unused", task_reader=lambda *_: [])
    else:
        cls, _ = frozen_classes()
        cls.push_live_frame.__globals__["time"] = Clock()
        client = cls(content_root="/unused")
    payloads = []
    client._send_recv = lambda message: (payloads.append(message) or {"ok": True})
    rgb = np.arange(180, dtype=np.uint8).reshape(6, 10, 3)
    depth = np.linspace(0.1, 3, 60, dtype=np.float64).reshape(6, 10)
    client.push_live_frame(rgb, depth, np.asarray(pose, dtype=np.float32))
    return payloads[0], client.n_live_frames, client._t_push_ms


@pytest.mark.parametrize(
    "pose",
    (
        [1.0, 0.0, 0.0, 0.0, 2.0, 1.0, -3.0],
        [0.70710678, 0.0, 0.70710678, 0.0, -1.0, 0.0, 4.0],
        [0.5, 0.5, 0.5, 0.5, 0.25, 2.0, -0.25],
    ),
)
def test_jpeg_depth_and_pose_payloads_match_frozen(monkeypatch, pose):
    assert packed_frame(monkeypatch, True, pose) == packed_frame(monkeypatch, False, pose)


@pytest.mark.parametrize(
    "command",
    (
        "reset",
        "frame",
        "missing_frame",
        "query",
        "batch",
        "status",
        "flush",
        "snapshot_disabled",
    ),
)
def test_command_handlers_and_model_inputs_match_frozen(monkeypatch, command):
    _, old_cls = frozen_classes()
    payload = packed_frame(monkeypatch, True, [1.0, 0.0, 0.0, 0.0, 2.0, 1.0, -3.0])[0]
    results = []
    for native in (False, True):
        mapper = Mapper()
        if native:
            service = MappingServer(mapper)
        else:
            service = old_cls.__new__(old_cls)
            service.top_k, service._live_mode, service._live_only = 10, True, True
            service._live_mapper, service._mapper = mapper, None
        if command == "reset":
            response = service.handle_reset_episode("scene", "7")
        elif command == "frame":
            response = service.handle_push_live_frame(
                **{k: v for k, v in payload.items() if k != "cmd"}
            )
        elif command == "missing_frame":
            response = service.handle_push_live_frame(0)
        elif command == "query":
            response = service.handle_query(0, "chair", 3)
        elif command == "batch":
            response = service.handle_batch_query(["chair", "bed"])
        elif command == "status":
            response = service.handle_status()
        elif command == "flush":
            response = service.handle_flush()
        else:
            response = service.handle_push_frame(0)
        results.append((response, mapper.calls, mapper.frames, mapper.flushes))
    assert results[0][:2] == results[1][:2]
    assert results[0][3] == results[1][3]
    for before, after in zip(results[0][2], results[1][2]):
        for a, b in zip(before, after):
            np.testing.assert_array_equal(a, b)


def test_client_state_and_numerical_methods_preserve_executable_source():
    old = ast.parse(subprocess.check_output(["git", "show", BASELINE + ":" + CLIENT], cwd=ROOT))
    new = ast.parse((ROOT / "runtime/mapping_client.py").read_text())
    previous = {n.name: n for n in ast.walk(old) if isinstance(n, ast.FunctionDef)}
    changed = {"__init__", "_load_tasks_for_episode", "_send_recv", "_ensure_connected"}

    def executable(node):
        node = deepcopy(node)
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            node.body.pop(0)
        return ast.dump(node, include_attributes=False)

    for node in ast.walk(new):
        if isinstance(node, ast.FunctionDef) and node.name not in changed:
            assert executable(node) == executable(previous[node.name])


@pytest.mark.parametrize(
    "ids,requested,expected",
    (
        (["0", "1"], "1", "bed"),
        (["1", "10"], "1", "bed"),
        (["1", "10"], "10", "bed"),
        (["1", "10"], "missing", None),
    ),
)
def test_dataset_task_reader_keeps_frozen_reindex_semantics(tmp_path, ids, requested, expected):
    episodes = [
        {"episode_id": key, "tasks": [[category, "object"]]}
        for key, category in zip(ids, ("chair", "bed"))
    ]
    with gzip.open(tmp_path / "scene.json.gz", "wt") as stream:
        json.dump({"episodes": episodes}, stream)
    old, _ = frozen_classes()
    previous = old(content_root=str(tmp_path))._load_tasks_for_episode("scene", requested)
    actual = load_runtime_tasks(tmp_path, "scene", requested)
    assert actual == previous
    assert (actual[0][0] if actual else None) == expected


def test_live_client_worker_roundtrip_cache_versions_and_shutdown(socket_path):
    mapper = Mapper()
    thread, stopped, errors = serve_in_thread(socket_path, mapper)
    client = MappingClient(
        str(socket_path),
        task_reader=lambda *_: [["chair", "object"], ["chair", "object"], ["bed", "object"]],
    )
    try:
        assert stat.S_IMODE(socket_path.stat().st_mode) == 0o600
        client.reset_episode("scene", "7")
        rgb, depth = np.zeros((6, 10, 3), dtype=np.uint8), np.full((6, 10), 1.25)
        client.push_live_frame(rgb, depth, np.array([1.0, 0.0, 0.0, 0.0, 2.0, 1.0, -3.0]))
        goals = client.get_all_goals()
        assert goals[0] == goals[1] == goals[2] == [[1.0, 0.0, 2.0]]
        batch = client.get_candidate_batch()
        assert batch.ordered_categories == ("chair", "chair", "bed")
        assert client.check_candidate_batch_status(batch)["would_invalidate"] is False
        assert mapper.queries == 2
        assert client.get_goals(0, "chair") == goals[0] and mapper.queries == 2
        np.testing.assert_array_equal(mapper.frames[0][1], depth.astype(np.float32))
        mapper.revision += 1
        assert client.check_candidate_batch_status(batch)["reason"] == "stale_version_token"
        assert client.flush_episode()["last_fused_keyframes"] == 8
        assert request(socket_path, {"cmd": "shutdown"})["ok"]
        thread.join(3)
        assert not thread.is_alive() and not errors
        assert mapper.closed == 1 and mapper.flushes == 2
        assert not socket_path.exists()
    finally:
        client.close()
        stopped.set()
        thread.join(3)


def test_protocol_supports_legacy_messages_and_rejects_unknown_versions(socket_path):
    mapper = Mapper()
    thread, stopped, errors = serve_in_thread(socket_path, mapper)
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.connect(str(socket_path))
            for payload, success in (
                (b'{"cmd":"status"}\n', True),
                (b'{"cmd":"status","protocol_version":99}\n', False),
                (b'{"cmd":"status","protocol_version":true}\n', False),
                (b'not-json\n', False),
                (encode_message({"cmd": "status"}), True),
            ):
                client.sendall(payload)
                line, remainder = receive_line(client)
                assert not remainder
                response = decode_message(line)
                assert response["ok"] is success
                assert response["protocol_version"] == 1
        request(socket_path, {"cmd": "shutdown"})
        thread.join(3)
        assert not thread.is_alive() and not errors and mapper.closed == 1
    finally:
        stopped.set()
        thread.join(3)


def test_fragmented_and_buffered_json_frames_preserve_message_boundaries():
    class Connection:
        def __init__(self):
            self.blocks = iter((b'{"ok":', b'true}\n{"ok":false}\n'))

        def recv(self, size):
            return next(self.blocks)

    connection = Connection()
    first, buffer = receive_line(connection)
    second, buffer = receive_line(connection, buffer)
    assert decode_message(first)["ok"] is True
    assert decode_message(second)["ok"] is False
    assert buffer == b""


@pytest.mark.parametrize("payload", (b"abc", [97, 98, 99], (97, 98, 99), "YWJj"))
def test_binary_payload_compatibility(payload):
    assert decode_binary_payload(payload) == b"abc"


def test_worker_preserves_existing_files_and_closes_supplied_model(socket_path):
    socket_path.write_text("keep this file")
    mapper = Mapper()
    with pytest.raises(FileExistsError):
        run_worker(MappingServer(mapper), str(socket_path))
    assert socket_path.read_text() == "keep this file" and mapper.closed == 1


def test_worker_does_not_replace_an_active_listener(socket_path):
    mapper, second = Mapper(), Mapper()
    thread, stopped, errors = serve_in_thread(socket_path, mapper)
    try:
        with pytest.raises(RuntimeError, match="already has a listener"):
            run_worker(MappingServer(second), str(socket_path))
        assert second.closed == 1 and mapper.closed == 0
        assert request(socket_path, {"cmd": "status"})["ok"]
    finally:
        stopped.set()
        thread.join(3)
    assert not errors and mapper.closed == 1


def test_worker_reuses_a_stale_socket_and_closes_on_external_stop(socket_path):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as stale:
        stale.bind(str(socket_path))
    mapper = Mapper()
    thread, stopped, errors = serve_in_thread(socket_path, mapper)
    stopped.set()
    thread.join(3)
    assert not thread.is_alive() and not errors
    assert mapper.closed == 1 and not socket_path.exists()


def test_query_failure_keeps_connection_usable_for_sequential_fallback(socket_path):
    class BrokenBatch(Mapper):
        def query_batch_with_records(self, categories, top_k):
            raise RuntimeError("batch query unavailable")

    mapper = BrokenBatch()
    thread, stopped, errors = serve_in_thread(socket_path, mapper)
    client = MappingClient(
        str(socket_path), task_reader=lambda *_: [["chair", "object"], ["bed", "object"]]
    )
    try:
        client.reset_episode("scene", "7")
        assert client.get_all_goals() == {0: [[1.0, 0.0, 2.0]], 1: [[1.0, 0.0, 2.0]]}
        assert client.get_candidate_batch().validation_status.value == "valid"
        assert client.check_candidate_batch_status()["would_invalidate"] is False
        assert request(socket_path, {"cmd": "shutdown"})["ok"]
        thread.join(3)
        assert not thread.is_alive() and not errors and mapper.closed == 1
    finally:
        client.close()
        stopped.set()
        thread.join(3)


def test_service_releases_other_backends_if_one_close_fails():
    class BrokenClose(Mapper):
        def close(self):
            super().close()
            raise RuntimeError("close failed")

    first, second = BrokenClose(), Mapper()
    server = MappingServer(first, snapshot_mapper=second)
    with pytest.raises(RuntimeError, match="close failed"):
        server.close()
    assert first.closed == second.closed == 1
    server.close()
    assert first.closed == second.closed == 1


def test_worker_cli_sigterm_releases_mapper_and_socket(tmp_path, socket_path):
    marker = tmp_path / "closed.txt"
    script = f'''
from pathlib import Path
from runtime import legacy_mapping
class Mapper:
    def get_status(self): return {{"ready": True}}
    def close(self): Path({str(marker)!r}).write_text("closed")
legacy_mapping.create_mapper = lambda **kwargs: Mapper()
from runtime.mapping_worker import main
main(["--socket_path", {str(socket_path)!r}])
'''
    environment = {**os.environ, "PYTHONPATH": str(ROOT)}
    for name in (
        "TDONAV_BLAS_NUM_THREADS",
        "TDONAV_TORCH_NUM_THREADS",
        "TDONAV_TORCH_NUM_INTEROP_THREADS",
        "TDONAV_OPENCV_NUM_THREADS",
    ):
        environment[name] = "0"
    process = subprocess.Popen(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        deadline = time.monotonic() + 5
        while True:
            assert process.poll() is None
            try:
                if request(socket_path, {"cmd": "status"}, timeout_s=0.2)["ok"]:
                    break
            except (ConnectionError, OSError):
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.02)
        process.send_signal(signal.SIGTERM)
        stdout, _ = process.communicate(timeout=5)
        assert process.returncode == 0, stdout
        assert marker.read_text() == "closed" and not socket_path.exists()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def test_native_runtime_imports_do_not_load_models_or_mutate_environment(tmp_path):
    code = '''
import os,sys
before=dict(os.environ)
import runtime.mapping_client, runtime.mapping_worker, runtime.protocol
assert dict(os.environ)==before
assert not any(name in sys.modules for name in ('torch','numpy','habitat','lagmemo','PIL'))
'''
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        check=True,
    )


def test_refused_connections_release_their_socket(monkeypatch):
    from types import SimpleNamespace

    class Refused:
        closed = 0

        def connect(self, address):
            raise ConnectionRefusedError()

        def close(self):
            self.closed += 1

    connection = Refused()
    now = [0.0]
    monkeypatch.setattr(
        mapping_client, "os", SimpleNamespace(path=SimpleNamespace(exists=lambda _: True))
    )
    monkeypatch.setattr(
        mapping_client,
        "socket",
        SimpleNamespace(AF_UNIX=1, SOCK_STREAM=1, socket=lambda *_: connection),
    )
    monkeypatch.setattr(
        mapping_client,
        "time",
        SimpleNamespace(
            time=lambda: now[0], sleep=lambda delay: now.__setitem__(0, now[0] + delay)
        ),
    )
    client = MappingClient("unused", task_reader=lambda *_: [], connect_timeout=0.5)
    with pytest.raises(RuntimeError, match="not available"):
        client._ensure_connected()
    assert connection.closed == 1 and client._sock is None
