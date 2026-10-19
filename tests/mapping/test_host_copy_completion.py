"""Host state must be complete before map counts or artifacts read it."""

from types import SimpleNamespace

import torch

from pbsp.mapping import memory


def test_nested_host_copies_complete_once_per_source_stream(monkeypatch):
    operations = []
    streams = {}

    class Stream:
        def __init__(self, device):
            self.device = device
            self.pending = []

        def synchronize(self):
            operations.append(("synchronize", str(self.device)))
            for result, payload in self.pending:
                result.value = payload

    class DelayedTensor:
        def __init__(self, device, payload):
            self.device = torch.device(device)
            self.payload = payload

        def to(self, device, non_blocking):
            assert device == "cpu" and non_blocking is True
            operations.append(("copy", str(self.device)))
            result = SimpleNamespace(value=None)
            streams[self.device].pending.append((result, self.payload))
            return result

    def current_stream(device):
        if device not in streams:
            streams[device] = Stream(device)
        return streams[device]

    monkeypatch.setattr(memory.torch, "is_tensor", lambda value: isinstance(value, DelayedTensor))
    monkeypatch.setattr(memory.torch.cuda, "current_stream", current_stream)
    state = SimpleNamespace(valid=DelayedTensor("cuda:0", 3088415), device="cuda:0")
    tree = {
        "online_state": [state, DelayedTensor("cuda:1", 12)],
        "globals": state,
        "extra": (DelayedTensor("cuda:0", 24),),
    }

    moved = memory._move_tree_to_device(tree, "cpu")

    assert moved is tree
    assert moved["globals"] is moved["online_state"][0]
    assert moved["globals"].valid.value == 3088415
    assert moved["globals"].device == "cpu"
    assert moved["online_state"][1].value == 12
    assert moved["extra"][0].value == 24
    assert operations == [
        ("copy", "cuda:0"),
        ("copy", "cuda:1"),
        ("copy", "cuda:0"),
        ("synchronize", "cuda:0"),
        ("synchronize", "cuda:1"),
    ]


def test_cpu_state_needs_no_cuda_stream(monkeypatch):
    def unexpected_stream(*args):
        raise AssertionError("CPU state must not initialize CUDA")

    monkeypatch.setattr(memory.torch.cuda, "current_stream", unexpected_stream)
    original = torch.tensor([True, False, True])
    state = {"valid": original, "metadata": ("scene-a", 2)}
    result = memory._move_tree_to_device(state, torch.device("cpu"))
    assert result is state
    assert result["valid"] is original
    assert result["metadata"] == ("scene-a", 2)
