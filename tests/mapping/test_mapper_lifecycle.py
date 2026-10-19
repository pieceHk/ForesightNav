"""Compare the extracted mapper against the frozen runtime with fixed RGB-D input."""

import ast
import collections
import math
import os
import subprocess
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from pbsp.mapping.mapper import OnlineMapper
from pbsp.mapping.model import OnlineMappingModel, preprocess_frame
from runtime.config import load_experiment
from runtime.legacy_mapping import ENVIRONMENT_FIELDS, benchmark_intrinsic, model_specification


ROOT = Path(__file__).resolve().parents[2]
BASELINE = "422cb8e"


class TextFeatures:
    def get_text_classifier(self, prompts, device):
        return torch.tensor([[1.0, 0.0] for _ in prompts], device=device)


class RecordingEncoder(torch.nn.Module):
    def __init__(self, started=None, proceed=None):
        super().__init__()
        self.clip = TextFeatures()
        self.contexts = []
        self.resumed = []
        self.started = started
        self.proceed = proceed

    def forward(self, contexts, **kwargs):
        context = contexts[0]
        if self.started is not None and not self.contexts:
            self.started.set()
            assert self.proceed.wait(5), "rebuild release was not signaled"
        self.contexts.append({key: value.detach().cpu().clone() if torch.is_tensor(value) else value
                              for key, value in context.items()})
        previous = kwargs["resume_state"]
        self.resumed.append(None if previous is None else previous["fused_frames"])
        n = context["image"].shape[1]
        assert kwargs["online_incremental"] is True and kwargs["return_state"] is True
        assert kwargs["export_ply"] is False and kwargs["is_testing"] is True
        assert kwargs["global_step"] == 0
        indices = torch.full((1, 3, 3), -1, dtype=torch.long)
        indices[0, :, 0] = 0
        weights = torch.zeros((1, 3, 3))
        weights[0, :, 0] = 1.0
        state = SimpleNamespace(
            valid=torch.ones((1, 3), dtype=torch.bool),
            means=torch.tensor([[[float(n), 0.0, 0.0], [n + .1, 0.0, .1], [n - .1, 0.0, -.1]]]),
            clip_features_2d={"feat": [torch.tensor([[1.0, 0.0]])], "idx": indices, "weight": weights},
        )
        return {"globals": state, "online_state": {"fused_frames": n}}


def _settings(**kwargs):
    base = load_experiment(ROOT / "configs/experiments/hm3d_4mon.yaml").method.mapping
    return replace(base, warmup_keyframes=2, rebuild_interval=2, **kwargs)


def _native(settings, tmp_path, encoder=None, **kwargs):
    encoder = encoder if encoder is not None else RecordingEncoder()
    model = OnlineMappingModel(encoder, torch.nn.Identity(), model_specification(), tmp_path)
    return OnlineMapper(settings=settings, model=model, device="cpu",
                        camera_intrinsic=benchmark_intrinsic(), host_memory_hook=lambda reason: None,
                        **kwargs), encoder


def _frozen(settings, encoder, monkeypatch):
    for field, name in ENVIRONMENT_FIELDS.items():
        value = getattr(settings, field)
        monkeypatch.setenv(name, str(int(value)) if type(value) is bool else str(value))
    source = subprocess.check_output([
        "git", "show", f"{BASELINE}:Mapping/runtime/server/online_es_mapper.py",
    ], cwd=ROOT)
    tree = ast.parse(source)
    constants = {"_H", "_W", "_HFOV_DEG", "_fx", "_fy", "_K_NORM", "_RUB2RDF",
                 "_MAX_GAUSSIANS", "_MAX_KEYFRAMES", "_FRAME_CHANGE_RATIO", "_FRAME_PIXEL_DELTA",
                 "_FRAME_COMPARE_H", "_FRAME_COMPARE_W", "_RAW_FRAME_BUFFER",
                 "_OFFLOAD_FASTSAM_AFTER_REBUILD", "_ENABLE_CLIP_COVERAGE_DIAGNOSTIC", "_NEAR", "_FAR", "_ES_H", "_ES_W"}
    nodes = [node for node in tree.body if (
        isinstance(node, ast.FunctionDef) and node.name != "load_es_model"
        or isinstance(node, ast.ClassDef) and node.name == "OnlineESMapper"
        or isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id in constants for target in node.targets)
    )]
    namespace = dict(np=np, torch=torch, F=F, os=os, math=math, Path=Path,
                     time=time, threading=threading, collections=collections,
                     _FINAL_BEV_MAX_POINTS=200000, _FINAL_BEV_DIR="", SNAP_LIVE="/tmp/pbsp-frozen",
                     ES_CKPT="unused", ES_CFG="unused", release_host_memory=lambda reason: None,
                     load_es_model=lambda *args: (encoder, torch.nn.Identity()))
    from pbsp.mapping.query import candidate_records
    namespace["candidate_records"] = candidate_records
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<frozen-mapping-runtime>", "exec"), namespace)
    return namespace["OnlineESMapper"](device="cpu", top_k=settings.top_k,
                                         warmup_n=settings.warmup_keyframes,
                                         rebuild_interval=settings.rebuild_interval)


def _frame(value):
    rgb = np.full((12, 16, 3), value, dtype=np.uint8)
    depth = np.linspace(.5, 3.0, 12 * 16, dtype=np.float32).reshape(12, 16)
    pose = np.eye(4, dtype=np.float32)
    pose[0, 3] = value / 255.0
    return rgb, depth, pose, np.eye(3, dtype=np.float32)


@pytest.mark.parametrize("limits", ({}, {"max_keyframes": 3}, {"max_gaussians": 3}))
def test_lifecycle_and_contexts_match_frozen_runtime(tmp_path, monkeypatch, limits):
    settings = _settings(**limits)
    extracted, current_encoder = _native(settings, tmp_path)
    reference_encoder = RecordingEncoder()
    reference = _frozen(settings, reference_encoder, monkeypatch)
    for mapper in (reference, extracted):
        mapper.reset_episode("scene-a", "episode-0")
    for value in (0, 255, 250, 128, 128, 0):
        for mapper in (reference, extracted):
            mapper.add_live_frame(*_frame(value))
            mapper.flush_episode()
        assert extracted.get_status() == reference.get_status()
        assert extracted.query_batch_with_records(["chair", "chair"]) == reference.query_batch_with_records(["chair", "chair"])
        assert len(extracted._frames) <= settings.raw_frame_buffer
        assert all(frame["image"].device.type == "cpu" for frame in extracted._keyframes)
    assert current_encoder.resumed == reference_encoder.resumed
    assert len(current_encoder.contexts) == len(reference_encoder.contexts)
    for actual, expected in zip(current_encoder.contexts, reference_encoder.contexts):
        assert actual.keys() == expected.keys()
        for key in actual:
            if torch.is_tensor(actual[key]):
                torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
            else:
                assert actual[key] == expected[key]
    for mapper in (reference, extracted):
        mapper.reset_episode("scene-b", "episode-1")
    assert extracted.get_status() == reference.get_status()
    assert extracted.query("chair") == []
    assert extracted._online_state is None
    extracted.close()
    extracted.close()
    assert extracted.model.encoder is None
    with pytest.raises(RuntimeError, match="closed"):
        extracted.add_live_frame(*_frame(0))


def test_pending_rebuild_fuses_new_frames_after_an_active_update(tmp_path):
    started, proceed = threading.Event(), threading.Event()
    encoder = RecordingEncoder(started, proceed)
    mapper, _ = _native(_settings(), tmp_path, encoder)
    mapper.reset_episode("scene-a", "0")
    mapper.add_live_frame(*_frame(0))
    mapper.add_live_frame(*_frame(255))
    assert started.wait(5), "background update did not start"
    mapper.add_live_frame(*_frame(128))
    mapper.add_live_frame(*_frame(0))
    assert mapper.get_status()["rebuild_pending"] is True
    proceed.set()
    status = mapper.flush_episode()
    assert status["last_fused_keyframes"] == status["n_keyframes"] == 4
    assert encoder.resumed == [None, 2]
    assert status["rebuilding"] is False
    mapper.close()


def test_frame_preparation_uses_explicit_intrinsics():
    spec = model_specification()
    rgb, depth, pose, intrinsic = _frame(128)
    frame = preprocess_frame(rgb, depth, pose, intrinsic, input_spec=spec)
    torch.testing.assert_close(frame["intrinsic"], torch.eye(3), rtol=0, atol=0)
    assert tuple(frame["image"].shape) == (3, spec.image_height, spec.image_width)


def test_explicit_controls_equal_the_frozen_defaults(monkeypatch):
    settings = load_experiment(ROOT / "configs/experiments/hm3d_4mon.yaml").method.mapping
    expected = {
        "_FRAME_COMPARE_H": "frame_compare_height", "_FRAME_COMPARE_W": "frame_compare_width",
        "_MAX_GAUSSIANS": "max_gaussians", "_MAX_KEYFRAMES": "max_keyframes",
        "_OFFLOAD_FASTSAM_AFTER_REBUILD": "offload_fastsam_after_rebuild",
        "_ENABLE_CLIP_COVERAGE_DIAGNOSTIC": "clip_coverage_diagnostic",
    }
    for field in expected.values():
        monkeypatch.delenv(ENVIRONMENT_FIELDS[field], raising=False)
    source = subprocess.check_output(["git", "show", f"{BASELINE}:Mapping/runtime/server/online_es_mapper.py"], cwd=ROOT)
    nodes = [node for node in ast.parse(source).body if isinstance(node, ast.Assign)
             and any(isinstance(target, ast.Name) and target.id in expected for target in node.targets)]
    namespace = {"os": os}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<frozen-mapping-defaults>", "exec"), namespace)
    assert {name: namespace[name] for name in expected} == {
        name: getattr(settings, field) for name, field in expected.items()
    }


def test_checkpoint_prefixes_load_into_the_owned_model(tmp_path):
    encoder = torch.nn.Linear(2, 2, bias=False)
    decoder = torch.nn.Linear(2, 2, bias=False)
    model = OnlineMappingModel(encoder, decoder, model_specification(), tmp_path)
    path = tmp_path / "checkpoint.pt"
    torch.save({"state_dict": {"encoder.weight": torch.ones((2, 2)),
                               "decoder.weight": torch.full((2, 2), 2.0)}}, path)
    report = model.load_checkpoint(path)
    assert all(not value["missing_keys"] and not value["unexpected_keys"] for value in report.values())
    torch.testing.assert_close(encoder.weight, torch.ones((2, 2)))
    torch.testing.assert_close(decoder.weight, torch.full((2, 2), 2.0))
