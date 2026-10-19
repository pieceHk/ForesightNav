"""Compare explicit state initialization with the original constructor."""

import ast
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from pbsp.agent import PBSPAgent
from pbsp.navigation.evidence import CandidateEvidenceBank
from pbsp.navigation.ranking import FrontierRerankConfig, FrontierUtilityRanker
from runtime.config import load_experiment
from runtime.legacy_navigation import (
    ENVIRONMENT_FIELDS,
    navigation_settings_from_environment,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE = (
    "Navigation/runtime/lagmemo/src/lagmemo/lagmemo/agent/lagmemo_agent/glue_agent.py"
)


@lru_cache()
def frozen_initialization():
    tree = ast.parse(
        subprocess.check_output(["git", "show", "422cb8e:" + SOURCE], cwd=ROOT)
    )
    agent = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "GLUEAgent"
    )
    constructor = next(
        node
        for node in agent.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    start = next(
        index
        for index, node in enumerate(constructor.body)
        if isinstance(node, ast.Assign)
        and ast.unparse(node.targets[0]) == "self.visited_candidate_positions"
    )
    return compile(
        ast.Module(body=constructor.body[start:], type_ignores=[]),
        "<frozen-agent-state>",
        "exec",
    )


@pytest.mark.parametrize(
    "variant", ("defaults", "full_method", "custom", "track_fallback", "bool_strings")
)
def test_explicit_navigation_state_matches_frozen_constructor(variant):
    full = load_experiment(
        ROOT / "configs/experiments/hm3d_4mon.yaml"
    ).method.navigation
    environment = {}
    if variant in {"full_method", "custom"}:
        for name, variable in ENVIRONMENT_FIELDS.items():
            value = getattr(full, name)
            environment[variable] = (
                str(int(value)) if type(value) is bool else str(value)
            )
    if variant == "custom":
        environment.update(
            ES_CANDIDATE_TRACK_RADIUS="2.0",
            ES_CANDIDATE_BLACKLIST_RADIUS="3.0",
            ES_CANDIDATE_EVIDENCE_RADIUS="1.2",
            ES_CANDIDATE_PROMOTION_MARGIN="3",
            HM3D_EXP_VERIFY_WINDOW_STEPS="9",
            HM3D_EXP_VERIFY_POSE_RADIUS="2.5",
            HM3D_EXP_VERIFY_MIN_OBSERVER_YAW_DEG="30",
            ES_FRONTIER_RERANK_ENABLED="1",
            ES_FRONTIER_W_GAIN="2.0",
            ES_TRAPPED_WINDOW_STEPS="10",
            ES_RANDOM_EXPLORATION_WARMUP_STEPS="50",
        )
    if variant == "track_fallback":
        environment = {"ES_CANDIDATE_TRACK_RADIUS": "2.3"}
    if variant == "bool_strings":
        environment = {
            variable: "true"
            for name, variable in ENVIRONMENT_FIELDS.items()
            if type(getattr(full, name)) is bool
        }
    old = SimpleNamespace()
    exec(
        frozen_initialization(),
        dict(
            self=old,
            os=SimpleNamespace(environ=environment),
            CandidateEvidenceBank=CandidateEvidenceBank,
            FrontierUtilityRanker=FrontierUtilityRanker,
            FrontierRerankConfig=FrontierRerankConfig,
        ),
    )
    current = PBSPAgent(
        settings=navigation_settings_from_environment(environment), actions=object()
    )
    assert old.__dict__ == {key: current.__dict__[key] for key in old.__dict__}


def test_agent_settings_ignore_inherited_environment_and_instances_do_not_share_state(
    monkeypatch,
):
    settings = load_experiment(
        ROOT / "configs/experiments/hm3d_4mon.yaml"
    ).method.navigation
    for variable in ENVIRONMENT_FIELDS.values():
        monkeypatch.setenv(variable, "invalid inherited value")
    first = PBSPAgent(settings=settings, actions=object())
    second = PBSPAgent(settings=settings, actions=object())
    assert first.candidate_blacklist_radius == 1.5
    assert first.two_stage_verification is True
    assert first.query_action_rerank_enabled is True
    before = deepcopy(second.visited_candidate_positions)
    first.visited_candidate_positions[0] = [[1, 2, 3]]
    first.trapped_events.append({"step": 30})
    assert second.visited_candidate_positions == before == {}
    assert second.trapped_events == []


def test_model_adapter_requires_explicit_settings_and_has_no_environment_reads():
    source = (ROOT / "adapters/lagmemo_agent.py").read_text()
    tree = ast.parse(source)
    adapter = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    constructor = next(
        node for node in adapter.body if isinstance(node, ast.FunctionDef)
    )
    assert "settings" in [argument.arg for argument in constructor.args.kwonlyargs]
    defaults = dict(zip((arg.arg for arg in constructor.args.kwonlyargs), constructor.args.kw_defaults))
    assert defaults["settings"] is None
    assert isinstance(defaults["navigation_backend"], ast.Constant)
    assert defaults["navigation_backend"].value is None
    assert "os.environ" not in source
    call = constructor.body[0].value
    assert ast.unparse(call.func) == "PBSPAgent.__init__"
    supplied = {keyword.arg: ast.unparse(keyword.value) for keyword in call.keywords}
    assert supplied["settings"] == "settings"
    assert supplied["navigation_backend"] == "navigation_backend"
