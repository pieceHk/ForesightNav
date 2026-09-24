import importlib.util
import math
import os
from copy import deepcopy
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from tdonav_navigation.scsp_contracts import BatchValidationStatus


ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault(
    "LAGMEMO_ROOT", str(ROOT / "Navigation" / "runtime" / "lagmemo")
)
PROVIDER_PATH = (
    ROOT / "Navigation" / "runtime" / "evaluators" / "goat" /
    "online_es_provider.py"
)
SPEC = importlib.util.spec_from_file_location("scsp_goat_online_es_provider", PROVIDER_PATH)
assert SPEC is not None and SPEC.loader is not None
PROVIDER_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROVIDER_MODULE)
OnlineESProvider = PROVIDER_MODULE.OnlineESProvider


TASKS = [
    {"object_category": "chair"},
    {"object_category": "table"},
    {"object_category": "chair"},
]


def _record(position, raw_rank, revision=7, **overrides):
    value = {
        "position": list(position),
        "raw_rank": raw_rank,
        "semantic_score": 0.8 - 0.01 * raw_rank,
        "support_count": 4,
        "support_score": 1.5,
        "map_revision": revision,
    }
    value.update(overrides)
    return value


def _status(
    revision=7,
    *,
    rebuilding=False,
    scene_id="scene-a",
    episode_id="episode-1",
    nested=False,
):
    fields = {
        "last_fused_keyframes": revision,
        "rebuilding": rebuilding,
        "scene_id": scene_id,
        "episode_id": episode_id,
    }
    return {"ok": True, "live": fields} if nested else {"ok": True, **fields}


def _batch_response(revision=7):
    chair = [[9.0, 0.0, 9.0], [1.0, 0.0, 1.0]]
    table = [[5.0, 0.0, 5.0]]
    return {
        "ok": True,
        "results": {"chair": chair, "table": table},
        "candidate_records_by_category": {
            "chair": [
                _record(chair[0], 4, revision),
                _record(chair[1], 1, revision),
            ],
            "table": [_record(table[0], 2, revision)],
        },
        "snap": "live",
    }


def _provider(responses, tasks=TASKS):
    provider = OnlineESProvider(content_root="/unused")
    provider._scene_id = "scene-a"
    provider._episode_id = "episode-1"
    provider._tasks = deepcopy(tasks)
    queue = list(responses)
    calls = []

    def fake_send_recv(message):
        calls.append(deepcopy(message))
        if not queue:
            raise AssertionError(f"unexpected IPC call: {message}")
        response = queue.pop(0)
        if isinstance(response, BaseException):
            raise response
        return deepcopy(response)

    provider._send_recv = fake_send_recv
    return provider, calls, queue


def test_batch_query_keeps_exact_legacy_order_and_builds_duplicate_task_records():
    response = _batch_response()
    expected = {
        0: response["results"]["chair"],
        1: response["results"]["table"],
        2: response["results"]["chair"],
    }
    provider, calls, queue = _provider(
        [_status(), response, _status(nested=True)]
    )

    actual = provider.get_all_goals()

    assert actual == expected
    assert calls[0] == {"cmd": "status"}
    assert calls[1] == {
        "cmd": "batch_query",
        "categories": ["chair", "table"],
        "top_k": 10,
    }
    assert calls[2] == {"cmd": "status"}
    assert not queue
    batch = provider.get_candidate_batch()
    assert batch.validation_status is BatchValidationStatus.VALID
    assert batch.ordered_categories == ("chair", "table", "chair")
    assert batch.version_token.query_serial == 1
    assert batch.version_token.map_revision == 7
    assert batch.records_for_task(0)[0].task_idx == 0
    assert batch.records_for_task(2)[0].task_idx == 2
    assert batch.records_for_task(2)[0].category == "chair"
    assert [item.raw_rank for item in batch.records_for_task(0)] == [4, 1]
    assert batch.positions_for_task(0) == tuple(map(tuple, expected[0]))
    assert provider.current_snap == "live"

    copied = provider.get_all_candidate_records()
    copied[0][0]["raw_rank"] = 999
    copied[0][0]["position"][0] = -999
    fresh = provider.get_all_candidate_records()
    assert fresh[0][0]["raw_rank"] == 4
    assert fresh[0][0]["position"][0] == 9.0
    with pytest.raises(FrozenInstanceError):
        batch.query_serial = 9


def test_successful_empty_batch_is_valid_and_uses_status_revision_not_live_snap():
    provider, _, _ = _provider(
        [
            _status(revision=0),
            {
                "ok": True,
                "results": {"chair": [], "table": []},
                "candidate_records_by_category": {"chair": [], "table": []},
                "snap": "live",
            },
            _status(revision=0),
        ]
    )

    assert provider.get_all_goals() == {0: [], 1: [], 2: []}
    batch = provider.get_candidate_batch()
    assert batch.validation_status is BatchValidationStatus.VALID
    assert batch.map_revision == 0
    assert batch.query_serial == 1
    assert all(not batch.records_for_task(index) for index in range(3))


def test_empty_batch_requires_a_stable_pre_and_post_query_revision():
    response = {
        "ok": True,
        "results": {"chair": [], "table": []},
        "candidate_records_by_category": {"chair": [], "table": []},
        "snap": "live",
    }
    provider, _, _ = _provider(
        [_status(revision=6), response, _status(revision=7)]
    )

    assert provider.get_all_goals() == {0: [], 1: [], 2: []}
    batch = provider.get_candidate_batch()
    assert batch.validation_status is BatchValidationStatus.INVALID
    assert batch.validation_reason.startswith("stale_version_token:")
    assert batch.map_revision == 7


def test_post_query_status_connection_error_keeps_legacy_coordinates():
    response = _batch_response()
    provider, _, _ = _provider(
        [_status(), response, ConnectionError("status connection lost")]
    )

    assert provider.get_all_goals()[0] == response["results"]["chair"]
    batch = provider.get_candidate_batch()
    assert batch.validation_status is BatchValidationStatus.INVALID
    assert batch.validation_reason.startswith("provider_status_unavailable:")
    assert provider.last_fused_keyframes is None


def test_missing_empty_sidecar_key_fails_closed_but_keeps_coordinates():
    response = _batch_response()
    del response["candidate_records_by_category"]["table"]
    expected = {
        0: response["results"]["chair"],
        1: response["results"]["table"],
        2: response["results"]["chair"],
    }
    provider, _, _ = _provider([_status(), response, _status()])

    assert provider.get_all_goals() == expected
    batch = provider.get_candidate_batch()
    assert batch.validation_status is BatchValidationStatus.INVALID
    assert "missing category 'table'" in batch.validation_reason
    assert batch.positions_for_task(0) == ()
    assert batch.records_for_task(0) == ()
    assert provider.get_all_candidate_records() == {}


@pytest.mark.parametrize(
    "mutate,reason",
    [
        (
            lambda response: response["candidate_records_by_category"]["chair"].pop(),
            "count mismatch",
        ),
        (
            lambda response: response["candidate_records_by_category"]["chair"][0].__setitem__(
                "position", [8.0, 0.0, 8.0]
            ),
            "alignment mismatch",
        ),
        (
            lambda response: response["candidate_records_by_category"]["chair"][0].__setitem__(
                "map_revision", 6
            ),
            "mixed map revisions",
        ),
        (
            lambda response: response["candidate_records_by_category"]["chair"][0].__setitem__(
                "semantic_score", math.nan
            ),
            "must be finite",
        ),
    ],
)
def test_invalid_metadata_publishes_empty_sentinel_without_touching_legacy_order(
    mutate, reason
):
    response = _batch_response()
    expected = deepcopy(response["results"]["chair"])
    mutate(response)
    provider, _, _ = _provider([_status(), response, _status()])

    goals = provider.get_all_goals()

    assert goals[0] == expected
    batch = provider.get_candidate_batch()
    assert batch.validation_status is BatchValidationStatus.INVALID
    assert reason in batch.validation_reason
    assert batch.positions_by_task == ((0, ()), (1, ()), (2, ()))
    assert batch.records_by_task == ((0, ()), (1, ()), (2, ()))


def test_status_episode_mismatch_invalidates_metadata_only():
    response = _batch_response()
    expected = deepcopy(response["results"])
    provider, _, _ = _provider(
        [_status(), response, _status(episode_id="other-episode")]
    )

    goals = provider.get_all_goals()

    assert goals[0] == expected["chair"]
    assert goals[1] == expected["table"]
    batch = provider.get_candidate_batch()
    assert batch.validation_status is BatchValidationStatus.INVALID
    assert batch.validation_reason.startswith("episode_mismatch:")


def test_live_snap_without_numeric_status_revision_is_never_a_version_token():
    status = _status()
    del status["last_fused_keyframes"]
    provider, _, _ = _provider([_status(), _batch_response(), status])

    provider.get_all_goals()

    batch = provider.get_candidate_batch()
    assert batch.validation_status is BatchValidationStatus.INVALID
    assert batch.validation_reason.startswith("provider_status_unavailable:")
    assert provider.last_fused_keyframes is None


@pytest.mark.parametrize("malformation", ["none_candidates", "overflow_score"])
def test_malformed_numeric_or_coordinate_payload_never_escapes(malformation):
    response = _batch_response()
    if malformation == "none_candidates":
        response["results"]["chair"] = None
    else:
        response["candidate_records_by_category"]["chair"][0][
            "semantic_score"
        ] = 10**10000
    provider, _, _ = _provider([_status(), response, _status()])

    goals = provider.get_all_goals()

    if malformation == "none_candidates":
        assert goals[0] is None
    else:
        assert goals[0] == response["results"]["chair"]
    batch = provider.get_candidate_batch()
    assert batch.validation_status is BatchValidationStatus.INVALID
    assert batch.validation_reason.startswith("invalid_candidate_batch:")


def test_sequential_fallback_aligns_each_response_and_counts_successful_queries():
    chair = [[2.0, 0.0, 2.0], [1.0, 0.0, 1.0]]
    table = [[3.0, 0.0, 3.0]]
    responses = [
        _status(),
        ["malformed batch protocol response"],
        {
            "ok": True,
            "candidates": chair,
            "candidate_records": [
                _record(chair[0], 5),
                _record(chair[1], 2),
            ],
            "snap": "live",
        },
        {
            "ok": True,
            "candidates": table,
            "candidate_records": [_record(table[0], 3)],
            "snap": "live",
        },
        {
            "ok": True,
            "candidates": chair,
            "candidate_records": [
                _record(chair[0], 5),
                _record(chair[1], 2),
            ],
            "snap": "live",
        },
        _status(),
    ]
    provider, calls, queue = _provider(responses)

    goals = provider.get_all_goals()

    assert goals == {0: chair, 1: table, 2: chair}
    assert provider.query_serial == 3
    assert [call["cmd"] for call in calls] == [
        "status", "batch_query", "query", "query", "query", "status"
    ]
    assert not queue
    batch = provider.get_candidate_batch()
    assert batch.validation_status is BatchValidationStatus.VALID
    assert batch.query_serial == 3
    assert [record.raw_rank for record in batch.records_for_task(0)] == [5, 2]


def test_sequential_mixed_revision_and_failed_subquery_both_fail_closed():
    tasks = TASKS[:2]
    first = [[1.0, 0.0, 1.0]]
    second = [[2.0, 0.0, 2.0]]
    provider, _, _ = _provider(
        [
            _status(),
            {"ok": False},
            {
                "ok": True,
                "candidates": first,
                "candidate_records": [_record(first[0], 0, revision=6)],
            },
            {
                "ok": True,
                "candidates": second,
                "candidate_records": [_record(second[0], 0, revision=7)],
            },
            _status(),
        ],
        tasks=tasks,
    )
    assert provider.get_all_goals() == {0: first, 1: second}
    assert provider.get_candidate_batch().validation_status is BatchValidationStatus.INVALID
    assert "mixed map revisions" in provider.get_candidate_batch().validation_reason

    provider, _, _ = _provider(
        [
            _status(),
            {"ok": False},
            {
                "ok": True,
                "candidates": first,
                "candidate_records": [_record(first[0], 0)],
            },
            ConnectionError("single query connection lost"),
            _status(),
        ],
        tasks=tasks,
    )
    assert provider.get_all_goals() == {0: first, 1: []}
    batch = provider.get_candidate_batch()
    assert batch.validation_status is BatchValidationStatus.INVALID
    assert "task 1 query failed" in batch.validation_reason
    assert provider.query_serial == 1


def test_sequential_empty_layer_requires_stable_revision_across_all_queries():
    tasks = TASKS[:2]
    point = [[2.0, 0.0, 2.0]]
    provider, _, _ = _provider(
        [
            _status(revision=6),
            {"ok": False},
            {"ok": True, "candidates": [], "candidate_records": []},
            {
                "ok": True,
                "candidates": point,
                "candidate_records": [_record(point[0], 0, revision=7)],
            },
            _status(revision=7),
        ],
        tasks=tasks,
    )

    assert provider.get_all_goals() == {0: [], 1: point}
    batch = provider.get_candidate_batch()
    assert batch.validation_status is BatchValidationStatus.INVALID
    assert batch.validation_reason.startswith("stale_version_token:")


def test_cache_invalidation_and_episode_reset_drop_stale_metadata_not_serial():
    provider, _, _ = _provider([_status(), _batch_response(), _status()])
    provider.get_all_goals()
    assert provider.get_candidate_batch() is not None
    assert provider.get_all_candidate_records()
    assert provider.query_serial == 1

    provider.invalidate_cache()

    assert provider._goals_cache == {}
    assert provider.get_all_candidate_records() == {}
    assert provider.get_candidate_batch() is None
    assert provider.last_fused_keyframes is None
    assert provider.query_serial == 1

    provider._load_tasks_for_episode = lambda scene, episode: deepcopy(TASKS)
    provider._send_recv = lambda message: {"ok": True}
    provider.reset_episode("scene-b", "episode-2")
    assert provider._scene_id == "scene-b"
    assert provider._episode_id == "episode-2"
    assert provider.get_candidate_batch() is None
    assert provider.query_serial == 1


def test_query_serial_does_not_advance_on_cache_hit_status_or_failed_query():
    tasks = TASKS[:1]
    point = [[1.0, 0.0, 1.0]]
    provider, calls, _ = _provider(
        [
            {
                "ok": True,
                "candidates": point,
                "candidate_records": [_record(point[0], 0)],
            },
        ],
        tasks=tasks,
    )
    assert provider.get_goals(0, "chair") == point
    assert provider.get_goals(0, "chair") == point
    assert provider.query_serial == 1
    assert len(calls) == 1
    assert provider.get_all_candidate_records() == {}

    provider.invalidate_cache()
    provider._send_recv = lambda message: {"ok": False, "error": "no map"}
    assert provider.get_goals(0, "chair") == []
    assert provider.query_serial == 1

    provider, _, _ = _provider(
        [["malformed single-query response"]], tasks=tasks
    )
    assert provider.get_goals(0, "chair") == []
    assert provider.query_serial == 0
    assert provider.get_all_candidate_records() == {}


def _valid_provider_and_batch():
    provider, _, _ = _provider([_status(), _batch_response(), _status()])
    provider.get_all_goals()
    return provider, provider.get_candidate_batch()


@pytest.mark.parametrize(
    "status,kwargs,reason",
    [
        (_status(rebuilding=True), {}, "rebuilding"),
        (_status(episode_id="other"), {}, "episode_mismatch"),
        (_status(revision=8), {}, "stale_version_token"),
        ({"ok": False}, {}, "provider_status_unavailable"),
        (
            _status(),
            {"ordered_categories": ("table", "chair", "chair")},
            "episode_mismatch",
        ),
    ],
)
def test_pre_dispatch_status_transition_table(status, kwargs, reason):
    provider, batch = _valid_provider_and_batch()

    event = provider.check_candidate_batch_status(
        batch, status_response=status, **kwargs
    )

    assert event["would_invalidate"] is True
    assert event["reason"] == reason
    assert provider.get_last_version_gate() == event


def test_pre_dispatch_safe_and_provider_token_changed_transitions():
    provider, old_batch = _valid_provider_and_batch()
    event = provider.check_candidate_batch_status(
        old_batch, status_response=_status()
    )
    assert event["would_invalidate"] is False
    assert event["reason"] == "none"

    provider.invalidate_cache()
    provider._send_recv = lambda message: (
        deepcopy(_batch_response(revision=8))
        if message["cmd"] == "batch_query"
        else _status(revision=8)
    )
    provider.get_all_goals()
    assert provider.get_candidate_batch().query_serial == old_batch.query_serial + 1

    event = provider.check_candidate_batch_status(
        old_batch, status_response=_status(revision=8)
    )
    assert event["would_invalidate"] is True
    assert event["reason"] == "provider_token_changed"


def test_failed_refresh_cannot_revalidate_an_old_equal_token_batch():
    provider, old_batch = _valid_provider_and_batch()
    assert old_batch.validation_status is BatchValidationStatus.VALID
    old_serial = old_batch.query_serial
    failed_responses = iter(
        [
            _status(),
            {"ok": False},
            {"ok": False, "error": "failed chair"},
            {"ok": False, "error": "failed table"},
            {"ok": False, "error": "failed duplicate chair"},
            _status(),
        ]
    )
    provider._send_recv = lambda message: deepcopy(next(failed_responses))

    assert provider.get_all_goals() == {0: [], 1: [], 2: []}
    current = provider.get_candidate_batch()
    assert current.validation_status is BatchValidationStatus.INVALID
    assert current.query_serial == old_serial
    assert current.version_token == old_batch.version_token

    event = provider.check_candidate_batch_status(
        old_batch, status_response=_status()
    )
    assert event["would_invalidate"] is True
    assert event["reason"] == "provider_token_changed"


def test_invalid_batch_gate_does_not_read_status_or_change_legacy_candidates():
    response = _batch_response()
    response["candidate_records_by_category"]["chair"][0]["map_revision"] = 8
    provider, calls, _ = _provider([_status(), response, _status()])
    goals = provider.get_all_goals()
    calls_before_gate = len(calls)

    event = provider.check_candidate_batch_status(provider.get_candidate_batch())

    assert event["reason"] == "invalid_candidate_batch"
    assert len(calls) == calls_before_gate
    assert goals[0] == response["results"]["chair"]
