"""Exercise resumable downloads against interrupted and invalid HTTP responses."""

import hashlib
import io
import urllib.error

import pytest

from scripts import prepare_models as models


class Response(io.BytesIO):
    def __init__(self, payload, status, headers):
        super().__init__(payload)
        self.status, self.headers = status, headers


def asset(payload=b"abcdefghij"):
    return {"path": "weights.bin", "url": "https://example.org/weights",
            "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def test_short_response_resumes_exact_prefix_and_uses_bounded_ranges(tmp_path, monkeypatch):
    item, calls = asset(), []
    monkeypatch.setattr(models, "CHUNK_BYTES", 4)
    monkeypatch.setattr(models.time, "sleep", lambda _: None)

    def serve(request, **kwargs):
        value = request.get_header("Range")
        calls.append(value)
        start, end = map(int, value.removeprefix("bytes=").split("-"))
        data = b"abcdefghij"[start:end + 1]
        if len(calls) == 1:
            data = data[:2]
        return Response(data, 206, {"Content-Range": f"bytes {start}-{end}/10",
                                    "Content-Length": str(end - start + 1)})

    monkeypatch.setattr(models.urllib.request, "urlopen", serve)
    assert models.fetch(tmp_path, item, False).read_bytes() == b"abcdefghij"
    assert calls == ["bytes=0-3", "bytes=2-5", "bytes=6-9"]


def test_full_response_restarts_at_zero_when_range_is_ignored(tmp_path, monkeypatch):
    item = asset()
    (tmp_path / (item["sha256"] + ".partial")).write_bytes(b"abc")
    monkeypatch.setattr(models.urllib.request, "urlopen", lambda *a, **k:
                        Response(b"abcdefghij", 200, {"Content-Length": "10"}))
    assert models.fetch(tmp_path, item, False).read_bytes() == b"abcdefghij"


@pytest.mark.parametrize("headers", [
    {"Content-Range": "bytes 1-9/10"},
    {"Content-Range": "bytes 0-9/11"},
    {"Content-Range": "bytes 0-9/10", "Content-Length": "9"},
    {"Content-Range": "bytes 0-9/10", "Content-Encoding": "gzip"},
])
def test_wrong_range_or_encoding_never_changes_partial(tmp_path, monkeypatch, headers):
    item = asset()
    partial = tmp_path / (item["sha256"] + ".partial")
    partial.write_bytes(b"abc")
    calls = []

    def serve(*args, **kwargs):
        calls.append(1)
        return Response(b"abcdefghij", 206, headers)

    monkeypatch.setattr(models.urllib.request, "urlopen", serve)
    with pytest.raises(ValueError):
        models.fetch(tmp_path, item, False)
    assert calls == [1]
    assert partial.read_bytes() == b"abc"


@pytest.mark.parametrize("code,expected_calls", [(404, 1), (429, 3), (503, 3)])
def test_http_failures_have_bounded_retries(tmp_path, monkeypatch, code, expected_calls):
    calls = []

    def serve(*args, **kwargs):
        calls.append(1)
        raise urllib.error.HTTPError("https://example.org/weights", code, "failure", {}, None)

    monkeypatch.setattr(models.urllib.request, "urlopen", serve)
    monkeypatch.setattr(models.time, "sleep", lambda _: None)
    with pytest.raises(urllib.error.HTTPError):
        models.fetch(tmp_path, asset(), False)
    assert len(calls) == expected_calls


def test_repeated_short_reads_retain_partial_then_stop(tmp_path, monkeypatch):
    calls = []

    def serve(request, **kwargs):
        calls.append(1)
        return Response(b"", 206, {"Content-Range": "bytes 0-9/10"})

    monkeypatch.setattr(models.urllib.request, "urlopen", serve)
    monkeypatch.setattr(models.time, "sleep", lambda _: None)
    with pytest.raises(models.ShortDownload):
        models.fetch(tmp_path, asset(), False)
    assert len(calls) == 3
    assert (tmp_path / (asset()["sha256"] + ".partial")).read_bytes() == b""


def test_full_partial_is_verified_without_another_request(tmp_path, monkeypatch):
    item = asset()
    partial = tmp_path / (item["sha256"] + ".partial")
    partial.write_bytes(b"abcdefghij")
    monkeypatch.setattr(models.urllib.request, "urlopen", lambda *a, **k: pytest.fail("unexpected request"))
    assert models.fetch(tmp_path, item, False).read_bytes() == b"abcdefghij"
    assert not partial.exists()


def test_wrong_full_hash_is_not_retried_or_published(tmp_path, monkeypatch):
    item, calls = asset(), []

    def serve(*args, **kwargs):
        calls.append(1)
        return Response(b"0123456789", 200, {})

    monkeypatch.setattr(models.urllib.request, "urlopen", serve)
    with pytest.raises(ValueError, match="checksum"):
        models.fetch(tmp_path, item, False)
    assert calls == [1]
    assert not (tmp_path / item["sha256"]).exists()
    assert (tmp_path / (item["sha256"] + ".partial")).read_bytes() == b"0123456789"


def test_oversized_partial_is_rejected_without_network(tmp_path, monkeypatch):
    item = asset()
    (tmp_path / (item["sha256"] + ".partial")).write_bytes(b"oversized file")
    monkeypatch.setattr(models.urllib.request, "urlopen", lambda *a, **k: pytest.fail("unexpected request"))
    with pytest.raises(ValueError, match="larger"):
        models.fetch(tmp_path, item, False)
