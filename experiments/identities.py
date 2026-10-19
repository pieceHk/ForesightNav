"""Read identity-only metadata for the retained experiment profiles."""

import hashlib
import json
from pathlib import Path


def load_frozen_identities(path, dataset_manifest_sha256, ordered_subtasks=3):
    """Load identity-only frozen metadata; never read policy outcomes."""

    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("dataset", {}).get("dataset_manifest_sha256")
        != dataset_manifest_sha256
    ):
        return {}
    if ordered_subtasks != 3:
        raise ValueError("SCSP frozen identities require three subtasks")
    identities = {}
    for row in payload.get("episodes", ()):
        identity = row.get("identity", {})
        key = (
            str(identity.get("scene_id", "")),
            str(identity.get("runtime_episode_id", "")),
        )
        if not all(key) or key in identities:
            raise ValueError("SCSP frozen manifest has an invalid identity key")
        source_ids = tuple(
            str(value) for value in identity.get("source_episode_ids", ())
        )
        categories = tuple(
            str(value) for value in identity.get("ordered_categories", ())
        )
        if len(source_ids) != 3 or len(categories) != 3:
            raise ValueError("SCSP frozen identity must contain three subtasks")
        identities[key] = {
            "scene_id": key[0],
            "runtime_episode_id": key[1],
            "source_episode_ids": source_ids,
            "ordered_categories": categories,
        }
    if len(identities) != 48:
        raise ValueError("SCSP frozen manifest must contain 48 identities")
    return identities


def load_identity_sidecar(path, dataset_manifest_sha256, ordered_subtasks=3):
    """Load optional identity-only smoke metadata without unrelated fields."""

    if not path:
        return {}, None
    source = Path(path).expanduser().resolve()
    raw = source.read_bytes()

    def reject_constant(value):
        raise ValueError(f"non-strict JSON constant: {value}")

    payload = json.loads(raw.decode("utf-8"), parse_constant=reject_constant)
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "dataset_manifest_sha256",
        "episodes",
    }:
        raise ValueError("SCSP identity sidecar contains non-identity fields")
    if payload.get("schema_version") != "scsp_identity_sidecar_v1":
        raise ValueError("SCSP identity sidecar schema mismatch")
    if (
        not isinstance(payload.get("dataset_manifest_sha256"), str)
        or payload["dataset_manifest_sha256"] != dataset_manifest_sha256
    ):
        raise ValueError("SCSP identity sidecar dataset hash mismatch")
    identities = {}
    rows = payload.get("episodes")
    if not isinstance(rows, list):
        raise ValueError("SCSP identity sidecar episodes must be a list")
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "scene_id",
            "runtime_episode_id",
            "source_episode_ids",
            "ordered_categories",
        }:
            raise ValueError("SCSP identity sidecar contains non-identity fields")
        if any(
            not isinstance(row[name], str) or not row[name].strip()
            for name in ("scene_id", "runtime_episode_id")
        ):
            raise ValueError("SCSP identity sidecar IDs must be non-empty strings")
        if not isinstance(row["source_episode_ids"], list) or not isinstance(
            row["ordered_categories"], list
        ):
            raise ValueError("SCSP identity sidecar identity values must be lists")
        if any(
            not isinstance(value, str) or not value.strip()
            for name in ("source_episode_ids", "ordered_categories")
            for value in row[name]
        ):
            raise ValueError(
                "SCSP identity sidecar identity values must be non-empty strings"
            )
        key = (row["scene_id"], row["runtime_episode_id"])
        source_ids = tuple(row["source_episode_ids"])
        categories = tuple(row["ordered_categories"])
        if (
            not all(key)
            or key in identities
            or len(source_ids) != ordered_subtasks
            or len(categories) != ordered_subtasks
            or not all(source_ids)
            or not all(categories)
        ):
            raise ValueError("SCSP identity sidecar has an invalid episode")
        identities[key] = {
            "scene_id": key[0],
            "runtime_episode_id": key[1],
            "source_episode_ids": source_ids,
            "ordered_categories": categories,
        }
    if not identities:
        raise ValueError("SCSP identity sidecar must contain at least one episode")
    return identities, hashlib.sha256(raw).hexdigest()
