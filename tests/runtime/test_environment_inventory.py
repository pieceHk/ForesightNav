"""Keep worker environment snapshots portable and distinct from policy code."""

from dataclasses import replace
import json
from pathlib import Path

import pytest

from runtime.environment import record_environment
from runtime.bootstrap import worker_environment
from tests.runtime.test_native_launch import plan


def test_worker_environment_disables_user_site_even_when_inherited(plan):
    assert worker_environment(plan, "navigation", {"PYTHONNOUSERSITE": "0"})["PYTHONNOUSERSITE"] == "1"


def test_environment_record_captures_real_metadata_and_refuses_overwrite(plan, tmp_path):
    directory = tmp_path / "inventory"
    directory.mkdir()
    selected = replace(plan, run_directory=directory)
    path = record_environment(selected, "mapping")
    data = json.loads(path.read_text())
    assert data["schema_version"] == "pbsp_environment_v1"
    assert data["identity_sha256"] == plan.identity_sha256
    assert data["launch_source_commit"] == plan.source_commit
    assert any(item["name"].lower() == "pytest" for item in data["packages"])
    assert all(Path(item["location"]).is_absolute() for item in data["packages"])
    with pytest.raises(FileExistsError):
        record_environment(selected, "mapping")
    assert path.read_text() == json.dumps(data, indent=2) + "\n"
    with pytest.raises(ValueError, match="selected scene"):
        record_environment(selected, "navigation", scene="not-selected")
