"""Reject foreign native-module origins and incompatible diagnostic modes."""

from pathlib import Path

import pytest

from scripts import check_install


def test_native_origin_check_resolves_symlinks_before_accepting_module(tmp_path):
    prefix = tmp_path / "environment"
    prefix.mkdir()
    external = tmp_path / "foreign.so"
    external.write_bytes(b"module")
    linked = prefix / "native.so"
    linked.symlink_to(external)
    with pytest.raises(ValueError, match="outside"):
        check_install.require_prefix(linked, prefix)
    local = prefix / "local.so"
    local.write_bytes(b"module")
    assert check_install.require_prefix(local, prefix) == str(local)


@pytest.mark.parametrize("args", [["--cpu-only", "--smoke"], ["--episode", "0"]])
def test_invalid_modes_never_start_installation_or_gpu_work(monkeypatch, args):
    monkeypatch.setattr(check_install, "check", lambda _: pytest.fail("started a diagnostic"))
    with pytest.raises(SystemExit) as error:
        check_install.main(args)
    assert error.value.code == 2
