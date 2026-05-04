import os

import pytest
from fastapi import HTTPException


def _import_validator():
    from routes.inference import _validate_native_mmproj_companion
    return _validate_native_mmproj_companion


def test_validate_mmproj_noop_when_mmproj_path_is_none(tmp_path):
    validator = _import_validator()
    gguf = tmp_path / "x.gguf"
    gguf.write_bytes(b"GGUF" + b"\x00" * 16)
    validator(None, str(gguf))


def test_validate_mmproj_noop_when_gguf_path_is_none(tmp_path):
    validator = _import_validator()
    mmproj = tmp_path / "mmproj.gguf"
    mmproj.write_bytes(b"GGUF" + b"\x00" * 16)
    validator(str(mmproj), None)


def test_validate_mmproj_noop_when_both_none():
    validator = _import_validator()
    validator(None, None)


def test_validate_mmproj_accepts_regular_sibling(tmp_path):
    validator = _import_validator()
    gguf = tmp_path / "model.gguf"
    mmproj = tmp_path / "mmproj-model.gguf"
    gguf.write_bytes(b"GGUF" + b"\x00" * 16)
    mmproj.write_bytes(b"GGUF" + b"\x00" * 16)
    validator(str(mmproj), str(gguf))


def test_validate_mmproj_rejects_symlinked_mmproj(tmp_path):
    if os.name == "nt":
        pytest.skip("symlink semantics differ on Windows without privileges")
    validator = _import_validator()
    gguf = tmp_path / "model.gguf"
    real_mmproj = tmp_path / "real-mmproj.gguf"
    gguf.write_bytes(b"GGUF" + b"\x00" * 16)
    real_mmproj.write_bytes(b"GGUF" + b"\x00" * 16)
    symlinked = tmp_path / "mmproj-link.gguf"
    symlinked.symlink_to(real_mmproj)
    with pytest.raises(HTTPException) as exc:
        validator(str(symlinked), str(gguf))
    assert exc.value.status_code == 400
    assert "regular file" in exc.value.detail


def test_validate_mmproj_rejects_directory_mmproj(tmp_path):
    validator = _import_validator()
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"GGUF" + b"\x00" * 16)
    bogus = tmp_path / "mmproj-as-dir"
    bogus.mkdir()
    with pytest.raises(HTTPException) as exc:
        validator(str(bogus), str(gguf))
    assert exc.value.status_code == 400


def test_validate_mmproj_rejects_missing_mmproj(tmp_path):
    validator = _import_validator()
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"GGUF" + b"\x00" * 16)
    missing = tmp_path / "nope-mmproj.gguf"
    with pytest.raises(HTTPException) as exc:
        validator(str(missing), str(gguf))
    assert exc.value.status_code == 400


def test_validate_mmproj_rejects_sibling_in_different_directory(tmp_path):
    validator = _import_validator()
    here = tmp_path / "model_dir"
    elsewhere = tmp_path / "other_dir"
    here.mkdir()
    elsewhere.mkdir()
    gguf = here / "model.gguf"
    mmproj = elsewhere / "mmproj-model.gguf"
    gguf.write_bytes(b"GGUF" + b"\x00" * 16)
    mmproj.write_bytes(b"GGUF" + b"\x00" * 16)
    with pytest.raises(HTTPException) as exc:
        validator(str(mmproj), str(gguf))
    assert exc.value.status_code == 400
    assert "next to" in exc.value.detail


def test_validate_mmproj_rejects_when_gguf_missing(tmp_path):
    validator = _import_validator()
    mmproj = tmp_path / "mmproj-model.gguf"
    mmproj.write_bytes(b"GGUF" + b"\x00" * 16)
    missing_gguf = tmp_path / "missing.gguf"
    with pytest.raises(HTTPException) as exc:
        validator(str(mmproj), str(missing_gguf))
    assert exc.value.status_code == 400


def test_validate_mmproj_accepts_when_both_under_resolved_symlinked_parent(tmp_path):
    if os.name == "nt":
        pytest.skip("symlink semantics differ on Windows without privileges")
    validator = _import_validator()
    real_dir = tmp_path / "real_models"
    real_dir.mkdir()
    gguf = real_dir / "model.gguf"
    mmproj = real_dir / "mmproj-model.gguf"
    gguf.write_bytes(b"GGUF" + b"\x00" * 16)
    mmproj.write_bytes(b"GGUF" + b"\x00" * 16)
    linked_dir = tmp_path / "linked_models"
    linked_dir.symlink_to(real_dir)
    validator(
        str(linked_dir / "mmproj-model.gguf"),
        str(linked_dir / "model.gguf"),
    )
