import os
import sys
import tempfile
import types

import pytest

from tensorflow_asr.callbacks import KaggleModelBackupAndRestore, upload_kaggle_model


@pytest.fixture
def fake_kagglehub(monkeypatch):
    """
    Stand in for the `kagglehub` package so the upload path can be exercised without credentials or
    the network. Records every `model_upload` call so the test can assert what would have been sent.
    """
    module = types.ModuleType("kagglehub")
    module.uploads = []
    module.model_upload = lambda **kwargs: module.uploads.append(kwargs)
    monkeypatch.setitem(sys.modules, "kagglehub", module)
    return module


def test_upload_kaggle_model_is_a_noop_without_a_handle(fake_kagglehub, tmp_path):
    """No handle means nothing to upload to -- and it must not even reach `kagglehub`."""
    upload_kaggle_model(str(tmp_path), None, notes="x")
    upload_kaggle_model(str(tmp_path), "", notes="x")
    assert fake_kagglehub.uploads == []


def test_upload_kaggle_model_sends_the_dir_and_always_ignores_dsstore(fake_kagglehub, tmp_path):
    (tmp_path / "kenlm.weights.h5").write_text("weights")
    upload_kaggle_model(str(tmp_path), "owner/lm/keras/kenlm", notes="built", ignore_patterns=["*.ids.txt"])

    assert len(fake_kagglehub.uploads) == 1
    call = fake_kagglehub.uploads[0]
    assert call["handle"] == "owner/lm/keras/kenlm"
    assert os.path.realpath(call["local_model_dir"]) == os.path.realpath(str(tmp_path))
    assert call["version_notes"] == "built"
    # the caller's patterns are kept, and .DS_Store is added whether or not they asked
    assert ".DS_Store" in call["ignore_patterns"] and "*.ids.txt" in call["ignore_patterns"]


def test_backup_kaggle_uploads_through_the_same_helper(fake_kagglehub, tmp_path):
    """The per-epoch callback path and the one-shot path must be the same upload."""
    (tmp_path / "model.h5").write_text("m")
    callback = KaggleModelBackupAndRestore(model_dir=str(tmp_path), model_handle="owner/lm/keras/external", save_freq=1)
    callback._backup_kaggle(logs={}, notes="epoch 1")  # pylint: disable=protected-access
    assert len(fake_kagglehub.uploads) == 1 and fake_kagglehub.uploads[0]["handle"] == "owner/lm/keras/external"


def test_backup_kaggle_skips_a_nonfinite_loss(fake_kagglehub, tmp_path):
    """A NaN/Inf epoch must not be uploaded over a good one -- the guard predates this refactor."""
    (tmp_path / "model.h5").write_text("m")
    callback = KaggleModelBackupAndRestore(model_dir=str(tmp_path), model_handle="owner/lm/keras/external", save_freq=1)
    callback._backup_kaggle(logs={"loss": float("nan")}, notes="bad")  # pylint: disable=protected-access
    assert fake_kagglehub.uploads == [], "a non-finite loss must never reach model_upload"


def test_kaggle_model_backup_and_restore():
    model_handle = os.getenv("TEST_MODEL_HANDLE")
    if not model_handle:
        return
    with tempfile.TemporaryDirectory() as temp_dir:
        os.environ["KAGGLEHUB_CACHE"] = os.path.join(temp_dir, "cache")
        os.makedirs(os.environ["KAGGLEHUB_CACHE"], exist_ok=True)
        model_dir = os.path.join(temp_dir, "model")
        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, "model.h5"), "w", encoding="utf-8") as f:
            f.write("dummy model data")
        callback = KaggleModelBackupAndRestore(
            model_handle=model_handle,
            model_dir=model_dir,
            save_freq=1,
        )
        callback._backup_kaggle(logs={}, notes="Backed up model at batch")
