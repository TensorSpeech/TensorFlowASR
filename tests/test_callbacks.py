import os
import tempfile

from tensorflow_asr.callbacks import KaggleModelBackupAndRestore


def test_kaggle_model_backup_and_restore():
    model_handle = os.getenv("TEST_MODEL_HANDLE")
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
