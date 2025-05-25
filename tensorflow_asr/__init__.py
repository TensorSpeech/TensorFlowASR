# pylint: disable=protected-access
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = os.environ.get("TF_CPP_MIN_LOG_LEVEL") or "3"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "true")

# import submodules to register keras objects
import glob
from os.path import basename, dirname, isdir, isfile, join

import keras
import tensorflow as tf  # for reference

from tensorflow_asr.utils import env_util  # import here fist to apply logging

for fd in glob.glob(join(dirname(__file__), "*")):
    if not isfile(fd) and not isdir(fd):
        continue
    if isfile(fd) and not fd.endswith(".py"):
        continue
    fd = fd if isdir(fd) else fd[:-3]
    fd = basename(fd)
    if fd.startswith("__"):
        continue
    __import__(f"{__name__}.{fd}")


__all__ = ["keras", "tf", "env_util"]
