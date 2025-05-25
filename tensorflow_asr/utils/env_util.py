# Copyright 2020 Huy Le Nguyen (@nglehuy)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import random
import sys
import warnings
from datetime import datetime, timezone
from typing import List, Union

TF_LOG_LEVEL = os.getenv("TF_LOG_LEVEL", "warning").upper()
TF_SOFT_PLACEMENT = os.getenv("TF_SOFT_PLACEMENT", "false").lower() == "true"
TF_ENABLE_CHECK_NUMERIC = os.getenv("TF_ENABLE_CHECK_NUMERIC", "false").lower() == "true"
TF_CUDNN = os.getenv("TF_CUDNN", "auto").lower()
TF_CUDNN = "auto" if TF_CUDNN == "auto" else TF_CUDNN == "true"
DEBUG = TF_LOG_LEVEL == "DEBUG"


def _logging_format_time(self, record, datefmt=None):
    return datetime.fromtimestamp(record.created, timezone.utc).astimezone().isoformat(sep="T", timespec="milliseconds")


logging.basicConfig(level=logging.INFO, format=logging.BASIC_FORMAT, stream=sys.stdout, force=True)
logging.Formatter.formatTime = _logging_format_time
logging.captureWarnings(True)
warnings.filterwarnings("ignore")

import keras
import numpy as np
import tensorflow as tf
from packaging import version
from tensorflow.python.util import deprecation  # pylint: disable = no-name-in-module

tf.get_logger().setLevel(TF_LOG_LEVEL)
deprecation._PRINT_DEPRECATION_WARNINGS = False  # comment this line to print deprecation warnings
if TF_ENABLE_CHECK_NUMERIC:
    tf.debugging.enable_check_numerics()


KERAS_SRC = "keras.src" if version.parse(tf.version.VERSION) >= version.parse("2.13.0") else "keras"

logger = logging.getLogger(__name__)


def setup_gpu(
    devices: List[int] = None,
):
    logger.info(f"Using TF_CUDNN={TF_CUDNN}, TF_SOFT_PLACEMENT={TF_SOFT_PLACEMENT}")
    tf.config.set_soft_device_placement(DEBUG or TF_SOFT_PLACEMENT)
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise RuntimeError("No GPUs found!")
    if devices is not None:
        gpus = [gpus[i] for i in devices]
    tf.config.set_visible_devices(gpus, "GPU")
    logger.info("Run on GPU")
    logger.info(f"All devices: {gpus}")
    return tf.distribute.MirroredStrategy()


def setup_tpu(
    tpu_address=None,
    tpu_vm: bool = False,
):
    # might cause performance penalty if ops fallback to cpu, see https://cloud.google.com/tpu/docs/tensorflow-ops
    tf.config.set_soft_device_placement(DEBUG)
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu_address)
    if not tpu_vm:
        tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    logger.info(f"Run on TPU {tpu_address}")
    logger.info(f"All devices: {tf.config.list_logical_devices('TPU')}")
    return tf.distribute.TPUStrategy(resolver)


def setup_strategy(
    device_type: str,
    devices: List[int] = None,
    tpu_address: str = None,
    tpu_vm: bool = False,
):
    if device_type.lower() == "tpu":
        return setup_tpu(tpu_address, tpu_vm)
    if device_type.lower() == "gpu":
        return setup_gpu(devices)
    return tf.distribute.get_strategy()


def has_devices(
    devices: Union[List[str], str],
):
    if isinstance(devices, list):
        return all((len(tf.config.list_logical_devices(d)) > 0 for d in devices))
    return len(tf.config.list_logical_devices(devices)) > 0


def setup_mxp(
    mxp: str = "strict",
):
    """
    Setup mixed precision

    Parameters
    ----------
    mxp : str, optional
        Either "strict", "auto" or "none", by default "strict"

    Raises
    ------
    ValueError
        Wrong value for mxp
    """
    options = ["strict", "strict_auto", "auto", "none"]
    if mxp not in options:
        raise ValueError(f"mxp must be in {options}")
    if mxp == "strict":
        policy = "mixed_bfloat16" if has_devices("TPU") else "mixed_float16"
        keras.mixed_precision.set_global_policy(policy)
        tf.config.optimizer.set_experimental_options({"auto_mixed_precision": False})
        logger.info(f"USING mixed precision policy {policy}")
    elif mxp == "strict_auto":
        policy = "mixed_bfloat16" if has_devices("TPU") else "mixed_float16"
        keras.mixed_precision.set_global_policy(policy)
        tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
        logger.info(f"USING auto mixed precision policy {policy}")
    elif mxp == "auto":
        tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
        logger.info("USING auto mixed precision policy")
    else:
        keras.mixed_precision.set_global_policy("float32")
        tf.config.optimizer.set_experimental_options({"auto_mixed_precision": False})
        logger.info("USING float32 precision policy")


def setup_seed(
    seed: int = 42,
):
    """
    The seed is given an integer value to ensure that the results of pseudo-random generation are reproducible
    Why 42?
    "It was a joke. It had to be a number, an ordinary, smallish number, and I chose that one.
    I sat at my desk, stared into the garden and thought 42 will do!"
    - Douglas Adams's popular 1979 science-fiction novel The Hitchhiker's Guide to the Galaxy

    Parameters
    ----------
    seed : int, optional
        Random seed, by default 42
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    keras.utils.set_random_seed(seed)
