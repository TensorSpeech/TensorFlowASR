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

import os
import random
from typing import List, Union

import numpy as np
import tensorflow as tf
from packaging import version

logger = tf.get_logger()

KERAS_SRC = "keras.src" if version.parse(tf.version.VERSION) >= version.parse("2.13.0") else "keras"

LENGTH_AS_OUTPUT = os.environ.get("LENGTH_AS_OUTPUT", "False").lower() == "true"


def setup_devices(
    devices: List[int] = None,
    cpu: bool = False,
):
    """
    Setting visible devices

    Parameters
    ----------
    devices : List[int], optional
        List of visible devices' indices, by default None
    cpu : bool, optional
        Use cpu or not, by default False
    """
    if cpu:
        cpus = tf.config.list_physical_devices("CPU")
        tf.config.set_visible_devices(cpus, "CPU")
        tf.config.set_visible_devices([], "GPU")
        logger.info(f"Run on {cpus}")
        return tf.config.list_logical_devices("CPU")
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        if devices is not None:
            gpus = [gpus[i] for i in devices]
            tf.config.set_visible_devices(gpus, "GPU")
    logger.info(f"Run on {gpus}")
    return tf.config.list_logical_devices("GPU")


def setup_tpu(
    tpu_address=None,
):
    if tpu_address is None:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    else:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="grpc://" + tpu_address)
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    return tf.distribute.TPUStrategy(resolver)


def setup_strategy(
    devices: List[int],
    tpu_address: str = None,
):
    """
    Setting mirrored strategy for training

    Parameters
    ----------
    devices : List[int]
        List of visible devices' indices
    tpu_address : str, optional
        An optional custom tpu address, by default None

    Returns
    -------
    f.distribute.Strategy
        TPUStrategy for training on tpus or MirroredStrategy for training on gpus
    """
    try:
        return setup_tpu(tpu_address)
    except (ValueError, tf.errors.NotFoundError) as e:
        logger.warning(e)
    available_devices = setup_devices(devices)
    if len(available_devices) == 1:
        return tf.distribute.get_strategy()
    return tf.distribute.MirroredStrategy(devices=[d.name for d in available_devices])


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
        tf.keras.mixed_precision.set_global_policy(policy)
        logger.info(f"USING mixed precision policy {policy}")
    elif mxp == "strict_auto":
        policy = "mixed_bfloat16" if has_devices("TPU") else "mixed_float16"
        tf.keras.mixed_precision.set_global_policy(policy)
        tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
        logger.info(f"USING auto mixed precision policy {policy}")
    elif mxp == "auto":
        tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
        logger.info("USING auto mixed precision policy")


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
    tf.keras.backend.experimental.enable_tf_random_generator()
    tf.keras.utils.set_random_seed(seed)
