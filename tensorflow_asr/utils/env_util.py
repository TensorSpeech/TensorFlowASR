# Copyright 2020 Huy Le Nguyen (@usimarit)
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

os.environ["TF_CPP_MIN_LOG_LEVEL"] = os.environ.get("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import random
import warnings
from typing import List, Union

import numpy as np
import tensorflow as tf

logger = tf.get_logger()
logger.setLevel(os.environ.get("LOG_LEVEL", "info").upper())
logger.propagate = False


def setup_environment():
    """Setting tensorflow running environment"""
    warnings.simplefilter("ignore")
    return tf.get_logger()


def setup_devices(
    devices: List[int] = None,
    cpu: bool = False,
):
    """Setting visible devices

    Args:
        devices (list): list of visible devices' indices
        cpu (bool): use cpu or not
    """
    if cpu:
        cpus = tf.config.list_physical_devices("CPU")
        tf.config.set_visible_devices(cpus, "CPU")
        tf.config.set_visible_devices([], "GPU")
        logger.info(f"Run on {len(cpus)} Physical CPUs")
    else:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            if devices is not None:
                gpus = [gpus[i] for i in devices]
                tf.config.set_visible_devices(gpus, "GPU")
        logger.info(f"Run on {len(gpus)} Physical GPUs")


def setup_tpu(
    tpu_address=None,
):
    if tpu_address is None:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    else:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="grpc://" + tpu_address)
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    logger.info(f"All TPUs: {tf.config.list_logical_devices('TPU')}")
    return tf.distribute.TPUStrategy(resolver)


def setup_strategy(
    devices: List[int],
    tpu_address: str = None,
):
    """Setting mirrored strategy for training

    Args:
        devices (list): list of visible devices' indices
        tpu_address (str): an optional custom tpu address

    Returns:
        tf.distribute.Strategy: TPUStrategy for training on tpus or MirroredStrategy for training on gpus
    """
    try:
        return setup_tpu(tpu_address)
    except (ValueError, tf.errors.NotFoundError) as e:
        logger.warning(e)
    setup_devices(devices)
    return tf.distribute.MirroredStrategy()


def has_devices(
    devices: Union[List[str], str],
):
    if isinstance(devices, list):
        return all((len(tf.config.list_logical_devices(d)) > 0 for d in devices))
    return len(tf.config.list_logical_devices(devices)) > 0


def setup_mxp(
    mxp=True,
):
    if not mxp:
        return
    policy = "mixed_bfloat16" if has_devices("TPU") else "mixed_float16"
    tf.keras.mixed_precision.set_global_policy(policy)
    logger.info(f"USING mixed precision policy {policy}")


def setup_seed(
    seed: int = 42,
):
    """
    The seed is given an integer value to ensure that the results of pseudo-random generation are reproducible
    Why 42?
    "It was a joke. It had to be a number, an ordinary, smallish number, and I chose that one.
    I sat at my desk, stared into the garden and thought 42 will do!"
    - Douglas Adams's popular 1979 science-fiction novel The Hitchhiker's Guide to the Galaxy

    Args:
        seed (int, optional): integer. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.backend.experimental.enable_tf_random_generator()
    tf.keras.utils.set_random_seed(seed)
