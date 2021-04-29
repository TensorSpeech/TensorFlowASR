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

from typing import Union, List
import warnings
import tensorflow as tf

logger = tf.get_logger()


def setup_environment():  # Set memory growth and only log ERRORs
    """ Setting tensorflow running environment """
    warnings.simplefilter("ignore")
    logger.setLevel("WARN")


def setup_devices(devices: List[int], cpu: bool = False):
    """Setting visible devices

    Args:
        devices (list): list of visible devices' indices
    """
    if cpu:
        cpus = tf.config.list_physical_devices("CPU")
        tf.config.set_visible_devices(cpus, "CPU")
    else:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            visible_gpus = [gpus[i] for i in devices]
            tf.config.set_visible_devices(visible_gpus, "GPU")
            print("Run on", len(visible_gpus), "Physical GPUs")


def setup_strategy(devices: List[int], tpu_address: str = None):
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
        logger.warn(e)
        pass
    setup_devices(devices)
    return tf.distribute.MirroredStrategy()


def setup_tpu(tpu_address=None):
    if tpu_address is None:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    else:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="grpc://" + tpu_address)
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    print("All TPUs: ", tf.config.list_logical_devices("TPU"))
    return tf.distribute.experimental.TPUStrategy(resolver)


def has_devices(devices: Union[List[str], str]):
    if isinstance(devices, list):
        return all([len(tf.config.list_logical_devices(d)) != 0 for d in devices])
    return len(tf.config.list_logical_devices(devices)) != 0
