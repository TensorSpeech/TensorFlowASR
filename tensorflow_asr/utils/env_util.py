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

import warnings
import tensorflow as tf


def setup_environment():  # Set memory growth and only log ERRORs
    """ Setting tensorflow running environment """
    warnings.simplefilter("ignore")
    tf.get_logger().setLevel("ERROR")


def setup_devices(devices, cpu=False):
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


def setup_strategy(devices):
    """Setting mirrored strategy for training

    Args:
        devices (list): list of visible devices' indices

    Returns:
        tf.distribute.Strategy: MirroredStrategy for training one or multiple gpus
    """
    setup_devices(devices)
    if has_tpu():
        return setup_tpu()
    return tf.distribute.MirroredStrategy()


def setup_tpu(tpu_address=None):
    if tpu_address is None:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    else:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + tpu_address)
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    print("All TPUs: ", tf.config.list_logical_devices('TPU'))
    return tf.distribute.experimental.TPUStrategy(resolver)


def has_gpu_or_tpu():
    gpus = tf.config.list_logical_devices("GPU")
    tpus = tf.config.list_logical_devices("TPU")
    if len(gpus) == 0 and len(tpus) == 0: return False
    return True


def has_tpu():
    tpus = tf.config.list_logical_devices("TPU")
    if len(tpus) == 0: return False
    return True
