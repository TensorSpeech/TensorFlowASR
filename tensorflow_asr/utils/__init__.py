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


def setup_environment():  # Set memory growth and only log ERRORs
    """ Setting tensorflow running environment """
    import os
    import warnings

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
    warnings.simplefilter("ignore")

    import tensorflow as tf

    tf.get_logger().setLevel("ERROR")

    tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})


def setup_devices(devices, cpu=False):
    """Setting visible devices

    Args:
        devices (list): list of visible devices' indices
    """
    import tensorflow as tf

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
    import tensorflow as tf

    setup_devices(devices)

    return tf.distribute.MirroredStrategy()


# def setup_tpu(tpu_address):
#     import tensorflow as tf

#     resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + tpu_address)
#     tf.config.experimental_connect_to_cluster(resolver)
#     tf.tpu.experimental.initialize_tpu_system(resolver)
#     print("All TPUs: ", tf.config.list_logical_devices('TPU'))
#     return resolver
