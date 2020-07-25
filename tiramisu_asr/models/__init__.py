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
import tensorflow as tf


def transfer_weights(source_model: tf.keras.Model,
                     target_model: tf.keras.Model):
    """
    Function to transfer weights from trained model to other one
    Args:
        source_model: trained `tf.keras.Model`
        target_model: target `tf.keras.Model`

    Returns:
        trained target_model
    """
    target_model.set_weights(source_model.get_weights())
    return target_model


def load_from_saved_model(model: tf.keras.Model,
                          saved_path: str):
    """
    Load model from saved model path
    Args:
        model: newly built `tf.keras.Model`
        saved_path: `str` path to saved model

    Returns:
        loaded model
    """
    try:
        saved_model = tf.keras.models.load_model(saved_path)
    except Exception as e:
        raise e

    model = transfer_weights(saved_model, model)
    return model


def load_from_weights(model: tf.keras.Model,
                      saved_path: str):
    """
    Load model from saved weights path
    Args:
        model: newly built `tf.keras.Model`
        saved_path: `str` path to saved weights

    Returns:
        loaded model
    """
    try:
        model.load_weights(saved_path)
    except Exception as e:
        raise e
    return model
