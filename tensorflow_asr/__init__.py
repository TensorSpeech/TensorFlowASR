# pylint: disable=protected-access
import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = os.environ.get("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import tensorflow as tf

# might cause performance penalty if ops fallback to cpu, see https://cloud.google.com/tpu/docs/tensorflow-ops
tf.config.set_soft_device_placement(True)

logger = tf.get_logger()
logger.setLevel(os.environ.get("LOG_LEVEL", "info").upper())
logger.propagate = False
warnings.simplefilter("ignore")

from keras.layers import Layer
from keras.utils import generic_utils, tf_utils


@property
def output_shape(self):
    if self._tfasr_output_shape is None:
        raise AttributeError(f"The layer {self.name} has never been called and thus has no defined output shape.")
    return self._tfasr_output_shape


@tf.__internal__.tracking.no_automatic_dependency_tracking
@generic_utils.default
def build(self, input_shape):
    self._tfasr_output_shape = tf_utils.convert_shapes(self.compute_output_shape(input_shape), to_tuples=True)
    self._build_input_shape = input_shape
    self.built = True


def compute_output_shape(self, input_shape):
    return input_shape


# monkey patch
Layer.output_shape = output_shape
Layer.build = build
Layer.compute_output_shape = compute_output_shape

from tensorflow_asr.models import *
from tensorflow_asr.optimizers import *
