# pylint: disable=protected-access
import importlib
import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = os.environ.get("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import tensorflow as tf
import keras
from tensorflow.python.util import deprecation  # pylint: disable = no-name-in-module

# might cause performance penalty if ops fallback to cpu, see https://cloud.google.com/tpu/docs/tensorflow-ops
tf.config.set_soft_device_placement(False)
deprecation._PRINT_DEPRECATION_WARNINGS = False  # comment this line to print deprecation warnings

logger = tf.get_logger()
logger.setLevel(os.environ.get("LOG_LEVEL", "info").upper())
logger.propagate = False
warnings.simplefilter("ignore")

from tensorflow_asr.utils import tf_util
from tensorflow_asr.utils.env_util import KERAS_SRC

compile_utils = importlib.import_module(f"{KERAS_SRC}.engine.compile_utils")


@property
def output_shape(self):
    if not hasattr(self, "_tfasr_output_shape") or self._tfasr_output_shape is None:
        raise AttributeError(f"The layer {self.name} has never been called and thus has no defined output shape.")
    return self._tfasr_output_shape


def build(self, input_shape):
    self._tfasr_output_shape = tf_util.convert_shapes(self.compute_output_shape(input_shape), to_tuples=True)
    self._build_input_shape = input_shape
    self.built = True


def compute_output_shape(self, input_shape):
    return input_shape


def match_dtype_and_rank(y_t, y_p, sw):
    return y_t, y_p, sw


# monkey patch
keras.layers.Layer.output_shape = output_shape
keras.layers.Layer.build = build
keras.layers.Layer.compute_output_shape = compute_output_shape
compile_utils.match_dtype_and_rank = match_dtype_and_rank

import tensorflow_asr.callbacks
from tensorflow_asr.models import *
from tensorflow_asr.optimizers import *
