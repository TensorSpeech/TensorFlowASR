import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = os.environ.get("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import tensorflow as tf

logger = tf.get_logger()
logger.setLevel(os.environ.get("LOG_LEVEL", "info").upper())
logger.propagate = False
warnings.simplefilter("ignore")

from tensorflow_asr.models import *
from tensorflow_asr.optimizers import *
