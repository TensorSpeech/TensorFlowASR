import os

os.environ["TFASR_PLOT_DIR"] = os.path.join(os.path.dirname(__file__), "figs")

# Hide accelerators for the whole session, before anything touches a device.
#
# This is not about speed. Keras picks the fused cuDNN-style LSTM kernel whenever a GPU is
# *visible* -- placement is irrelevant, `tf.device("/CPU:0")` does not change it -- and that kernel
# converts to a `CudnnRNNV3` custom op which no TFLite interpreter can resolve:
#
#     RuntimeError: Encountered unresolved custom op: CudnnRNNV3
#
# So on any machine with a GPU (including Apple silicon once `tensorflow-metal` is installed via
# the `apple` extra) every recurrent model would export an unrunnable flatbuffer. Hiding the GPU
# here keeps `tests/test_tflite.py` exporting the portable graph that a deployment actually needs.
# The same applies outside the tests -- see the note on `app_util.convert_tflite`.
#
# Must run at import time: `set_visible_devices` raises once the device context is initialised,
# and conftest is imported before any test module.
import tensorflow as tf  # noqa: E402

tf.config.set_visible_devices([], "GPU")
