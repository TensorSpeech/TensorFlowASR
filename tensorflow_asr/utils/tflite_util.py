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

"""
Read and write a JSON blob inside a `.tflite` flatbuffer.

A TFLite model's root table has a `metadata` field: a list of `(name, buffer index)` pairs
pointing at arbitrary bytes in the model's own buffer list. The interpreter never reads them, so
anything stored there rides along inside the single file without touching the graph, the
signatures or inference. Every export already carries two such entries, `min_runtime_version` and
`CONVERSION_METADATA`, written by TensorFlow during conversion.

`app_util.convert_tflite` uses this to record what a streaming client needs to drive the model --
chunk geometry, sample rate, blank id, beam width -- so a deployed `.tflite` needs neither a
sidecar file nor the Python model that produced it. See `BaseModel.get_tflite_metadata`.

Reading it back is not part of `tf.lite.Interpreter`'s API, which exposes only the signature
methods, so a non-Python client writes the equivalent of `read_metadata` against the same schema.
"""

import json

import flatbuffers
import numpy as np
from tensorflow.lite.python import schema_py_generated as schema

from tensorflow_asr import tf
from tensorflow_asr.utils import file_util

METADATA_NAME = "TFASR_METADATA"


def _entry_name(entry) -> str:
    """Entry names come back as `bytes` when unpacked and stay `str` when we set them."""
    return entry.name.decode() if isinstance(entry.name, bytes) else entry.name


def write_metadata(tflite_model: bytes, metadata: dict) -> bytes:
    """
    Return `tflite_model` with `metadata` stored under `METADATA_NAME`, replacing any prior copy.

    The flatbuffer is immutable, so this unpacks the whole model and rebuilds it. Weights survive
    that as numpy arrays rather than Python lists -- `BufferT` takes the `DataAsNumpy`/
    `CreateNumpyVector` path on both sides -- so the cost is one bulk copy and roughly twice the
    model in peak memory, not the per-byte blowup the unpack suggests.

    Replacing leaves the superseded buffer orphaned in the buffer list. Dropping it would mean
    reindexing every other `buffer` reference in the model to no real end, since the blob is a few
    dozen bytes.

    `file_identifier` is not optional. Finishing without it produces bytes that look like a model
    and that every interpreter refuses to load.
    """
    model = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(bytearray(tflite_model), 0))

    blob = schema.BufferT()
    blob.data = np.frombuffer(json.dumps(metadata).encode(), dtype=np.uint8)
    model.buffers.append(blob)

    entry = schema.MetadataT()
    entry.name = METADATA_NAME
    entry.buffer = len(model.buffers) - 1
    model.metadata = [each for each in (model.metadata or []) if _entry_name(each) != METADATA_NAME] + [entry]

    builder = flatbuffers.Builder(0)
    builder.Finish(model.Pack(builder), file_identifier=b"TFL3")
    return bytes(builder.Output())


def read_metadata(tflite_model) -> dict:
    """
    Read the blob back, as a dict. Empty when the model carries none.

    Parameters
    ----------
    tflite_model : bytes or str
        The flatbuffer itself, or a path to it.
    """
    if isinstance(tflite_model, str):
        with tf.io.gfile.GFile(file_util.preprocess_paths(tflite_model), "rb") as tflite_file:
            tflite_model = tflite_file.read()

    model = schema.Model.GetRootAsModel(bytearray(tflite_model), 0)
    for index in range(model.MetadataLength()):
        entry = model.Metadata(index)
        if entry.Name().decode() == METADATA_NAME:
            return json.loads(bytes(model.Buffers(entry.Buffer()).DataAsNumpy()))
    return {}
