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
Transcribe one audio file with an exported `.tflite`, in a single pass, through `ASRInference`::

    python examples/inferences/tflite.py \\
        --audio-file-path /path/to/audio.wav \\
        --tflite /path/to/model.tflite

`streaming=False` sends the whole signal in one call with fresh state and keeps nothing, which is
exact for every architecture -- offline models included. To feed it a piece at a time instead, see
`streaming_tflite.py` (a file read in blocks) and `live_streaming_tflite.py` (a microphone).

Nothing has to be configured on this side. `tensorflow_asr tflite` records the sample rate, the
blank id and the chunk geometry in the flatbuffer's own metadata, and the transcript is produced
inside the graph, so there is no config to load, no tokenizer to build and no rate to pass in --
one wrong number there would transcribe nonsense rather than fail. See
`tensorflow_asr/utils/tflite_util.py` for what is stored and `tensorflow_asr/inferences.py` for the
interpreter work this hides: locating each tensor in a signature the converter reordered, seeding
the carried state, and decoding the transcript bytes.

Any export batch size works. One file is one session, and the engine under it decodes up to
`--bs` sessions per call, so with a single session the other slots decode silence.
"""

import logging

from tensorflow_asr.inferences import ASRInference
from tensorflow_asr.utils import cli_util, data_util

logger = logging.getLogger(__name__)


def main(
    audio_file_path: str,
    tflite: str,
):
    """
    Parameters
    ----------
    audio_file_path : str
        Audio to transcribe. Any format `librosa` reads; it is resampled to the rate recorded in
        the model's metadata.
    tflite : str
        An export from `tensorflow_asr tflite`.
    """
    asr = ASRInference(tflite=tflite, streaming=False)
    metadata = asr.engine.metadata
    logger.info(f"Model metadata: {metadata}")

    signal = data_util.read_raw_audio(data_util.load_and_convert_to_wav(audio_file_path, sample_rate=metadata["sample_rate"]))

    transcript = asr(signal)
    logger.info(f"Transcript: {transcript}")


if __name__ == "__main__":
    cli_util.run(main)
