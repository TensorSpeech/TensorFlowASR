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
Transcribe an audio file a block at a time with an exported `.tflite`, through `ASRInference`.

The file stands in for a live source: the loop below only ever sees the next block, never the whole
signal, so swapping `blocks()` for a microphone callback or a socket read changes nothing else.
That is the point of the example -- `ASRInference` buffers whatever arrives, decodes each whole
chunk as soon as it is complete, and carries the encoder and decoder state across calls, so the
caller never has to know the chunk geometry.

Block size is deliberately unrelated to that geometry. Feed 100 samples or 100000; the cache holds
back what does not yet fill a chunk. The default is one `signal_chunk_step`, which makes each call
after the first complete exactly one chunk and so prints one chunk of transcript at a time.

Everything it needs comes out of the file itself. `tensorflow_asr tflite` records the sample rate,
the chunk geometry and the blank id in the flatbuffer's metadata, so there is no config to load and
no tokenizer to build -- see `tensorflow_asr/utils/tflite_util.py`.

The export must be traced at batch size 1 (`tensorflow_asr tflite --bs=1`), since one file is one
signal. A larger one refuses the call with a message saying so.

For a whole-utterance decode instead, see `examples/inferences/tflite.py`, or pass
`streaming=False` here. For a microphone in place of the file, `live_streaming_tflite.py`.
"""

import logging

import numpy as np

from tensorflow_asr.inferences import ASRInference
from tensorflow_asr.utils import cli_util, data_util

logger = logging.getLogger(__name__)


def main(
    audio_file_path: str,
    tflite: str,
    blocksize: int = None,
):
    """
    Parameters
    ----------
    audio_file_path : str
        Audio to transcribe. Resampled to the rate recorded in the model's metadata.
    tflite : str
        An export from `tensorflow_asr tflite`, traced at `--bs=1`.
    blocksize : int, optional
        Samples handed over per call, standing in for however much a live source would deliver at
        once. Defaults to one `signal_chunk_step`, so each call after the first decodes one chunk.
    """
    asr = ASRInference(tflite=tflite)
    logger.info(f"Model metadata: {asr.metadata}")

    signal = np.asarray(data_util.read_raw_audio(data_util.load_and_convert_to_wav(audio_file_path, sample_rate=asr.metadata["sample_rate"])))
    blocksize = blocksize or asr.metadata["signal_chunk_step"]

    def blocks():
        """The live source, faked. A microphone callback or a socket read would go here instead."""
        for start in range(0, len(signal), blocksize):
            yield signal[start : start + blocksize]

    # `start()` opens the session, clearing whatever a previous one left behind.
    asr.start()
    transcript = ""
    try:
        for block in blocks():
            # Each call returns only what this block completed -- empty until a chunk is full -- so
            # the caller appends. Row 0 because a flat block is a batch of one.
            piece = asr(block)[0]
            # Only the new text is printed, never the whole transcript again: it grows past the
            # width of a terminal, and rewriting the line would then wrap instead of overwrite.
            print(piece, end="", flush=True)
            transcript += piece
    except KeyboardInterrupt:
        logger.info("Interrupted; flushing what has been received so far")
    # `end()` zero-pads the last partial chunk up to a whole one, decodes it, and closes the
    # session. Without it the tail of the audio is never transcribed.
    transcript += asr.end()[0]

    print()
    logger.info(f"Transcript: {transcript}")


if __name__ == "__main__":
    cli_util.run(main)
