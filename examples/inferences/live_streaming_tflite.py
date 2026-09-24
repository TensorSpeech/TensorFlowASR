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
Transcribe the microphone live with an exported `.tflite`, through `ASRInference`.

The same loop as `examples/inferences/streaming_tflite.py`, with a real source in place of a file
read in blocks: audio arrives a block at a time, `ASRInference` buffers it, decodes each chunk as
soon as it is complete, and carries the state across calls. Press Ctrl-C to stop -- the last
partial chunk is padded and decoded on the way out, so the tail of what you said is not lost::

    python examples/inferences/live_streaming_tflite.py --tflite /path/to/model.tflite

The recording device is the system default. To choose another, list them and pass an id or a name::

    python -c "import sounddevice; print(sounddevice.query_devices())"
    python examples/inferences/live_streaming_tflite.py --tflite model.tflite --device 2

The sample rate is not a choice: it comes from the model's metadata and the device is opened at
exactly that rate, because feeding a model audio at the wrong rate transcribes nonsense rather
than failing. A device that cannot do it raises instead of silently resampling.

Reading blocks straight off the stream keeps this short, at the cost of dropping audio if decoding
falls behind the microphone. The stream reports that, and the count is logged at the end. If it
happens, either raise `--blocksize` -- bigger blocks mean fewer, larger calls and more latency, but
the same transcript -- or move decoding onto its own thread with a `queue.Queue` between it and an
`sd.InputStream` callback, which buffers instead of dropping.
"""

import logging

import sounddevice as sd

from tensorflow_asr.inferences import ASRInference
from tensorflow_asr.utils import cli_util

logger = logging.getLogger(__name__)


def main(
    tflite: str,
    device=None,
    blocksize: int = None,
):
    """
    Parameters
    ----------
    tflite : str
        An export from `tensorflow_asr tflite`.
    device : int or str, optional
        Recording device, by id or by name. The system default when unset.
    blocksize : int, optional
        Samples read per call, which is also the latency: `blocksize / sample_rate` seconds pass
        before a block can be decoded. Defaults to one `signal_chunk_step`, so each read completes
        exactly one chunk.
    """
    asr = ASRInference(tflite=tflite)
    metadata = asr.engine.metadata
    sample_rate = metadata["sample_rate"]
    blocksize = blocksize or metadata["signal_chunk_step"]
    logger.info(f"Model metadata: {metadata}")
    logger.info(f"Recording at {sample_rate} Hz in blocks of {blocksize} samples ({blocksize / sample_rate:.2f}s of latency)")

    asr.start()
    transcript = ""
    overflows = 0
    try:
        with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32", blocksize=blocksize, device=device) as stream:
            logger.info("Listening -- press Ctrl-C to stop")
            while True:
                # Blocks until the microphone has delivered `blocksize` frames. `overflowed` says
                # the device had to discard input while this thread was busy decoding the last one.
                block, overflowed = stream.read(blocksize)
                overflows += bool(overflowed)
                # [frames, channels] -> the single mono channel. Each call returns only what this
                # block completed, empty until a chunk is full, so the caller appends.
                piece = asr(block[:, 0])
                print(piece, end="", flush=True)
                transcript += piece
    except KeyboardInterrupt:
        pass
    # Whatever is left in the cache is padded up to a whole chunk and decoded, so the last words
    # spoken before Ctrl-C still land in the transcript.
    transcript += asr.end()

    print()
    if overflows:
        logger.warning(f"The microphone dropped input on {overflows} block(s): decoding could not keep up")
    logger.info(f"Transcript: {transcript}")


if __name__ == "__main__":
    cli_util.run(main)
