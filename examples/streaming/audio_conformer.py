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

import sys
import argparse
import soundfile as sf
import time
import sounddevice as sd
from multiprocessing import Process, Queue, Event

import numpy as np
import tensorflow as tf


def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text


parser = argparse.ArgumentParser(prog="Conformer audio file streaming")

parser.add_argument(
    '-l', '--list-devices', action='store_true',
    help='show list of audio devices and exit')
args, remaining = parser.parse_known_args()
if args.list_devices:
    print(sd.query_devices())
    parser.exit(0)

parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    parents=[parser])

parser.add_argument('filename', metavar='FILENAME',
                    help='audio file to be played back')

parser.add_argument('-d', '--device', type=int_or_str,
                    help='output device (numeric ID or substring)')

parser.add_argument('-b', '--blocksize', type=int, default=2048,
                    help='block size (default: %(default)s)')

parser.add_argument('-q', '--buffersize', type=int, default=20,
                    help='number of blocks used for buffering (default: %(default)s)')

parser.add_argument("--tflite", type=str, default=None,
                    help="Path to conformer tflite")

args = parser.parse_args(remaining)

if args.blocksize == 0:
    parser.error('blocksize must not be zero')
if args.buffersize < 1:
    parser.error('buffersize must be at least 1')

q = Queue(maxsize=args.buffersize)
Q = Queue()
E = Event()


def recognizer():
    tflitemodel = tf.lite.Interpreter(model_path=args.tflite)

    input_details = tflitemodel.get_input_details()
    output_details = tflitemodel.get_output_details()

    tflitemodel.resize_tensor_input(input_details[0]["index"], [args.blocksize])
    tflitemodel.allocate_tensors()

    def recognize(signal, lastid, states):
        tflitemodel.set_tensor(input_details[0]["index"], signal)
        tflitemodel.set_tensor(input_details[1]["index"], lastid)
        tflitemodel.set_tensor(input_details[2]["index"], states)
        tflitemodel.invoke()
        upoints, lastid, states = tflitemodel.get_tensor(output_details[0]["index"])
        text = "".join([chr(u) for u in upoints])
        return text, lastid, states

    lastid = 0
    states = np.zeros(shape=[1, 1, 1, 320], dtype=np.float32)
    transcript = ""

    while True:
        try:
            data = Q.get()
            text, lastid, states = recognize(data, lastid, states)
            transcript += text
            print(transcript)
        except Exception:
            pass


tflite_process = Process(target=recognizer)
tflite_process.start()


def callback(outdata, frames, time, status):
    assert frames == args.blocksize
    if status.output_underflow:
        print('Output underflow: increase blocksize?', file=sys.stderr)
        raise sd.CallbackAbort
    assert not status
    try:
        data = q.get_nowait()
        Q.put(data)
    except Queue.Empty as e:
        print('Buffer is empty: increase buffersize?', file=sys.stderr)
        raise sd.CallbackAbort from e
    if len(data) < len(outdata):
        outdata[:len(data)] = data
        outdata[len(data):] = b'\x00' * (len(outdata) - len(data))
        raise sd.CallbackStop
    else:
        outdata[:] = data


try:
    with sf.SoundFile(args.audio) as f:
        for _ in range(args.buffersize):
            data = f.buffer_read(args.blocksize, dtype='float32')
            if not data:
                break
            q.put_nowait(data)  # Pre-fill queue
        stream = sd.RawOutputStream(
            samplerate=f.samplerate, blocksize=args.blocksize,
            device=args.device, channels=f.channels, dtype='float32',
            callback=callback, finished_callback=E.set)
        with stream:
            timeout = args.blocksize * args.buffersize / f.samplerate
            while data:
                data = f.buffer_read(args.blocksize, dtype='float32')
                q.put(data, timeout=timeout)
            E.wait()

except KeyboardInterrupt:
    parser.exit('\nInterrupted by user')
except Queue.Full:
    # A timeout occurred, i.e. there was an error in the callback
    parser.exit(1)
except Exception as e:
    parser.exit(type(e).__name__ + ': ' + str(e))
