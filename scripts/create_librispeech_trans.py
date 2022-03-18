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

import glob
import os
import unicodedata

import librosa
from tqdm.auto import tqdm

from tensorflow_asr.utils import cli_util, file_util


def main(
    directory: str,
    output: str,
):
    directory = file_util.preprocess_paths(directory, isdir=True)
    output = file_util.preprocess_paths(output)

    transcripts = []

    text_files = glob.glob(os.path.join(directory, "**", "*.txt"), recursive=True)

    for text_file in tqdm(text_files, desc="[Loading]"):
        current_dir = os.path.dirname(text_file)
        with open(text_file, "r", encoding="utf-8") as txt:
            lines = txt.read().splitlines()
        for line in lines:
            line = line.split(" ", maxsplit=1)
            audio_file = os.path.join(current_dir, line[0] + ".flac")
            y, sr = librosa.load(audio_file, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
            text = unicodedata.normalize("NFKC", line[1])
            transcripts.append(f"{audio_file}\t{duration}\t{text.lower()}\n")

    with open(output, "w", encoding="utf-8") as out:
        out.write("PATH\tDURATION\tTRANSCRIPT\n")
        for line in tqdm(transcripts, desc="[Writing]"):
            out.write(line)


if __name__ == "__main__":
    cli_util.run(main)
