# Copyright 2020 M. Yusuf Sarıgöz (@monatis)
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

import argparse
import os

import librosa
import tqdm
import tensorflow as tf

# example usage: python create_mls_trans.py -dataset-home /mnt/datasets/mls --language polish --opus

base_url = "https://dl.fbaipublicfiles.com/mls/"

langs = [
    "dutch",
    "english",
    "german",
    "french",
    "italian",
    "portuguese",
    "polish",
    "spanish"
]

splits = [
    "dev",
    "test",
    "train"
]

chars = set()

def prepare_split(dataset_dir, split, opus=False):
    # Setup necessary paths
    split_home = os.path.join(dataset_dir, split)
    transcripts_infile = os.path.join(split_home, 'transcripts.txt')
    transcripts_outfile  = os.path.join(split_home, 'transcripts_tfasr.tsv')
    audio_home = os.path.join(split_home, "audio")
    extension = ".opus" if opus else ".flac"
    transcripts = []

    # Make paths absolute, get durations and read chars to form alphabet later on
    with open(transcripts_infile, 'r', encoding='utf8') as infile:
        for line in tqdm.tqdm(infile.readlines(), desc=f"Reading from {transcripts_infile}..."):
            file_id, transcript = line.strip().split('\t')
            speaker_id, book_id, _ = file_id.split('_')
            audio_path = os.path.join(audio_home, speaker_id, book_id, f"{file_id}{extension}")
            y, sr = librosa.load(audio_path, sr=None)
            duration = librosa.get_duration(y, sr)
            transcripts.append(f"{audio_path}\t{duration:2f}\t{transcript}\n")
            for char in transcript:
                chars.add(char)

    # Write transcripts to file
    with open(transcripts_outfile, 'w', encoding='utf8') as outfile:
        outfile.write("PATH\tDURATION\tTRANSCRIPT\n")
        for t in tqdm.tqdm(transcripts, desc=f"Writing to {transcripts_outfile}"):
            outfile.write(t)


def make_alphabet_file(filepath, chars_list, lang):
    print(f"Writing alphabet to {filepath}...")
    with open(filepath, 'w', encoding='utf8') as outfile:
        outfile.write(f"# Alphabet file for language {lang}\n")
        outfile.write("Automatically generated. Do not edit\n#\n")
        for char in sorted(list(chars_list)):
            outfile.write(f"{char}\n")

        outfile.write("# end of file")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Download and prepare MLS dataset in a given language")
    ap.add_argument("--dataset-home", "-d", help="Path to home directory to download and prepare dataset. Default to ~/.keras", default=None, required=False)
    ap.add_argument("--language", "-l", type=str, choices=langs, help="Any name of language included in MLS", default=None, required=True)
    ap.add_argument("--opus", help="Whether to use dataset in opus format or not", default=False, action='store_true')
    
    args = ap.parse_args()
    fname = "mls_{}{}.tar.gz".format(args.language, "_opus" if args.opus else "")
    subdir = fname[:-7]
    dataset_home = os.path.abspath(args.dataset_home)
    dataset_dir = os.path.join(dataset_home, subdir)
    full_url = base_url + fname

    downloaded_file = tf.keras.utils.get_file(
        fname,
        full_url,
        cache_subdir=dataset_home,
        extract=True
        )

    print(f"Dataset extracted to {dataset_dir}. Preparing...")

    for split in splits:
        prepare_split(dataset_dir=dataset_dir, split=split, opus=args.opus)

    make_alphabet_file(os.path.join(dataset_dir, "alphabet.txt"), chars, args.language)