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

import argparse
from tqdm.auto import tqdm

parser = argparse.ArgumentParser(prog="Create vocabulary file from transcripts")

parser.add_argument("--output", type=str, default=None, help="The output .txt vocabulary file path")

parser.add_argument("transcripts", nargs="+", type=str, default=None, help="Transcript .tsv files")

args = parser.parse_args()

assert args.output and args.transcripts

lines = []
for trans in args.transcripts:
    with open(trans, "r", encoding="utf-8") as t:
        lines.extend(t.read().splitlines()[1:])

vocab = {}
for line in tqdm(lines, desc="[Processing]"):
    line = line.split("\t")[-1]
    for c in line:
        vocab[c] = 1

with open(args.output, "w", encoding="utf-8") as out:
    for key in vocab.keys():
        out.write(f"{key}\n")
