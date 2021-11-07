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

from setuptools import find_packages, setup
from typing import List
from collections import defaultdict


def parse_requirements(lines: List[str]):
    extras_requires = defaultdict(list)
    extra = "requires"
    for line in lines:
        line = line.strip()
        if line.startswith("# extra="):
            extra = line.split("=")[1].strip()
            continue
        if line and line[0] != "#":
            lib_package = line.split("#")[0].strip()  # split comments
            extras_requires[extra].append(lib_package)
    install_requires = extras_requires.pop("requires")
    return install_requires, extras_requires


with open("requirements.txt", "r", encoding="utf-8") as fr:
    install_requires, extras_requires = parse_requirements(fr.readlines())

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setup(
    name="TensorFlowASR",
    version="1.0.3",
    author="Huy Le Nguyen",
    author_email="nlhuy.cs.16@gmail.com",
    description="Almost State-of-the-art Automatic Speech Recognition using Tensorflow 2",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TensorSpeech/TensorFlowASR",
    packages=find_packages(include=("tensorflow_asr", "tensorflow_asr.*")),
    install_requires=install_requires,
    extras_require=extras_requires,
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX :: Linux",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.6, <4",
)
