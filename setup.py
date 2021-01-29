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

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

requirements = [
    "tensorflow-datasets>=3.2.1,<4.0.0",
    "tensorflow-addons>=0.10.0",
    "setuptools>=47.1.1",
    "librosa>=0.8.0",
    "soundfile>=0.10.3",
    "PyYAML>=5.3.1",
    "matplotlib>=3.2.1",
    "sox>=1.4.1",
    "tqdm>=4.54.1",
    "colorama>=0.4.4",
    "nlpaug>=1.1.1",
    "nltk>=3.5",
    "sentencepiece>=0.1.94"
]

setuptools.setup(
    name="TensorFlowASR",
    version="0.7.0",
    author="Huy Le Nguyen",
    author_email="nlhuy.cs.16@gmail.com",
    description="Almost State-of-the-art Automatic Speech Recognition using Tensorflow 2",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TensorSpeech/TensorFlowASR",
    packages=setuptools.find_packages(include=["tensorflow_asr*"]),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX :: Linux",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ],
    python_requires='>=3.6',
)
