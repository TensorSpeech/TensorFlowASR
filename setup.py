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
    "tensorflow>=2.2.0",
    "tensorflow-addons>=0.9.1",
    "setuptools>=40.0.0",
    "librosa>=0.7.0",
    "soundfile>=0.10.2",
    "PyYAML>=3.12",
    "tqdm>=4.26.1",
    "matplotlib",
    "numpy",
    "sox",
    "nltk",
    "ctc-decoders",
    "warprnnt-tensorflow",
    "semetrics"
]

setuptools.setup(
    name="tiramisu-asr",
    version="0.0.1",
    author="Huy Le Nguyen",
    author_email="nlhuy.cs.16@gmail.com",
    description="ASR using Tensorflow keras",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/usimarit/vnasr",
    packages=setuptools.find_packages(include=["tiramisu_asr*"]),
    package_data={
        "tiramisu_asr": ["configs/*.yml", "datasets/*.txt"]
    },
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX :: Linux",
        "License :: OSI Approved :: Apache-2.0 License",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ],
    python_requires='>=3.6',
)
