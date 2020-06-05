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
from __future__ import absolute_import

import yaml
from collections import UserDict
from utils.utils import preprocess_paths


def load_yaml(path):
    with open(preprocess_paths(path), "r", encoding="utf-8") as file:
        return yaml.load(file, Loader=yaml.FullLoader)


def fill_missing(default: dict, custom: dict, level: int = 0):
    if level > 1:  # Only fill default value up to level 1 from 0 of config dict
        return custom
    for key, value in default.items():
        if not custom.get(key, None):
            custom[key] = value
        elif isinstance(value, dict):
            custom[key] = fill_missing(value, custom[key], level + 1)
    return custom


class UserConfig(UserDict):
    """ User config class for training, testing or infering """

    def __init__(self, default: str, custom: str, learning: bool = True):
        assert default, "Default dict for config must be set"
        default = load_yaml(default)
        custom = fill_missing(default, load_yaml(custom))
        super(UserConfig, self).__init__(custom)
        if not learning:  # No need to have learning_config on Inferencer
            del self.data["learning_config"]

    def __missing__(self, key):
        return None
