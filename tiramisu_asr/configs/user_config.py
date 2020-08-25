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
import re
import yaml
from collections import UserDict

from ..utils.utils import preprocess_paths, append_default_keys_dict, check_key_in_dict


def load_yaml(path):
    # Fix yaml numbers https://stackoverflow.com/a/30462009/11037553
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))
    with open(preprocess_paths(path), "r", encoding="utf-8") as file:
        return yaml.load(file, Loader=loader)


class UserConfig(UserDict):
    """ User config class for training, testing or infering """

    def __init__(self, default: str, custom: str, learning: bool = True):
        assert default, "Default dict for config must be set"
        default = load_yaml(default)
        custom = append_default_keys_dict(default, load_yaml(custom))
        super(UserConfig, self).__init__(custom)
        if not learning and self.data.get("learning_config", None) is not None:
            # No need to have learning_config on Inferencer
            del self.data["learning_config"]
        elif learning:
            # Check keys
            check_key_in_dict(
                self.data["learning_config"],
                ["augmentations", "dataset_config", "optimizer_config", "running_config"]
            )
            check_key_in_dict(
                self.data["learning_config"]["dataset_config"],
                ["train_paths", "eval_paths", "test_paths"]
            )
            check_key_in_dict(
                self.data["learning_config"]["running_config"],
                ["batch_size", "num_epochs", "outdir", "log_interval_steps",
                    "save_interval_steps", "eval_interval_steps"]
            )

    def __missing__(self, key):
        return None
