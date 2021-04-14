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

import os
import re
import yaml
import tempfile
import contextlib
from typing import Union, List
import tensorflow as tf


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
    with open(path, "r", encoding="utf-8") as file:
        return yaml.load(file, Loader=loader)


def is_hdf5_filepath(filepath: str) -> bool:
    return (filepath.endswith('.h5') or filepath.endswith('.keras') or filepath.endswith('.hdf5'))


def is_cloud_path(path: str) -> bool:
    """ Check if the path is on cloud (which requires tf.io.gfile)

    Args:
        path (str): Path to directory or file

    Returns:
        bool: True if path is on cloud, False otherwise
    """
    return bool(re.match(r"^[a-z]+://", path))


def preprocess_paths(paths: Union[List[str], str], isdir: bool = False) -> Union[List[str], str]:
    """ Expand the path to the root "/" and makedirs

    Args:
        paths (Union[List, str]): A path or list of paths

    Returns:
        Union[List, str]: A processed path or list of paths, return None if it's not path
    """
    if isinstance(paths, list):
        paths = [path if is_cloud_path(path) else os.path.abspath(os.path.expanduser(path)) for path in paths]
        for path in paths:
            dirpath = path if isdir else os.path.dirname(path)
            if not tf.io.gfile.exists(dirpath): tf.io.gfile.makedirs(dirpath)
        return paths
    if isinstance(paths, str):
        paths = paths if is_cloud_path(paths) else os.path.abspath(os.path.expanduser(paths))
        dirpath = paths if isdir else os.path.dirname(paths)
        if not tf.io.gfile.exists(dirpath): tf.io.gfile.makedirs(dirpath)
        return paths
    return None


@contextlib.contextmanager
def save_file(filepath: str):
    if is_cloud_path(filepath) and is_hdf5_filepath(filepath):
        _, ext = os.path.splitext(filepath)
        with tempfile.NamedTemporaryFile(suffix=ext) as tmp:
            yield tmp.name
            tf.io.gfile.copy(tmp.name, filepath, overwrite=True)
    else:
        yield filepath


@contextlib.contextmanager
def read_file(filepath: str):
    if is_cloud_path(filepath) and is_hdf5_filepath(filepath):
        _, ext = os.path.splitext(filepath)
        with tempfile.NamedTemporaryFile(suffix=ext) as tmp:
            tf.io.gfile.copy(filepath, tmp.name, overwrite=True)
            yield tmp.name
    else:
        yield filepath
