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
import tempfile
from typing import Union, List
import tensorflow as tf


def is_hdf5_filepath(filepath):
    return (filepath.endswith('.h5') or filepath.endswith('.keras') or filepath.endswith('.hdf5'))


def is_cloud_path(path):
    """ Check if the path is on cloud (which requires tf.io.gfile)

    Args:
        path (str): Path to directory or file

    Returns:
        bool: True if path is on cloud, False otherwise
    """
    return bool(re.match(r"^[a-z]+://", path))


def preprocess_paths(paths: Union[List, str]):
    """Expand the path to the root "/"

    Args:
        paths (Union[List, str]): A path or list of paths

    Returns:
        Union[List, str]: A processed path or list of paths, return None if it's not path
    """
    if isinstance(paths, list):
        return [path if is_cloud_path(path) else os.path.abspath(os.path.expanduser(path)) for path in paths]
    elif isinstance(paths, str):
        return paths if is_cloud_path(paths) else os.path.abspath(os.path.expanduser(paths))
    else:
        return None


def read_bytes(path: str) -> tf.Tensor:
    with tf.io.gfile.GFile(path, "rb") as f:
        content = f.read()
    return tf.convert_to_tensor(content, dtype=tf.string)


def save_file(filepath):
    if is_cloud_path(filepath) and is_hdf5_filepath(filepath):
        _, ext = os.path.splitext(filepath)
        with tempfile.NamedTemporaryFile(suffix=ext) as tmp:
            yield tmp.name
            tf.io.gfile.copy(tmp.name, filepath, overwrite=True)
    else:
        yield filepath


def read_file(filepath):
    if is_cloud_path(filepath) and is_hdf5_filepath(filepath):
        _, ext = os.path.splitext(filepath)
        with tempfile.NamedTemporaryFile(suffix=ext) as tmp:
            tf.io.gfile.copy(filepath, tmp.name, overwrite=True)
            yield tmp.name
    else:
        yield filepath
