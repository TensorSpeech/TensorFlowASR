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
import abc
import tempfile
import tensorflow as tf

from ..utils.utils import is_cloud_path, is_hdf5_filepath


class Model(tf.keras.Model):
    def __init__(self, name, **kwargs):
        super(Model, self).__init__(name=name, **kwargs)

    def save(self, filepath, overwrite=True, include_optimizer=True, save_format=None,
             signatures=None, options=None, save_traces=True):
        if is_cloud_path(filepath) and is_hdf5_filepath(filepath):
            _, ext = os.path.splitext(filepath)
            with tempfile.NamedTemporaryFile(suffix=ext) as tmp:
                super(Model, self).save(
                    tmp.name, overwrite=overwrite, include_optimizer=include_optimizer,
                    save_format=save_format, signatures=signatures, options=options, save_traces=save_traces
                )
                tf.io.gfile.copy(tmp.name, filepath, overwrite=True)
        else:
            super(Model, self).save(
                filepath, overwrite=overwrite, include_optimizer=include_optimizer,
                save_format=save_format, signatures=signatures, options=options, save_traces=save_traces
            )

    def save_weights(self, filepath, overwrite=True, save_format=None, options=None):
        if is_cloud_path(filepath) and is_hdf5_filepath(filepath):
            _, ext = os.path.splitext(filepath)
            with tempfile.NamedTemporaryFile(suffix=ext) as tmp:
                super(Model, self).save_weights(tmp.name, overwrite=overwrite, save_format=save_format, options=options)
                tf.io.gfile.copy(tmp.name, filepath, overwrite=True)
        else:
            super(Model, self).save_weights(filepath, overwrite=overwrite, save_format=save_format, options=options)

    def load_weights(self, filepath, by_name=False, skip_mismatch=False, options=None):
        if is_cloud_path(filepath) and is_hdf5_filepath(filepath):
            _, ext = os.path.splitext(filepath)
            with tempfile.NamedTemporaryFile(suffix=ext) as tmp:
                tf.io.gfile.copy(filepath, tmp.name, overwrite=True)
                super(Model, self).load_weights(tmp.name, by_name=by_name, skip_mismatch=skip_mismatch, options=options)
        else:
            super(Model, self).load_weights(filepath, by_name=by_name, skip_mismatch=skip_mismatch, options=options)

    @abc.abstractmethod
    def _build(self, *args, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def call(self, inputs, training=False, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def recognize(self, features, input_lengths, **kwargs):
        pass

    @abc.abstractmethod
    def recognize_beam(self, features, input_lengths, **kwargs):
        pass
