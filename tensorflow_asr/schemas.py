# Copyright 2023 Huy Le Nguyen (@nglehuy)
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

import typing

import tensorflow as tf


class TrainInput(typing.NamedTuple):
    inputs: tf.Tensor
    inputs_length: tf.Tensor
    predictions: tf.Tensor
    predictions_length: tf.Tensor


class TrainOutput(typing.NamedTuple):
    logits: tf.Tensor
    logits_length: tf.Tensor


class TrainLabel(typing.NamedTuple):
    labels: tf.Tensor
    labels_length: tf.Tensor


class TrainData(typing.NamedTuple):
    inputs: TrainInput
    labels: TrainLabel


class PredictInput(typing.NamedTuple):
    inputs: tf.Tensor
    inputs_length: tf.Tensor
    previous_tokens: typing.Optional[tf.Tensor] = None
    previous_encoder_states: typing.Optional[tf.Tensor] = None
    previous_decoder_states: typing.Optional[tf.Tensor] = None


class PredictOutput(typing.NamedTuple):
    tokens: tf.Tensor
    next_tokens: tf.Tensor
    next_encoder_states: typing.Optional[tf.Tensor] = None
    next_decoder_states: typing.Optional[tf.Tensor] = None


class PredictOutputWithTranscript(typing.NamedTuple):
    transcript: tf.Tensor
    tokens: tf.Tensor
    next_tokens: tf.Tensor
    next_encoder_states: typing.Optional[tf.Tensor] = None
    next_decoder_states: typing.Optional[tf.Tensor] = None
