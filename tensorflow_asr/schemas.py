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

import collections


def TrainInput(inputs, inputs_length, predictions, predictions_length, caching=None):
    outputs = {
        "inputs": inputs,
        "inputs_length": inputs_length,
        "predictions": predictions,
        "predictions_length": predictions_length,
    }
    if caching is not None:
        outputs["caching"] = caching
    return outputs


def TrainOutput(logits, logits_length, caching=None):
    outputs = {
        "logits": logits,
        "logits_length": logits_length,
    }
    if caching is not None:
        outputs["caching"] = caching
    return outputs


def TrainLabel(labels, labels_length):
    return {
        "labels": labels,
        "labels_length": labels_length,
    }


PredictInput = collections.namedtuple(
    "PredictInput",
    ("inputs", "inputs_length", "previous_tokens", "previous_encoder_states", "previous_decoder_states"),
    defaults=(None, None, None),
)
PredictOutput = collections.namedtuple(
    "PredictOutput",
    ("tokens", "next_tokens", "next_encoder_states", "next_decoder_states"),
    defaults=(None, None),
)
PredictOutputWithTranscript = collections.namedtuple(
    "PredictOutputWithTranscript",
    ("transcript", "tokens", "next_tokens", "next_encoder_states", "next_decoder_states"),
    defaults=(None, None),
)
