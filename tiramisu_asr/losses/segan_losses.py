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
import tensorflow as tf


@tf.function
def generator_loss(y_true, y_pred, l1_lambda, d_fake_logit):
    """ Reduce mean will be performed in distributed training """
    l1_loss = l1_lambda * tf.abs(tf.subtract(y_pred, y_true))
    g_adv_loss = tf.math.squared_difference(d_fake_logit, 1.)
    return l1_loss, g_adv_loss


@tf.function
def discriminator_loss(d_real_logit, d_fake_logit):
    """ Reduce mean will be performed in distributed training """
    real_loss = tf.math.squared_difference(d_real_logit, 1.)
    fake_loss = tf.math.squared_difference(d_fake_logit, 0.)
    return real_loss + fake_loss
