# Copyright 2020 Huy Le Nguyen (@usimarit) and Huy Phan (@puochuy)
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

import numpy as np


class GradPolicy(object):
    """ Gradient policy for one network branch """

    def __init__(self,
                 train_size: int = 0,
                 valid_size: int = 0,
                 smooth_win_size: int = 50,
                 hist_size: int = 100,
                 policy_name: str = 'simple'):
        """
        train_size: size of the training (sub)set used for gradient policing
        valid_size: size of the validation set used for gradient policing
        smooth_win_size: window size for loss smoothing
        hist_size: how many previous loss values we should use for line fitting
        policy_name: name of the used policy
        ("simple", "fully_adaptive", "fully_adaptive_slope")
        """

        self.train_size = train_size
        self.valid_size = valid_size
        self.smooth_win_size = smooth_win_size
        self.hist_size = hist_size
        self.policy_name = policy_name
        self.train_loss = np.array([])
        self.valid_loss = np.array([])
        self.smoothed_train_loss = np.array([])
        self.smoothed_valid_loss = np.array([])
        self.prev_best_train_loss = None
        self.prev_best_valid_loss = None
        self.train_slope_ref = None
        self.valid_slope_ref = None
        self.train_loss_ref = None
        self.valid_loss_ref = None

        self.G = np.array([])
        self.O = np.array([])

    def get_O(self):
        return self.O

    def get_G(self):
        return self.G

    def update_losses(self, train_loss, valid_loss):
        self.train_loss = np.append(self.train_loss, [train_loss])
        self.valid_loss = np.append(self.valid_loss, [valid_loss])
        self._moving_average_smoothing()

    def _moving_average_smoothing(self):
        N = len(self.train_loss)
        smoothed_val = np.mean(self.train_loss[np.max([0, (N - self.smooth_win_size)]):])
        self.smoothed_train_loss = np.append(self.smoothed_train_loss, [smoothed_val])

        smoothed_val = np.mean(self.valid_loss[np.max([0, (N - self.smooth_win_size)]):])
        self.smoothed_valid_loss = np.append(self.smoothed_valid_loss, [smoothed_val])

    def _compute_weight_simple(self):
        # the losses at training step 0 are references
        if (self.train_loss_ref is None and self.valid_loss_ref is None):
            self.train_loss_ref = self.train_loss[0]
            self.valid_loss_ref = self.valid_loss[0]

        cur_train_loss = self.train_loss[-1]
        cur_valid_loss = self.valid_loss[-1]

        # overfitting rate
        Ok = (self.train_loss_ref - cur_train_loss) / self.train_size - \
             (self.valid_loss_ref - cur_valid_loss) / self.valid_size
        self.O = np.append(self.O, [Ok])

        # generalization rate
        Gk = (self.valid_loss_ref - cur_valid_loss) / self.valid_size
        self.G = np.append(self.G, [Gk])

        # generalization to overfitting ratio
        w = Gk / (Ok * Ok + 1e-6)
        if (w < 0.):
            w = 0.

        return w

    def _compute_weight_fully_adaptive(self):
        # the losses at training step 0 are references
        if (self.train_loss_ref is None and self.valid_loss_ref is None):
            self.train_loss_ref = self.smoothed_train_loss[0]
            self.valid_loss_ref = self.smoothed_valid_loss[0]

        cur_train_loss = self.smoothed_train_loss[-1]
        cur_valid_loss = self.smoothed_valid_loss[-1]

        # overfitting rate
        Ok = (self.train_loss_ref - cur_train_loss) / self.train_size - \
             (self.valid_loss_ref - cur_valid_loss) / self.valid_size
        self.O = np.append(self.O, [Ok])

        # generalization rate
        Gk = (self.valid_loss_ref - cur_valid_loss) / self.valid_size
        self.G = np.append(self.G, [Gk])

        # generalization to overfitting ratio
        w = Gk / (Ok * Ok + 1e-6)
        if (w < 0.):
            w = 0.

        # update references to the best-performance current model snapshot
        if (self.prev_best_valid_loss is None or
                self.prev_best_valid_loss > self.valid_loss[-1]):
            self.train_loss_ref = cur_train_loss
            self.valid_loss_ref = cur_valid_loss
            self.prev_best_valid_loss = self.valid_loss[-1]
            self.prev_best_train_loss = self.train_loss[-1]

        return w

    def _compute_weight_fully_adaptive_slope(self):
        # the losses at training step 0 are references
        N = len(self.smoothed_train_loss)
        if (self.train_slope_ref is None and self.valid_slope_ref is None):
            train_loss = self.smoothed_train_loss[max([0, N - self.hist_size - 1]): -1]
            valid_loss = self.smoothed_valid_loss[max([0, N - self.hist_size - 1]): -1]
            self.train_slope_ref, self.valid_slope_ref = self._line_fit(train_loss, valid_loss)

        cur_train_loss = self.smoothed_train_loss[max([0, N - self.hist_size]):]
        cur_valid_loss = self.smoothed_valid_loss[max([0, N - self.hist_size]):]
        train_slope, valid_slope = self._line_fit(cur_train_loss, cur_valid_loss)

        Ok = (valid_slope - train_slope) - (self.valid_slope_ref - self.train_slope_ref)
        self.O = np.append(self.O, [Ok])

        Gk = valid_slope - self.valid_slope_ref
        self.G = np.append(self.G, [Gk])

        w = Gk / (Ok * Ok + 1e-6)
        if (w < 0.):
            w = 0.

        # update references
        if (self.prev_best_valid_loss is None or
                self.prev_best_valid_loss > self.valid_loss[-1]):
            self.train_slope_ref = train_slope
            self.valid_slope_ref = valid_slope
            self.prev_best_valid_loss = self.valid_loss[-1]
            self.prev_best_train_loss = self.train_loss[-1]

        return w

    def _line_fit(self, train_loss, valid_loss):
        size = len(train_loss)
        train_val = train_loss[np.max([0, (size - self.hist_size)]):]
        t = np.arange(np.min([len(train_val), self.hist_size]))
        p_train = np.polyfit(t, train_val, 1)
        assert (len(p_train) == 2)
        p_train = p_train[0]

        size = len(valid_loss)
        valid_val = valid_loss[np.max([0, (size - self.hist_size)]):]
        t = np.arange(np.min([len(valid_val), self.hist_size]))
        p_valid = np.polyfit(t, valid_val, 1)
        assert (len(p_valid) == 2)
        p_valid = p_valid[0]

        return p_train, p_valid

    def compute_weight(self):
        if self.policy_name == "simple":
            w = self._compute_weight_simple()
        elif self.policy_name == "fully_adaptive":
            w = self._compute_weight_fully_adaptive()
        elif self.policy_name == "fully_adaptive_slope":
            w = self._compute_weight_fully_adaptive_slope()
        else:
            raise ValueError(
                "policy_name must be either 'simple', "
                "'fully_adaptive' or 'fully_adaptive_slope'")
        return w

