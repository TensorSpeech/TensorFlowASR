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

from .grad_policy import GradPolicy


class MultiviewGradPolicy(object):
    """ Gradient policy for a multi-view network with multiple branches """

    def __init__(self,
                 num_branches: int = 1,
                 train_size: int = 0,
                 valid_size: int = 0,
                 smooth_win_size: int = 50,
                 hist_size: int = 100,
                 policy_name: str = 'simple'):
        """
        num_branches: number of network branches in the multiview network
        train_size: size of the training (sub)set used for gradient policing
        valid_size: size of the validation set used for gradient policing
        smooth_win_size: window size for loss smoothing
        hist_size: how many previous loss values we should use for line fitting
        policy_name: name of the used policy
        ("simple", "fully_adaptive", "fully_adaptive_slope")
        """

        self.num_branches = num_branches
        self.agents = list()
        self.train_size = train_size
        for i in range(self.num_branches):
            agent = GradPolicy(train_size=train_size,
                               valid_size=valid_size,
                               smooth_win_size=smooth_win_size,
                               hist_size=hist_size,
                               policy_name=policy_name)
            self.agents.append(agent)

    def update_losses(self, train_loss: list, valid_loss: list):
        assert len(train_loss) == len(valid_loss), \
            "the list of training and validation losses must be equal size"
        assert len(train_loss) == self.num_branches, \
            "the loss list size must be equal to the number of branches"
        for i in range(self.num_branches):
            self.agents[i].update_losses(train_loss=train_loss[i], valid_loss=valid_loss[i])

    def compute_weights(self) -> list:
        w = list()
        w_sum = 0.
        for i in range(self.num_branches):
            wi = self.agents[i].compute_weight()
            w.append(wi)
            w_sum += wi
        for i in range(self.num_branches):
            w[i] = 1. / self.num_branches if w_sum == 0. else w[i] / w_sum
        return w
