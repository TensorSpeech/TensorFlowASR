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
from tiramisu_asr.gradpolicy.multiview_grad_policy import MultiviewGradPolicy

gp = MultiviewGradPolicy(num_branches=2,
                         train_size=100,
                         valid_size=80,
                         policy_name='simple')

gp.update_losses([1.5, 1.5], [1.2, 1.2]) # first loss values as references
gp.update_losses([1.2, 1.1], [1.1, 1.0]) # first loss values as references
w = gp.compute_weights()
print(w)
gp.update_losses([1.1, 0.9], [1.1, 0.9]) # first loss values as references
w = gp.compute_weights()
print(w)
gp.update_losses([0.9, 0.8], [1.1, 1.0]) # first loss values as references
w = gp.compute_weights()
print(w)
gp.update_losses([1.3, 0.5], [1.1, 1.3]) # first loss values as references
w = gp.compute_weights()
print(w)
gp.update_losses([0.8, 0.5], [1.4, 1.3]) # first loss values as references
w = gp.compute_weights()
print(w)

gp = MultiviewGradPolicy(num_branches=2,
                         train_size=100,
                         valid_size=80,
                         smooth_win_size=2,
                         policy_name='fully_adaptive')

gp.update_losses([1.5, 1.5], [1.2, 1.2]) # first loss values as references
gp.update_losses([1.2, 1.1], [1.1, 1.0]) # first loss values as references
w = gp.compute_weights()
print(w)
gp.update_losses([1.1, 0.9], [1.1, 0.9]) # first loss values as references
w = gp.compute_weights()
print(w)
gp.update_losses([0.9, 0.8], [1.1, 1.0]) # first loss values as references
w = gp.compute_weights()
print(w)
gp.update_losses([1.3, 0.5], [1.1, 1.3]) # first loss values as references
w = gp.compute_weights()
print(w)
gp.update_losses([0.8, 0.5], [1.4, 1.3]) # first loss values as references
w = gp.compute_weights()
print(w)


gp = MultiviewGradPolicy(num_branches=2,
                         train_size=100,
                         valid_size=80,
                         smooth_win_size=50,
                         hist_size=50,
                         policy_name='fully_adaptive_slope')

for i in range(100):
    train_loss1 = np.random.random_sample()
    train_loss2 = np.random.random_sample()
    valid_loss1 = np.random.random_sample()
    valid_loss2 = np.random.random_sample()
    gp.update_losses(train_loss = [train_loss1, train_loss2],
                     valid_loss = [valid_loss1, valid_loss2])
    if(i >= 50):
        w = gp.compute_weights()
        print(w)
