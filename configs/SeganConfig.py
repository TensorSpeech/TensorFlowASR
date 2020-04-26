from __future__ import absolute_import

batch_size = 16

num_epochs = 86

kwidth = 31

ratio = 2

noise_std = 0.

denoise_epoch = 5

noise_decay = 0.9

noise_std_lbound = 0.

l1_lambda = 100.

pre_emph = 0.95

window_size = 2 ** 14

stride = 0.5

g_learning_rate = 0.0002

d_learning_rate = 0.0002

clean_train_data_dir = "/mnt/Data/ML/SEGAN/clean_trainset_wav_16k"

noisy_train_data_dir = "/mnt/Data/ML/SEGAN/noisy_trainset_wav_16k"

clean_test_data_dir = "/mnt/Data/ML/SEGAN/"

noisy_test_data_dir = "/mnt/Data/ML/SEGAN/"

checkpoint_dir = "/mnt/Projects/asrk16/trained/segan/ckpts/"

log_dir = "/mnt/Projects/asrk16/trained/segan/logs/"
