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

sample_rate = 16000

stride = 0.5

noise_conf = {
    "snr": (-1, 2.5, 7.5, 12.5, 17.5),
    "max_noises": 3
}

noises_dir = "/mnt/Data/ML/ASR/Preprocessed/Noises"

g_learning_rate = 0.0002

d_learning_rate = 0.0002

clean_train_data_dir = "/mnt/Data/ML/ASR/Preprocessed/VivosClean"

clean_test_data_dir = "/mnt/Data/ML/ASR/Preprocessed/VLSP/test_transcripts.tsv"

checkpoint_dir = "/mnt/Projects/asrk16/trained/segan/ckpts/"

log_dir = "/mnt/Projects/asrk16/trained/segan/logs/"
