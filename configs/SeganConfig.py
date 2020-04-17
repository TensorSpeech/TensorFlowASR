from __future__ import absolute_import

batch_size = 16

num_epochs = 86

kwidth = 31

ratio = 2

noise_std = 0.

l1_lambda = 100.

pre_emph = 0.95

g_learning_rate = 0.002

d_learning_rate = 0.002

train_data_paths = [
  "/mnt/Data/ML/ASR/Preprocessed/SmallFixed/Train/transcripts.tsv"
]

eval_data_paths = [
  "/mnt/Data/ML/ASR/Preprocessed/SmallFixed/Dev/transcripts.tsv"
]

test_data_paths = [
  "/mnt/Data/ML/ASR/Preprocessed/SmallFixed/Test/transcripts.tsv"
]

checkpoint_dir = "/mnt/Projects/asrk16/trained/segan/checkpoints/"

log_dir = "/mnt/Projects/asrk16/trained/segan/logs/"
