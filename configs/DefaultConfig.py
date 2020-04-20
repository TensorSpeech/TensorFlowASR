from __future__ import absolute_import

from models.deepspeech2.DeepSpeech2 import DeepSpeech2
from augmentations.Augments import TimeWarping, TimeMasking, \
  FreqMasking

base_model = DeepSpeech2(num_conv=2, num_rnn=3, rnn_units=256, is_bidirectional=True, kernel_size=(31, 11))

streaming_size = None

decoder = {
  "name": "beamsearch",
  "beam_width": 500,
  "lm_path": "/mnt/Data/ML/NLP/vntc_5gram_probing.binary",
  "alpha": 1.0,
  "beta": 0.5
}

augmentations = [
  #TimeMasking(num_time_mask=1, time_mask_param=30, p_upperbound=0.2),
  #FreqMasking(num_freq_mask=1, freq_mask_param=10),
  #TimeWarping(time_warp_param=40, direction="right")
]

batch_size = 16

num_epochs = 10

vocabulary_file_path = "/mnt/Projects/asrk16/code/data/vocabulary.txt"

learning_rate = 0.001

min_lr = 0.0

sample_rate = 16384

frame_ms = 20

stride_ms = 10

num_feature_bins = 86

feature_type = "mfcc"

train_data_transcript_paths = [
  "/mnt/Data/ML/ASR/Preprocessed/SmallFixed/Train/transcripts.tsv"
]

eval_data_transcript_paths = [
  "/mnt/Data/ML/ASR/Preprocessed/SmallFixed/Dev/transcripts.tsv"
]

test_data_transcript_paths = [
  "/mnt/Data/ML/ASR/Preprocessed/SmallFixed/Test/transcripts.tsv"
]

checkpoint_dir = "/mnt/Projects/asrk16/trained/med-bilstm/checkpoints/"

log_dir = "/mnt/Projects/asrk16/trained/med-bilstm/tensorboard/"
