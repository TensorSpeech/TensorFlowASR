from __future__ import absolute_import

from models.DeepSpeech2 import DeepSpeech2RowConv
from augmentations.Augments import TimeWarping, TimeMasking, \
  FreqMasking

base_model = DeepSpeech2RowConv()

decoder = {
  "name": "beamsearch",
  "beam_width": 500,
  "lm_path": "~/drives/e/ML/NLP/vntc_5gram_probing.binary",
  "alpha": 0.0,
  "beta": 0.0
}

augmentations = [
  TimeMasking(num_time_mask=1, time_mask_param=30, p_upperbound=0.2),
  FreqMasking(num_freq_mask=1, freq_mask_param=10),
  TimeWarping(time_warp_param=40, direction="right")
]

batch_size = 32

num_epochs = 10

vocabulary_file_path = "~/drives/d/asrk16/code/data/vocabulary.txt"

learning_rate = 0.005

min_lr = 0.0

sample_rate = 16000

frame_ms = 20

stride_ms = 10

num_feature_bins = 128

feature_type = "mfcc"

train_data_transcript_paths = [
  "/media/nlhuy/Miscellanea/Datasets/asr/SmallFixed/Train"
  "/transcripts.tsv"
]

eval_data_transcript_paths = [
  "/media/nlhuy/Miscellanea/Datasets/asr/SmallFixed/Dev/transcripts"
  ".tsv"
]

test_data_transcript_paths = [
  "/media/nlhuy/Miscellanea/Datasets/asr/SmallFixed/Test"
  "/transcripts.tsv"
]

checkpoint_dir = "/tmp/asr/checkpoint_dir/"

log_dir = "/tmp/asr/tensorboard/"
