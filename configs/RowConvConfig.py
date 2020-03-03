from __future__ import absolute_import

from models.DeepSpeech2 import DeepSpeech2RowConv
from augmentations.Augments import TimeWarping, TimeMasking, \
  FreqMasking

base_model = DeepSpeech2RowConv()

decoder = 'beamsearch'

augmentations = [
  TimeMasking(num_time_mask=1, time_mask_param=30, p_upperbound=0.2),
  FreqMasking(num_freq_mask=1, freq_mask_param=10),
  TimeWarping(time_warp_param=40, direction="right")
]

beam_width = 500

batch_size = 32

num_epochs = 10

vocabulary_file_path = "/mnt/Projects/asrk16/code/data/vocabulary.txt"

learning_rate = 0.005

min_lr = 0.0

sample_rate = 16000

frame_ms = 20

stride_ms = 10

num_feature_bins = 128

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
