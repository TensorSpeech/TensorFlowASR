from __future__ import absolute_import

from models.DeepSpeech2 import DeepSpeech2
from augmentations.Augments import TimeWarping, TimeMasking, FreqMasking

base_model = DeepSpeech2()

decoder = 'beamsearch'

augmentations = [
    TimeMasking(num_time_mask=1, time_mask_param=30, p_upperbound=0.2),
    FreqMasking(num_freq_mask=1, freq_mask_param=10),
    TimeWarping(time_warp_param=20, direction="right")
]

beam_width = 500

batch_size = 24

num_epochs = 20

vocabulary_file_path = "/app/data/vocabulary.txt"

learning_rate = 0.0006

min_lr = 0.0

sample_rate = 16000

frame_ms = 20

stride_ms = 10

num_feature_bins = 128
