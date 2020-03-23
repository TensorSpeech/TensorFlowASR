from __future__ import absolute_import

from models.DeepSpeech2 import DeepSpeech2RowConv

base_model = DeepSpeech2RowConv()

decoder = 'beamsearch'

beam_width = 128

batch_size = 24

num_epochs = 20

vocabulary_file_path = "/app/data/vocabulary.txt"

learning_rate = 0.0006

min_lr = 0.0

sample_rate = 16000

frame_ms = 20

stride_ms = 10

num_feature_bins = 128
