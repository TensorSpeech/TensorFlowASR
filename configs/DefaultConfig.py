from __future__ import absolute_import

from models.deepspeech2.DeepSpeech2 import DeepSpeech2
from augmentations.Augments import Noise

base_model = DeepSpeech2(conv_type=1, rnn_type="gru", num_rnn=7, rnn_units=256,
                         kernel_size=(5, 5, 5), strides=(2, 1, 1), pre_fc_units=0,
                         filters=(32, 32, 96), is_bidirectional=True)

streaming_size = None

decoder = {
  "name": "beamsearch",
  "beam_width": 500,
  "lm_path": "/mnt/Data/ML/NLP/vntc_5gram_probing.binary",
  "alpha": 1.0,
  "beta": 0.5
}

# augmentations = [Noise(min_snr=3, max_snr=10,
#                        min_noises=1, max_noises=3,
#                        noise_dir="/mnt/Data/ML/ASR/Preprocessed/Noises")]
augmentations = []

batch_size = 8

num_epochs = 10

vocabulary_file_path = "/mnt/Projects/asrk16/vnasr/data/vocabulary.txt"

last_activation = 'relu'

speech_conf = {
  "sample_rate": 16000,
  "frame_ms": 25,
  "stride_ms": 10,
  "num_feature_bins": 40,
  "feature_type": "mfcc",
  "pre_emph": 0.97,
  "delta": True,
  "normalize_signal": True,
  "normalize_feature": True,
  "pitch": True
}

train_data_transcript_paths = [
  "/mnt/Data/ML/ASR/Preprocessed/Large/Train/transcripts.tsv"
]

eval_data_transcript_paths = [
  "/mnt/Data/ML/ASR/Preprocessed/Large/Dev/transcripts.tsv"
]

test_data_transcript_paths = [
  "/mnt/Data/ML/ASR/Preprocessed/Large/Test/transcripts.tsv"
]

tfrecords_dir = "/mnt/Data/ML/ASR/Preprocessed/Large/TFRecords"

checkpoint_dir = "/mnt/Projects/asrk16/trained/large/ckpts/"

log_dir = "/mnt/Projects/asrk16/trained/large/logs/"
