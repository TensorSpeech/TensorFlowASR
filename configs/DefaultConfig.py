from __future__ import absolute_import

from models.deepspeech2.DeepSpeech2 import DeepSpeech2
from augmentations.Augments import Noise

base_model = DeepSpeech2()

streaming_size = None

decoder = {
  "name": "beamsearch",
  "beam_width": 500,
  "lm_path": "/mnt/Data/ML/NLP/vntc_5gram_probing.binary",
  "alpha": 1.0,
  "beta": 0.5
}

# augmentations = [Noise(snr_list=[0, 5, 10, 15],
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
  "delta_delta": True,
  "normalize_signal": True,
  "normalize_feature": True,
  "norm_per_feature": False,
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
