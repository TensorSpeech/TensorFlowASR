from __future__ import absolute_import

from models.deepspeech2.DeepSpeech2 import DeepSpeech2
from augmentations.Augments import Noise

base_model = DeepSpeech2(num_conv=3, num_rnn=3, rnn_units=256, filters=(16, 32, 64),
                         is_bidirectional=True, kernel_size=(31, 11))

streaming_size = None

decoder = {
  "name":       "beamsearch",
  "beam_width": 500,
  "lm_path":    "/mnt/Data/ML/NLP/vntc_5gram_probing.binary",
  "alpha":      1.0,
  "beta":       0.5
}

# augmentations = [Noise(min_snr=3, max_snr=10,
#                        min_noises=1, max_noises=3,
#                        noise_dir="/mnt/Data/ML/ASR/Preprocessed/Noises")]
augmentations = []

batch_size = 8

num_epochs = 10

vocabulary_file_path = "/mnt/Projects/asrk16/vnasr/data/vocabulary.txt"

last_activation = 'linear'

speech_conf = {
  "sample_rate":       16000,
  "frame_ms":          20,
  "stride_ms":         10,
  "num_feature_bins":  12,
  "feature_type":      "mfcc",
  "pre_emph":          0.95,
  "is_delta":          True,
  "normalize_signal":  True,
  "normalize_feature": False
}

train_data_transcript_paths = [
  "/mnt/Data/ML/ASR/Preprocessed/SmallFixed/Train/transcripts.tsv"
]

eval_data_transcript_paths = [
  "/mnt/Data/ML/ASR/Preprocessed/SmallFixed/Dev/transcripts.tsv"
]

test_data_transcript_paths = [
  "/mnt/Data/ML/ASR/Preprocessed/SmallFixed/Test/transcripts.tsv"
]

tfrecords_dir = "/mnt/Data/ML/ASR/Preprocessed/SmallFixed/TFRecords"

checkpoint_dir = "/mnt/Projects/asrk16/trained/med-bilstm/ckpts/"

log_dir = "/mnt/Projects/asrk16/trained/med-bilstm/tensorboard/"
