from __future__ import absolute_import

from models.deepspeech2.DeepSpeech2 import DeepSpeech2
from models.aaconvds2.AAConvDeepSpeech2 import AAConvDeepSpeech2
from augmentations.Augments import Noise, TimeStretch, TimeMasking, FreqMasking, TimeWarping

base_model = DeepSpeech2(
    rnn_conf={
        "rnn_type": "lstm",
        "rnn_layers": 5,
        "rnn_bidirectional": True,
        "rnn_rowconv": False,
        "rnn_dropout": 0.2,
        "rnn_rowconv_context": 2,
        "rnn_units": 512,
        "rnn_activation": "tanh"
    },
    fc_conf={
        "fc_units": None
    }
)

streaming_size = None

decoder = {
    "name": "beamsearch",
    "beam_width": 500,
    "lm_path": "/mnt/Data/ML/NLP/vntc_asr_5g_pruned_probing.binary",
    "alpha": 2.0,
    "beta": 1.0
}

augmentations = [
    Noise(snr_list=[0, 5, 10, 15], max_noises=3, noise_dir="/mnt/Data/ML/ASR/Preprocessed/Noises"),
    TimeStretch(min_ratio=0.5, max_ratio=2.0),
    TimeMasking(),
    FreqMasking(),
    TimeWarping()
]

batch_size = 8

num_epochs = 10

vocabulary_file_path = "/mnt/Projects/asrk16/vnasr/data/vocabulary.txt"

sortagrad = False

speech_conf = {
    "sample_rate": 16000,
    "frame_ms": 25,
    "stride_ms": 10,
    "num_feature_bins": 80,
    "feature_type": "logfbank",
    "pre_emph": 0.97,
    "delta": True,
    "delta_delta": True,
    "normalize_signal": True,
    "normalize_feature": True,
    "norm_per_feature": False,
    "pitch": False
}

train_data_transcript_paths = [
    "/mnt/Data/ML/ASR/Preprocessed/Vivos/train/transcripts.tsv"
]

eval_data_transcript_paths = [
    "/mnt/Data/ML/ASR/Preprocessed/Vivos/test/transcripts.tsv"
]

test_data_transcript_paths = [
    "/mnt/Data/ML/ASR/Preprocessed/Vivos/test/transcripts.tsv"
]

tfrecords_dir = "/mnt/Data/ML/ASR/Preprocessed/Vivos/TFRecords"

checkpoint_dir = "/mnt/Projects/asrk16/trained/local/vivos/ckpts/"

log_dir = "/mnt/Projects/asrk16/trained/local/vivos/logs/"
