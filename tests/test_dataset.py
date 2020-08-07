from tiramisu_asr.datasets.asr_dataset import ASRSliceDataset
from tiramisu_asr.featurizers.speech_featurizers import TFSpeechFeaturizer
from tiramisu_asr.featurizers.text_featurizers import TextFeaturizer


augments = {
    "before": {
        "methods": {
            "loudness": {
                "zone": (0.3, 0.7)
            },
            "speed": None,
            "noise": {
                "noises": "/mnt/Data/ML/ASR/Preprocessed/Noises/train"
            }
        },
        "sometimes": True
    },
    "after": {
        "methods": {
            "time_masking": {
                "num_masks": 10,
                "mask_factor": 100,
                "p_upperbound": 0.05
            },
            "freq_masking": {
                "mask_factor": 27
            }
        },
        "sometimes": False
    },
    "include_original": False
}

data = "/mnt/Data/ML/ASR/Raw/LibriSpeech/train-clean-100/transcripts.tsv"

text_featurizer = TextFeaturizer({
    "vocabulary": None,
    "blank_at_zero": True,
    "beam_width": 5,
    "norm_score": True
})

speech_featurizer = TFSpeechFeaturizer({
    "sample_rate": 16000,
    "frame_ms": 25,
    "stride_ms": 10,
    "num_feature_bins": 80,
    "feature_type": "logfbank",
    "preemphasis": 0.97,
    # "delta": True,
    # "delta_delta": True,
    "normalize_signal": True,
    "normalize_feature": True,
    "normalize_per_feature": False,
    # "pitch": False,
})


dataset = ASRSliceDataset(stage="train", speech_featurizer=speech_featurizer,
                          text_featurizer=text_featurizer, data_paths=[data],
                          augmentations=augments, shuffle=True).create(4)

for i, batch in enumerate(dataset):
    print(f"Processed batch {i}")
