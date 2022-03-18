# %%
from tensorflow_asr.configs.config import Config
from tensorflow_asr.helpers import featurizer_helpers
from tensorflow_asr.models.ctc.conformer import Conformer
from tensorflow_asr.utils import env_util

logger = env_util.setup_environment()

env_util.setup_seed()

config_dict = {
    "speech_config": {
        "sample_rate": 16000,
        "frame_ms": 25,
        "stride_ms": 10,
        "num_feature_bins": 80,
        "feature_type": "log_mel_spectrogram",
    },
    "decoder_config": {
        "type": "wordpiece",
        "blank_index": 0,
        "unknown_token": "[PAD]",
        "unknown_index": 0,
        "beam_width": 0,
        "norm_score": True,
        "lm_config": None,
        "vocabulary": "../../../vocabularies/librispeech/wordpiece/train_1000_50.tokens",
        "vocab_size": 1000,
        "max_token_length": 50,
        "max_unique_chars": 1000,
        "reserved_tokens": ["[PAD]"],
        "normalization_form": "NFKC",
        "num_iterations": 4,
    },
    "model_config": {
        "name": "conformer",
        "encoder_subsampling": {
            "type": "conv2d_blurpool",
            "filters": 144,
            "kernel_size": 3,
            "strides": 2,
            "conv_padding": "same",
            "pool_padding": "reflect",
            "activation": "relu",
        },
        "encoder_dmodel": 144,
        "encoder_num_blocks": 16,
        "encoder_head_size": 36,
        "encoder_num_heads": 4,
        "encoder_mha_type": "relmha",
        "encoder_use_attention_mask": True,
        "encoder_kernel_size": 32,
        "encoder_fc_factor": 0.5,
        "encoder_dropout": 0.1,
        "encoder_padding": "same",
    },
}

config = Config(config_dict)

speech_featurizer, text_featurizer = featurizer_helpers.prepare_featurizers(config=config)

global_batch_size = 2
speech_featurizer.update_length(1200)
text_featurizer.update_length(700)

conformer = Conformer(
    **config.model_config,
    vocab_size=text_featurizer.num_classes,
)
conformer.make(speech_featurizer.shape, batch_size=global_batch_size)
conformer.add_featurizers(speech_featurizer, text_featurizer)
conformer.summary()
# %%
import tensorflow as tf

from tensorflow_asr.models.layers.multihead_attention import compute_self_attention_mask

compute_self_attention_mask(tf.zeros([4, 10, 3]), [8, 7, 9, 10])
# %%
conformer.save_weights("./conformer.h5")
conformer.load_weights("./conformer.h5")
# %%
conformer.save("./saved_model")

# %%

import tf2onnx

tf2onnx.convert.from_keras(conformer, output_path="./conformer.onnx")

# %%
