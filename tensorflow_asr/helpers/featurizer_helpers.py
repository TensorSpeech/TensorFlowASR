import tensorflow as tf

from tensorflow_asr.configs.config import Config
from tensorflow_asr.featurizers import speech_featurizers, text_featurizers

logger = tf.get_logger()


def prepare_featurizers(
    config: Config,
    subwords: bool = True,
    sentence_piece: bool = False,
):
    speech_featurizer = speech_featurizers.TFSpeechFeaturizer(config.speech_config)
    if sentence_piece:
        logger.info("Loading SentencePiece model ...")
        text_featurizer = text_featurizers.SentencePieceFeaturizer(config.decoder_config)
    elif subwords:
        logger.info("Loading subwords ...")
        text_featurizer = text_featurizers.SubwordFeaturizer(config.decoder_config)
    else:
        logger.info("Use characters ...")
        text_featurizer = text_featurizers.CharFeaturizer(config.decoder_config)
    return speech_featurizer, text_featurizer
