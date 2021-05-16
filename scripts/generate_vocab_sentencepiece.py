import os
import argparse
from tensorflow_asr.utils.env_util import setup_environment, setup_strategy

logger = setup_environment()
import tensorflow as tf

DEFAULT_YAML = os.path.join(os.path.abspath(os.path.dirname(__file__)), "config.yml")

tf.keras.backend.clear_session()

parser = argparse.ArgumentParser(prog="Vocab Training with SentencePiece")

parser.add_argument("--config", type=str, default=DEFAULT_YAML,
                    help="The file path of model configuration file")

parser.add_argument("--devices", type=int, nargs="*", default=[0],
                    help="Devices' ids to apply distributed training")

args = parser.parse_args()

strategy = setup_strategy(args.devices)

from tensorflow_asr.configs.config import Config
from tensorflow_asr.featurizers.text_featurizers import SentencePieceFeaturizer

config = Config(args.config)

logger.info("Generating subwords ...")
text_featurizer = SentencePieceFeaturizer.build_from_corpus(
    config.decoder_config
)
