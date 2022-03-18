import argparse
import os

from tensorflow_asr.configs.config import Config
from tensorflow_asr.featurizers.text_featurizers import SubwordFeaturizer

DEFAULT_YAML = os.path.join(os.path.abspath(os.path.dirname(__file__)), "config_wp.j2")

parser = argparse.ArgumentParser(prog="Vocab Training with Subwords")

parser.add_argument("corpus", nargs="*", type=str, default=[], help="Transcript files for generating subwords")

parser.add_argument("--config", type=str, default=DEFAULT_YAML, help="The file path of model configuration file")

parser.add_argument("--output_file", type=str, default=None, help="Path to file that stores generated subwords")

args = parser.parse_args()

config = Config(args.config)

print("Generating subwords ...")

SubwordFeaturizer.build_from_corpus(config.decoder_config)
