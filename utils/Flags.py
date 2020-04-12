from __future__ import absolute_import

import os
import argparse

current_path = os.path.dirname(os.path.abspath(__file__))
_CONF_FILE = os.path.join(current_path,
                          "..", "configs",
                          "DefaultConfig.py")

parser = argparse.ArgumentParser(description="ASR Commands")

parser.add_argument("--mode", "-m", type=str, default="",
                    help="Mode for training, testing or infering")

parser.add_argument("--config", "-c", type=str, default=_CONF_FILE,
                    help="The file path of model configuration file")

parser.add_argument("--input_file_path", "-i", type=str, default=None,
                    help="Path to input file")

parser.add_argument("--export_file", "-e", type=str, default=None,
                    help="Path to the model file to be exported")

parser.add_argument("--pretrained", "-p", type=str, default=None,
                    help="Path to the previous checkpoint to retrain")

parser.add_argument("--output_file_path", "-o", type=str, default=None,
                    help="Path to output file")

args_parser = parser.parse_args()
