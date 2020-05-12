import os
import argparse

from data.Dataset import Dataset
from utils.Utils import get_asr_config

argparser = argparse.ArgumentParser()

argparser.add_argument("config", type=str, help="Path to config file")

args = argparser.parse_args()

config = get_asr_config(args.config)

tfrecords_dir = config["tfrecords_dir"]
eval_data = config["eval_data_transcript_paths"]
augmentations = config["augmentations"]

train_dataset = Dataset(config["train_data_transcript_paths"], tfrecords_dir, mode="train")
train_dataset.create_tfrecords(augmentations=augmentations)
test_dataset = Dataset(config["test_data_transcript_paths"], tfrecords_dir, mode="test")
test_dataset.create_tfrecords(augmentations=[None])
if eval_data:
  eval_dataset = Dataset(eval_data, tfrecords_dir, mode="eval")
  eval_dataset.create_tfrecords(augmentations=[None])
