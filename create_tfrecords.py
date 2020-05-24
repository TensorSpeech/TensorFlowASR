from __future__ import absolute_import

import argparse
import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter('ignore')

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from data.Dataset import Dataset

parser = argparse.ArgumentParser(description="Create tfrecords for asr")

parser.add_argument("--tfrecords_dir", "-d", type=str, help="TFRecords directory")
parser.add_argument("--mode", "-m", type=str, help="Mode either train, eval or test")
parser.add_argument("transcripts", nargs="+", type=str, help="Transcripts paths")

args = parser.parse_args()


def main():
    assert args.mode in ["train", "eval", "test"]
    train_dataset = Dataset(args.transcripts, mode=args.mode)
    train_dataset.create_tfrecords(args.tfrecords_dir, True)


if __name__ == "__main__":
    main()
