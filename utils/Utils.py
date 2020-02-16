from __future__ import absolute_import

import runpy
from nltk.metrics import distance

conf_options = ["base_model",
                "decoder",
                "batch_size",
                "num_epochs",
                "vocabulary_file_path",
                "learning_rate",
                "sample_rate",
                "frame_ms",
                "stride_ms",
                "num_feature_bins",
                "train_data_transcript_paths",
                "eval_data_transcript_paths",
                "test_data_transcript_paths",
                "checkpoint_file",
                "log_dir"]


def get_config(config_path):
    conf_dict = runpy.run_path(config_path)
    for option in conf_options:
        if conf_dict.get(option, None) is None:
            raise ValueError(
                '{} has to be defined in the config file'.format(option))
    return conf_dict


def wer(decode, target):
    words = set(decode.split() + target.split())
    word2char = dict(zip(words, range(len(words))))

    new_decode = [chr(word2char[w]) for w in decode.split()]
    new_target = [chr(word2char[w]) for w in target.split()]

    return distance.edit_distance(''.join(new_decode), ''.join(new_target))


def cer(decode, target):
    return distance.edit_distance(decode, target)
