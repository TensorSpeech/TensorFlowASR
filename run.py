from __future__ import absolute_import
from asr.SpeechToText import SpeechToText
from featurizers.TextFeaturizer import TextFeaturizer
from featurizers.SpeechFeaturizer import SpeechFeaturizer
from data.Dataset import Dataset
from utils.Flags import app, flags_obj

import tensorflow as tf
from logging import ERROR

tf.get_logger().setLevel(ERROR)


def main(argv):
    if flags_obj.export_file is None:
        raise ValueError("Flag 'export_file' must be set")
    if flags_obj.mode == "train":
        asr = SpeechToText(configs_path=flags_obj.config, mode="train")
        asr(model_file=flags_obj.export_file)
    elif flags_obj.mode == "save":
        asr = SpeechToText(configs_path=flags_obj.config, mode="infer")
        asr.save_model(flags_obj.export_file)
    elif flags_obj.mode == "test":
        asr = SpeechToText(configs_path=flags_obj.config, mode="test")
        if flags_obj.output_file_path is None:
            raise ValueError("Flag 'output_file_path must be set")
        asr(model_file=flags_obj.export_file,
            output_file_path=flags_obj.output_file_path)
    elif flags_obj.mode == "infer":
        if flags_obj.output_file_path is None:
            raise ValueError("Flag 'output_file_path must be set")
        if flags_obj.speech_file_path is None:
            raise ValueError("Flag 'speech_file_path must be set")
        asr = SpeechToText(configs_path=flags_obj.config, mode="infer")
        asr(model_file=flags_obj.export_file,
            speech_file_path=flags_obj.speech_file_path,
            output_file_path=flags_obj.output_file_path)
    else:
        raise ValueError("Flag 'mode' must be either 'save', 'train', \
                         'test' or 'infer'")


if __name__ == '__main__':
    app.run(main)
