from __future__ import absolute_import

import tensorflow as tf

tf.get_logger().setLevel("ERROR")

from utils.Flags import app, flags_obj
from utils.Utils import get_config
from data.Dataset import Dataset
from featurizers.SpeechFeaturizer import SpeechFeaturizer
from featurizers.TextFeaturizer import TextFeaturizer
from asr.SpeechToText import SpeechToText


def main(argv):
    configs = get_config(flags_obj.config)
    # Initiate featurizers
    speech_featurizer = SpeechFeaturizer(
        sample_rate=configs["sample_rate"],
        frame_ms=configs["frame_ms"],
        stride_ms=configs["stride_ms"],
        num_feature_bins=configs["num_feature_bins"])
    text_featurizer = TextFeaturizer(configs["vocabulary_file_path"])

    if flags_obj.mode == "train":
        # Initiate datasets
        train_dataset = Dataset(data_path=configs["train_data_transcript_paths"], mode="train")
        eval_dataset = Dataset(data_path=configs["eval_data_transcript_paths"], mode="eval")
        # Initiate speech to try:
        asr = SpeechToText(
            speech_featurizer=speech_featurizer,
            text_featurizer=text_featurizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            configs=configs)
        asr.train_and_eval()
        if flags_obj.export_file is not None:
            asr.save_model(flags_obj.export_file)
    elif flags_obj.mode == "save":
        if flags_obj.export_file is None:
            raise ValueError("Flag 'export_file' must be set")
        asr = SpeechToText(
            speech_featurizer=speech_featurizer,
            text_featurizer=text_featurizer,
            configs=configs)
        asr.save_model(flags_obj.export_file)
    elif flags_obj.mode == "test":
        if flags_obj.export_file is None:
            raise ValueError("Flag 'export_file' must be set")
        test_dataset = Dataset(data_path=configs["test_data_transcript_paths"], mode="test")
        asr = SpeechToText(
            speech_featurizer=speech_featurizer,
            text_featurizer=text_featurizer,
            test_dataset=test_dataset,
            configs=configs
        )
        asr.test(flags_obj.export_file)
    elif flags_obj.mode == "infer":
        if flags_obj.infer_file_path == "":
            raise ValueError("Flag 'infer_file_path' must be set")
        if flags_obj.export_file is None:
            raise ValueError("Flag 'export_file' must be set")
        asr = SpeechToText(
            speech_featurizer=speech_featurizer,
            text_featurizer=text_featurizer,
            configs=configs
        )
        asr.infer(speech_file_path=flags_obj.infer_file_path, model_file=flags_obj.export_file)
    else:
        raise ValueError("Flag 'mode' must be either 'save', 'train', 'test' or 'infer'")


if __name__ == '__main__':
    app.run(main)
