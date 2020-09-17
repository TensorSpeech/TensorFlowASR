import os
import argparse
from tiramisu_asr.utils import setup_environment, setup_devices

setup_environment()
import tensorflow as tf

DEFAULT_YAML = os.path.join(os.path.abspath(os.path.dirname(__file__)), "config.yml")

parser = argparse.ArgumentParser(prog="Self Attention DS2")

parser.add_argument("--config", "-c", type=str, default=DEFAULT_YAML,
                    help="The file path of model configuration file")

parser.add_argument("--saved", type=str, default=None,
                    help="Path to saved model")

parser.add_argument("--tfrecords", default=False, action="store_true",
                    help="Whether to use tfrecords")

parser.add_argument("--mxp", default=False, action="store_true",
                    help="Enable mixed precision")

parser.add_argument("--device", type=int, default=0,
                    help="Device's id to run test on")

parser.add_argument("--bs", type=int, default=None, help="Batch size")

args = parser.parse_args()

tf.config.optimizer.set_experimental_options({"auto_mixed_precision": args.mxp})

setup_devices([args.device])

from tiramisu_asr.featurizers.speech_featurizers import TFSpeechFeaturizer
from tiramisu_asr.featurizers.text_featurizers import CharFeaturizer
from tiramisu_asr.configs.user_config import UserConfig
from tiramisu_asr.datasets.asr_dataset import ASRTFRecordDataset, ASRSliceDataset
from model import SelfAttentionDS2
from tiramisu_asr.runners.base_runners import BaseTester
from ctc_decoders import Scorer

tf.random.set_seed(0)
assert args.saved

config = UserConfig(DEFAULT_YAML, args.config, learning=True)
speech_featurizer = TFSpeechFeaturizer(config["speech_config"])
text_featurizer = CharFeaturizer(config["decoder_config"])

text_featurizer.add_scorer(Scorer(**text_featurizer.decoder_config["lm_config"],
                                  vocabulary=text_featurizer.vocab_array))

# Build DS2 model
satt_ds2_model = SelfAttentionDS2(
    input_shape=speech_featurizer.shape,
    arch_config=config["model_config"],
    num_classes=text_featurizer.num_classes
)
satt_ds2_model._build(speech_featurizer.shape)
satt_ds2_model.load_weights(args.saved, by_name=True)
satt_ds2_model.summary(line_length=150)
satt_ds2_model.add_featurizers(speech_featurizer, text_featurizer)

if args.tfrecords:
    test_dataset = ASRTFRecordDataset(
        data_paths=config["learning_config"]["dataset_config"]["test_paths"],
        tfrecords_dir=config["learning_config"]["dataset_config"]["tfrecords_dir"],
        speech_featurizer=speech_featurizer,
        text_featurizer=text_featurizer,
        stage="test", shuffle=False
    )
else:
    test_dataset = ASRSliceDataset(
        data_paths=config["learning_config"]["dataset_config"]["test_paths"],
        speech_featurizer=speech_featurizer,
        text_featurizer=text_featurizer,
        stage="test", shuffle=False
    )

ctc_tester = BaseTester(config=config["learning_config"]["running_config"])
ctc_tester.compile(satt_ds2_model)
ctc_tester.run(test_dataset, batch_size=args.bs)
