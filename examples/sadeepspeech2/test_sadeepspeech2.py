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

parser.add_argument("--tfrecords", type=bool, default=False,
                    help="Whether to use tfrecords")

parser.add_argument("--device", type=int, default=0,
                    help="Device's id to run test on")

parser.add_argument("--bs", type=int, default=None, help="Batch size")

args = parser.parse_args()

setup_devices([args.device])

from tiramisu_asr.featurizers.speech_featurizers import TFSpeechFeaturizer
from tiramisu_asr.featurizers.text_featurizers import TextFeaturizer
from tiramisu_asr.configs.user_config import UserConfig
from tiramisu_asr.datasets.asr_dataset import ASRTFRecordDataset, ASRSliceDataset
from model import SelfAttentionDS2
from tiramisu_asr.runners.base_runners import BaseTester
from ctc_decoders import Scorer

tf.random.set_seed(0)
assert args.saved

config = UserConfig(DEFAULT_YAML, args.config, learning=True)
speech_featurizer = TFSpeechFeaturizer(config["speech_config"])
text_featurizer = TextFeaturizer(config["decoder_config"])

text_featurizer.add_scorer(Scorer(**text_featurizer.decoder_config["lm_config"],
                                  vocabulary=text_featurizer.vocab_array))

# Build DS2 model
satt_ds2_model = SelfAttentionDS2(
    input_shape=speech_featurizer.shape,
    arch_config=config["model_config"],
    num_classes=text_featurizer.num_classes
)
satt_ds2_model._build(speech_featurizer.shape)
satt_ds2_model.summary(line_length=150)
satt_ds2_model.load_weights(args.saved)
satt_ds2_model.add_featurizers(speech_featurizer, text_featurizer)

if args.tfrecords:
    test_dataset = ASRTFRecordDataset(
        config["learning_config"]["dataset_config"]["test_paths"],
        config["learning_config"]["dataset_config"]["tfrecords_dir"],
        speech_featurizer, text_featurizer, "test",
        augmentations=config["learning_config"]["augmentations"], shuffle=False
    )
else:
    test_dataset = ASRSliceDataset(
        stage="test", speech_featurizer=speech_featurizer,
        text_featurizer=text_featurizer,
        data_paths=config["learning_config"]["dataset_config"]["test_paths"],
        augmentations=config["learning_config"]["augmentations"], shuffle=False
    )

ctc_tester = BaseTester(config=config["learning_config"]["running_config"])
ctc_tester.compile(satt_ds2_model)
ctc_tester.run(test_dataset, batch_size=args.bs)
