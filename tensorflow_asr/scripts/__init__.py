from tensorflow_asr.scripts import save, test, tflite, train
from tensorflow_asr.scripts.utils import create_librispeech_trans, create_mls_trans, create_tfrecords, prepare_vocab_and_metadata
from tensorflow_asr.utils import cli_util


def main():
    cli_util.run(
        {
            "train": train.main,
            "test": test.main,
            "tflite": tflite.main,
            "save": save.main,
            "utils": {
                "create_librispeech_trans": create_librispeech_trans.main,
                "create_mls_trans": create_mls_trans.main,
                "create_tfrecords": create_tfrecords.main,
                "prepare_vocab_and_metadata": prepare_vocab_and_metadata.main,
            },
        }
    )
