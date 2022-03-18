import os

from tensorflow_asr.utils import file_util


def test():
    a = file_util.load_yaml(f"{os.path.dirname(__file__)}/../examples/conformer/config_wp.yml")
    print(a)
