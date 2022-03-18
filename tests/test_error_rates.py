from tensorflow_asr.utils import metric_util


def test_wer():
    decode = [
        "hello i am huy",
    ]
    target = [
        "hello i am huy",
    ]
    a, b = metric_util.tf_wer(decode, target)
    print(a.numpy())
    print(b.numpy())
