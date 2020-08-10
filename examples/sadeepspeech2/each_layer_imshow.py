import os
import argparse

from tiramisu_asr.utils import setup_environment

setup_environment()
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import tensorflow as tf
import numpy as np

from tiramisu_asr.featurizers.speech_featurizers import SpeechFeaturizer, read_raw_audio
from tiramisu_asr.featurizers.text_featurizers import TextFeaturizer
from tiramisu_asr.configs.user_config import UserConfig
from tiramisu_asr.utils.utils import bytes_to_string
from ctc_decoders import Scorer
from model import SelfAttentionDS2


def main():
    parser = argparse.ArgumentParser(prog="SelfAttentionDS2 Histogram")

    parser.add_argument("--config", type=str, default=None,
                        help="Config file")

    parser.add_argument("--audio", type=str, default=None,
                        help="Audio file")

    parser.add_argument("--saved_model", type=str, default=None,
                        help="Saved model")

    parser.add_argument("--from_weights", type=bool, default=False,
                        help="Load from weights")

    parser.add_argument("--output", type=str, default=None,
                        help="Output dir storing histograms")

    args = parser.parse_args()

    config = UserConfig(args.config, args.config, learning=False)
    speech_featurizer = SpeechFeaturizer(config["speech_config"])
    text_featurizer = TextFeaturizer(config["decoder_config"])
    text_featurizer.add_scorer(Scorer(**text_featurizer.decoder_config["lm_config"],
                                      vocabulary=text_featurizer.vocab_array))

    f, c = speech_featurizer.compute_feature_dim()
    satt_ds2_model = SelfAttentionDS2(input_shape=[None, f, c],
                                      arch_config=config["model_config"],
                                      num_classes=text_featurizer.num_classes)
    satt_ds2_model._build([1, 50, f, c])

    if args.from_weights:
        satt_ds2_model.load_weights(args.saved_model)
    else:
        saved_model = tf.keras.models.load_model(args.saved_model)
        satt_ds2_model.set_weights(saved_model.get_weights())

    satt_ds2_model.summary(line_length=100)

    satt_ds2_model.add_featurizers(speech_featurizer, text_featurizer)

    signal = read_raw_audio(args.audio, speech_featurizer.sample_rate)
    features = speech_featurizer.extract(signal)
    decoded = satt_ds2_model.recognize_beam(tf.expand_dims(features, 0), lm=True)
    print(bytes_to_string(decoded.numpy()))

    # for i in range(1, len(satt_ds2_model.base_model.layers)):
    #     func = tf.keras.backend.function([satt_ds2_model.base_model.input],
    #                                      [satt_ds2_model.base_model.layers[i].output])
    #     data = func([np.expand_dims(features, 0), 1])[0][0]
    #     print(data.shape)
    #     plt.figure(figsize=(16, 5))
    #     ax = plt.gca()
    #     im = ax.imshow(data.T, origin="lower", aspect="auto")
    #     ax.set_title(f"{satt_ds2_model.base_model.layers[i].name}", fontweight="bold")
    #     divider = make_axes_locatable(ax)
    #     cax = divider.append_axes("right", size="5%", pad=0.05)
    #     plt.colorbar(im, cax=cax)
    #     plt.savefig(os.path.join(
    #         args.output, f"{i}_{satt_ds2_model.base_model.layers[i].name}.png"))
    #     plt.clf()
    #     plt.cla()
    #     plt.close()

    fc = satt_ds2_model(tf.expand_dims(features, 0), training=False)
    plt.figure(figsize=(16, 5))
    ax = plt.gca()
    ax.set_title(f"{satt_ds2_model.layers[-1].name}", fontweight="bold")
    im = ax.imshow(fc[0].numpy().T, origin="lower", aspect="auto")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.savefig(os.path.join(args.output, f"{satt_ds2_model.layers[-1].name}.png"))
    plt.clf()
    plt.cla()
    plt.close()
    fc = tf.nn.softmax(fc)
    plt.figure(figsize=(16, 5))
    ax = plt.gca()
    ax.set_title("Softmax", fontweight="bold")
    im = ax.imshow(fc[0].numpy().T, origin="lower", aspect="auto")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.savefig(os.path.join(args.output, "softmax.png"))
    plt.clf()
    plt.cla()
    plt.close()
    plt.figure(figsize=(16, 5))
    ax = plt.gca()
    ax.set_title("Log Mel Spectrogram", fontweight="bold")
    im = ax.imshow(features[:, :, 0].T, origin="lower", aspect="auto")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.savefig(os.path.join(args.output, "features.png"))
    plt.clf()
    plt.cla()
    plt.close()


if __name__ == "__main__":
    main()
