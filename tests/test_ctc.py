import tensorflow as tf

from ctc_decoders import Scorer
from tensorflow_asr.models.ctc import CtcModel
from tensorflow_asr.featurizers.text_featurizers import CharFeaturizer
from tensorflow_asr.featurizers.speech_featurizers import TFSpeechFeaturizer
from tensorflow_asr.utils.utils import bytes_to_string, merge_two_last_dims

decoder_config = {
    "vocabulary": "/mnt/Projects/asrk16/TiramisuASR/vocabularies/vietnamese.txt",
    "beam_width": 100,
    "blank_at_zero": False,
    "lm_config": {
        "model_path": "/mnt/Data/ML/NLP/vntc_asrtrain_5gram_trie.binary",
        "alpha": 2.0,
        "beta": 2.0
    }
}
text_featurizer = CharFeaturizer(decoder_config)
text_featurizer.add_scorer(Scorer(**decoder_config["lm_config"],
                                  vocabulary=text_featurizer.vocab_array))
speech_featurizer = TFSpeechFeaturizer({
    "sample_rate": 16000,
    "frame_ms": 25,
    "stride_ms": 10,
    "num_feature_bins": 80,
    "feature_type": "spectrogram",
    "preemphasis": 0.97,
    # "delta": True,
    # "delta_delta": True,
    "normalize_signal": True,
    "normalize_feature": True,
    "normalize_per_feature": False,
    # "pitch": False,
})

inp = tf.keras.Input(shape=[None, 80, 3])


class BaseModel(tf.keras.Model):
    def __init__(self, name="basemodel", **kwargs):
        super().__init__(name=name, **kwargs)
        self.dense = tf.keras.layers.Dense(350)
        self.lstm = tf.keras.layers.LSTM(350, return_sequences=True)
        self.time_reduction_factor = 1

    @tf.function
    def call(self, inputs, training=False, **kwargs):
        outputs = merge_two_last_dims(inputs)
        outputs = self.lstm(outputs, training=training)
        return self.dense(outputs, training=training)


model = CtcModel(base_model=BaseModel(), num_classes=text_featurizer.num_classes)

model._build(speech_featurizer.shape)
model.summary(line_length=100)
model.add_featurizers(
    speech_featurizer=speech_featurizer,
    text_featurizer=text_featurizer
)

features = tf.random.normal(shape=[5, 50, 80, 1], dtype=tf.float32)
hyp = model.recognize(features)
print(bytes_to_string(hyp.numpy()))

hyp = model.recognize_beam(features)
print(bytes_to_string(hyp.numpy()))

hyp = model.recognize_beam(features, lm=True)
print(bytes_to_string(hyp.numpy()))

# signal = read_raw_audio("/home/nlhuy/Desktop/test/11003.wav", speech_featurizer.sample_rate)
signal = tf.random.normal(shape=[500], dtype=tf.float32)

hyp = model.recognize_tflite(signal)
print(hyp.numpy())

hyp = model.recognize_beam_tflite(signal)
print(hyp.numpy())
#
# hyp = model.recognize_beam_tflite(signal, lm=True)
# print(hyp.numpy().decode("utf-8"))

concrete_func = model.make_tflite_function(greedy=False).get_concrete_function()
converter = tf.lite.TFLiteConverter.from_concrete_functions(
    [concrete_func]
)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                       tf.lite.OpsSet.SELECT_TF_OPS]
tflite = converter.convert()

tflitemodel = tf.lite.Interpreter(model_content=tflite)

input_details = tflitemodel.get_input_details()
output_details = tflitemodel.get_output_details()
tflitemodel.resize_tensor_input(input_details[0]["index"], signal.shape)
tflitemodel.allocate_tensors()
tflitemodel.set_tensor(input_details[0]["index"], signal)
tflitemodel.invoke()
hyp = tflitemodel.get_tensor(output_details[0]["index"])

print(hyp)
