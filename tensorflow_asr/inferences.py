from tensorflow_asr.models.base_model import BaseModel


class ASRInference:
    def __init__(
        self,
        tflite: str = None,
        model: BaseModel = None,
    ):
        if not tflite and not model:
            raise ValueError("Either `tflite` or `model` must be provided.")
        self.tflite = tflite
        self.model = model

    def _tflite_inference(self, *args, **kwargs):
        raise NotImplementedError("TFLite inference is not implemented yet.")

    def _tflite_streaming_inference(self, *args, **kwargs):
        raise NotImplementedError("TFLite streaming inference is not implemented yet.")

    def _model_inference(self, *args, **kwargs):
        raise NotImplementedError("Model inference is not implemented yet.")

    def _model_streaming_inference(self, *args, **kwargs):
        raise NotImplementedError("Model streaming inference is not implemented yet.")
