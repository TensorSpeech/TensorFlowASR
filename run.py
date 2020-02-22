from __future__ import absolute_import

import tensorflow as tf
import os
from logging import handlers, ERROR


def emit(self, record):
    """
    Overwrite the logging.handlers.SMTPHandler.emit function with SMTP_SSL.
    Emit a record.
    Format the record and send it to the specified addressees.
    """
    try:
        import smtplib
        from email.utils import formatdate
        port = self.mailport
        if not port:
            port = smtplib.SMTP_PORT
        smtp = smtplib.SMTP_SSL(self.mailhost, port, timeout=self._timeout)
        msg = self.format(record)
        msg = "From: %s\r\nTo: %s\r\nSubject: %s\r\nDate: %s\r\n\r\n%s" % (
            self.fromaddr, ", ".join(self.toaddrs), self.getSubject(record), formatdate(), msg)
        if self.username:
            smtp.ehlo()
            smtp.login(self.username, self.password)
        smtp.sendmail(self.fromaddr, self.toaddrs, msg)
        smtp.quit()
    except (KeyboardInterrupt, SystemExit):
        raise
    except:
        self.handleError(record)


tf.get_logger().setLevel(ERROR)
mailHandler = handlers.SMTPHandler(mailhost=("smtp.gmail.com", 587), fromaddr="nlhuy1998@gmail.com",
                                   toaddrs=["nlhuy.cs.16@gmail.com"], subject="ASR Error",
                                   credentials=("nlhuy1998@gmail.com", "oqehnspyvxnniiyj"), secure=(None), timeout=30)
mailHandler.emit = emit
mailHandler.setLevel(ERROR)
tf.get_logger().addHandler(mailHandler)

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
        if flags_obj.export_file is not None and os.path.isfile(flags_obj.export_file):
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
        error_rates = asr.test(flags_obj.export_file)
        if flags_obj.output_file_path is not None and os.path.isfile(flags_obj.output_file_path):
            asr.save_test_result(results=error_rates, output_file_path=flags_obj.output_file_path)
        else:
            print("WER: ", error_rates[0])
            print("CER: ", error_rates[-1])

    elif flags_obj.mode == "infer":
        if flags_obj.infer_file_path == "" or not os.path.isfile(flags_obj.infer_file_path):
            raise ValueError("Flag 'infer_file_path' must be set")
        if flags_obj.export_file is None or not os.path.isfile(flags_obj.export_file):
            raise ValueError("Flag 'export_file' must be set")
        asr = SpeechToText(
            speech_featurizer=speech_featurizer,
            text_featurizer=text_featurizer,
            configs=configs
        )
        predictions = asr.infer(speech_file_path=flags_obj.infer_file_path, model_file=flags_obj.export_file)
        if flags_obj.output_file_path is not None and os.path.isfile(flags_obj.output_file_path):
            asr.save_inference(predictions=predictions, output_file_path=flags_obj.output_file_path)
        else:
            print(predictions)
    else:
        raise ValueError("Flag 'mode' must be either 'save', 'train', 'test' or 'infer'")


if __name__ == '__main__':
    app.run(main)
