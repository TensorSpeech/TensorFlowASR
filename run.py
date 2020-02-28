from __future__ import absolute_import
from asr.SpeechToText import SpeechToText
from featurizers.TextFeaturizer import TextFeaturizer
from featurizers.SpeechFeaturizer import SpeechFeaturizer
from data.Dataset import Dataset
from utils.Flags import app, flags_obj

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
        msg = "From: %s\r\nTo: %s\r\nSubject: %s\r\nDate: %s\r\n\r\n%s" % \
              (self.fromaddr, ", ".join(self.toaddrs),
               self.getSubject(record), formatdate(), msg)
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
mailHandler = handlers.SMTPHandler(
    mailhost=("smtp.gmail.com", 587), fromaddr="nlhuy1998@gmail.com",
    toaddrs=["nlhuy.cs.16@gmail.com"], subject="ASR Error",
    credentials=("nlhuy1998@gmail.com", "oqehnspyvxnniiyj"),
    secure=(None), timeout=30)
mailHandler.emit = emit
mailHandler.setLevel(ERROR)
tf.get_logger().addHandler(mailHandler)


def main(argv):
    if flags_obj.export_file is None:
        raise ValueError("Flag 'export_file' must be set")
    if flags_obj.mode == "train":
        asr = SpeechToText(configs_path=flags_obj.config, mode="train")
        asr(model_file=flags_obj.export_file)
    elif flags_obj.mode == "save":
        asr = SpeechToText(configs_path=flags_obj.config, mode="infer")
        asr.save_model(flags_obj.export_file)
    elif flags_obj.mode == "test":
        asr = SpeechToText(configs_path=flags_obj.config, mode="test")
        if flags_obj.output_file_path is None:
            raise ValueError("Flag 'output_file_path must be set")
        asr(model_file=flags_obj.export_file,
            output_file_path=flags_obj.output_file_path)
    elif flags_obj.mode == "infer":
        if flags_obj.output_file_path is None:
            raise ValueError("Flag 'output_file_path must be set")
        if flags_obj.speech_file_path is None:
            raise ValueError("Flag 'speech_file_path must be set")
        asr = SpeechToText(configs_path=flags_obj.config, mode="infer")
        asr(model_file=flags_obj.export_file,
            speech_file_path=flags_obj.speech_file_path,
            output_file_path=flags_obj.output_file_path)
    else:
        raise ValueError("Flag 'mode' must be either 'save', 'train', \
                         'test' or 'infer'")


if __name__ == '__main__':
    app.run(main)
