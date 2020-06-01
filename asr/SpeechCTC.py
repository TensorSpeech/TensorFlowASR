from __future__ import absolute_import

import os
import sys
import time
import pathlib
import tensorflow as tf

from models.CTCModel import create_ctc_model, ctc_loss, create_ctc_train_model
from decoders.CTCDecoders import create_ctc_decoder
from featurizers.TextFeaturizer import TextFeaturizer
from utils.Utils import get_asr_config, check_key_in_dict, bytes_to_string
from utils.Metrics import wer, cer
from featurizers.SpeechFeaturizer import speech_feature_extraction, compute_feature_dim, compute_time_dim
from utils.Checkpoint import Checkpoint
from utils.TimeHistory import TimeHistory
from data.Dataset import Dataset


class SpeechCTC:
    def __init__(self, configs_path, noise_filter=None):
        self.configs = get_asr_config(configs_path)
        self.text_featurizer = TextFeaturizer(self.configs["vocabulary_file_path"])
        self.decoder = create_ctc_decoder(decoder_config=self.configs["decoder"],
                                          text_featurizer=self.text_featurizer)
        self.model, self.optimizer = create_ctc_model(num_classes=self.text_featurizer.num_classes,
                                                      base_model=self.configs["base_model"],
                                                      streaming_size=self.configs["streaming_size"],
                                                      speech_conf=self.configs["speech_conf"])
        self.noise_filter = noise_filter
        self.writer = None

    def _create_checkpoints(self, model):
        if not self.configs["checkpoint_dir"]:
            raise ValueError("Must set checkpoint_dir")
        if not os.path.exists(self.configs["checkpoint_dir"]):
            os.makedirs(self.configs["checkpoint_dir"])
        self.ckpt = tf.train.Checkpoint(model=model, optimizer=self.optimizer)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.configs["checkpoint_dir"], max_to_keep=None)

    @tf.function
    def train(self, dataset, loss, epoch, num_epochs, num_steps=0, gpu=0):
        step = tf.zeros(shape=(), dtype=tf.int64)
        for step, [features, input_length, label, label_length] in dataset.enumerate(start=0):
            start = time.time()
            with tf.GradientTape() as tape:
                y_pred = self.model(features, training=True)
                train_loss = loss(y_true=label, y_pred=y_pred,
                                  input_length=input_length, label_length=label_length,
                                  num_classes=self.text_featurizer.num_classes)
                scaled_train_loss = self.optimizer.get_scaled_loss(train_loss) if gpu else train_loss
            scaled_gradients = tape.gradient(scaled_train_loss, self.model.trainable_variables)
            gradients = self.optimizer.get_unscaled_gradients(scaled_gradients) if gpu else scaled_gradients
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

            sys.stdout.write("\033[K")
            tf.print("\rEpoch: ", epoch + 1, "/", num_epochs,
                     ", step: ", step + 1, "/", num_steps,
                     ", duration: ", int(time.time() - start), "s",
                     ", train_loss = ", train_loss,
                     sep="", end="", output_stream=sys.stdout)

            if self.writer:
                with self.writer.as_default():
                    tf.summary.scalar("train_loss", train_loss, step=(((epoch + 1) * num_steps) + step))
        return step + 1

    def validate(self, model, decoder, dataset, loss, num_classes):
        eval_loss_count = 0;
        epoch_eval_loss = 0.0
        total_wer = 0.0;
        wer_count = 0.0
        total_cer = 0.0;
        cer_count = 0.0

        @tf.function
        def val_step(features, inp_length, y_true, lab_length):
            start = time.time()
            y_pred = model(features, training=False)
            _loss = loss(y_true=y_true, y_pred=y_pred,
                         input_length=inp_length, label_length=lab_length,
                         num_classes=num_classes)
            _pred = decoder(probs=y_pred, input_length=inp_length)

            sys.stdout.write("\033[K")
            tf.print("\rVal_duration: ", int(time.time() - start), "s",
                     ", val_loss = ", _loss, sep="", end="", output_stream=sys.stdout)

            return _loss, _pred

        for feature, input_length, transcript, label_length in dataset:
            _val_loss, _eval_pred = val_step(feature, input_length, transcript, label_length)
            predictions = bytes_to_string(_eval_pred.numpy())
            transcripts = self.decoder.convert_to_string(transcript)

            for idx, decoded in enumerate(predictions):
                _wer, _wer_count = wer(decode=decoded, target=transcripts[idx])
                _cer, _cer_count = cer(decode=decoded, target=transcripts[idx])
                total_wer += _wer;
                wer_count += _wer_count
                total_cer += _cer;
                cer_count += _cer_count

            epoch_eval_loss += _val_loss;
            eval_loss_count += 1

        epoch_eval_loss = epoch_eval_loss / eval_loss_count
        total_wer = total_wer / wer_count
        total_cer = total_cer / cer_count
        return epoch_eval_loss, total_wer, total_cer

    def train_and_eval(self, model_file=None, gpu=0):
        print("Training and evaluating model ...")
        self.optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(self.optimizer, loss_scale="dynamic")
        self._create_checkpoints(self.model)

        check_key_in_dict(dictionary=self.configs,
                          keys=["tfrecords_dir", "checkpoint_dir", "augmentations",
                                "log_dir", "train_data_transcript_paths"])
        augmentations = self.configs["augmentations"]

        train_dataset = Dataset(data_path=self.configs["train_data_transcript_paths"], mode="train")
        tf_train_dataset = train_dataset(text_featurizer=self.text_featurizer,
                                         speech_conf=self.configs["speech_conf"],
                                         batch_size=self.configs["batch_size"],
                                         augmentations=augmentations, sortagrad=False,
                                         builtin=False, fext=True, tfrecords_dir=self.configs["tfrecords_dir"])

        tf_eval_dataset = None

        if self.configs["eval_data_transcript_paths"]:
            eval_dataset = Dataset(data_path=self.configs["eval_data_transcript_paths"], mode="eval")
            tf_eval_dataset = eval_dataset(text_featurizer=self.text_featurizer,
                                           speech_conf=self.configs["speech_conf"],
                                           batch_size=self.configs["batch_size"],
                                           sortagrad=False, builtin=False, fext=True,
                                           tfrecords_dir=self.configs["tfrecords_dir"])

        self.model.summary()

        initial_epoch = 0
        if self.ckpt_manager.latest_checkpoint:
            initial_epoch = int(self.ckpt_manager.latest_checkpoint.split('-')[-1])
            # restoring the latest checkpoint in checkpoint_path
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)

        if self.configs["log_dir"]:
            if not os.path.exists(self.configs["log_dir"]):
                os.makedirs(self.configs["log_dir"])
            with open(os.path.join(self.configs["log_dir"], "model.json"), "w") as f:
                f.write(self.model.to_json())
            self.writer = tf.summary.create_file_writer(os.path.join(self.configs["log_dir"], "train"))

        epochs = self.configs["num_epochs"]
        steps = 0

        for epoch in range(initial_epoch, epochs, 1):
            epoch_eval_loss = None;
            epoch_eval_wer = None;
            epoch_eval_cer = None
            start = time.time()

            steps = self.train(tf_train_dataset, ctc_loss, epoch, epochs, steps, gpu)

            print(f"\nEnd training on epoch = {epoch}")

            self.ckpt_manager.save()
            print(f"Saved checkpoint at epoch {epoch + 1}")

            if tf_eval_dataset:
                print("Validating ... ")
                epoch_eval_loss, epoch_eval_wer, epoch_eval_cer = self.validate(
                    self.model, self.decoder, tf_eval_dataset, ctc_loss,
                    self.text_featurizer.num_classes
                )
                print(f"Average_val_loss = {epoch_eval_loss:.2f}, val_wer = {epoch_eval_wer:.2f}, val_cer = {epoch_eval_cer:.2f}")

            time_epoch = time.time() - start
            print(f"Time for epoch {epoch + 1} is {time_epoch} secs")

            if self.writer:
                with self.writer.as_default():
                    if epoch_eval_loss and epoch_eval_wer and epoch_eval_cer:
                        tf.summary.scalar("eval_loss", epoch_eval_loss, step=epoch)
                        tf.summary.scalar("eval_wer", epoch_eval_wer, step=epoch)
                        tf.summary.scalar("eval_cer", epoch_eval_cer, step=epoch)
                    tf.summary.scalar("epoch_time", time_epoch, step=epoch)
                self.writer.flush()

        if model_file:
            self.save_model(model_file)

    def train_and_eval_builtin(self, model_file=None):
        print("Training and evaluating model ...")

        check_key_in_dict(dictionary=self.configs,
                          keys=["tfrecords_dir", "checkpoint_dir", "augmentations",
                                "log_dir", "train_data_transcript_paths", "sortagrad"])
        augmentations = self.configs["augmentations"]

        train_dataset = Dataset(data_path=self.configs["train_data_transcript_paths"], mode="train")
        if self.configs["sortagrad"]:
            tf_train_dataset_sortagrad = train_dataset(text_featurizer=self.text_featurizer,
                                                       speech_conf=self.configs["speech_conf"],
                                                       batch_size=self.configs["batch_size"],
                                                       augmentations=augmentations, sortagrad=True, builtin=True,
                                                       fext=True, tfrecords_dir=self.configs["tfrecords_dir"])
        tf_train_dataset = train_dataset(text_featurizer=self.text_featurizer,
                                         speech_conf=self.configs["speech_conf"],
                                         batch_size=self.configs["batch_size"],
                                         augmentations=augmentations, sortagrad=False, builtin=True,
                                         fext=True, tfrecords_dir=self.configs["tfrecords_dir"])

        tf_eval_dataset = None
        if self.configs["eval_data_transcript_paths"]:
            eval_dataset = Dataset(data_path=self.configs["eval_data_transcript_paths"], mode="eval")
            tf_eval_dataset = eval_dataset(text_featurizer=self.text_featurizer,
                                           speech_conf=self.configs["speech_conf"],
                                           batch_size=self.configs["batch_size"],
                                           sortagrad=False, builtin=True, fext=True,
                                           tfrecords_dir=self.configs["tfrecords_dir"])

        train_model = create_ctc_train_model(self.model, num_classes=self.text_featurizer.num_classes)
        self._create_checkpoints(train_model)

        self.model.summary()

        initial_epoch = 0
        if self.ckpt_manager.latest_checkpoint:
            initial_epoch = int(self.ckpt_manager.latest_checkpoint.split('-')[-1])
            # restoring the latest checkpoint in checkpoint_path
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)

        train_model.compile(optimizer=self.optimizer, loss={"ctc_loss": lambda y_true, y_pred: y_pred})

        callback = [Checkpoint(self.ckpt_manager)]
        if self.configs["log_dir"]:
            if not os.path.exists(self.configs["log_dir"]):
                os.makedirs(self.configs["log_dir"])
            with open(os.path.join(self.configs["log_dir"], "model.json"), "w") as f:
                f.write(self.model.to_json())
            callback.append(TimeHistory(os.path.join(self.configs["log_dir"], "time.txt")))
            callback.append(tf.keras.callbacks.TensorBoard(log_dir=self.configs["log_dir"], update_freq="batch",
                                                           write_graph=True, histogram_freq=1))

        if tf_eval_dataset is not None:
            if initial_epoch == 0 and self.configs["sortagrad"]:
                train_model.fit(x=tf_train_dataset_sortagrad, epochs=1,
                                validation_data=tf_eval_dataset, shuffle="batch",
                                initial_epoch=initial_epoch, callbacks=callback)
                initial_epoch = 1

            train_model.fit(x=tf_train_dataset, epochs=self.configs["num_epochs"],
                            validation_data=tf_eval_dataset, shuffle="batch",
                            initial_epoch=initial_epoch, callbacks=callback)
        else:
            if initial_epoch == 0 and self.configs["sortagrad"]:
                train_model.fit(x=tf_train_dataset_sortagrad, epochs=1, shuffle="batch",
                                initial_epoch=initial_epoch, callbacks=callback)
                initial_epoch = 1

            train_model.fit(x=tf_train_dataset, epochs=self.configs["num_epochs"], shuffle="batch",
                            initial_epoch=initial_epoch, callbacks=callback)

        if model_file:
            self.save_model(model_file)

    def test(self, model_file, output_file_path):
        print("Testing model ...")
        check_key_in_dict(dictionary=self.configs,
                          keys=["test_data_transcript_paths", "tfrecords_dir"])
        test_dataset = Dataset(data_path=self.configs["test_data_transcript_paths"], mode="test")
        msg = self.load_saved_model(model_file)
        if msg:
            raise Exception(msg)

        tf_test_dataset = test_dataset(text_featurizer=self.text_featurizer,
                                       speech_conf=self.configs["speech_conf"],
                                       batch_size=self.configs["batch_size"],
                                       sortagrad=False, builtin=False, fext=True,
                                       tfrecords_dir=self.configs["tfrecords_dir"])

        def test_step(features, inp_length, transcripts):
            predictions = self.predict(features, inp_length)
            predictions = bytes_to_string(predictions.numpy())

            transcripts = self.decoder.convert_to_string(transcripts)

            b_wer = 0.0
            b_wer_count = 0.0
            b_cer = 0.0
            b_cer_count = 0.0

            for idx, decoded in enumerate(predictions):
                print(f"Pred: {decoded}")
                print(f"Groundtruth: {transcripts[idx]}")
                _wer, _wer_count = wer(decode=decoded, target=transcripts[idx])
                _cer, _cer_count = cer(decode=decoded, target=transcripts[idx])
                b_wer += _wer
                b_cer += _cer
                b_wer_count += _wer_count
                b_cer_count += _cer_count

            return b_wer, b_wer_count, b_cer, b_cer_count

        total_wer = 0.0
        wer_count = 0.0
        total_cer = 0.0
        cer_count = 0.0

        for feature, input_length, label, _ in tf_test_dataset:
            batch_wer, batch_wer_count, batch_cer, batch_cer_count = test_step(feature, input_length, label)
            total_wer += batch_wer
            total_cer += batch_cer
            wer_count += batch_wer_count
            cer_count += batch_cer_count

        results = (total_wer / wer_count, total_cer / cer_count)

        print(f"WER: {results[0]}, CER: {results[-1]}")

        with open(output_file_path, "w", encoding="utf-8") as of:
            of.write("WER: " + str(results[0]) + "\n")
            of.write("CER: " + str(results[-1]) + "\n")

    def test_with_noise_filter(self, model_file, output_file_path):
        print("Testing model ...")
        if not self.noise_filter:
            raise ValueError("noise_filter must be defined")

        check_key_in_dict(dictionary=self.configs,
                          keys=["test_data_transcript_paths", "tfrecords_dir"])
        test_dataset = Dataset(data_path=self.configs["test_data_transcript_paths"], mode="test")
        msg = self.load_saved_model(model_file)
        if msg:
            raise Exception(msg)

        tf_test_dataset = test_dataset(text_featurizer=self.text_featurizer,
                                       speech_conf=self.configs["speech_conf"],
                                       batch_size=1, sortagrad=False, builtin=False, fext=False,
                                       tfrecords_dir=self.configs["tfrecords_dir"])

        def test_step(signal, label):
            prediction = self.infer_single(signal)
            label = self.decoder.convert_to_string_single(label)

            print(f"Pred: {prediction}")
            print(f"Groundtruth: {label}")
            _wer, _wer_count = wer(decode=prediction, target=label)
            _cer, _cer_count = cer(decode=prediction, target=label)

            return _wer, _wer_count, _cer, _cer_count

        total_wer = 0.0
        wer_count = 0.0
        total_cer = 0.0
        cer_count = 0.0

        for signal, label in tf_test_dataset.as_numpy_iterator():
            batch_wer, batch_wer_count, batch_cer, batch_cer_count = test_step(signal, label)
            total_wer += batch_wer
            total_cer += batch_cer
            wer_count += batch_wer_count
            cer_count += batch_cer_count

        results = (total_wer / wer_count, total_cer / cer_count)

        print(f"WER: {results[0]}, CER: {results[-1]}")

        with open(output_file_path, "w", encoding="utf-8") as of:
            of.write("WER: " + str(results[0]) + "\n")
            of.write("CER: " + str(results[-1]) + "\n")

    def infer(self, input_file_path, model_file, output_file_path):
        print("Infering ...")
        check_key_in_dict(dictionary=self.configs,
                          keys=["tfrecords_dir"])
        msg = self.load_model(model_file)
        if msg:
            raise Exception(msg)
        tf_infer_dataset = Dataset(data_path=input_file_path, mode="infer")
        tf_infer_dataset = tf_infer_dataset(batch_size=self.configs["batch_size"],
                                            text_featurizer=self.text_featurizer,
                                            speech_conf=self.configs["speech_conf"],
                                            sortagrad=False, builtin=False, fext=True,
                                            tfrecords_dir=self.configs["tfrecords_dir"])

        def infer_step(feature, input_length):
            prediction = self.predict(feature, input_length)
            return bytes_to_string(prediction.numpy())

        for features, inp_length in tf_infer_dataset:
            predictions = infer_step(features, inp_length)

            with open(output_file_path, "a", encoding="utf-8") as of:
                of.write("Predictions\n")
                for pred in predictions:
                    of.write(pred + "\n")

    def infer_single(self, signal):
        if self.noise_filter:
            signal = self.noise_filter.generate(signal)
        features = speech_feature_extraction(signal, self.configs["speech_conf"])
        input_length = tf.cast(tf.shape(features)[0], tf.int32)
        pred = self.predict(tf.expand_dims(features, 0), tf.expand_dims(input_length, 0))
        return bytes_to_string(pred.numpy())[0]

    def infer_single_interpreter(self, signal, length):
        # Inference only on "length" seconds due to limitation of tflite
        features = speech_feature_extraction(signal, self.configs["speech_conf"])
        input_length = tf.cast(tf.shape(features)[0], tf.int32)
        features = tf.expand_dims(features, 0)
        length = compute_time_dim(self.configs["speech_conf"], length)
        if input_length < length:
            features = tf.pad(features, paddings=[[0, 0], [0, length - input_length], [0, 0], [0, 0]])
        elif input_length > length:
            features = features[:, :length, :, :]

        input_index = self.model.get_input_details()[0]["index"]
        output_index = self.model.get_output_details()[0]["index"]

        self.model.set_tensor(input_index, features)
        self.model.invoke()

        pred = self.model.get_tensor(output_index)

        pred = self.decoder(probs=pred, input_length=input_length)

        return bytes_to_string(pred.numpy())[0]

    def load_interpreter(self, tflite_file):
        try:
            self.model = tf.lite.Interpreter(model_path=str(tflite_file))
            self.model.allocate_tensors()
        except ValueError as e:
            return f"Model is not trained: {e}"
        return None

    def load_model(self, model_file):
        try:
            self.model = tf.keras.models.load_model(model_file)
            print(self.model.summary())
        except Exception as e:
            return f"Model is not trained: {e}"
        return None

    def load_saved_model(self, model_file):
        try:
            self.model = tf.saved_model.load(model_file)
        except Exception as e:
            return f"Model is not trained: {e}"
        return None

    def load_model_from_weights(self, model_file):
        try:
            self.model.load_weights(model_file)
        except Exception as e:
            return f"Model is not trained: {e}"
        return None

    @tf.function
    def predict(self, feature, input_length):
        logits = self.model(feature, training=False)
        return self.decoder(probs=logits, input_length=input_length)

    def save_model(self, model_file):
        print("Saving whole ASR model ...")
        self.model.save(model_file)

    def save_from_checkpoint(self, model_file, idx, is_builtin=False):
        if is_builtin:
            train_model = create_ctc_train_model(self.model, num_classes=self.text_featurizer.num_classes)
        else:
            train_model = self.model
        self._create_checkpoints(train_model)
        if len(self.ckpt_manager.checkpoints) <= 0:
            raise ValueError("No checkpoint to save from")
        if idx == -1:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
        else:
            self.ckpt.restore(self.ckpt_manager.checkpoints[idx])
        self.save_model(model_file)

    def save_weights(self, model_file):
        print("Saving ASR model's weights ...")
        self.model.save_weights(model_file)

    def convert_to_tflite(self, model_file, length, output_file_path):
        # Convert model to tflite with input [1, num frames of "length" seconds, f, c]
        if os.path.exists(output_file_path):
            return
        model = tf.saved_model.load(model_file)
        concrete_func = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
        f, c = compute_feature_dim(speech_conf=self.configs["speech_conf"])
        concrete_func.inputs[0].set_shape([1, compute_time_dim(self.configs["speech_conf"], length), f, c])
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        # converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir=args.export_file)
        converter.experimental_new_converter = True
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()

        tflite_model_dir = pathlib.Path(os.path.dirname(output_file_path))
        tflite_model_dir.mkdir(exist_ok=True, parents=True)

        tflite_model_file = tflite_model_dir / f"{os.path.basename(output_file_path)}"
        tflite_model_file.write_bytes(tflite_model)
