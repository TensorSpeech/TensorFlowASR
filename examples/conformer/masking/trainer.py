import tensorflow as tf

from masking import create_padding_mask
from tensorflow_asr.runners.transducer_runners import TransducerTrainer, TransducerTrainerGA
from tensorflow_asr.losses.rnnt_losses import rnnt_loss


class TrainerWithMasking(TransducerTrainer):
    @tf.function(experimental_relax_shapes=True)
    def _train_step(self, batch):
        _, features, input_length, labels, label_length, pred_inp = batch

        mask = create_padding_mask(features, input_length, self.model.time_reduction_factor)

        with tf.GradientTape() as tape:
            logits = self.model([features, pred_inp], training=True, mask=mask)
            tape.watch(logits)
            per_train_loss = rnnt_loss(
                logits=logits, labels=labels, label_length=label_length,
                logit_length=(input_length // self.model.time_reduction_factor),
                blank=self.text_featurizer.blank
            )
            train_loss = tf.nn.compute_average_loss(per_train_loss,
                                                    global_batch_size=self.global_batch_size)

        gradients = tape.gradient(train_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_metrics["transducer_loss"].update_state(per_train_loss)


class TrainerWithMaskingGA(TransducerTrainerGA):
    @tf.function(experimental_relax_shapes=True)
    def _train_step(self, batch):
        _, features, input_length, labels, label_length, pred_inp = batch

        mask = create_padding_mask(features, input_length, self.model.time_reduction_factor)

        with tf.GradientTape() as tape:
            logits = self.model([features, pred_inp], training=True, mask=mask)
            tape.watch(logits)
            per_train_loss = rnnt_loss(
                logits=logits, labels=labels, label_length=label_length,
                logit_length=(input_length // self.model.time_reduction_factor),
                blank=self.text_featurizer.blank
            )
            train_loss = tf.nn.compute_average_loss(
                per_train_loss,
                global_batch_size=self.global_batch_size
            )

        gradients = tape.gradient(train_loss, self.model.trainable_variables)
        self.accumulation.accumulate(gradients)
        self.train_metrics["transducer_loss"].update_state(per_train_loss)
